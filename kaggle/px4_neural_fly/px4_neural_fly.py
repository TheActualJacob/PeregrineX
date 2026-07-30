"""Fly the PX4 neural controller (mc_nn_control) headless in SIH and log it.

Attaches the build kernel's output (px4_build.tar.gz via kernel_sources).
Everything mode/param-specific is DISCOVERED at runtime -- airframe number,
MC_NN params, module status -- because mainline moves and the docs lag.

Flight: SIH quad boot -> arm -> takeoff -> hover -> try to engage the neural
mode -> offboard square (setpoints feed whatever controller is active) ->
land -> disarm. Telemetry sampled over MAVLink throughout.

Output (/kaggle/working):
  console.txt   -- full pxh console (the ground truth for what ran)
  recon.txt     -- commander/module/param status probes
  flight.json   -- phase-by-phase telemetry summary (alt, tracking error)
  *.ulg         -- PX4 ulog(s) from the run
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BUILD_INPUT = Path("/kaggle/input/aviationsim-px4-neural-build")
WORK = Path("/kaggle/working")
RUN = Path("/tmp/px4_run")
BOARD = "px4_sitl_neural"
TAKEOFF_ALT_M = 5.0


def sh(cmd: str, timeout: int = 60) -> str:
    print(f"+ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True,
                       timeout=timeout, cwd=RUN)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[:2000], flush=True)
    return out


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymavlink"],
                   check=True)
    from pymavlink import mavutil

    # -- locate + unpack build ------------------------------------------------
    # Kernel-source outputs do not always mount at /kaggle/input/<slug>/<file>;
    # discover instead of assuming.
    mounts = sorted(str(p) for p in Path("/kaggle/input").rglob("*")
                    if p.is_file())[:200]
    print("\n".join(mounts[:40]))
    tarball = next((Path(m) for m in mounts if m.endswith("px4_build.tar.gz")),
                   None)
    assert tarball, f"px4_build.tar.gz not found under /kaggle/input: {mounts}"
    global BUILD_INPUT
    BUILD_INPUT = tarball.parent
    base = Path("/tmp/px4")
    base.mkdir(exist_ok=True)
    subprocess.run(["tar", "xzf", str(tarball), "-C", str(base)], check=True)
    px4 = base / f"build/{BOARD}/bin/px4"
    etc = base / f"build/{BOARD}/etc"
    commit = (BUILD_INPUT / "PX4_COMMIT").read_text().strip()
    print(f"commit {commit}")

    af_dir = etc / "init.d-posix/airframes"
    sih = next((p.name for p in sorted(af_dir.glob("*sihsim_quadx"))), None)
    assert sih, f"no sihsim_quadx airframe in {af_dir}"
    autostart = sih.split("_")[0]
    print(f"airframe {sih} -> PX4_SYS_AUTOSTART={autostart}")

    # -- boot -----------------------------------------------------------------
    RUN.mkdir(exist_ok=True)
    env = dict(os.environ, PX4_SYS_AUTOSTART=autostart,
               PX4_SIMULATOR="sihsim", HEADLESS="1")
    # px4 client tools (px4-commander etc.) resolve the instance from the cwd
    os.environ.update(PX4_SYS_AUTOSTART=autostart)
    console = (WORK / "console.txt").open("w")
    # -d (daemon): no pxh console. Without it, stdin at EOF makes the shell
    # spin re-printing its prompt -- 191 MB of "pxh> " in the build kernel's
    # 60 s boot probe.
    proc = subprocess.Popen([str(px4), "-d", str(etc),
                             "-s", "etc/init.d-posix/rcS"],
                            cwd=RUN, env=env, stdout=console,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    path_hint = base / f"build/{BOARD}/bin"
    os.environ["PATH"] = f"{path_hint}:{os.environ['PATH']}"

    conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
    hb = conn.wait_heartbeat(timeout=90)
    assert hb, "no MAVLink heartbeat from PX4"
    print(f"heartbeat: sysid {hb.get_srcSystem()}")
    time.sleep(10)  # let EKF settle

    # -- recon ----------------------------------------------------------------
    recon = []
    for cmd in ("px4-commander status", "px4-mc_nn_control status",
                "px4-param show MC_NN*", "px4-param show SIH_*",
                "px4-commander check"):
        try:
            recon.append(f"$ {cmd}\n{sh(cmd)}")
        except Exception as exc:  # noqa: BLE001
            recon.append(f"$ {cmd}\nEXC {exc}")
    (WORK / "recon.txt").write_text("\n".join(recon))

    summary: dict = {"commit": commit, "airframe": sih, "phases": {}}

    def rel_alt(timeout=2.0):
        msg = conn.recv_match(type="GLOBAL_POSITION_INT", blocking=True,
                              timeout=timeout)
        return msg.relative_alt / 1000.0 if msg else None

    def wait_alt(target, cmp, deadline_s):
        t_end = time.time() + deadline_s
        alt = None
        while time.time() < t_end:
            a = rel_alt()
            if a is not None:
                alt = a
                if cmp(a, target):
                    return alt
        return alt

    # -- arm + takeoff (with retries while preflight settles) -----------------
    took_off = False
    for attempt in range(6):
        sh("px4-commander arm -f", timeout=30)
        sh(f"px4-commander takeoff", timeout=30)
        alt = wait_alt(TAKEOFF_ALT_M * 0.5, lambda a, t: a >= t, 25)
        if alt and alt >= TAKEOFF_ALT_M * 0.5:
            took_off = True
            break
        print(f"takeoff attempt {attempt}: alt={alt}")
        time.sleep(10)
    summary["phases"]["takeoff"] = {"ok": took_off, "alt_m": rel_alt()}
    assert took_off, "never left the ground -- see console.txt"
    time.sleep(8)

    # -- hover telemetry sampler ----------------------------------------------
    def hover_stats(seconds: float) -> dict:
        xs, ys, zs = [], [], []
        t_end = time.time() + seconds
        while time.time() < t_end:
            pos = conn.recv_match(type="LOCAL_POSITION_NED", blocking=True,
                                  timeout=1.0)
            if pos:
                xs.append(pos.x); ys.append(pos.y); zs.append(pos.z)
        if not xs:
            return {"n": 0}
        import statistics as st
        return {"n": len(xs),
                "drift_m": round(((xs[-1] - xs[0]) ** 2
                                  + (ys[-1] - ys[0]) ** 2) ** 0.5, 3),
                "xy_std_m": round((st.pstdev(xs) ** 2
                                   + st.pstdev(ys) ** 2) ** 0.5, 3),
                "z_std_m": round(st.pstdev(zs), 3),
                "mean_alt_m": round(-sum(zs) / len(zs), 2)}

    # baseline: 15 s hover under the CLASSICAL stack
    summary["phases"]["hover_classical"] = hover_stats(15)
    print("classical hover:", summary["phases"]["hover_classical"])

    # -- engage NeuralControl -------------------------------------------------
    # The module auto-starts and registers "NeuralControl" (mode_id 23 in the
    # build kernel's boot probe) -- the first external-mode slot, i.e. ext1.
    engage_log = [sh("px4-mc_nn_control status", timeout=20)]
    engaged = False
    for attempt in ("ext1", "ext 1", "23"):
        out_ = sh(f"px4-commander mode {attempt}", timeout=20)
        engage_log.append(out_)
        status = sh("px4-commander status", timeout=20)
        engage_log.append(status)
        if "NeuralControl" in status or "ext" in status.lower():
            engaged = True
            break
        sh("px4-param set MC_NN_EN 1", timeout=20)
    (WORK / "recon.txt").open("a").write("\n== engage ==\n" + "\n".join(engage_log))
    summary["phases"]["engage"] = {"engaged": engaged}
    print(f"neural mode engaged: {engaged}")

    # -- the actual event: the network holding the aircraft in the air --------
    summary["phases"]["hover_neural"] = hover_stats(30)
    print("neural hover:", summary["phases"]["hover_neural"])

    # best-effort reposition under the neural mode (may be rejected; logged)
    conn.mav.command_long_send(
        hb.get_srcSystem(), 1, mavutil.mavlink.MAV_CMD_DO_REPOSITION, 0,
        -1, 1, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    ack = conn.recv_match(type="COMMAND_ACK", blocking=True, timeout=5)
    summary["phases"]["reposition_ack"] = getattr(ack, "result", None)
    summary["phases"]["hover_neural_2"] = hover_stats(15)
    status_after = sh("px4-commander status", timeout=20)
    summary["phases"]["nav_state_line"] = [
        ln for ln in status_after.splitlines() if "nav" in ln.lower()][:3]

    # -- land -----------------------------------------------------------------
    sh("px4-commander land", timeout=30)
    landed_alt = wait_alt(0.3, lambda a, t: a <= t, 60)
    summary["phases"]["land"] = {"alt_m": landed_alt}
    time.sleep(5)
    proc.kill()
    proc.wait()
    console.close()

    # -- collect logs ---------------------------------------------------------
    for ulg in glob.glob(str(RUN / "**/*.ulg"), recursive=True):
        shutil.copy2(ulg, WORK / Path(ulg).name)
        print(f"ulog: {Path(ulg).name}")
    txt = (WORK / "console.txt").read_text()
    summary["nn_console_mentions"] = txt.lower().count("nn")
    (WORK / "flight.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
