"""Build PX4 with the mainline neural-network controller (mc_nn_control) on Kaggle.

The board config `px4/sitl/neural` builds SITL with the TFLite-Micro neural
position controller compiled in -- an autopilot whose control law is a learned
network, not PID. Docs: https://docs.px4.io/main/en/neural_networks/

This kernel (CPU, internet ON):
  1. clones PX4-Autopilot main + submodules, records the commit
  2. installs build deps (Tools/setup/ubuntu.sh --no-sim-tools --no-nuttx)
  3. `make px4_sitl_neural`
  4. explores what shipped: sihsim airframes, neural module files, params
  5. attempts a first headless SIH boot (60 s, logged) -- best-effort recon
     for the flight kernel; failure here is data, not fatal
  6. packages build/px4_sitl_neural as output for reuse by flight kernels

Output (/kaggle/working):
  px4_build.tar.gz  -- the built tree (bin + ROMFS etc), attach to fly
  PX4_COMMIT        -- pinned commit for every later run to cite
  explore.txt       -- airframes/modules/params recon
  boot_attempt.txt  -- pxh console output from the SIH boot try
"""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SRC = Path("/tmp/PX4-Autopilot")  # /tmp: NOT kernel output (that ships everything)
BOARD_TARGET = "px4_sitl_neural"


def run(cmd, check=True, timeout=None, capture=False, **kw):
    print(f"+ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check,
                          timeout=timeout, text=True,
                          capture_output=capture, **kw)


def main() -> None:
    t0 = time.time()
    run("lsb_release -a || cat /etc/os-release | head -3")

    # -- source ---------------------------------------------------------------
    run(["git", "clone", "--recursive", "--shallow-submodules", "--depth", "1",
         "https://github.com/PX4/PX4-Autopilot.git", str(SRC)])
    commit = run(["git", "rev-parse", "HEAD"], cwd=SRC, capture=True).stdout.strip()
    (WORK / "PX4_COMMIT").write_text(commit + "\n")
    print(f"PX4 commit: {commit}")

    # -- deps -----------------------------------------------------------------
    run(f"bash {SRC}/Tools/setup/ubuntu.sh --no-sim-tools --no-nuttx",
        timeout=1200)

    # -- build ----------------------------------------------------------------
    run(f"ls {SRC}/boards/px4/sitl/")
    run(["make", BOARD_TARGET, f"-j{os.cpu_count()}"], cwd=SRC, timeout=3000)
    px4_bin = SRC / f"build/{BOARD_TARGET}/bin/px4"
    assert px4_bin.is_file(), "build produced no px4 binary"
    print(f"built in {(time.time() - t0) / 60:.1f} min")

    # -- recon: what did we get? ---------------------------------------------
    lines = [f"commit {commit}"]
    af_dir = SRC / "ROMFS/px4fmu_common/init.d-posix/airframes"
    lines.append("== sihsim / sitl airframes ==")
    for p in sorted(af_dir.glob("*")):
        if "sih" in p.name or "neural" in p.name:
            lines.append(p.name)
    lines.append("== neural module files ==")
    mod = SRC / "src/modules/mc_nn_control"
    for p in sorted(mod.rglob("*")):
        if p.is_file() and p.suffix in (".c", ".cpp", ".hpp", ".yaml", ".px4b",
                                        ".tflite", ""):
            lines.append(str(p.relative_to(SRC)))
    lines.append("== MC_NN params ==")
    r = run(f"grep -rh 'MC_NN' {mod} --include='*.c' --include='*.yaml' | head -40",
            check=False, capture=True)
    lines.append(r.stdout or "(none found)")
    lines.append("== board config ==")
    for cand in (SRC / "boards/px4/sitl").glob("neural*"):
        lines.append(f"--- {cand.name} ---")
        lines.append(cand.read_text())
    (WORK / "explore.txt").write_text("\n".join(lines))
    print("\n".join(lines[:40]))

    # -- best-effort first boot: SIH quad, headless, 60 s ---------------------
    # sihsim airframe number discovered from the airframes dir at runtime.
    sih_quad = next((p.name for p in af_dir.glob("*sihsim_quadx")), None)
    boot_log = WORK / "boot_attempt.txt"
    if sih_quad:
        autostart = sih_quad.split("_")[0]
        env = dict(os.environ,
                   PX4_SYS_AUTOSTART=autostart,
                   PX4_SIMULATOR="sihsim",
                   HEADLESS="1")
        rundir = Path("/tmp/px4_run")
        rundir.mkdir(exist_ok=True)
        print(f"boot attempt: airframe {sih_quad} (autostart {autostart})")
        with boot_log.open("w") as fh:
            proc = subprocess.Popen(
                [str(px4_bin), str(SRC / f"build/{BOARD_TARGET}/etc"),
                 "-s", "etc/init.d-posix/rcS"],
                cwd=rundir, env=env, stdout=fh, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL)
            time.sleep(60)
            proc.kill()
            proc.wait()
        txt = boot_log.read_text()
        print(f"boot log: {len(txt)} bytes; "
              f"nn mentions: {txt.lower().count('nn_control')}, "
              f"ready: {'Ready for takeoff' in txt}")
    else:
        boot_log.write_text("no sihsim_quadx airframe found\n")
        print("no sihsim_quadx airframe found -- see explore.txt")

    # -- package the build for flight kernels ---------------------------------
    with tarfile.open(WORK / "px4_build.tar.gz", "w:gz") as tar:
        tar.add(SRC / f"build/{BOARD_TARGET}/bin",
                arcname=f"build/{BOARD_TARGET}/bin")
        tar.add(SRC / f"build/{BOARD_TARGET}/etc",
                arcname=f"build/{BOARD_TARGET}/etc")
    size = (WORK / "px4_build.tar.gz").stat().st_size / 1e6
    print(f"px4_build.tar.gz: {size:.0f} MB; total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
