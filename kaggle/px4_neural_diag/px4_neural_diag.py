"""Diagnostic: which adversarial levers does SIH actually honor?

The scenario probe showed identical margins across mass 0.5-2.5 kg and a
"killed" motor, which means either the perturbations never reached the plant
or the oracle could not see them. This run answers, empirically:

  A) pre-boot params  -- set SIH_MASS in a throwaway boot (persists to
     parameters.bson in cwd), reboot, hover. Does hover behavior change?
  B) live params      -- step SIH_MASS mid-hover. Does the aircraft sag?
  C) motor failure    -- `failure motor off -i N` mid-hover. Any effect?
  D) thrust ceiling   -- step SIH_F_T_MAX down mid-hover (degradation proxy).
  E) hold altitude    -- is MIS_TAKEOFF_ALT honored? what is the real setpoint?

Each test hovers, applies one lever, and reports the position/altitude
response so we can see which levers move the aircraft at all. Both arms
(classical Hold and neural ext1) for the levers that bite.

Output: /kaggle/working/diag.json + diag_log.txt
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
RUN = Path("/tmp/px4_run")
BOARD = "px4_sitl_neural"
SPEED = 5
LOG = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


def sh(cmd: str, timeout: int = 40) -> str:
    """Never raises. A hung px4 client tool must not end the diagnostic."""
    try:
        r = subprocess.run(cmd, shell=True, text=True, capture_output=True,
                           timeout=timeout, cwd=RUN)
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return f"__TIMEOUT__({cmd})"
    except Exception as exc:  # noqa: BLE001
        return f"__EXC__({type(exc).__name__})"


def reap() -> None:
    """Kill every px4 and clear the daemon's IPC leftovers.

    Rebooting in a directory a killed px4 was using leaves a stale daemon
    socket, and the next `px4-commander` blocks on it forever (this ended the
    first diagnostic run). Params still persist: parameters.bson is copied
    forward explicitly instead of reusing the directory.
    """
    subprocess.run("pkill -9 -f 'bin/px4' || true", shell=True)
    time.sleep(2)
    # ONLY the daemon's IPC endpoints. Globbing /tmp/px4* would delete the
    # unpacked build (/tmp/px4build) and the run dir (/tmp/px4_run).
    subprocess.run("rm -f /tmp/px4-* /tmp/pxh-* 2>/dev/null || true",
                   shell=True)


class Vehicle:
    """One PX4 boot in RUN (params in RUN persist across boots)."""

    def __init__(self, px4, etc, autostart, mavutil, fresh_dir: bool,
                 carry_params: bool = False):
        self.mavutil = mavutil
        reap()
        saved = None
        if carry_params and (RUN / "parameters.bson").is_file():
            saved = (RUN / "parameters.bson").read_bytes()
        if fresh_dir or carry_params:
            shutil.rmtree(RUN, ignore_errors=True)
        RUN.mkdir(parents=True, exist_ok=True)
        if saved is not None:
            (RUN / "parameters.bson").write_bytes(saved)
        env = dict(os.environ, PX4_SYS_AUTOSTART=autostart,
                   PX4_SIMULATOR="sihsim", HEADLESS="1",
                   PX4_SIM_SPEED_FACTOR=str(SPEED))
        self.proc = subprocess.Popen(
            [str(px4), "-d", str(etc), "-s", "etc/init.d-posix/rcS"],
            cwd=RUN, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        self.conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
        assert self.conn.wait_heartbeat(timeout=60), "no heartbeat"
        self.conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                     mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                                     0, 0, 0)
        for mid in (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,):
            self.conn.mav.command_long_send(
                self.conn.target_system or 1, 1,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mid, 100000, 0, 0, 0, 0, 0)
        assert self.conn.recv_match(type="LOCAL_POSITION_NED", blocking=True,
                                    timeout=20), "no position stream"

    def sample(self, seconds: float) -> dict:
        xs, ys, zs = [], [], []
        t_end = time.time() + seconds / SPEED + 3
        while time.time() < t_end:
            p = self.conn.recv_match(type="LOCAL_POSITION_NED", blocking=True,
                                     timeout=2.0)
            if p:
                xs.append(p.x); ys.append(p.y); zs.append(-p.z)
        if not zs:
            return {"n": 0}
        dev = [((x - xs[0]) ** 2 + (y - ys[0]) ** 2) ** 0.5
               for x, y in zip(xs, ys)]
        return {"n": len(zs), "alt_mean": round(sum(zs) / len(zs), 2),
                "alt_min": round(min(zs), 2), "alt_max": round(max(zs), 2),
                "max_dev": round(max(dev), 3)}

    def takeoff(self, target: float) -> float:
        sh("px4-param set SYS_FAILURE_EN 1")
        best = -99.0
        for _ in range(4):
            sh("px4-commander arm -f")
            sh("px4-commander takeoff")
            t_end = time.time() + 30 / SPEED + 12
            while time.time() < t_end:
                p = self.conn.recv_match(type="LOCAL_POSITION_NED",
                                         blocking=True, timeout=2.0)
                if p:
                    best = max(best, -p.z)
                    if best >= target:
                        return best
        return best

    def close(self):
        try:
            self.conn.close()
        finally:
            self.proc.kill()
            self.proc.wait()
            reap()


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymavlink"],
                   check=True)
    from pymavlink import mavutil

    tb = next(Path("/kaggle/input").rglob("px4_build.tar.gz"))
    base = Path("/tmp/px4build"); base.mkdir(exist_ok=True)
    subprocess.run(["tar", "xzf", str(tb), "-C", str(base)], check=True)
    px4 = base / f"build/{BOARD}/bin/px4"
    etc = base / f"build/{BOARD}/etc"
    os.environ["PATH"] = f"{base / f'build/{BOARD}/bin'}:{os.environ['PATH']}"
    autostart = next((etc / "init.d-posix/airframes")
                     .glob("*sihsim_quadx")).name.split("_")[0]
    out: dict = {}

    def guarded(name, fn):
        """One test failing must never end the diagnostic."""
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            reap()
        log(f"[{name}] {out[name]}")

    # -- E + reference: default plant, both arms ------------------------------
    def reference(arm):
        v = Vehicle(px4, etc, autostart, mavutil, fresh_dir=True)
        try:
            sh("px4-param set MIS_TAKEOFF_ALT 5.0")
            alt = v.takeoff(3.5)
            engaged = None
            if arm == "neural":
                engaged = sh("px4-commander mode ext1").strip()[:120]
            time.sleep(3)
            return {**v.sample(20), "alt_after_takeoff": round(alt, 2),
                    "engage": engaged}
        finally:
            v.close()

    for arm in ("classical", "neural"):
        guarded(f"ref_{arm}", lambda a=arm: reference(a))

    # -- A: pre-boot mass. Set params, reboot carrying parameters.bson --------
    def preboot_mass():
        v = Vehicle(px4, etc, autostart, mavutil, fresh_dir=True)
        try:
            sh("px4-param set SIH_MASS 2.5")
            sh("px4-param set MIS_TAKEOFF_ALT 5.0")
            sh("px4-param save")
            before = sh("px4-param show SIH_MASS").strip()[-60:]
        finally:
            v.close()
        v = Vehicle(px4, etc, autostart, mavutil, fresh_dir=False,
                    carry_params=True)
        try:
            after = sh("px4-param show SIH_MASS").strip()[-60:]
            alt = v.takeoff(3.5)
            time.sleep(3)
            return {**v.sample(20), "alt_after_takeoff": round(alt, 2),
                    "readback_before_reboot": before,
                    "readback_after_reboot": after}
        finally:
            v.close()

    guarded("A_preboot_mass2.5", preboot_mass)

    # -- B/C/D: live levers, applied mid-hover on a default plant ------------
    def lever(cmd):
        v = Vehicle(px4, etc, autostart, mavutil, fresh_dir=True)
        try:
            sh("px4-param set MIS_TAKEOFF_ALT 5.0")
            alt = v.takeoff(3.5)
            time.sleep(3)
            before = v.sample(10)
            resp = sh(cmd).strip()[:200]
            after = v.sample(20)
            return {"before": before, "after": after, "cmd_output": resp,
                    "alt_after_takeoff": round(alt, 2),
                    "alt_drop": round(before.get("alt_mean", 0)
                                      - after.get("alt_min", 0), 2)}
        finally:
            v.close()

    for name, cmd in {
        "B_live_mass_3.0": "px4-param set SIH_MASS 3.0",
        "C_motor_fail": "px4-failure motor off -i 3",
        "D_thrust_ceiling_0.5": "px4-param set SIH_F_T_MAX 1.0",
    }.items():
        guarded(name, lambda c=cmd: lever(c))

    (WORK / "diag.json").write_text(json.dumps(out, indent=2))
    (WORK / "diag_log.txt").write_text("\n".join(LOG))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
