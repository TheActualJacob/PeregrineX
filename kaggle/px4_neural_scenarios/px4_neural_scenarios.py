"""Adversarial scenario runs against PX4's neural controller (mc_nn_control).

Each scenario perturbs the SIH plant away from the network's training regime
and optionally kills a motor mid-hold:

  ixx_mul     x SIH_IXX/IYY   0.5 - 3.0   inertia mismatch      (static)
  kdv         = SIH_KDV       0.0 - 3.0   linear drag           (static)
  mass_mul    = SIH_MASS      0.5 - 2.5   mass step             (at t+10 s)
  tmax_mul    x SIH_F_T_MAX   0.5 - 1.2   thrust ceiling step   (at t+10 s)
  motor_fail  none | 0-3      `failure motor off -i N`          (at t+10 s)
                              NOTE: recorded but INERT - the shakeout showed
                              PX4 accepts the command while the SIH backend
                              does not implement it, so it changes nothing.
                              Kept as a dim so the null result stays visible.
  arm         "neural" (ext1) | "classical" (Hold)  -- control arm

The step at t+10 s is the adversarial event: the controller is already trimmed
for the original plant when it changes underneath it. Verified lever - the
diagnostic showed SIH_MASS 1->3 mid-hover dropping the aircraft 5 m to the
ground.

Flight: boot -> takeoff to 5 m -> settle -> engage arm's mode -> 40 s hold
(failure injected 10 s in, if any) -> score -> teardown. One fresh PX4 boot
per run so no state leaks between scenarios.

Oracle (margins normalized, 0 = limit, positive = safe, min() wins). Both
terms measure how well the controller HELD what it was asked to hold, scored
only after the aircraft has settled at the setpoint:
  m_pos   = (POS_LIM - max_xy_drift) / POS_LIM       POS_LIM = 3 m
  m_alt   = (ALT_LIM - max_alt_error) / ALT_LIM      ALT_LIM = 2.5 m
  crash (alt < 0.3 m mid-test or no samples): clamp to min(margins, -1)

Output (/kaggle/working):
  results.jsonl  -- one row per run: scenario, arm, metrics, margin
  worst_*.ulg    -- ulogs for the lowest-margin runs (cap WORST_KEEP)

MODE=shakeout (default): 8 LHS scenarios x 2 arms x speed factors (1, 5, 10)
to validate speedup fidelity and measure throughput.
MODE=baseline: N_BASE LHS scenarios x 2 arms at the shakeout-validated speed.
Set via the MODE line below (kernel push == run; edit + push to switch).
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

MODE = "pair"              # "probe" | "shakeout" | "baseline" | "pair"
N_BASE = 400               # baseline scenarios (x2 arms); ~11 s/run => ~2.5 h
BASE_SPEED = 10            # validated by shakeout before use
SEED = 42
WORST_KEEP = 12

WORK = Path("/kaggle/working")
RUN = Path("/tmp/px4_run")
BOARD = "px4_sitl_neural"
TAKEOFF_ALT = 5.0
POS_LIM = 3.0    # lateral drift from the hold point that counts as failure
ALT_LIM = 2.5    # altitude error against the 5 m setpoint at the limit
HOLD_S, FAIL_AT_S = 40.0, 10.0

DIMS = {   # name: (lo, hi)
    "mass_mul": (0.5, 2.5), "ixx_mul": (0.5, 3.0),
    "tmax_mul": (0.5, 1.2), "kdv": (0.0, 3.0),
}


def lhs(n: int, seed: int) -> list[dict]:
    import numpy as np
    rng = np.random.default_rng(seed)
    grid = (rng.permuted(np.arange(n).reshape(-1, 1).repeat(len(DIMS) + 1, 1),
                         axis=0) + rng.random((n, len(DIMS) + 1))) / n
    out = []
    for i in range(n):
        s = {k: round(lo + grid[i, j] * (hi - lo), 4)
             for j, (k, (lo, hi)) in enumerate(DIMS.items())}
        # last coordinate: motor failure on ~40% of scenarios, motor 0-3
        u = grid[i, len(DIMS)]
        s["motor_fail"] = int(u * 10) % 4 if u < 0.4 else None
        s["scenario_id"] = i
        out.append(s)
    return out


def sh(cmd: str, timeout: int = 60) -> str:
    """Never raises: a hung px4 client tool must not end the batch."""
    try:
        r = subprocess.run(cmd, shell=True, text=True, capture_output=True,
                           timeout=timeout, cwd=RUN)
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return f"__TIMEOUT__({cmd})"
    except Exception as exc:  # noqa: BLE001
        return f"__EXC__({type(exc).__name__})"


def reap() -> None:
    """Kill every px4 and clear daemon IPC endpoints.

    Rebooting over a killed px4's socket makes the next px4-commander block
    forever. Globs are narrow on purpose: /tmp/px4* would delete the unpacked
    build and the run dir.
    """
    subprocess.run("pkill -9 -f 'bin/px4' || true", shell=True)
    time.sleep(1.5)
    subprocess.run("rm -f /tmp/px4-* /tmp/pxh-* 2>/dev/null || true",
                   shell=True)


class Keepalive(threading.Thread):
    """PX4 stops streaming to a GCS that goes quiet, and a silent stream reads
    exactly like a failed flight (the diagnostic scored real hovers as zero
    samples). Heartbeat once a second for the life of the run.
    """

    def __init__(self, conn, mavutil):
        super().__init__(daemon=True)
        self.conn, self.mavutil, self.stop = conn, mavutil, threading.Event()

    def run(self):
        while not self.stop.wait(1.0):
            try:
                self.conn.mav.heartbeat_send(
                    self.mavutil.mavlink.MAV_TYPE_GCS,
                    self.mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            except Exception:  # noqa: BLE001 - teardown races are harmless
                return


def fly(px4: Path, etc: Path, autostart: str, scenario: dict, arm: str,
        speed: int, mavutil) -> dict:
    """One boot, one flight, one score. Never raises; failure is data."""
    row = {**scenario, "arm": arm, "speed": speed, "status": "error",
           "margin": -2.0}
    reap()
    shutil.rmtree(RUN, ignore_errors=True)
    RUN.mkdir()
    env = dict(os.environ, PX4_SYS_AUTOSTART=autostart,
               PX4_SIMULATOR="sihsim", HEADLESS="1",
               PX4_SIM_SPEED_FACTOR=str(speed))
    proc = subprocess.Popen(
        [str(px4), "-d", str(etc), "-s", "etc/init.d-posix/rcS"],
        cwd=RUN, env=env, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    conn = None
    t0 = time.time()
    try:
        conn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
        if not conn.wait_heartbeat(timeout=60):
            row["status"] = "no-heartbeat"
            return row
        # PX4 streams position ONLY to an endpoint it has heard from. Driving
        # the vehicle purely through the px4-commander CLI means we never send
        # anything, so every recv_match returns None and a perfectly good
        # flight scores as takeoff-failed. Announce ourselves, then ask for the
        # two streams the oracle needs.
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0)
        sysid = conn.target_system or 1
        for msg_id in (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                       mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT):
            conn.mav.command_long_send(
                sysid, 1, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, 100000, 0, 0, 0, 0, 0)   # 100 ms => 10 Hz
        keepalive = Keepalive(conn, mavutil)
        keepalive.start()
        if conn.recv_match(type="LOCAL_POSITION_NED", blocking=True,
                           timeout=20) is None:
            row["status"] = "no-position-stream"
            return row
        time.sleep(8 / speed + 2)

        sh("px4-param set SYS_FAILURE_EN 1")
        sh(f"px4-param set MIS_TAKEOFF_ALT {TAKEOFF_ALT}")
        # Static plant perturbation, applied pre-takeoff. SIH reads these live
        # (diagnostic: SIH_MASS 1->3 mid-hover dropped the aircraft 5 m), so no
        # reboot dance is needed.
        sh(f"px4-param set SIH_IXX {0.025 * scenario['ixx_mul']:.5f}")
        sh(f"px4-param set SIH_IYY {0.025 * scenario['ixx_mul']:.5f}")
        sh(f"px4-param set SIH_KDV {scenario['kdv']:.4f}")

        def climb_to(target: float, budget_wall_s: float) -> float:
            best = -99.0
            t_end = time.time() + budget_wall_s
            while time.time() < t_end:
                p = conn.recv_match(type="LOCAL_POSITION_NED", blocking=True,
                                    timeout=2.0)
                if p is not None:
                    best = max(best, -p.z)
                    if best >= target:
                        return best
            return best

        alt_reached = -99.0
        for _ in range(4):
            sh("px4-commander arm -f", timeout=30)
            sh("px4-commander takeoff", timeout=30)
            alt_reached = climb_to(TAKEOFF_ALT * 0.7, 30 / speed + 12)
            if alt_reached >= TAKEOFF_ALT * 0.7:
                break
        else:
            row.update(status="takeoff-failed", alt_reached_m=round(alt_reached, 2),
                       wall_s=round(time.time() - t0, 1))
            return row
        time.sleep(6 / speed + 1)

        if arm == "neural":
            sh("px4-commander mode ext1", timeout=20)
        time.sleep(2 / speed + 0.5)

        # Settle before scoring. The probe scored the climb transient, which
        # saturated every margin at the same value and hid the controllers'
        # actual differences.
        settle_end = time.time() + 25 / speed + 8
        stable = 0
        while time.time() < settle_end and stable < 5:
            p = conn.recv_match(type="LOCAL_POSITION_NED", blocking=True,
                                timeout=2.0)
            if p is not None:
                stable = stable + 1 if abs(-p.z - TAKEOFF_ALT) < 0.5 else 0
        row["settled"] = stable >= 5

        # the hold window: sample deviation, inject failure at FAIL_AT_S
        xs, ys, alts = [], [], []
        x0 = y0 = None
        injected = False
        t_hold = time.time()
        sim_t0 = sim_t = None
        wall_budget = HOLD_S / speed + 30
        while time.time() - t_hold < wall_budget:
            pos = conn.recv_match(type="LOCAL_POSITION_NED", blocking=True,
                                  timeout=2.0)
            if pos is None:
                continue
            if x0 is None:
                x0, y0 = pos.x, pos.y
            # elapsed SIM time from the vehicle's own clock, not wall * speed:
            # the speed factor is a request, and measuring it is the point of
            # the probe run.
            if sim_t0 is None:
                sim_t0 = pos.time_boot_ms / 1000.0
            sim_t = pos.time_boot_ms / 1000.0
            elapsed_sim = sim_t - sim_t0
            xs.append(pos.x - x0); ys.append(pos.y - y0); alts.append(-pos.z)
            if not injected and elapsed_sim >= FAIL_AT_S:
                # The adversarial event: a step change to the plant the
                # controller is already trimmed for. mass_mul/tmax_mul are the
                # proven levers (live SIH params); motor_fail additionally
                # exercises PX4's failure-injection path.
                sh(f"px4-param set SIH_MASS {scenario['mass_mul']:.4f}")
                sh(f"px4-param set SIH_F_T_MAX "
                   f"{2.0 * scenario['tmax_mul']:.4f}")
                if scenario["motor_fail"] is not None:
                    sh(f"px4-failure motor off -i "
                       f"{scenario['motor_fail'] + 1}", timeout=20)
                injected = True
            if elapsed_sim >= HOLD_S:
                break
            if alts[-1] < 0.3 and elapsed_sim > 2:
                break  # ground contact mid-test = crash

        if not xs:
            row["status"] = "no-telemetry"
            return row
        max_dev = max((x * x + y * y) ** 0.5 for x, y in zip(xs, ys))
        alt_min = min(alts)
        alt_err = max(abs(a - TAKEOFF_ALT) for a in alts)
        # Margins measure how well the controller HELD what it was asked to
        # hold: lateral drift from the hold point, and altitude error against
        # the 5 m setpoint. Both normalized so 0 = at the limit.
        m_pos = (POS_LIM - max_dev) / POS_LIM
        m_alt = (ALT_LIM - alt_err) / ALT_LIM
        margin = min(m_pos, m_alt)
        crashed = alt_min < 0.3
        if crashed:
            margin = min(margin, -1.0)
        hold_wall = time.time() - t_hold
        hold_sim = (sim_t - sim_t0) if sim_t0 is not None else 0.0
        row.update(status="crashed" if crashed else "held",
                   injected=injected, max_dev_m=round(max_dev, 3),
                   alt_err_m=round(alt_err, 3),
                   alt_min_m=round(alt_min, 3), n_samples=len(xs),
                   margin=round(margin, 4),
                   alt_reached_m=round(alt_reached, 2),
                   hold_sim_s=round(hold_sim, 1),
                   hold_wall_s=round(hold_wall, 1),
                   measured_speedup=round(hold_sim / max(hold_wall, 1e-6), 2),
                   wall_s=round(time.time() - t0, 1))
        return row
    except Exception as exc:  # noqa: BLE001 - one bad run must not kill the batch
        row["status"] = f"exc:{type(exc).__name__}"
        return row
    finally:
        try:
            keepalive.stop.set()
        except NameError:
            pass
        if conn is not None:
            conn.close()
        proc.kill()
        proc.wait()
        reap()
        # keep this run's ulog only until scored; caller prunes
        for ulg in glob.glob(str(RUN / "**/*.ulg"), recursive=True):
            dest = WORK / f"run{row.get('scenario_id', 'x')}_{arm}.ulg"
            shutil.copy2(ulg, dest)
            row["ulog"] = dest.name


def main() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "pymavlink", "numpy"], check=True)
    from pymavlink import mavutil

    mounts = [str(p) for p in Path("/kaggle/input").rglob("px4_build.tar.gz")]
    assert mounts, "px4_build.tar.gz not found in inputs"
    base = Path("/tmp/px4")
    base.mkdir(exist_ok=True)
    subprocess.run(["tar", "xzf", mounts[0], "-C", str(base)], check=True)
    px4 = base / f"build/{BOARD}/bin/px4"
    etc = base / f"build/{BOARD}/etc"
    os.environ["PATH"] = f"{base / f'build/{BOARD}/bin'}:{os.environ['PATH']}"
    af = next((etc / "init.d-posix/airframes").glob("*sihsim_quadx")).name
    autostart = af.split("_")[0]

    if MODE == "pair":
        # Re-fly the demo scenario with both arms and keep both ulogs.
        # Scenario 115 of the 800-run campaign: classical holds (margin +0.43,
        # drift 0.13 m), neural crashes with 2.0 m lateral drift — one of the
        # ten asymmetric scenarios. Speed 5 for a denser MAVLink trace.
        s115 = {"scenario_id": 115, "mass_mul": 1.8251, "ixx_mul": 2.5282,
                "tmax_mul": 1.1901, "kdv": 0.0782, "motor_fail": 2}
        plan = [(s115, arm, 5) for arm in ("classical", "neural")]
    elif MODE == "probe":
        # smallest run that answers: does the loop score a real flight, and
        # what speedup do we actually get? 2 scenarios x 2 arms x 3 speeds.
        probe = lhs(8, SEED)[:2]
        plan = [(s, arm, sp) for sp in (1, 5, 10)
                for s in probe for arm in ("neural", "classical")]
    elif MODE == "shakeout":
        # speed factor already validated by the probe (13.5x measured at 10);
        # this pass checks the fixed oracle actually spreads margins.
        plan = [(s, arm, 10) for s in lhs(10, SEED)
                for arm in ("neural", "classical")]
    else:
        plan = [(s, arm, BASE_SPEED)
                for s in lhs(N_BASE, SEED) for arm in ("neural", "classical")]
    print(f"{MODE}: {len(plan)} runs")

    results = []
    out = WORK / "results.jsonl"
    for i, (scenario, arm, speed) in enumerate(plan):
        row = fly(px4, etc, autostart, scenario, arm, speed, mavutil)
        results.append(row)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(plan)}] id={scenario['scenario_id']} {arm} "
              f"x{speed}: {row['status']} margin={row['margin']} "
              f"wall={row.get('wall_s')}s", flush=True)
        # ulog retention: keep only the WORST_KEEP lowest margins so far
        keep = {r["ulog"] for r in sorted(
                    (r for r in results if r.get("ulog")),
                    key=lambda r: r["margin"])[:WORST_KEEP]}
        for ulg in WORK.glob("run*.ulg"):
            if ulg.name not in keep:
                ulg.unlink()

    n_ok = sum(r["status"] in ("held", "crashed") for r in results)
    fails = sum(r["margin"] < 0 for r in results if r["status"] != "error")
    print(f"done: {len(results)} runs, {n_ok} scored, {fails} failures")


if __name__ == "__main__":
    main()
