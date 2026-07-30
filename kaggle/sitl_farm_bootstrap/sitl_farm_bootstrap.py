"""Build ArduPilot SITL (ArduPlane) on Kaggle at the pinned commit and smoke-test it.

Needs internet enabled (github clone) -- if the account is not phone-verified,
Kaggle disables internet and the clone fails immediately; verify at
kaggle.com/settings before running.

Output (/kaggle/working):
  sitl_plane_build.tar.gz -- arduplane binary + autotest model params. Attach
      this kernel's output as an input to future farm kernels and skip the
      ~20-40 min rebuild.
  smoke_ok.txt            -- written only if the heartbeat smoke test passed.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import time
from pathlib import Path

DATA = Path("/kaggle/input/aviationsim-week1-evidence-v3")
WORK = Path("/kaggle/working")
# NOT under WORK: everything in /kaggle/working becomes kernel output, and a
# first run that cloned there shipped the whole source tree (+.git) as output,
# making it slow to attach to other kernels. /tmp is scratch, discarded.
SRC = Path("/tmp/ardupilot")
COMMIT = (DATA / "ARDUPILOT_COMMIT").read_text().strip()
HOME_ARG = "-35.363261,149.165230,584,353"  # CMAC, same as the Mac harness


def run(cmd: list[str] | str, **kw) -> None:
    print(f"+ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def main() -> None:
    t0 = time.time()
    print(f"pinned commit: {COMMIT}")
    run([sys.executable, "-m", "pip", "install", "-q",
         "pymavlink", "empy==3.3.4", "pexpect", "future", "packaging", "lxml"])

    if not SRC.exists():
        run(["git", "clone", "--filter=blob:none",
             "https://github.com/ArduPilot/ardupilot.git", str(SRC)])
    run(["git", "checkout", COMMIT], cwd=SRC)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=SRC)

    run(["./waf", "configure", "--board", "sitl"], cwd=SRC)
    run(["./waf", "plane"], cwd=SRC)
    binary = SRC / "build/sitl/bin/arduplane"
    assert binary.is_file(), "build produced no arduplane binary"
    print(f"built in {(time.time() - t0) / 60:.1f} min")

    # -- smoke test: headless boot, one heartbeat, clean kill -----------------
    from pymavlink import mavutil  # installed above

    rundir = WORK / "smoke"
    rundir.mkdir(exist_ok=True)
    proc = subprocess.Popen(
        [str(binary), "--model", "plane", "--speedup", "10", "-I0",
         "--defaults", str(SRC / "Tools/autotest/models/plane.parm"),
         "--home", HOME_ARG],
        cwd=rundir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 60
        conn = None
        while conn is None and time.time() < deadline:
            try:
                conn = mavutil.mavlink_connection("tcp:127.0.0.1:5760")
            except ConnectionRefusedError:
                time.sleep(1)
        assert conn is not None, "SITL never opened its MAVLink port"
        hb = conn.wait_heartbeat(timeout=60)
        assert hb is not None, "no heartbeat within 60 s"
        print(f"smoke test: heartbeat from sysid {hb.get_srcSystem()}")
        (WORK / "smoke_ok.txt").write_text(f"heartbeat ok, commit {COMMIT}\n")
    finally:
        proc.kill()
        proc.wait()

    # -- package the build for reuse by future kernels ------------------------
    with tarfile.open(WORK / "sitl_plane_build.tar.gz", "w:gz") as tar:
        tar.add(binary, arcname="build/sitl/bin/arduplane")
        tar.add(SRC / "Tools/autotest/models", arcname="Tools/autotest/models")
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
