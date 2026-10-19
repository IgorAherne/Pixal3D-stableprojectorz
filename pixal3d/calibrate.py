"""Calibrate Pixal3D for this PC's graphics card, once, at install: Spellcast3D's installer runs

    python -m pixal3d.calibrate --version <N> [--quiet-wait 900] [--record <file>]

as the last step of Pixal3D's install (and again only when the card changes or the installer asks for a new
version of it). What it measures is kept in the install's calibration record (pixal3d/calibration.py), so that no
process times or tunes anything while the artist works:

  1. The card is let go quiet first: another program using it (a render in Blender, a game, a video) would make
     every timing below wrong. NVML's utilisation is read every second (nvidia-smi); the steps start after five
     quiet seconds in a row (QUIET_UTIL %), or after --quiet-wait seconds whatever the card does (said in the
     record and the log).
  2. The Triton kernels the engine compiles at first use are compiled now, through their own self-checks (the
     fused elementwise kernels, the int8 GEMMs, the convolutions' row codes): the first 3D job of the install no
     longer waits for them (the compiled kernels stay in Triton's cache).
  3. Which of the SageAttention wheel's two CUDA kernels is faster on this card (modules/sage.py `calibrate`; they
     give the same bits).
  4. The decoders' sparse convolutions' tile configurations for every size a prop can have
     (modules/sparse/conv/tuning.py `calibrate`, written into flex_gemm's autotune_cache.json): afterwards a size
     that was not calibrated takes the configuration of the nearest one that was, instead of being tuned while the
     artist waits (every configuration gives the same values; only the time differs).

Nothing here changes a result: every choice is between kernels that give the same numbers (the convolutions' split-K
choice aside, which flex_gemm has always timed per install). A step that fails is recorded and the others go on; the
exit code is 0 unless the card cannot be used at all (2). The record is written once, at the end, with
"complete": true - an interrupted run leaves the previous record (or none), and the installer runs it again.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

FORK = Path(__file__).resolve().parents[1]
# flex_gemm reads its cache path when it is imported (app.py sets the same before its imports).
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(FORK / "autotune_cache.json"))
# The Triton kernels compiled here must land where the launcher's processes look for them: Spellcast3D's launcher
# (pixal3d_server.configure_env) keeps them with the venv, not in %USERPROFILE%\.triton (Triton's default), which
# would leave every kernel compiled by this calibration to be compiled again at the artist's first prop.
os.environ.setdefault("TRITON_CACHE_DIR", str(Path(sys.prefix) / "triton-cache"))

QUIET_UTIL = 15          # % of the card's time busy (NVML) that still counts as quiet
QUIET_SECONDS = 5        # quiet samples in a row before the steps start
SAY_EVERY = 30.0         # while waiting: one line this often


def say(text: str) -> None:
    print(f"[CALIBRATE] {text}", flush=True)


def _nvidia_smi() -> Optional[str]:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in (Path(r"C:\Windows\System32\nvidia-smi.exe"),
                      Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")):
        if candidate.exists():
            return str(candidate)
    return None


def _query(fields: str) -> Optional[list[str]]:
    """One nvidia-smi query of the first card (`fields` comma-separated), or None when it cannot be asked."""
    exe = _nvidia_smi()
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    return [part.strip() for part in lines[0].split(",")] if out.returncode == 0 and lines else None


def wait_until_quiet(limit_seconds: float) -> dict:
    """Wait until the card has been at most QUIET_UTIL % busy for QUIET_SECONDS samples in a row, at most
    `limit_seconds`. Returns what was seen: {"quiet": bool, "waited_s", "util_max_last", "how"}."""
    began = time.monotonic()
    told = -SAY_EVERY
    calm = 0
    last: list[int] = []
    while True:
        got = _query("utilization.gpu")
        if got is None:
            return {"quiet": None, "waited_s": round(time.monotonic() - began, 1),
                    "how": "nvidia-smi could not be asked: not checked"}
        try:
            util = int(float(got[0]))
        except ValueError:
            util = 100
        last = (last + [util])[-QUIET_SECONDS:]
        calm = calm + 1 if util <= QUIET_UTIL else 0
        waited = time.monotonic() - began
        if calm >= QUIET_SECONDS:
            return {"quiet": True, "waited_s": round(waited, 1), "util_last": last, "how": "quiet"}
        if waited >= limit_seconds:
            say(f"the graphics card is still busy after {waited / 60:.0f} minutes (last {util} %): calibrating "
                f"anyway - the timings may be a little off")
            return {"quiet": False, "waited_s": round(waited, 1), "util_last": last, "how": "busy: gave up waiting"}
        if util > QUIET_UTIL and waited - told >= SAY_EVERY:
            told = waited
            say(f"the graphics card is busy ({util} % in use): waiting for it to be free before measuring. Please "
                f"close heavy programs (Blender, games, videos, Photoshop, a browser with many tabs).")
        time.sleep(1.0)


def _timed(name: str, step: Callable[[], Any], record: dict) -> Any:
    began = time.perf_counter()
    try:
        found = step()
    except Exception as exc:        # noqa: BLE001 - one step's failure is recorded; the others go on
        record[name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "seconds": round(time.perf_counter() - began, 2)}
        say(f"{name}: FAILED ({type(exc).__name__}: {str(exc)[:200]}) - Pixal3D still works; that part is then "
            f"done at its first use")
        return None
    if isinstance(found, dict):
        found.setdefault("ok", True)
        found["seconds"] = round(time.perf_counter() - began, 2)
        record[name] = found
    return found


def _triton(device: Any) -> dict:
    """Compile the engine's Triton kernels through their self-checks (each says whether it can run here)."""
    import torch

    done: dict[str, Any] = {}
    from pixal3d.modules import fused, int8mm

    done["fused"] = bool(fused._check(device))
    ok, why = int8mm.ready(device)
    done["int8"] = bool(ok) if ok else f"off: {why}"
    from pixal3d.modules.sparse.conv import codes

    done["codes"] = bool(codes._usable(torch.zeros(1, device=device), 27))     # its self-check, on its own table
    torch.cuda.synchronize(device)
    return done


def _sage(device: Any) -> dict:
    from pixal3d.modules import sage

    if not sage.from_env():
        return {"tuned": False, "why": "PIXAL3D_SAGE is off", "summary": "off"}
    return sage.calibrate(device)


def _convolutions(device: Any) -> dict:
    from pixal3d.modules.sparse.conv import tuning

    calibrate = getattr(tuning, "calibrate", None)
    if calibrate is None:
        return {"ok": False, "why": "this build has no convolution calibration"}
    return calibrate(device, say=say)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate Pixal3D for this PC's graphics card (once, at install).")
    ap.add_argument("--version", type=int, default=1, help="the calibration's version (the installer's)")
    ap.add_argument("--record", default="", help="the record file (default: pixal3d/calibration.py's)")
    ap.add_argument("--quiet-wait", type=float, default=900.0, help="seconds to wait for a quiet card at most")
    ap.add_argument("--skip", default="", help="comma-separated steps to leave out: triton, sage, convolutions")
    args = ap.parse_args(argv)
    if args.record:
        os.environ["PIXAL3D_CALIBRATION_FILE"] = args.record
    skip = {part.strip() for part in args.skip.split(",") if part.strip()}

    say("checking that the graphics card is free (close heavy programs: Blender, games, videos, Photoshop)")
    quiet = wait_until_quiet(args.quiet_wait)

    import torch

    from pixal3d import calibration

    if not torch.cuda.is_available():
        say("PyTorch cannot use the graphics card: nothing calibrated")
        return 2
    device = torch.device("cuda", torch.cuda.current_device())
    driver = _query("driver_version")
    record: dict[str, Any] = {
        "version": args.version, "complete": False,
        "card": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "driver": driver[0] if driver else None,
        "made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "quiet": quiet, "torch": torch.__version__, "python": sys.version.split()[0],
    }
    say(f"calibrating for {record['card']} (sm_{record['capability'][0]}{record['capability'][1]})")
    began = time.monotonic()
    if "triton" not in skip:
        say("compiling Pixal3D's GPU kernels")
        _timed("triton", lambda: _triton(device), record)
    if "sage" not in skip:
        say("timing the attention kernels")
        found = _timed("sage", lambda: _sage(device), record)
        if found:
            say(f"attention: the {'tuned' if found.get('tuned') else 'shipped'} kernel ({found.get('summary', '')})")
    if "convolutions" not in skip:
        say("tuning the 3D decoders' convolutions for this card (the longest step: several minutes)")
        found = _timed("flex_gemm", lambda: _convolutions(device), record)
        if found:
            say(f"convolutions: {found.get('summary', found)}")
    record["seconds"] = round(time.monotonic() - began, 1)
    record["complete"] = True
    where = calibration.save(record)
    say(f"done in {record['seconds'] / 60:.1f} minutes; kept in {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
