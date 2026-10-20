"""Calibrate Pixal3D for this PC's graphics card, once, at install: Spellcast3D's installer runs

    python -m pixal3d.calibrate --version <N> [--quiet-wait 900] [--record <file>]

as the last step of Pixal3D's install (and again only when the card changes or the installer asks for a new
version of it). What it measures is kept in the install's calibration record (pixal3d/calibration.py), so that no
process times or tunes anything while the artist works:

  1. The card is let go quiet first: another program using it (a render in Blender, a game, a video) would make
     every timing below wrong. NVML's utilisation is read every second (nvidia-smi); the steps start after five
     quiet seconds in a row (QUIET_UTIL %), and the card is looked at again before every convolution key: a run
     that finds it busy waits again, within one budget of --quiet-wait seconds for the whole run. A run that had to
     go on with a busy card is written "complete": false, so that the installer tries once more at the next start
     (at most MAX_ATTEMPTS times, then the record stands as it is).
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
record is "complete": true only when every step passed on a quiet card, and the exit code is then 0 (1 when a step
failed or the card was busy: the installer says so and tries again at its next start; 2 when the card cannot be used
at all). The record is written once, at the end - an interrupted run leaves the previous record (or none). The card
is named by its UUID (torch's; the installer compares it with nvidia-smi's list, whatever the device order).
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
# (pixal3d_server.configure_env) forces the venv's folder, not %USERPROFILE%\.triton (Triton's default) nor a global
# TRITON_CACHE_DIR; forced here too, so the calibration's kernels are the ones the artist's first prop finds.
os.environ["TRITON_CACHE_DIR"] = str(Path(sys.prefix) / "triton-cache")

QUIET_UTIL = 15          # % of the card's time busy (NVML) that still counts as quiet
QUIET_SECONDS = 5        # quiet samples in a row before the steps start
# A quick look between keys first lets the run's own kernels drop out of NVML's sample window: utilization.gpu is
# the share of the LAST sample period (up to a second) that had a kernel running, so a sample taken right after a
# benchmark reads 100 % of the run's own work (the first install blamed itself and waited after every key). The
# card is looked at again at most every CHECK_EVERY seconds of work (a look costs SETTLE_SECONDS), and two busy
# samples in a row are needed before a quick look counts the card as busy.
SETTLE_SECONDS = 1.0
CHECK_EVERY = 15.0
SAY_EVERY = 30.0         # while waiting: one line this often
MAX_ATTEMPTS = 3         # runs that ended with a busy card or a failed step before the record stands as it is
# Where the run's bar stands when each step begins (the installer shows it on its page: components._calibrate reads
# the "NN%" of a progress line).
SHARE = {"start": 0.0, "triton": 0.04, "sage": 0.14, "convolutions": 0.18, "done": 1.0}


def say(text: str) -> None:
    print(f"[CALIBRATE] {text}", flush=True)


def progress(fraction: float, text: str) -> None:
    """A line the installer reads for its page: how far the run is (0..1) and what it does now."""
    print(f"[CALIBRATE] {int(round(max(0.0, min(1.0, fraction)) * 100)):3d}% {text}", flush=True)


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


class Quiet:
    """The card watched through nvidia-smi: `wait()` before the steps and again before every convolution key, within
    one budget of `limit_seconds` for the whole run; a wait that runs out of budget lets the run go on and marks it
    (`busy_during`: the record is then not complete)."""

    def __init__(self, limit_seconds: float):
        self.limit = limit_seconds
        self.spent = 0.0
        self.checks = 0
        self.waits = 0              # times the card was found busy
        self.busy_during = False    # a wait ran out of budget: the steps went on beside another program
        self.unknown = False        # nvidia-smi could not be asked
        self.told = -SAY_EVERY
        self.looked: Optional[float] = None     # time.monotonic() of the last look (a quick one is skipped for CHECK_EVERY)

    def wait(self, quick: bool = False) -> bool:
        """Until the card has been at most QUIET_UTIL % busy for QUIET_SECONDS samples in a row. `quick` (between
        keys: the card is the run's own unless another program started): nothing within CHECK_EVERY seconds of the
        last look; else SETTLE_SECONDS for the run's own kernels to leave NVML's window, then one sample - quiet,
        done; busy twice in a row, a proper quiet spell is waited for. True when it was quiet, False when the budget
        ran out."""
        self.checks += 1
        if self.busy_during:
            return False                # the budget is spent: said once, the run goes on without more waiting
        if quick and self.looked is not None and time.monotonic() - self.looked < CHECK_EVERY:
            return True
        if quick:
            time.sleep(SETTLE_SECONDS)
        began = time.monotonic()
        calm, need, found_busy, strikes = 0, (1 if quick else QUIET_SECONDS), False, 0
        try:
            while True:
                got = _query("utilization.gpu")
                if got is None:
                    self.unknown = True
                    return True
                try:
                    util = int(float(got[0]))
                except ValueError:
                    util = 100
                if util <= QUIET_UTIL:
                    calm += 1
                    strikes = 0
                else:
                    calm = 0
                    strikes += 1
                    if quick and not found_busy and strikes < 2:
                        time.sleep(1.0)     # one busy sample right after the run's own work proves nothing
                        continue
                    need = QUIET_SECONDS
                    if not found_busy:
                        found_busy = True
                        self.waits += 1
                if calm >= need:
                    self.spent += time.monotonic() - began
                    return True
                waited = self.spent + time.monotonic() - began
                if waited >= self.limit:
                    self.spent = waited
                    self.busy_during = True
                    say(f"the graphics card is still busy after {waited / 60:.0f} minutes of waiting (last {util} %): "
                        f"going on anyway - this run is not kept as final, the installer tries again at its next start")
                    return False
                if util > QUIET_UTIL and waited - self.told >= SAY_EVERY:
                    self.told = waited
                    say(f"the graphics card is busy ({util} % in use): waiting for it to be free before measuring. "
                        f"Please close heavy programs (Blender, games, videos, Photoshop, a browser with many tabs).")
                time.sleep(1.0)
        finally:
            self.looked = time.monotonic()

    def record(self) -> dict:
        return {"quiet": not self.busy_during and not self.unknown, "waited_s": round(self.spent, 1),
                "checks": self.checks, "found_busy": self.waits, "busy_during": self.busy_during,
                "how": ("nvidia-smi could not be asked: not checked" if self.unknown else
                        "busy: gave up waiting" if self.busy_during else "quiet")}


def wait_until_quiet(limit_seconds: float) -> dict:
    """One wait (the old entry point): see Quiet."""
    quiet = Quiet(limit_seconds)
    quiet.wait()
    return quiet.record()


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
    """Compile the engine's Triton kernels: each family's self-check (which says whether it can run here), then the
    kernels as the flows and decoders call them - the int8 layers of the shipped policy at the flows' widths, the
    fused elementwise kernels, the row codes, NAF's RoPE - at a row count divisible by 16 and one that is not
    (Triton compiles a kernel again for each), so that the first 3D job finds them in the cache."""
    import torch

    done: dict[str, Any] = {}
    from pixal3d.modules import fused, int8mm

    done["fused"] = bool(fused._check(device))
    ok, why = int8mm.ready(device)
    done["int8"] = bool(ok) if ok else f"off: {why}"
    from pixal3d.modules.sparse.conv import codes

    done["codes"] = bool(codes._usable(torch.zeros(1, device=device), 27))     # its self-check, on its own table
    torch.cuda.synchronize(device)
    warmed: dict[str, Any] = {}
    for name, step in (("int8_layers", _warm_int8), ("fused", _warm_fused), ("codes", _warm_codes),
                       ("rope", _warm_rope)):
        began = time.perf_counter()
        try:
            count = step(device)
            warmed[name] = {"calls": count, "seconds": round(time.perf_counter() - began, 2)}
        except Exception as exc:        # noqa: BLE001 - a kernel that cannot be warmed compiles at first use
            warmed[name] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    done["warmed"] = warmed
    return done


WARM_ROWS = (4096, 4099)        # a row count divisible by 16 and one that is not: both of Triton's specialisations


def _warm_int8(device: Any) -> int:
    """The int8 layers of the shipped policy at the flows' widths (1536 channels, 12 x 128 heads, 8192 hidden,
    2048 of conditioning): Int8Linear / Int8MLP built from random nn.Linear layers as int8.convert builds them."""
    import torch
    from torch import nn

    from pixal3d.modules import int8

    if int8.policy() is None or not int8.available(device):
        return 0
    rules = int8.POLICIES[int8.policy()].get("hr") or {}
    shapes = {"qkv": (1536, 4608), "out": (1536, 1536), "xq": (1536, 1536), "xout": (1536, 1536),
              "proj": (2048, 1536)}
    calls = 0
    with torch.no_grad():
        g = torch.Generator(device=device).manual_seed(0)
        for name, (k, n) in shapes.items():
            how = rules.get(name)
            if how is None or how in ("cs",):           # a calibration vector it has not got: the engine's case
                continue
            layer = nn.Linear(k, n).to(device=device, dtype=torch.bfloat16)
            q = int8.Int8Linear(layer, how)
            for rows in WARM_ROWS:
                q(torch.randn(rows, k, device=device, dtype=torch.bfloat16, generator=g))
                calls += 1
        how = rules.get("fc1")
        if how is not None and how != "cs":
            fc1 = nn.Linear(1536, 8192).to(device=device, dtype=torch.bfloat16)
            fc2 = nn.Linear(8192, 1536).to(device=device, dtype=torch.bfloat16)
            mlp = int8.Int8MLP(fc1, fc2, how, rules.get("fc2") if rules.get("fc2") != "cs" else how)
            for rows in WARM_ROWS:
                mlp(torch.randn(rows, 1536, device=device, dtype=torch.bfloat16, generator=g))
                calls += 1
    return calls


def _warm_fused(device: Any) -> int:
    """The fused elementwise kernels at the flows' widths, with and without the modulation and the gate."""
    import torch

    from pixal3d.modules import fused

    calls = 0
    with torch.no_grad():       # the fused path serves inference only: active() says no with autograd on
        if not fused.active(torch.zeros(1, 1536, device=device, dtype=torch.bfloat16)):
            return 0
        g = torch.Generator(device=device).manual_seed(0)
        for rows in WARM_ROWS:
            x = torch.randn(rows, 1536, device=device, dtype=torch.bfloat16, generator=g)
            h = torch.randn(rows, 1536, device=device, dtype=torch.bfloat16, generator=g)
            scale = torch.randn(1, 1536, device=device, dtype=torch.bfloat16, generator=g)
            shift = torch.randn(1, 1536, device=device, dtype=torch.bfloat16, generator=g)
            weight = torch.ones(1536, device=device, dtype=torch.float32)
            fused.norm_modulate(x, 1e-6, scale=scale, shift=shift)
            fused.norm_modulate(x, 1e-6, weight=weight, bias=weight)
            fused.residual(x, h, gate=scale)
            fused.residual(x, h, second=h)
            fused.residual(x, h)
            qk = torch.randn(rows, 12, 128, device=device, dtype=torch.bfloat16, generator=g)
            gamma = torch.ones(128, device=device, dtype=torch.bfloat16)
            fused.qk_norm_rope(qk, gamma, 128 ** -0.5)
            calls += 6
    return calls


def _warm_codes(device: Any) -> int:
    import torch

    from pixal3d.modules.sparse.conv import codes

    calls = 0
    g = torch.Generator(device=device).manual_seed(0)
    for rows in WARM_ROWS:
        nmap = torch.randint(-1, 1 << 20, (rows, 27), dtype=torch.int32, device=device, generator=g)
        codes.codes(nmap.view(torch.uint32), 27)
        calls += 1
    return calls


def _warm_rope(device: Any) -> int:
    """NAF's RoPE kernel on a guide map of the texture conditioning's size (1024 x 1024 at 1536; 512 x 512 too),
    through the module's own entry (the eager rope_apply serves if the kernel cannot run)."""
    import torch

    from pixal3d.modules import naf_sliced

    if not naf_sliced.FUSED_ROPE:
        return 0
    warm = getattr(naf_sliced, "warm_rope", None)
    if warm is None:
        return 0
    calls = 0
    with torch.no_grad():
        for side in (512, 1024):
            calls += int(bool(warm(device, side)))
    return calls


def _sage(device: Any) -> dict:
    from pixal3d.modules import sage

    if not sage.from_env():
        return {"tuned": False, "why": "PIXAL3D_SAGE is off", "summary": "off"}
    return sage.calibrate(device)


def _convolutions(device: Any, quiet: Quiet) -> dict:
    from pixal3d.modules.sparse.conv import calibrate_conv, tuning

    calibrate = getattr(tuning, "calibrate", None)
    if calibrate is None:
        return {"ok": False, "why": "this build has no convolution calibration"}
    shapes = max(1, len(getattr(calibrate_conv, "GRID", {})) or 1)
    done = [0]

    def tell(text: str) -> None:
        """The step's own lines; one per finished shape ("convolutions 1024x1024: 9 sizes ...") moves the bar."""
        if text.startswith("convolutions ") and " sizes (" in text:
            done[0] += 1
            progress(SHARE["convolutions"] + (SHARE["done"] - SHARE["convolutions"]) * min(done[0], shapes) / shapes,
                     f"{text} - shape {done[0]} of {shapes}")
        else:
            say(text)
    return calibrate(device, say=tell, pause=lambda: quiet.wait(quick=True))


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

    progress(SHARE["start"], "checking that the graphics card is free (close heavy programs: Blender, games, videos, "
                              "Photoshop)")
    quiet = Quiet(args.quiet_wait)
    quiet.wait()

    import torch

    from pixal3d import calibration

    if not torch.cuda.is_available():
        say("PyTorch cannot use the graphics card: nothing calibrated")
        return 2
    device = torch.device("cuda", torch.cuda.current_device())
    driver = _query("driver_version")
    earlier = calibration.load() or {}
    attempts = int(earlier.get("attempts") or 0) + 1 if not earlier.get("complete") else 1
    record: dict[str, Any] = {
        "version": args.version, "complete": False, "attempts": attempts,
        "card": torch.cuda.get_device_name(device),
        "card_uuid": _card_uuid(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "driver": driver[0] if driver else None,
        "made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "torch": torch.__version__, "python": sys.version.split()[0],
    }
    say(f"calibrating for {record['card']} (sm_{record['capability'][0]}{record['capability'][1]})"
        + (f", attempt {attempts}" if attempts > 1 else ""))
    began = time.monotonic()
    if "triton" not in skip:
        progress(SHARE["triton"], "compiling Pixal3D's GPU kernels (step 1 of 3)")
        _timed("triton", lambda: _triton(device), record)
    if "sage" not in skip:
        progress(SHARE["sage"], "timing the attention kernels (step 2 of 3)")
        quiet.wait(quick=True)
        found = _timed("sage", lambda: _sage(device), record)
        if found:
            say(f"attention: the {'tuned' if found.get('tuned') else 'shipped'} kernel ({found.get('summary', '')})")
    if "convolutions" not in skip:
        progress(SHARE["convolutions"], "tuning the 3D decoders' convolutions for this card (step 3 of 3, the longest: "
                                        "several minutes)")
        found = _timed("flex_gemm", lambda: _convolutions(device, quiet), record)
        if found:
            say(f"convolutions: {found.get('summary', found)}")
    record["seconds"] = round(time.monotonic() - began, 1)
    record["quiet"] = quiet.record()
    steps = {"triton": "triton", "sage": "sage", "convolutions": "flex_gemm"}      # --skip name -> record section
    failed = [key for step, key in steps.items() if step not in skip and not (record.get(key) or {}).get("ok", False)]
    record["failed"] = failed
    record["complete"] = not failed and not quiet.busy_during
    where = calibration.save(record)
    if record["complete"]:
        progress(SHARE["done"], f"done in {record['seconds'] / 60:.1f} minutes; kept in {where}")
        return 0
    why = (f"the card was busy with another program during the run" if quiet.busy_during else "") \
        + (f"{'; ' if quiet.busy_during and failed else ''}{', '.join(failed)} failed" if failed else "")
    if attempts >= MAX_ATTEMPTS:
        say(f"done in {record['seconds'] / 60:.1f} minutes, with a reservation ({why}); this was attempt "
            f"{attempts} of {MAX_ATTEMPTS}: the record stands as it is (kept in {where})")
        return 0
    say(f"done in {record['seconds'] / 60:.1f} minutes, with a reservation ({why}): the installer tries once more "
        f"at its next start (attempt {attempts} of {MAX_ATTEMPTS}; kept in {where})")
    return 1


def _card_uuid(device: Any) -> Optional[str]:
    """The card's UUID as nvidia-smi spells it (GPU-xxxxxxxx-...), from torch; None when torch has none."""
    try:
        import torch

        uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        return f"GPU-{uuid}" if uuid is not None else None
    except Exception:       # noqa: BLE001
        return None


if __name__ == "__main__":
    sys.exit(main())
