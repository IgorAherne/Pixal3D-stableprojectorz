"""The calibration record of this install: what was measured once, on this PC's own graphics card, when Pixal3D was
installed (Spellcast3D's installer runs `python -m pixal3d.calibrate`), so that nothing is timed or tuned again while
the artist works.

    record = calibration.load()             # the whole record, or None
    sage = calibration.section("sage")      # one section, only if the record was made on this card (else None)

The file is <the fork's folder>/calibration.json (beside flex_gemm's autotune_cache.json; both belong to the install,
not to the repository), or the path in PIXAL3D_CALIBRATION_FILE (the checks point it elsewhere). It holds:

    {"version": <the calibration's version, the installer's PIXAL3D_CALIBRATION>, "card": <torch's device name>,
     "capability": [major, minor], "driver": <driver version>, "made": <ISO time>,
     "quiet": {...how the card was before and during the run...},
     "sage": {...modules/sage.py: the wheel's version and which of its two CUDA kernels serves...},
     "triton": {...which kernel families were compiled...},
     "flex_gemm": {...the convolution keys tuned...}}

A record made on another card (the graphics card was changed) is not used: every reader then behaves as an
uncalibrated install does (sage.py times its kernels at first use, flex_gemm tunes what it meets), and the
installer calibrates again at the next start.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

FILE_NAME = "calibration.json"
ENV = "PIXAL3D_CALIBRATION_FILE"

_cache: dict[str, Any] = {}


def path() -> Path:
    override = os.environ.get(ENV, "").strip()
    return Path(override) if override else Path(__file__).resolve().parents[1] / FILE_NAME


def load() -> Optional[dict]:
    """The record, or None when there is none or it cannot be read (read once per process)."""
    where = str(path())
    if where not in _cache:
        try:
            record = json.loads(Path(where).read_text(encoding="utf-8"))
            _cache[where] = record if isinstance(record, dict) else None
        except (OSError, ValueError):
            _cache[where] = None
    return _cache[where]


def card_name(device: Any = None) -> Optional[str]:
    try:
        import torch

        return torch.cuda.get_device_name(device)
    except Exception:       # noqa: BLE001 - no card, no CUDA
        return None


def section(name: str, device: Any = None) -> Optional[dict]:
    """The record's section `name` when the record was made on this card, else None."""
    record = load()
    if not record or record.get("card") != card_name(device):
        return None
    found = record.get(name)
    return found if isinstance(found, dict) else None


def save(record: dict) -> Path:
    """Write the record (atomically: a reader never sees half a file); the process's cached copy follows."""
    where = path()
    where.parent.mkdir(parents=True, exist_ok=True)
    tmp = where.with_name(f"{where.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, where)
    _cache[str(where)] = record
    return where


def forget() -> None:
    """Drop the cached copy (a test, or a process that wrote the file by other means)."""
    _cache.clear()
