# File: pixal3d/utils/local_models.py
"""
Resolve model weights that ship with the installer instead of coming from a
gated HuggingFace repo.

`briaai/RMBG-2.0` and `facebook/dinov3-*` both require accepting a license
before they can be downloaded, which breaks an unattended install. install.py
mirrors them as plain zips into `MODELS/<name>/`, and the loaders below fall
back to the original repo id only when that folder is absent.
"""
import os
from typing import Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_local_model(model_name: str, folder: str) -> Tuple[str, bool]:
    """Return (path_or_repo_id, is_local) for a bundled model folder.

    Looks for `MODELS/<folder>/config.json` relative to the current working
    directory first, then relative to this file, so it works whether the app is
    launched from the repo root or from somewhere else.
    """
    candidates = [
        os.path.join(os.getcwd(), "MODELS", folder),
        os.path.join(_REPO_ROOT, "MODELS", folder),
    ]

    for path in candidates:
        if os.path.exists(os.path.join(path, "config.json")):
            print(f"[INFO] Local {folder} found. Loading from: {path}")
            return path, True

    print(f"[INFO] Local {folder} NOT found. Attempting download/access: {model_name}")
    return model_name, False
