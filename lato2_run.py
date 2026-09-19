# lato2_run.py
# File: lato2_run.py
"""
Entry point for LATO.2's e2e inference with the Windows render patch applied.

Runs third_party/lato2/scripts/e2e_inference.py unchanged, having first swapped
its Open3D conditioning renderer for the nvdiffrast one (see
lato2_render_patch.py). Keeping the patch here rather than in the submodule means
third_party/lato2 stays a clean checkout of the pinned upstream commit.

Every argument is forwarded to e2e_inference.py.
"""
import os
import runpy
import sys
from pathlib import Path

CODE_DIR = Path(__file__).parent.resolve()
LATO2_DIR = CODE_DIR / "third_party" / "lato2"
SCRIPT = LATO2_DIR / "scripts" / "e2e_inference.py"


def main():
    if not SCRIPT.exists():
        sys.exit(f"LATO.2 not found at {LATO2_DIR}. Run install.py first.")

    # LATO.2 imports its own top-level `models` / `modules` / `utils` / `dataset`
    # packages, so its root has to come first on the path.
    sys.path.insert(0, str(LATO2_DIR))
    os.chdir(str(LATO2_DIR))

    sys.path.insert(0, str(CODE_DIR))
    import lato2_render_patch
    lato2_render_patch.apply()

    sys.argv = [str(SCRIPT)] + sys.argv[1:]
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
