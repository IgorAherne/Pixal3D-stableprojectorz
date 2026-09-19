# install.py
# File: install.py
#
# One-shot installer for Pixal3D on Windows. No CUDA Toolkit / MSVC required:
# every native extension is installed from a pre-compiled cp311 wheel.
#
# The stack is fixed, because the wheels are ABI-locked to it:
#     Python 3.11  +  torch 2.8.0  +  CUDA 12.8  (cp311-win_amd64)
#
# Run it with the interpreter you want to install into, e.g.
#     venv\Scripts\python.exe install.py
import subprocess
import sys
import os
import time
import argparse
from typing import Optional, Tuple, List
from pathlib import Path
import urllib.request
import urllib.error
import socket
import shutil
import tempfile
import zipfile

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
REQUIRED_PYTHON = (3, 11)  # Matches cp311 wheels

# The weights are ~29 GB over a handful of 5.5 GB files, so a dropped connection
# is normal rather than exceptional. Every attempt resumes from the partial blob
# on disk, so retrying is nearly free - be generous.
HF_MAX_RETRIES = 12
# huggingface_hub defaults to 8 parallel workers, which on a slow line splits the
# pipe until each stream falls under the read timeout and thrashes. One stream at
# full speed finishes sooner. Raise it with --dl-workers on a fast connection.
HF_DEFAULT_WORKERS = 1
# Default is 10s, which a congested CDN connection trips constantly.
HF_DOWNLOAD_TIMEOUT = "60"

TORCH_VERSION = "2.8.0"
TORCHVISION_VERSION = "0.23.0"
TORCHAUDIO_VERSION = "2.8.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

# Flash-attention and NATTEN kernels are only built for Ampere and newer.
MIN_CC_FOR_FLASH_ATTN = 8
MIN_CC_FOR_NATTEN = 8

# Mirror for the big wheels. Populate this release and it will be preferred over
# the upstream community builds below.
SPZ_WHEEL_MIRROR = "https://github.com/IgorAherne/Pixal3D-stableprojectorz/releases/download/wheels"

# Wheels that are too large to keep in git (>100 MB), so they're fetched at install time.
# Each entry is tried mirror-by-mirror; the first one that answers a HEAD request wins.
DOWNLOAD_WHEELS = [
    {
        "name": "flash_attn",
        "filename": "flash_attn-2.8.3+cu128torch2.8-cp311-cp311-win_amd64.whl",
        "min_cc": MIN_CC_FOR_FLASH_ATTN,
        "skip_reason": "GPU is pre-Ampere; xformers will be used for attention instead",
        "urls": [
            f"{SPZ_WHEEL_MIRROR}/flash_attn-2.8.3%2Bcu128torch2.8-cp311-cp311-win_amd64.whl",
            "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/flash_attn-2.8.3%2Bcu128torch2.8-cp311-cp311-win_amd64.whl",
        ],
    },
    {
        # NAF (the DINOv3 feature upsampler used by the shape/texture stages)
        # is built on neighborhood attention, so NATTEN is not optional.
        "name": "natten",
        "filename": "natten-0.21.6+cu128torch2.8-cp311-cp311-win_amd64.whl",
        "min_cc": MIN_CC_FOR_NATTEN,
        "skip_reason": "NATTEN has no pre-Ampere kernels; NAF upsampling will not run on this GPU",
        "urls": [
            f"{SPZ_WHEEL_MIRROR}/natten-0.21.6%2Bcu128torch2.8-cp311-cp311-win_amd64.whl",
            "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/natten-latest/natten-0.21.6%2Bcu128torch2.8-cp311-cp311-win_amd64.whl",
        ],
    },
]

# flex_gemm dispatches sparse conv to Triton kernels that live in a pure-python
# `flex_gemm/kernels/triton/` subpackage. A build that omits it still imports fine,
# because flex_gemm/kernels/__init__.py swallows the ImportError - the failure only
# shows up mid-generation as:
#   AttributeError: module 'flex_gemm.kernels' has no attribute 'triton'
# Source for those kernels when the installed flex_gemm is missing them. Note this
# wheel cannot be pip-installed: its dist-info directory is named `flex-gemm-...`
# instead of `flex_gemm-...`, so pip reads the name as 'flex' and refuses it. We
# only unzip the subpackage out of it, so that doesn't matter.
FLEX_GEMM_COMPLETE = {
    "filename": "flex_gemm-1.0.0+cu128torch2.8-cp311-cp311-win_amd64.whl",
    "urls": [
        f"{SPZ_WHEEL_MIRROR}/flex_gemm-1.0.0%2Bcu128torch2.8-cp311-cp311-win_amd64.whl",
        "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm-latest/flex_gemm-1.0.0%2Bcu128torch2.8-cp311-cp311-win_amd64.whl",
    ],
}

# Pure-python packages pinned by URL so the install never needs a git executable.
UTILS3D_WHEEL = "https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl"
MOGE_ARCHIVE = "https://github.com/microsoft/MoGe/archive/74fbce054ebed49800de42d0ad0e83495065719a.zip"
UTILS3D_MOGE_ARCHIVE = "https://github.com/EasternJournalist/utils3d-moge/archive/62f09d58509485564e24d5d9f6aac9ee9ebc0c37.zip"

# torch.hub repo + checkpoint for the NAF feature upsampler.
NAF_HUB_REPO = "valeoai/NAF"
NAF_CHECKPOINT = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"

# LATO.2 retopologises the generated high-poly mesh into a low-poly one
# (V-Flow vertices -> T-Flow connectivity). It is tracked as a git submodule under
# third_party/lato2 for development, but the installer never shells out to git: if
# the folder is already populated - by the release zip, or by `git clone
# --recursive` - it is left alone, and only an empty folder triggers a fetch of the
# pinned source archive.
LATO2_COMMIT = "fbb1f5a5755e6db8700cf6922fd506830b7cdccd"
LATO2_SOURCE_URLS = [
    f"https://github.com/IgorAherne/LATO2-stableprojectorz/archive/{LATO2_COMMIT}.zip",
]
# Ungated, ~3.6 GB of *.pt weights.
LATO2_CKPT_REPO = "0x4c48/LATO.2"
# LATO.2 conditions on a rendered view encoded by DINOv2, pulled through torch.hub.
LATO2_DINO_HUB_REPO = "facebookresearch/dinov2"
# spconv is LATO.2's default sparse-conv backend. Unlike everything else here it is
# NOT a torch extension - it declares no torch dependency and is tagged by CUDA
# alone - so the official PyPI build is used instead of a torch-tagged mirror
# wheel. There is no cu128 release, but CUDA minor-version compatibility makes the
# cu126 binaries fine on a 12.8 driver. Installed *with* dependencies, because
# spconv genuinely needs pccm, ccimport, cumm-cu126, pybind11 and fire to import.
SPCONV_PACKAGE = "spconv-cu126==2.3.8"

# The two CUDA extensions LATO.2 needs that Pixal3D itself doesn't: torch_scatter
# and vox2seq (z-order / Hilbert serialization). `torchsparse` is only touched when
# SPARSE_BACKEND=torchsparse, so it is skipped.
LATO2_WHEELS = [
    {
        # From PyG's own wheel index rather than the community mirror: the
        # community build's dist-info directory is named `torch-scatter-...`
        # instead of `torch_scatter-...`, so pip reads the name as 'torch' and
        # refuses it ("inconsistent name: expected 'torch-scatter', but metadata
        # has 'torch'"). PyG publishes a correctly-packaged build for this exact
        # torch/CUDA combination.
        "name": "torch_scatter",
        "filename": "torch_scatter-2.1.2+pt28cu128-cp311-cp311-win_amd64.whl",
        "urls": [
            f"{SPZ_WHEEL_MIRROR}/torch_scatter-2.1.2%2Bpt28cu128-cp311-cp311-win_amd64.whl",
            "https://data.pyg.org/whl/torch-2.8.0%2Bcu128/torch_scatter-2.1.2%2Bpt28cu128-cp311-cp311-win_amd64.whl",
        ],
    },
    {
        "name": "vox2seq",
        "filename": "vox2seq-0.0.0+cu128torch2.8-cp311-cp311-win_amd64.whl",
        "urls": [
            f"{SPZ_WHEEL_MIRROR}/vox2seq-0.0.0%2Bcu128torch2.8-cp311-cp311-win_amd64.whl",
            "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/vox2seq-latest/vox2seq-0.0.0%2Bcu128torch2.8-cp311-cp311-win_amd64.whl",
        ],
    },
]

# HuggingFace repos pulled during install so the first run doesn't stall.
PIXAL3D_REPO = "TencentARC/Pixal3D"
MOGE_REPO = "Ruicheng/moge-2-vitl"
DINOV3_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"

# Single-view cascade only. The *_mv checkpoints are another ~17 GB and are
# fetched only with --mv.
PIXAL3D_SINGLE_VIEW_PATTERNS = [
    "pipeline.json",
    "ckpts/ss_dec_conv3d_16l8_fp16.*",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16.*",
    "ckpts/shape_dec_next_dc_f16c32_fp16.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.*",
    "ckpts/tex_dec_next_dc_f16c32_fp16.*",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.*",
]
PIXAL3D_MULTI_VIEW_PATTERNS = [
    "pipeline_mv.json",
    "ckpts/ss_flow_img_dit_1_3B_64_bf16_mv.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv.*",
    "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv.*",
    "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv.*",
]


class InstallationError(Exception):
    """Custom exception for installation failures"""
    pass


def get_current_script_dir() -> Path:
    """Helper to get the directory of the current script."""
    try:
        return Path(__file__).parent.resolve()
    except NameError:
        return Path(os.getcwd()).resolve()


CODE_DIR = get_current_script_dir()
# Keep every download inside the app folder so the install stays portable.
# The launcher must export the same two variables, otherwise the runtime will
# look in the user-wide caches instead and download everything a second time.
HF_HOME = CODE_DIR / "models"
TORCH_HOME = CODE_DIR / "models" / "torch"


def check_python_version():
    """Ensure we are running on the correct Python version for the wheels."""
    current = sys.version_info[:2]
    if current != REQUIRED_PYTHON:
        print(f"Error: This installer requires Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}")
        print(f"You are currently using Python {current[0]}.{current[1]}")
        print("Please use the embedded python or correct environment.")
        sys.exit(1)


def check_connectivity(url: str = "https://pytorch.org", timeout: int = 5) -> Tuple[bool, Optional[str]]:
    """Check internet connectivity."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True, None
    except urllib.error.URLError as e:
        reason = getattr(e, 'reason', str(e))
        if isinstance(reason, socket.gaierror):
            return False, f"DNS resolution failed: {reason}"
        elif isinstance(reason, socket.timeout) or 'timed out' in str(e):
            return False, "Connection timed out"
        else:
            return False, f"Connection failed: {reason}"
    except Exception as e:
        return False, f"Unknown error: {str(e)}"


def get_git_env() -> dict:
    """Return a copy of the current environment configured to use the portable Git."""
    env = os.environ.copy()
    PORTABLE_GIT_BASE = (CODE_DIR / ".." / "tools" / "git").resolve()

    if PORTABLE_GIT_BASE.exists():
        git_paths = [
            str(PORTABLE_GIT_BASE / "mingw64" / "bin"),
            str(PORTABLE_GIT_BASE / "cmd"),
            str(PORTABLE_GIT_BASE / "usr" / "bin"),
        ]
        existing_path = env.get("PATH", "")
        env["PATH"] = ";".join(git_paths) + (";" + existing_path if existing_path else "")

        ca_bundle = PORTABLE_GIT_BASE / "mingw64" / "etc" / "ssl" / "certs" / "ca-bundle.crt"
        if ca_bundle.exists():
            env["GIT_SSL_CAINFO"] = str(ca_bundle)
            env["SSL_CERT_FILE"] = str(ca_bundle)

    return env


def _gpu_compute_capability() -> Optional[int]:
    """Major compute capability of GPU 0, or None if it can't be determined."""
    try:
        result = subprocess.run(
            f'"{sys.executable}" -c "import torch; major, _ = torch.cuda.get_device_capability(); print(major)"',
            shell=True, capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except Exception:
        return None  # unknown; callers assume a modern GPU


def run_command_with_retry(cmd: str, desc: Optional[str] = None, max_retries: int = MAX_RETRIES, fatal: bool = True) -> subprocess.CompletedProcess:
    """Run a command with retry logic."""
    last_error = None
    env = get_git_env()

    if cmd.startswith('pip install'):
        args = cmd[11:]
        cmd = f'"{sys.executable}" -m pip install --no-cache-dir --isolated {args}'

    if "pip install" in cmd and "--progress-bar" not in cmd:
        cmd += " --progress-bar=on"

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\nRetry attempt {attempt + 1}/{max_retries} for: {desc or cmd}")
                connected, error_msg = check_connectivity()
                if not connected:
                    print(f"Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                    continue

            if "pip install" in cmd:
                result = subprocess.run(cmd, shell=True, text=True, stdout=sys.stdout, stderr=subprocess.PIPE, env=env)
            else:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

            if result.returncode == 0:
                return result

            last_error = result
            print(f"\nCommand failed (attempt {attempt + 1}/{max_retries}):")
            if hasattr(result, 'stderr') and result.stderr:
                print(f"Error output:\n{result.stderr}")

        except Exception as e:
            last_error = e
            print(f"\nException during {desc or cmd} (attempt {attempt + 1}/{max_retries}):")
            print(str(e))

        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY)

    if fatal:
        raise InstallationError(f"Command failed after {max_retries} attempts: {last_error}")
    else:
        print(f"Warning: Command '{desc}' failed. Continuing...")
        return last_error


def _url_exists(url: str, timeout: int = 15) -> bool:
    """Cheap HEAD probe so a mirror that isn't populated yet costs one request."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def download_file(urls: List[str], dest: Path, label: str) -> None:
    """Download to `dest`, walking the mirror list until one succeeds."""
    def _reporthook(block_num, block_size, total_size):
        dl = block_num * block_size
        if total_size > 0:
            pct = min(dl * 100 / total_size, 100)
            print(f"\r  {dl / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB ({pct:.0f}%)", end="", flush=True)

    if len(urls) > 1:
        # Probe first so an unpopulated mirror costs one request instead of three
        # failed downloads. Some hosts reject HEAD, so if nothing answers we still
        # try every URL rather than giving up.
        candidates = [u for u in urls if _url_exists(u)]
        for u in urls:
            if u not in candidates:
                print(f"  {label}: mirror unavailable, skipping ({u})")
        urls = candidates or urls

    for url_idx, url in enumerate(urls):
        mirror_label = f"mirror {url_idx + 1}/{len(urls)}"
        print(f"\nDownloading {label} from {mirror_label}: {url}")
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    print(f"  Retry attempt {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(RETRY_DELAY)
                urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
                print()  # newline after progress
                return
            except Exception as e:
                print(f"\n  Download failed: {e}")
                if dest.exists():
                    dest.unlink()
                if attempt == MAX_RETRIES - 1:
                    print(f"  All {MAX_RETRIES} attempts failed for {mirror_label}. Trying next mirror...")

    raise InstallationError(f"Failed to download {label} from all {len(urls)} mirrors")


def download_models():
    """Download and extract model zips from GitHub releases if not already present.

    briaai/RMBG-2.0 is a gated HuggingFace repo, and facebook's DINOv3 weights are
    gated too, so both are mirrored as plain zips and loaded from MODELS/ instead
    (see pixal3d/utils/local_models.py).
    """
    models_dir = CODE_DIR / "MODELS"
    models_dir.mkdir(exist_ok=True)

    MODELS = [
        {
            "name": "dinov3",
            "urls": [
                "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/dinov3.zip",
                "https://sourceforge.net/projects/trellis-2-stableprojectorz/files/extra-models/dinov3.zip/download",
            ],
            "check_file": models_dir / "dinov3" / "model.safetensors",
        },
        {
            "name": "RMBG-2.0",
            "urls": [
                "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/RMBG-2.0.zip",
                "https://sourceforge.net/projects/trellis-2-stableprojectorz/files/extra-models/RMBG-2.0.zip/download",
            ],
            "check_file": models_dir / "RMBG-2.0" / "model.safetensors",
        },
    ]

    for model in MODELS:
        if model["check_file"].exists():
            print(f"[INFO] {model['name']} already present, skipping download.")
            continue

        zip_path = models_dir / f"{model['name']}.zip"
        download_file(model["urls"], zip_path, model["name"])

        print(f"  Extracting {model['name']}...")
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(models_dir))
        zip_path.unlink()

        if not model["check_file"].exists():
            raise InstallationError(
                f"Extraction succeeded but {model['check_file'].name} not found. "
                f"Check that the zip contains a '{model['name']}/' folder at its root."
            )
        print(f"  {model['name']} ready.")


def download_hf_models(include_mv: bool = False, workers: int = HF_DEFAULT_WORKERS):
    """Pre-download HuggingFace model weights so the app doesn't timeout on first launch.

    Safe to re-run: partial files are kept as `<sha>.incomplete` blobs under
    models/hub and resumed over HTTP Range, so an interrupted install picks up
    where it stopped instead of starting over.
    """

    size = "~46GB" if include_mv else "~29GB"
    print("\n============================================================================= ")
    print(" Downloading Pixal3D model weights from HuggingFace.")
    print(f" This may take a long time on first install ({size} total). Please wait")
    print(" Interrupted downloads resume - just re-run install.py.")
    print("============================================================================= ")

    # Imported here because huggingface_hub is installed in a previous step.
    from huggingface_hub import snapshot_download

    cache_dir = str(HF_HOME / "hub")

    patterns = list(PIXAL3D_SINGLE_VIEW_PATTERNS)
    if include_mv:
        patterns += PIXAL3D_MULTI_VIEW_PATTERNS

    repos = [
        (PIXAL3D_REPO, patterns),
        (MOGE_REPO, None),
    ]
    # DINOv3 comes from MODELS/dinov3; only pull the mirror repo if that failed.
    if not (CODE_DIR / "MODELS" / "dinov3" / "config.json").exists():
        repos.append((DINOV3_REPO, None))

    for repo_id, allow_patterns in repos:
        print(f"  Downloading {repo_id}...")
        for attempt in range(HF_MAX_RETRIES):
            try:
                snapshot_download(
                    repo_id,
                    cache_dir=cache_dir,
                    allow_patterns=allow_patterns,
                    max_workers=workers,
                )
                print(f"  {repo_id} ready.")
                break
            except Exception as e:
                print(f"  Download interrupted (attempt {attempt + 1}/{HF_MAX_RETRIES}): {e}")
                if attempt < HF_MAX_RETRIES - 1:
                    print(f"  Resuming in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    raise InstallationError(
                        f"Failed to download {repo_id} after {HF_MAX_RETRIES} attempts. "
                        f"Progress is kept - re-run install.py to continue."
                    )


def download_naf():
    """Pre-fetch the NAF upsampler (torch.hub repo + checkpoint) into models/torch."""
    print("\n--- Fetching NAF feature upsampler ---")
    import torch

    try:
        # Populates models/torch/hub/valeoai_NAF_main without importing natten.
        torch.hub._get_cache_or_reload(
            NAF_HUB_REPO, force_reload=False, trust_repo=True,
            calling_fn=None, verbose=True, skip_validation=False,
        )
    except TypeError:
        # Older/newer torch tweak this private signature; the public path also works.
        torch.hub.load(NAF_HUB_REPO, "naf", pretrained=False, device="cpu", trust_repo=True)

    torch.hub.load_state_dict_from_url(NAF_CHECKPOINT, progress=True, map_location="cpu")
    print("  NAF ready.")


def _pip_install_wheel(wheel: Path, fatal: bool = True) -> bool:
    # --no-deps: these wheels declare loose torch pins that would otherwise
    # let pip replace the cu128 build with a CPU one from PyPI.
    result = run_command_with_retry(
        f'pip install --no-deps "{wheel}"', f"Installing {wheel.name}",
        max_retries=MAX_RETRIES if fatal else 1, fatal=fatal,
    )
    return getattr(result, "returncode", 1) == 0


def install_downloaded_wheel(spec: dict, whl_dir: Path, cc_major: Optional[int]):
    """Install one wheel from the download table, preferring a local copy."""
    # A copy already in whl/ wins, so you can ship your own build and so a
    # re-run doesn't re-download a few hundred MB.
    local = sorted(whl_dir.glob(f"{spec['name']}*.whl"))
    if local:
        wheel = local[0]
        print(f"Installing: {wheel.name} (local)")
        if _pip_install_wheel(wheel, fatal=False):
            return
        # A stale or malformed local wheel shouldn't wedge the install - a
        # mis-named dist-info directory is enough for pip to reject one.
        print(f"  {wheel.name} was rejected; falling back to the download mirror.")

    if cc_major is not None and cc_major < spec.get("min_cc", 0):
        print(f"Skipping {spec['name']}: {spec['skip_reason']}")
        return

    wheel = whl_dir / spec["filename"]
    if not wheel.exists():
        download_file(spec["urls"], wheel, spec["filename"])
    print(f"Installing: {wheel.name}")
    _pip_install_wheel(wheel)


LATO2_DIR = CODE_DIR / "third_party" / "lato2"
LATO2_MARKER = LATO2_DIR / "scripts" / "e2e_inference.py"
LATO2_CKPT_DIR = CODE_DIR / "MODELS" / "lato2"
LATO2_CKPT_FILES = ["vflow.pt", "vvae.pt", "offset_head.pt", "tflow.pt", "tvae.pt", "voxel_encoder.pt"]


def ensure_lato2_source():
    """Make sure third_party/lato2 holds the LATO.2 source.

    Present already (release zip, or a recursive clone) -> leave it alone.
    Empty -> download the pinned archive, so a plain `git clone` without
    --recursive still ends up with a working install and no git executable is
    ever required.
    """
    print("\n--- Checking LATO.2 source ---")
    if LATO2_MARKER.exists():
        print(f"[INFO] LATO.2 already present at {LATO2_DIR}, skipping download.")
        return

    LATO2_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="pixal3d_lato2_"))
    try:
        archive = tmp_dir / "lato2.zip"
        download_file(LATO2_SOURCE_URLS, archive, "LATO.2 source")
        with zipfile.ZipFile(str(archive)) as zf:
            zf.extractall(str(tmp_dir))
        # GitHub archives wrap everything in a <repo>-<sha>/ folder.
        roots = [p for p in tmp_dir.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise InstallationError(f"Unexpected LATO.2 archive layout: {[p.name for p in roots]}")
        for item in roots[0].iterdir():
            target = LATO2_DIR / item.name
            if target.exists():
                continue
            shutil.move(str(item), str(target))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not LATO2_MARKER.exists():
        raise InstallationError(f"LATO.2 extracted but {LATO2_MARKER} is missing.")
    print(f"  LATO.2 ready at {LATO2_DIR}")


def download_lato2_models():
    """Fetch the LATO.2 weights (~3.6 GB) plus its DINOv2 conditioning backbone.

    Non-fatal: Pixal3D generates fine without them, only the low-poly retopology
    step is unavailable.
    """
    print("\n--- Downloading LATO.2 weights (~3.6 GB) ---")
    LATO2_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if all((LATO2_CKPT_DIR / f).exists() for f in LATO2_CKPT_FILES):
        print("[INFO] LATO.2 checkpoints already present, skipping download.")
    else:
        try:
            from huggingface_hub import snapshot_download
            for attempt in range(HF_MAX_RETRIES):
                try:
                    snapshot_download(
                        LATO2_CKPT_REPO,
                        local_dir=str(LATO2_CKPT_DIR),
                        allow_patterns=["*.pt"],
                        max_workers=HF_DEFAULT_WORKERS,
                    )
                    break
                except Exception as e:
                    print(f"  Download interrupted (attempt {attempt + 1}/{HF_MAX_RETRIES}): {e}")
                    if attempt == HF_MAX_RETRIES - 1:
                        raise
                    print(f"  Resuming in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"\n  [WARNING] Could not download {LATO2_CKPT_REPO}: {e}")
            print("  Pixal3D generation still works; only the low-poly output is unavailable.\n")
            return False

    missing = [f for f in LATO2_CKPT_FILES if not (LATO2_CKPT_DIR / f).exists()]
    if missing:
        print(f"  [WARNING] LATO.2 checkpoints missing: {missing}")
        return False
    print("  LATO.2 checkpoints ready.")

    # DINOv2 encodes the conditioning render. LATO.2 keeps its own hub dir, so
    # point torch.hub there and warm it now rather than on first generation.
    print("  Fetching DINOv2 conditioning backbone...")
    import torch
    hub_dir = LATO2_CKPT_DIR / "dinov2"
    hub_dir.mkdir(parents=True, exist_ok=True)
    prev_dir = torch.hub.get_dir()
    try:
        torch.hub.set_dir(str(hub_dir))
        torch.hub._get_cache_or_reload(
            LATO2_DINO_HUB_REPO, force_reload=False, trust_repo=True,
            calling_fn=None, verbose=True, skip_validation=False,
        )
        print("  DINOv2 hub checkout ready.")
    except Exception as e:
        print(f"  [WARNING] DINOv2 hub prefetch failed: {e}")
        print("  It will be retried on first low-poly generation.")
    finally:
        torch.hub.set_dir(prev_dir)

    return True


def _has_flex_gemm_triton() -> bool:
    check = subprocess.run(
        f'"{sys.executable}" -c "import torch, flex_gemm.kernels as k; k.triton"',
        shell=True, capture_output=True, text=True
    )
    return check.returncode == 0


def ensure_flex_gemm_triton():
    """Graft the Triton kernels into flex_gemm if the installed build lacks them.

    Pixal3D's sparse conv defaults to the 'masked_implicit_gemm_splitk' algorithm,
    which lives in flex_gemm.kernels.triton. Catching it here beats discovering it
    several minutes into a generation, in the shape decoder's upsample.

    Only the missing pure-python subpackage is copied in, rather than swapping the
    whole wheel: it imports nothing but math/torch/triton/typing, so it is
    independent of which build produced the compiled kernels/cuda pyd next to it.
    """
    print("\n--- Checking flex_gemm Triton kernels ---")
    if _has_flex_gemm_triton():
        print("[OK] flex_gemm ships the Triton kernels.")
        return

    print("Installed flex_gemm has no Triton kernels. Fetching them...")
    located = subprocess.run(
        f'"{sys.executable}" -c "import torch, flex_gemm.kernels as k; print(k.__path__[0])"',
        shell=True, capture_output=True, text=True
    )
    if located.returncode != 0:
        raise InstallationError(f"Could not locate flex_gemm.kernels: {located.stderr.strip()}")
    kernels_dir = Path(located.stdout.strip())

    tmp_dir = Path(tempfile.mkdtemp(prefix="pixal3d_flexgemm_"))
    try:
        wheel = tmp_dir / FLEX_GEMM_COMPLETE["filename"]
        download_file(FLEX_GEMM_COMPLETE["urls"], wheel, wheel.name)

        prefix = "flex_gemm/kernels/triton/"
        with zipfile.ZipFile(str(wheel)) as zf:
            members = [n for n in zf.namelist() if n.startswith(prefix) and not n.endswith("/")]
            if not members:
                raise InstallationError(f"{wheel.name} does not contain {prefix}")
            for name in members:
                target = kernels_dir / "triton" / name[len(prefix):]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(name))
        print(f"  Copied {len(members)} files into {kernels_dir / 'triton'}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not _has_flex_gemm_triton():
        raise InstallationError(
            "flex_gemm still has no Triton kernels. Sparse convolution will fail "
            "during generation."
        )
    print("[OK] flex_gemm Triton kernels installed.")


def install_pillow_simd(whl_dir: Path):
    """Swap standard Pillow for Pillow-SIMD, reverting if the wheel doesn't load."""
    print("\n--- Configuring Pillow ---")
    simd_wheels = list(whl_dir.glob("Pillow_SIMD*.whl"))

    simd_success = False
    if simd_wheels:
        print("Uninstalling standard Pillow...")
        subprocess.run(f'"{sys.executable}" -m pip uninstall -y pillow', shell=True, stdout=subprocess.DEVNULL)

        print(f"Installing Pillow-SIMD: {simd_wheels[0].name}")
        try:
            run_command_with_retry(f'pip install "{simd_wheels[0]}"', "Installing Pillow-SIMD")

            # --- SAFETY CHECK ---
            # Verify immediately if Pillow-SIMD actually works
            print("Verifying Pillow-SIMD...")
            check_cmd = f'"{sys.executable}" -c "from PIL import Image; print(\'Pillow OK\')"'
            check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)

            if check_result.returncode == 0:
                print("Pillow-SIMD installed and verified successfully.")
                simd_success = True
            else:
                print(f"Warning: Pillow-SIMD installed but failed to load. (Error: {check_result.stderr.strip()})")
                print("Falling back to standard Pillow...")
        except Exception:
            print("Failed to install Pillow-SIMD wheel. Falling back to standard Pillow...")

    if not simd_success:
        # Revert to standard Pillow if SIMD missing or broken.
        # We uninstall Pillow-SIMD first just in case.
        subprocess.run(f'"{sys.executable}" -m pip uninstall -y Pillow-SIMD', shell=True, stdout=subprocess.DEVNULL)
        run_command_with_retry("pip install pillow", "Installing Standard Pillow")


def install_dependencies(args):
    """Install Pixal3D dependencies."""
    whl_dir = CODE_DIR / "whl"
    check_python_version()

    try:
        connected, error_msg = check_connectivity()
        if not connected:
            print(f"Error: Internet connectivity check failed: {error_msg}")
            sys.exit(1)

        # 0. Model weights that live behind gated HF repos (dinov3, RMBG-2.0)
        if not args.skip_models:
            download_models()

        if args.models_only:
            # Resuming an interrupted weights download - the packages are already in.
            download_naf()
            download_hf_models(include_mv=args.mv, workers=args.dl_workers)
            if not args.no_lato2:
                download_lato2_models()
            print("\nModel downloads completed successfully!")
            return

        # 1. PyTorch 2.8.0 + CUDA 12.8 — every wheel below is built against this exact ABI
        print(f"\n--- Installing PyTorch {TORCH_VERSION} (CUDA 12.8) ---")
        torch_cmd = (
            f"pip install torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} "
            f"torchaudio=={TORCHAUDIO_VERSION} --index-url {TORCH_INDEX_URL}"
        )
        run_command_with_retry(torch_cmd, "Installing PyTorch")

        # 2. General Dependencies
        print("\n--- Installing General Dependencies ---")
        general_deps = [
            "imageio==2.37.2", "imageio-ffmpeg==0.6.0", "tqdm==4.67.1", "easydict==1.13",
            "opencv-python-headless==4.12.0.88", "ninja", "trimesh==4.10.1",
            "transformers==4.57.3", "gradio==6.0.1", "tensorboard", "pandas", "lpips",
            "zstandard==0.25.0", "kornia==0.8.2", "timm==1.0.22", "diffusers==0.37.1",
            "plyfile==1.1.3", "huggingface_hub", "accelerate==1.13.0", "psutil", "scipy",
            "nest_asyncio", "einops", "triton-windows==3.4.0.post21",
        ]
        run_command_with_retry(f"pip install {' '.join(general_deps)}", "Installing pip packages")

        # 2.0.1. app.py decorates its handlers with @spaces.GPU. The package is a
        # no-op outside HuggingFace Spaces, but it has to be importable. Not fatal:
        # inference.py doesn't need it.
        run_command_with_retry("pip install spaces", "Installing spaces", fatal=False)

        # 2.1. xformers — fallback attention for pre-Ampere GPUs that can't run flash-attention
        print("\n--- Installing xformers ---")
        run_command_with_retry(
            f"pip install xformers==0.0.32.post2 --index-url {TORCH_INDEX_URL}",
            "Installing xformers"
        )

        # 2.2. utils3d — pinned wheel, pure python (no git executable needed)
        print("\n--- Installing utils3d ---")
        run_command_with_retry(f'pip install "{UTILS3D_WHEEL}"', "Installing utils3d")

        # 2.3. MoGe-2 (camera/FOV estimation). Installed --no-deps because its
        # metadata pulls torch and a source build of FlexGEMM, both of which
        # would fight the pinned stack. It ships its own DINOv2, so the only
        # extra runtime dependency is utils3d_moge.
        print("\n--- Installing MoGe-2 ---")
        run_command_with_retry(f'pip install "{UTILS3D_MOGE_ARCHIVE}"', "Installing utils3d_moge")
        run_command_with_retry(f'pip install --no-deps "{MOGE_ARCHIVE}"', "Installing MoGe")

        # 3. Local Wheels (cumesh, flex_gemm, o_voxel, nvdiffrast, nvdiffrec_render)
        print("\n--- Installing Custom Wheels ---")
        handled_elsewhere = (
            # spconv comes from PyPI, so ignore any stale mirror wheel left in whl/.
            ("pillow", "spconv")
            + tuple(spec["name"] for spec in DOWNLOAD_WHEELS)
            + tuple(spec["name"] for spec in LATO2_WHEELS)
        )
        for whl_file in sorted(whl_dir.glob("*.whl")):
            if whl_file.name.lower().startswith(handled_elsewhere):
                continue
            print(f"Installing: {whl_file.name}")
            run_command_with_retry(f'pip install "{whl_file}"', f"Installing {whl_file.name}")

        # 4. Large CUDA wheels fetched from release mirrors
        print("\n--- Installing Downloaded CUDA Wheels ---")
        cc_major = _gpu_compute_capability()
        if cc_major is None:
            print("Warning: could not read GPU compute capability; assuming Ampere or newer.")
        for spec in DOWNLOAD_WHEELS:
            install_downloaded_wheel(spec, whl_dir, cc_major)

        # 4.5. Some flex_gemm builds omit the Triton kernels Pixal3D's conv needs
        ensure_flex_gemm_triton()

        # 4.6. LATO.2 retopology: source (submodule contents, or the pinned
        # archive) plus the three CUDA extensions Pixal3D doesn't already have.
        if not args.no_lato2:
            ensure_lato2_source()
            print("\n--- Installing LATO.2 CUDA wheels ---")
            for spec in LATO2_WHEELS:
                install_downloaded_wheel(spec, whl_dir, cc_major)
            # A mirror wheel named plain `spconv` would shadow the module that
            # spconv-cu126 provides, so clear it out before installing.
            subprocess.run(f'"{sys.executable}" -m pip uninstall -y spconv',
                           shell=True, stdout=subprocess.DEVNULL)
            run_command_with_retry(f"pip install {SPCONV_PACKAGE}", "Installing spconv")
            # No open3d: LATO.2 only used it for the conditioning render, and its
            # OffscreenRenderer cannot run on Windows at all (see
            # lato2_render_patch.py). nvdiffrast does that job instead, which also
            # spares the install ~70 MB plus an ipython/ipywidgets dependency tree.

        # 5. NAF upsampler weights (needs natten to be importable at runtime, not now)
        if not args.skip_models:
            download_naf()

        # 6. Pre-download the HuggingFace weights
        if not args.skip_models:
            download_hf_models(include_mv=args.mv, workers=args.dl_workers)
            if not args.no_lato2:
                download_lato2_models()

        # 7. Pillow last, so nothing pulls standard Pillow back in afterwards
        if args.no_pillow_simd:
            run_command_with_retry("pip install pillow", "Installing Standard Pillow")
        else:
            install_pillow_simd(whl_dir)

        print("\nInstallation completed successfully!")

    except InstallationError as e:
        print(f"\nInstallation failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)


def verify_installation():
    """Verify installation."""
    try:
        import torch
        print(f"\nVerification successful.")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        ok = True
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            cc = torch.cuda.get_device_capability()
            print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})")

            # moge.model.v2 rather than plain `moge`: it's the only import that
            # actually exercises the utils3d_moge wiring.
            required = ["cumesh", "flex_gemm", "o_voxel", "nvdiffrast", "nvdiffrec_render",
                        "utils3d", "moge.model.v2"]
            for mod in required:
                try:
                    __import__(mod)
                    print(f"[OK] {mod} detected.")
                except ImportError:
                    print(f"[ERROR] {mod} not found.")
                    ok = False

            if _has_flex_gemm_triton():
                print("[OK] flex_gemm Triton kernels detected.")
            else:
                print("[ERROR] flex_gemm has no Triton kernels - sparse conv will fail "
                      "during generation. Re-run install.py to replace the wheel.")
                ok = False

            # Attention: flash_attn on Ampere+, xformers everywhere else.
            has_attn = False
            for mod in ["flash_attn", "xformers"]:
                try:
                    __import__(mod)
                    print(f"[OK] {mod} detected.")
                    has_attn = True
                except ImportError:
                    print(f"[WARNING] {mod} not found.")
            if not has_attn:
                print("[ERROR] No attention backend available (need flash_attn or xformers).")
                ok = False

            # LATO.2 is an optional post-step, so report but don't fail on it.
            lato2_ckpts_ok = all((LATO2_CKPT_DIR / f).exists() for f in LATO2_CKPT_FILES)
            if LATO2_MARKER.exists() and lato2_ckpts_ok:
                for mod in ["spconv", "torch_scatter", "vox2seq", "nvdiffrast"]:
                    try:
                        __import__(mod)
                        print(f"[OK] {mod} detected (LATO.2).")
                    except ImportError:
                        print(f"[WARNING] {mod} not found - LATO.2 low-poly output disabled.")
                print("[OK] LATO.2 source and checkpoints detected.")
            elif LATO2_MARKER.exists():
                print("[WARNING] LATO.2 source present but checkpoints missing - "
                      "low-poly output disabled.")
            else:
                print("[WARNING] LATO.2 not installed - low-poly output disabled.")

            # NATTEN drives the NAF upsampler; without libnatten the shape/tex
            # stages have no high-res DINOv3 features to condition on.
            try:
                import natten
                print(f"[OK] natten {natten.__version__} detected (HAS_LIBNATTEN={getattr(natten, 'HAS_LIBNATTEN', '?')}).")
            except ImportError:
                print("[WARNING] natten not found - NAF upsampling will fail. "
                      "Expected on pre-Ampere GPUs (sm_75 and older).")

        try:
            from PIL import Image
            print("[OK] PIL (Pillow) detected.")
        except ImportError:
            print("[ERROR] PIL (Pillow) not found! This is required.")
            ok = False

        return ok
    except ImportError as e:
        print(f"Verification failed: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install Pixal3D (Windows, Python 3.11 / torch 2.8.0 / CUDA 12.8)")
    parser.add_argument("--skip-models", action="store_true",
                        help="Install python packages only; skip the ~29GB of model weights.")
    parser.add_argument("--mv", action="store_true",
                        help="Also download the multi-view checkpoints used by inference_mv.py (+~17GB).")
    parser.add_argument("--no-pillow-simd", action="store_true",
                        help="Keep standard Pillow instead of trying the Pillow-SIMD wheel.")
    parser.add_argument("--models-only", action="store_true",
                        help="Skip the python packages and only (re)download weights. "
                             "Use this to resume an interrupted weights download.")
    parser.add_argument("--no-lato2", action="store_true",
                        help="Skip LATO.2 entirely (source, CUDA wheels and its ~3.6GB weights). "
                             "Pixal3D still generates; only the low-poly output is lost.")
    parser.add_argument("--dl-workers", type=int, default=HF_DEFAULT_WORKERS,
                        help=f"Parallel HuggingFace download streams (default {HF_DEFAULT_WORKERS}). "
                             f"Raise it to 4-8 on a fast connection; leave at 1 on a slow one, "
                             f"where extra streams just starve each other into timeouts.")
    args = parser.parse_args()
    if args.models_only and args.skip_models:
        parser.error("--models-only and --skip-models are contradictory.")

    # Keep model caches inside the app folder. The launcher has to set these too.
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", HF_DOWNLOAD_TIMEOUT)

    install_dependencies(args)
    if verify_installation():
        print("\nInstallation completed and verified!")
        print("\nSet these before running, so the app finds the downloaded weights:")
        print(f'  set HF_HOME={HF_HOME}')
        print(f'  set TORCH_HOME={TORCH_HOME}')
        print("\nThen:  python inference.py --image assets/images/0_img.png --output ./output.glb")
    else:
        print("\nInstallation completed but verification failed.")
        sys.exit(1)
