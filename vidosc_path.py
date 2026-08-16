"""
Locate the upstream VidOSC checkout and make its modules importable.

This repo contains only our own code. The upstream model definitions
(`model.py`, `task.py`, `dataset.py`, ...) live in a separate clone of
facebookresearch/VidOSC that is NOT vendored here -- see README.md for the
clone command and the pinned commit.

Resolution order for the checkout location:
    1. $VIDOSC_ROOT, if set
    2. ../VidOSC, i.e. a sibling of this repo (the layout README.md sets up)

Usage:
    from vidosc_path import ensure_on_path
    ensure_on_path()
    from model import FeatTimeTransformer
"""
import os
import sys
from pathlib import Path

# Upstream commit this repo's patches and scripts are written against.
# Keep in sync with README.md.
PINNED_COMMIT = "59575773c97878ea30a0228f28ab00a3ee2f1ea2"

ENV_VAR = "VIDOSC_ROOT"
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "VidOSC"

# Files that must exist for a directory to look like a real VidOSC checkout.
_MARKERS = ("model.py", "task.py", "dataset.py")

_HELP = f"""
Set {ENV_VAR} to your VidOSC checkout, or clone it as a sibling of this repo:

    git clone https://github.com/facebookresearch/VidOSC.git {DEFAULT_ROOT}
    git -C {DEFAULT_ROOT} checkout {PINNED_COMMIT}
    git -C {DEFAULT_ROOT} apply {Path(__file__).resolve().parent}/patches/*.patch
""".rstrip()


def vidosc_root() -> Path:
    """Return the VidOSC checkout path, raising if it is missing or incomplete."""
    env_value = os.environ.get(ENV_VAR)
    root = Path(env_value).expanduser().resolve() if env_value else DEFAULT_ROOT
    source = f"${ENV_VAR}" if env_value else "default sibling path"

    if not root.is_dir():
        raise FileNotFoundError(
            f"VidOSC checkout not found at {root} (from {source}).\n{_HELP}"
        )

    missing = [m for m in _MARKERS if not (root / m).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{root} (from {source}) does not look like a VidOSC checkout -- "
            f"missing {', '.join(missing)}.\n{_HELP}"
        )

    return root


def ensure_on_path() -> Path:
    """Prepend the VidOSC checkout to sys.path (idempotent) and return it."""
    root = vidosc_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


if __name__ == "__main__":
    print(ensure_on_path())
