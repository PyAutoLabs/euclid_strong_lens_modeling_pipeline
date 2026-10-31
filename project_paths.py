"""
Where this project's datasets live — stdlib only, so light tools can import it.

The datasets used to sit inside the project at ``dataset/``. On the laptop they now live in
the shared Science dataset folder (``Science/dataset/euclid_dr1/``) and the project points at
them through ``DATASET_ROOT`` in ``hpc/sync.conf`` — a machine-local file that ``hpc/sync``
never pushes. RAL has no such key, so it keeps ``<project>/dataset``.

Resolution order:

1. the ``DATASET_ROOT`` environment variable (tests pin it to ``<project>/dataset``);
2. ``DATASET_ROOT=...`` in ``hpc/sync.conf``;
3. ``<project>/dataset``.

A relative value is resolved against the project root.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SYNC_CONF = PROJECT_ROOT / "hpc" / "sync.conf"


def _from_sync_conf() -> str:
    if not SYNC_CONF.is_file():
        return ""
    for line in SYNC_CONF.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key == "DATASET_ROOT":
            return value.split("#", 1)[0].strip().strip("'\"")
    return ""


def dataset_root() -> Path:
    """The directory holding ``<sample>/<dataset>/`` trees for this machine."""
    value = os.environ.get("DATASET_ROOT", "").strip() or _from_sync_conf()
    if not value:
        return PROJECT_ROOT / "dataset"
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else PROJECT_ROOT / path
