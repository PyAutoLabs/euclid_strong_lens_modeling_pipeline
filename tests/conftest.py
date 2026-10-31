"""Pin the dataset root for the test suite.

Tests build and read samples under ``<project>/dataset`` (shipped fixtures and temporary
samples), so they must not follow a machine-local ``DATASET_ROOT`` in ``hpc/sync.conf``,
which on the laptop points at the shared Science dataset folder.
"""
import os
from pathlib import Path

os.environ["DATASET_ROOT"] = str(Path(__file__).resolve().parent.parent / "dataset")
