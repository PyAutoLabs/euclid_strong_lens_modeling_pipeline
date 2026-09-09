"""
``latest_result_per_lens_band`` must survive temporary zip extraction.

``catalogue/scripts/magnitudes.py`` opens its aggregator with
``unzip_temporary=True``, and ``config/general.yaml`` sets
``hpc.hpc_mode: true``. Together those mean a completed search leaves **only**
its ``<hash>.zip``: with no extracted directory beside it the aggregator
extracts into a temporary root instead of in place, so ``result.directory``
points into ``/tmp`` rather than under ``sample_root``.

Two things broke on that path, and this module pins both:

* the de-duplication key was ``result.directory.relative_to(sample_root)``,
  which raises ``ValueError`` for a temporary directory — a crash that takes
  down stage 7 of ``scripts/build_inspection_bundle.sh``;
* recency was ``Path(f"{result.directory}.zip").stat().st_mtime``, and no zip
  sits beside a temporary extraction, so it fell through to the extracted
  directory's own mtime — the time *this process* unpacked it, which is
  effectively identical for every result and makes the choice between
  duplicates arbitrary. That one produces a wrong catalogue rather than an
  error, so it is the more dangerous of the two.

The fix keys off the trailing ``<lens>/<stage>/<band>/<hash>`` parts (the
temporary root mirrors the scanned layout, so they are the same either way) and
reconstructs the zip under ``sample_root`` to date completion. These tests use a
stub aggregator: the logic under test is purely path and timestamp arithmetic,
and building real ``SearchOutput``s would test ``autofit``'s loader instead.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "_magnitudes_under_test",
    PROJECT_ROOT / "catalogue" / "scripts" / "magnitudes.py",
)
magnitudes = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(magnitudes)

TAIL = ("TileABC", "sersic_lens_model", "VIS", "0" * 32)


class _Result:
    """A ``SearchOutput`` stand-in: the function only reads ``.directory``."""

    def __init__(self, directory):
        self.directory = Path(directory)


class _Aggregator(list):
    """``latest_result_per_lens_band`` iterates its argument and takes ``len``."""

    grid_search_outputs = []


def _zipped(sample_root, tail, mtime):
    """Write the zip a completed search leaves under ``sample_root``."""
    zip_path = sample_root.joinpath(*tail[:-1]) / f"{tail[-1]}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(b"")
    os.utime(zip_path, (mtime, mtime))
    return zip_path


def test_a_temporary_extraction_does_not_raise(tmp_path):
    """
    The regression: a directory outside ``sample_root`` used to raise
    ``ValueError`` from ``relative_to``.
    """
    sample_root = tmp_path / "output_sed" / "dr1_prelim_grade_ab"
    _zipped(sample_root, TAIL, mtime=1_000_000)

    temporary = tmp_path / "autofit_aggregator_xyz"
    result = _Result(temporary.joinpath(*TAIL))
    result.directory.mkdir(parents=True)

    selected = magnitudes.latest_result_per_lens_band(
        _Aggregator([result]), sample_root=sample_root
    )

    assert len(selected) == 1


def test_recency_comes_from_the_zip_not_the_extraction(tmp_path):
    """
    Two extractions of the same lens/band, unpacked in the *opposite* order to
    their completion. The newer **zip** must win, so extraction order cannot
    decide the catalogue.
    """
    sample_root = tmp_path / "output_sed" / "dr1_prelim_grade_ab"
    older_tail = TAIL[:-1] + ("a" * 32,)
    newer_tail = TAIL[:-1] + ("b" * 32,)
    _zipped(sample_root, older_tail, mtime=1_000_000)
    _zipped(sample_root, newer_tail, mtime=2_000_000)

    temporary = tmp_path / "autofit_aggregator_xyz"
    results = []
    # Unpack the newer result first so its extraction mtime is the *older* of
    # the two: keying on extraction time would pick the wrong one.
    for tail in (newer_tail, older_tail):
        directory = temporary.joinpath(*tail)
        directory.mkdir(parents=True)
        results.append(_Result(directory))
    os.utime(results[0].directory, (10, 10))
    os.utime(results[1].directory, (20, 20))

    selected = magnitudes.latest_result_per_lens_band(
        _Aggregator(results), sample_root=sample_root
    )

    assert len(selected) == 1
    assert list(selected)[0].directory.name == newer_tail[-1]


def test_in_place_extraction_still_works(tmp_path):
    """
    The non-HPC case — an unzipped directory beside its zip, under
    ``sample_root`` — must behave exactly as before.
    """
    sample_root = tmp_path / "output_sed" / "dr1_prelim_grade_ab"
    _zipped(sample_root, TAIL, mtime=1_000_000)

    result = _Result(sample_root.joinpath(*TAIL))
    result.directory.mkdir(parents=True, exist_ok=True)

    selected = magnitudes.latest_result_per_lens_band(
        _Aggregator([result]), sample_root=sample_root
    )

    assert len(selected) == 1


def test_a_short_path_is_skipped(tmp_path):
    """A directory with fewer than four parts is not a per-band result."""
    sample_root = tmp_path / "output_sed"
    sample_root.mkdir(parents=True)
    result = _Result(sample_root / "stray")
    result.directory.mkdir()

    selected = magnitudes.latest_result_per_lens_band(
        _Aggregator([result]), sample_root=sample_root
    )

    assert len(selected) == 0
