"""
Tests for the positions rewrite (``scripts/tools/rewrite_positions.py``) and the
``write_positions`` writer it shares with ``preprocess/segmentation.py``.

The tool runs on temp copies of the committed simulated tile
(``dataset/simulated/euclid_dr1_like``, a clean quad the finder keeps) and of a
variant whose source-flux map is flat (no peaks: status ``n_lt_2``). PyAutoLens
is used only as the oracle for the ``positions.json`` bytes and for the
``positions_meta.json`` contract ``util.load_vis_dataset`` reads.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "preprocess"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))

import positions_finder as pf  # noqa: E402
import rewrite_positions as rp  # noqa: E402
import segmentation  # noqa: E402

SIMULATED = PROJECT_ROOT / "dataset" / "simulated" / "euclid_dr1_like"
OLD_POSITIONS = [[1.0, 1.0], [-1.0, -1.0], [0.5, -0.5]]


def _old_positions_json(d):
    segmentation.write_positions_json(d, OLD_POSITIONS)
    return (d / "positions.json").read_bytes()


def _tile(root, name="tile_keep", flat=False):
    """A temp copy of the simulated tile under ``root/name`` with an 'old' positions.json."""
    from astropy.io import fits

    d = Path(root) / name
    d.mkdir(parents=True)
    shutil.copy(SIMULATED / "info.json", d / "info.json")
    shutil.copy(SIMULATED / "euclid_dr1_like.fits", d / f"{name}.fits")
    shutil.copytree(SIMULATED / "segmentation", d / "segmentation")
    if flat:
        sf = d / "segmentation" / "source_flux.fits"
        fits.writeto(sf, np.zeros_like(fits.getdata(sf), dtype=np.float32), overwrite=True)
    _old_positions_json(d)
    return d


def _positions(d):
    return json.loads((d / "positions.json").read_text())["arguments"]["values"]["array"]


def _meta(d):
    return json.loads((d / pf.META_NAME).read_text())


def _csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# write_positions (segmentation.py)
# ---------------------------------------------------------------------------


def test_positions_json_bytes_match_output_to_json(tmp_path):
    import autolens as al

    values = [[0.15000000000000002, 0.55], [-0.25, -0.25], [0.05, -0.35000000000000003]]
    al.output_to_json(obj=al.Grid2DIrregular(values=values), file_path=tmp_path / "al.json")
    segmentation.write_positions_json(tmp_path, values)
    assert (tmp_path / "positions.json").read_bytes() == (tmp_path / "al.json").read_bytes()
    assert al.from_json(file_path=tmp_path / "positions.json").array.tolist() == values


def test_write_positions_matches_the_finder(tmp_path):
    d = _tile(tmp_path)
    result, meta = segmentation.write_positions(d, verbose=False)
    assert result.status == "keep" and len(result.positions) == 4
    assert _positions(d) == result.positions
    assert _meta(d) == json.loads(json.dumps(meta))
    assert meta["positions_sha"] == pf.positions_sha(d)
    assert meta["tile"] == d.name and meta["finder_version"] == pf.FINDER_VERSION


def test_write_positions_require_noise_raises(tmp_path):
    from astropy.io import fits

    d = _tile(tmp_path)
    with fits.open(d / f"{d.name}.fits") as hdul:
        fits.HDUList(hdul[:2]).writeto(d / "cut.fits")
    (d / "cut.fits").replace(d / f"{d.name}.fits")
    with pytest.raises(Exception):
        segmentation.write_positions(d, require_noise=True, verbose=False)


# ---------------------------------------------------------------------------
# rewrite_positions.py
# ---------------------------------------------------------------------------


def test_backup_is_made_once_and_never_overwritten(tmp_path):
    d = _tile(tmp_path)
    original = (d / "positions.json").read_bytes()

    row = rp.rewrite_tile(d)
    assert row["action"] == "written" and row["backup"] is True
    assert (d / rp.BACKUP_NAME).read_bytes() == original
    assert _positions(d) != OLD_POSITIONS and len(_positions(d)) == 4
    assert _meta(d)["backup"] == rp.BACKUP_NAME

    # A forced rerun rewrites positions.json but keeps the original backup.
    row = rp.rewrite_tile(d, force=True)
    assert row["action"] == "written" and row["backup"] is False
    assert (d / rp.BACKUP_NAME).read_bytes() == original


def test_rerun_skips_a_current_tile_and_redoes_a_stale_one(tmp_path):
    d = _tile(tmp_path)
    assert not rp.is_current(d)
    rp.rewrite_tile(d)
    assert rp.is_current(d)
    row = rp.rewrite_tile(d)
    assert row["action"] == "skipped" and row["status"] == "keep" and row["n_positions"] == 4

    # positions.json no longer matching the sidecar's sha: not current.
    _old_positions_json(d)
    assert not rp.is_current(d)
    assert rp.rewrite_tile(d)["action"] == "written"
    assert len(_positions(d)) == 4

    # A sidecar from segmentation.py (no backup key) is not "rewritten".
    segmentation.write_positions(d, verbose=False)
    assert not rp.is_current(d)


def test_no_positions_removes_positions_json_and_keeps_the_sidecar(tmp_path):
    import util

    d = _tile(tmp_path, "tile_flat", flat=True)
    original = (d / "positions.json").read_bytes()
    row = rp.rewrite_tile(d)
    assert (row["action"], row["status"], row["n_positions"]) == ("written", "n_lt_2", 0)
    assert not (d / "positions.json").exists()
    assert (d / rp.BACKUP_NAME).read_bytes() == original
    meta = _meta(d)
    assert meta["positions_used"] == [] and meta["positions_sha"] is None
    # load_vis_dataset reads the sidecar: fit with the positions penalty off.
    assert util.positions_likelihood_list_from_meta(d) == (True, None)
    # Rerun: current (no positions.json expected), skipped.
    assert rp.rewrite_tile(d)["action"] == "skipped"


def test_missing_inputs_are_an_error_row_not_a_crash(tmp_path):
    d = _tile(tmp_path)
    (d / "segmentation" / "source_flux.fits").unlink()
    original = (d / "positions.json").read_bytes()
    row = rp.rewrite_tile(d)
    assert row["action"] == "error" and "source_flux" in row["error"]
    assert (d / "positions.json").read_bytes() == original
    assert not (d / pf.META_NAME).exists()


def test_batch_csv_parts_and_merge(tmp_path):
    root = tmp_path / "sample"
    _tile(root, "tile_a")
    _tile(root, "tile_b", flat=True)
    _tile(root, "tile_c")
    (root / "tile_c" / "info.json").unlink()

    for part in (0, 1):
        rp.main(["--root", str(root), "--part", str(part), "--nparts", "2"])
    part0 = _csv(root / "positions_rewrite_part0.csv")
    part1 = _csv(root / "positions_rewrite_part1.csv")
    assert [r["tile"] for r in part0] == ["tile_a", "tile_c"]
    assert [r["tile"] for r in part1] == ["tile_b"]
    assert tuple(part0[0]) == rp.CSV_COLUMNS

    rp.main(["--root", str(root), "--merge"])
    rows = {r["tile"]: r for r in _csv(root / "positions_rewrite.csv")}
    assert set(rows) == {"tile_a", "tile_b", "tile_c"}
    assert (rows["tile_a"]["action"], rows["tile_a"]["status"], rows["tile_a"]["n_positions"]) == (
        "written",
        "keep",
        "4",
    )
    assert float(rows["tile_a"]["threshold"]) == pytest.approx(0.3)
    assert (rows["tile_b"]["status"], rows["tile_b"]["n_positions"]) == ("n_lt_2", "0")
    assert rows["tile_c"]["action"] == "error" and "info.json" in rows["tile_c"]["error"]

    # --tiles-file selects tiles; a name absent under the root is an error row.
    tiles = tmp_path / "tiles.txt"
    tiles.write_text("tile_a\nnot_a_tile\n")
    rp.main(["--root", str(root), "--tiles-file", str(tiles)])
    rows = {r["tile"]: r for r in _csv(root / "positions_rewrite.csv")}
    assert set(rows) == {"tile_a", "not_a_tile"}
    assert rows["tile_a"]["action"] == "skipped" and rows["not_a_tile"]["action"] == "error"


def test_cli_help_runs():
    import subprocess

    out = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "tools" / "rewrite_positions.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0 and "--nparts" in out.stdout
