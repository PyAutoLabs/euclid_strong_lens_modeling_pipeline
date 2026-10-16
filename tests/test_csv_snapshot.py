"""
``scripts/tools/csv_snapshot/build.py`` on a synthetic five-tile project
(``tests/release_fixture.py``): membership from the manifest, the cutoff rule,
the 2026-10-11 table set and master column layout, and no literal counts.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools" / "csv_snapshot"))

import build  # noqa: E402
from release_fixture import CUTOFF, TILES, UIDS, make_project  # noqa: E402

RELEASE_1011 = PROJECT_ROOT / "inspect_download" / "dr1_catalogue_15067_20261011_csv_only"


def run_build(root: Path, out: Path, *extra):
    return build.main([
        f"--project_root={root}", f"--manifest={root / 'manifest.csv'}", f"--out_dir={out}",
        f"--objects={root / 'objects.csv'}", f"--uncovered={root / 'uncovered.csv'}",
        f"--cutoff={CUTOFF}", "--workers=2", *extra,
    ])


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = make_project(tmp_path_factory.mktemp("project"))
    out = root / "inspect" / "cat_csv"
    run_build(root, out)
    return root, out


def rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def test_table_set_matches_the_1011_release(snapshot):
    _root, out = snapshot
    names = {p.name for p in out.iterdir() if p.is_file()}
    expected = {
        "README.txt", "astrometric_offsets.csv", "dr1_catalogue_5_master.csv",
        "excluded_one_candidates.csv", "file_manifest.csv", "lens_mass.csv", "lens_sersic.csv",
        "master_columns.csv", "metadata_errors.csv", "photometry.csv", "result_provenance.csv",
        "sample_manifest.csv", "source_clumps.csv", "source_sersic.csv", "validation.csv", "wcs.csv",
    }
    assert names == expected
    if RELEASE_1011.is_dir():
        old = {p.name for p in RELEASE_1011.iterdir()}
        renamed = {"dr1_catalogue_15067_master.csv", "excluded_three_candidates.csv"}
        assert old - renamed == expected - {"dr1_catalogue_5_master.csv", "excluded_one_candidates.csv"}


def test_membership_and_counts_come_from_the_manifest(snapshot):
    _root, out = snapshot
    master = rows(out / "dr1_catalogue_5_master.csv")
    assert [r["tile_id"] for r in master] == TILES
    assert {r["unique_identifier"] for r in master} == set(UIDS.values())
    validation = {r["check"]: r["value"] for r in rows(out / "validation.csv")}
    assert validation["rows"] == "5" and validation["parent_ab_count"] == "6"
    assert validation["excluded_parent_count"] == "1"
    assert "parity_comparisons" not in validation  # no parity claim without --parity_json
    readme = (out / "README.txt").read_text()
    assert "15067" not in readme and "15,067" not in readme and "47,412" not in readme
    assert "exactly 5 distinct" in readme and "—" not in readme
    assert len(rows(out / "sample_manifest.csv")) == 5


def test_cutoff_ignores_results_finished_later(snapshot):
    _root, out = snapshot
    mass = {r["tile_id"]: r for r in rows(out / "lens_mass.csv")}
    assert set(mass) == {"T1", "T3", "T4", "T5"}  # T2 has no vis_pix result
    assert mass["T1"]["id"] == "pix1" and float(mass["T1"]["einstein_radius"]) == pytest.approx(1.1)
    provenance = rows(out / "result_provenance.csv")
    assert not any("pix_late" in r["result_path"] for r in provenance)
    snap = json.loads((out / "_build" / "snapshot.json").read_text())
    assert snap["cutoff_unix"] == pytest.approx(CUTOFF) and snap["n_included"] == 5


def test_master_columns_and_flags(snapshot):
    _root, out = snapshot
    master = {r["tile_id"]: r for r in rows(out / "dr1_catalogue_5_master.csv")}
    assert master["T2"]["vis_pix_completed"] == "False"
    assert master["T3"]["sed_fully_complete"] == "False" and master["T3"]["missing_bands"] == "nir_y;vis"
    assert master["T1"]["sed_fully_complete"] == "True"
    columns = [r["column"] for r in rows(out / "master_columns.csv")]
    assert columns == list(master["T1"])
    if RELEASE_1011.is_dir():
        old = [r["column"] for r in rows(RELEASE_1011 / "master_columns.csv")]
        # Same base/status/mass/light block; band columns follow the bands present.
        n_fixed = old.index(next(c for c in old if c.endswith("_lens_flux")))
        assert columns[:n_fixed] == old[:n_fixed]
        assert set(columns) <= set(old)


def test_file_manifest_hashes(snapshot):
    import hashlib

    _root, out = snapshot
    for row in rows(out / "file_manifest.csv"):
        data = (out / row["file"]).read_bytes()
        assert len(data) == int(row["bytes"])
        assert hashlib.sha256(data).hexdigest() == row["sha256"]


def test_assemble_refuses_to_overwrite(snapshot):
    root, out = snapshot
    with pytest.raises(SystemExit):
        run_build(root, out, "--stage=assemble")
