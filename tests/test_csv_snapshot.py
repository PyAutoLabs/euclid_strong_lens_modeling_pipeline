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
from assemble import INSPECTION  # noqa: E402
from release_fixture import CUTOFF, TILES, UIDS, make_project  # noqa: E402

RELEASE_1011 = PROJECT_ROOT / "inspect" / "dr1_catalogue_15067_20261011_csv_only"


def run_build(root: Path, out: Path, *extra):
    return build.main([
        f"--project_root={root}", f"--manifest={root / 'manifest.csv'}", f"--out_dir={out}",
        f"--objects={root / 'objects.csv'}", f"--uncovered={root / 'uncovered.csv'}",
        f"--inspections={root / 'inspections.csv'}",
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
        "README.txt", "astrometric_offsets.csv", "dr1_catalogue_6_master.csv",
        "excluded_one_candidates.csv", "file_manifest.csv", "lens_mass.csv", "lens_sersic.csv",
        "master_columns.csv", "metadata_errors.csv", "photometry.csv", "result_provenance.csv",
        "sample_manifest.csv", "source_clumps.csv", "source_sersic.csv", "validation.csv", "wcs.csv",
    }
    assert names == expected
    if RELEASE_1011.is_dir():
        old = {p.name for p in RELEASE_1011.iterdir()}
        renamed = {"dr1_catalogue_15067_master.csv", "excluded_three_candidates.csv"}
        assert old - renamed == expected - {"dr1_catalogue_6_master.csv", "excluded_one_candidates.csv"}


def test_membership_and_counts_come_from_the_manifest(snapshot):
    _root, out = snapshot
    master = rows(out / "dr1_catalogue_6_master.csv")
    # Manifest file order, with the excluded object's placeholder at its manifest row.
    assert [r["tile_id"] for r in master] == TILES[:2] + [""] + TILES[2:]
    assert {r["unique_identifier"] for r in master} == set(UIDS.values()) | {"EUCL-DR1-L000099"}
    validation = {r["check"]: r["value"] for r in rows(out / "validation.csv")}
    assert validation["rows"] == "5" and validation["parent_ab_count"] == "6"
    assert validation["excluded_parent_count"] == "1" and validation["not_modelled_rows"] == "1"
    assert validation["master_rows"] == "6" and validation["unique_identifiers"] == "6"
    stats = json.loads((out / "_build" / "validation_counts.json").read_text())
    assert stats["n_included"] == 5 and stats["master_file"] == "dr1_catalogue_6_master.csv"
    assert "parity_comparisons" not in validation  # no parity claim without --parity_json
    readme = (out / "README.txt").read_text()
    assert "15067" not in readme and "15,067" not in readme and "47,412" not in readme
    assert "exactly 5 distinct" in readme and "—" not in readme
    assert "lens-finding catalogue\nv5" in readme and "v2" not in readme
    assert "(the fits stand):\n1." in readme  # T1
    assert "EUCL-DR1-L000099 (no_model): no FITS made" in readme
    assert "one row per parent v5 A/B object (6)" in readme and "one placeholder row (inspection_status\nnot_modelled)" in readme
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
    master = {r["tile_id"]: r for r in rows(out / "dr1_catalogue_6_master.csv")}
    assert master["T2"]["vis_pix_completed"] == "False"
    assert master["T3"]["sed_fully_complete"] == "False" and master["T3"]["missing_bands"] == "nir_y;vis"
    assert master["T1"]["sed_fully_complete"] == "True"
    columns = [r["column"] for r in rows(out / "master_columns.csv")]
    assert columns == list(master["T1"])
    if RELEASE_1011.is_dir():
        old = [r["column"] for r in rows(RELEASE_1011 / "master_columns.csv")]
        # Same base/status/mass/light block; band columns follow the bands present.
        # v5 renamed three identity columns; nothing else moved.
        renamed = {"v2_grade": "grade", "v2_score": "score", "origin_sample": "cutout_batch"}
        old = [renamed.get(c, c) for c in old]
        n_fixed = old.index(next(c for c in old if c.endswith("_lens_flux")))
        # The inspection block (new after 10-11) sits between the status and mass columns.
        kept = [c for c in columns if c not in INSPECTION]
        assert columns[columns.index("metadata_error_count") + 1:][:len(INSPECTION)] == INSPECTION
        assert kept[:n_fixed] == old[:n_fixed]
        assert set(kept) <= set(old)


def test_inspection_votes_and_pending_placeholders(snapshot):
    _root, out = snapshot
    master = {r["tile_id"]: r for r in rows(out / "dr1_catalogue_6_master.csv")}
    t1 = master["T1"]
    assert t1["inspection_status"] == "inspected" and t1["inspection_votes"] == "4"
    assert (t1["success_votes"], t1["failure_votes"], t1["multi_galaxy_lens_votes"]) == ("3", "1", "0")
    assert t1["inspection_consensus_75"] == "Success"
    assert master["T2"]["inspection_consensus_75"] == "No Consensus"  # 1 Multi Galaxy Lens / 1 Failure tie
    for tile in ("T3", "T4", "T5"):  # empty row, or no row at all
        assert master[tile]["inspection_status"] == "pending"
        assert all(master[tile][c] == "" for c in INSPECTION[1:])
    validation = {r["check"]: r["value"] for r in rows(out / "validation.csv")}
    assert (validation["inspection_inspected"], validation["inspection_pending"]) == ("2", "3")
    readme = (out / "README.txt").read_text()
    assert "Inspected models: 2. Pending: 3." in readme and "later snapshot" in readme
    described = {r["column"]: r for r in rows(out / "master_columns.csv")}
    assert all(described[c]["source_table"] == "catalogue/inspection/final_classifications.csv" for c in INSPECTION)


def test_not_modelled_placeholder_row(snapshot):
    _root, out = snapshot
    master = {r["unique_identifier"]: r for r in rows(out / "dr1_catalogue_6_master.csv")}
    row = master["EUCL-DR1-L000099"]
    assert (row["grade"], row["score"], row["RA_obj"], row["DEC_obj"]) == ("B", "2.1", "2.0", "2.0")
    assert row["tile_id"] == "" and row["cutout_batch"] == ""
    assert row["inspection_status"] == "not_modelled"
    flags = [c for c in row if c.endswith(("_completed", "_available", "_complete"))]
    assert len(flags) == 8 and all(row[c] == "False" for c in flags)
    assert all(row[c] == "0" for c in row if c.endswith("_count"))
    # Same literals as a modelled row with nothing done (T3 has no SED results).
    assert master[UIDS["T3"]]["photometry_band_count"] == "0" and master[UIDS["T3"]]["photometry_bands"] == ""
    filled = {"unique_identifier", "grade", "score", "RA_obj", "DEC_obj", "inspection_status"}
    blank = [c for c in row if c not in filled and c not in flags and not c.endswith("_count")]
    assert blank and all(row[c] == "" for c in blank)
    assert len(rows(out / "sample_manifest.csv")) == 5
    described = {r["column"]: r["description"] for r in rows(out / "master_columns.csv")}
    assert "not_modelled" in described["inspection_status"] and "not_modelled" in described["tile_id"]


def test_package_rejects_a_missing_placeholder(snapshot, tmp_path):
    import shutil

    import package

    root, out = snapshot
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    master = rows(copy / "dr1_catalogue_6_master.csv")
    with open(copy / "dr1_catalogue_6_master.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, list(master[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(r for r in master if r["inspection_status"] != "not_modelled")
    with pytest.raises(SystemExit):
        package.package(copy, root / "manifest.csv", root / "uncovered.csv", log=lambda *_: None)


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
