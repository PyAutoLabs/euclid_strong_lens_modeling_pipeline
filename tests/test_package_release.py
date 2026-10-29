"""
``scripts/tools/package_release.py`` on a synthetic five-tile catalogue +
dataset tree (``tests/release_fixture.py``; T5 has no lens folder, T2/T3/T4
lack some files).

Pinned: tier membership (vis never ships a multi-band product, full ships the
whole lens folder, both ship the dataset cutout), exactly ``--n_parts`` parts,
SHA256SUMS and file_manifest.csv agree with the zips, missing-file accounting,
``--excluded`` handling, the manifest-sha check and ``--dry_run``. Also that
``merge_catalogue.py``'s zip writer is unchanged by the shared-module refactor.
"""

import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools" / "csv_snapshot"))

import build as csv_build  # noqa: E402
import package_release  # noqa: E402
from release_fixture import CATALOGUE_MISSING, CUTOFF, FULL_FILES, make_project  # noqa: E402

MULTI_BAND = {"fit_multi_wavelength.png", "fit_sersic.png", "coolest_sersic.json", "magnitudes.csv", "astrometric_offsets.csv"}


def _excluded(root):
    path = root / "excluded.csv"
    path.write_text("tile,sample,reason\nT5,dr1_sep1,no finished vis_pix\n")
    return path


def _args(root, out, *extra):
    return [
        f"--catalogue_dir={root / 'inspect' / 'cat'}",
        f"--dataset_dir={root / 'dataset' / 'dr1_sep1'}",
        f"--manifest={root / 'manifest.csv'}",
        f"--csv_dir={root / 'inspect' / 'cat_csv'}",
        f"--out_dir={out}",
        *extra,
    ]


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    root = make_project(tmp_path_factory.mktemp("project"))
    csv_build.main([
        f"--project_root={root}", f"--manifest={root / 'manifest.csv'}",
        f"--out_dir={root / 'inspect' / 'cat_csv'}", f"--objects={root / 'objects.csv'}",
        f"--uncovered={root / 'uncovered.csv'}", f"--cutoff={CUTOFF}", "--workers=2",
    ])
    out = root / "inspect" / "cat_release"
    package_release.main(_args(root, out, "--tier=all", "--n_parts=2", "--workers=2",
                               f"--excluded={_excluded(root)}"))
    return root, out


def members(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        return archive.namelist()


def lens_members(tier_dir):
    out = {}
    for part in sorted(tier_dir.glob("catalogue_part_*.zip")):
        for name in members(part):
            _root, tile, rest = name.split("/", 2)
            out.setdefault(tile, set()).add(rest)
    return out


def read_rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def zip_text(zip_path, suffix):
    with zipfile.ZipFile(zip_path) as archive:
        # ``suffix`` is a root-level name ("/file_manifest.csv"), never csv/<name>.
        name = next(n for n in archive.namelist() if "/" + n.split("/", 1)[1] == suffix)
        return archive.read(name).decode()


def test_tier_layout(release):
    _root, out = release
    assert sorted(p.name for p in (out / "csv").iterdir()) == ["SHA256SUMS", "catalogue_csvs.zip", "snapshot.json"]
    for tier in ("vis", "full"):
        names = sorted(p.name for p in (out / tier).iterdir())
        assert names == ["SHA256SUMS", "catalogue_csvs.zip", "catalogue_part_01.zip", "catalogue_part_02.zip",
                         "missing_files.csv", "parts_manifest.csv", "snapshot.json"]
        assert not list((out / tier).glob(".*"))  # no staged temp zips left behind


def test_vis_never_ships_multi_band_products(release):
    _root, out = release
    lenses = lens_members(out / "vis")
    assert set(lenses) == {"T1", "T2", "T3", "T4"}  # T5 excluded
    for tile, names in lenses.items():
        assert not names & MULTI_BAND, tile
        assert {n for n in names if not n.startswith("dataset/")} <= set(FULL_FILES) - MULTI_BAND
        assert f"dataset/{tile}.fits" in names and "dataset/segmentation/lens_flux.fits" in names
        assert "dataset/positions_meta.json" in names and "dataset/rgb.jpg" not in names
    # model.fits / pre_psf.fits ship whole (every band's HDU pair) in both tiers (human choice, 2026-10-15).
    from astropy.io import fits

    expected = ["PRIMARY", "VIS_GALAXY_0", "VIS_GALAXY_1", "NIR_Y_GALAXY_0", "NIR_Y_GALAXY_1"]
    for tier in ("vis", "full"):
        for part in (out / tier).glob("catalogue_part_*.zip"):
            with zipfile.ZipFile(part) as archive:
                for name in archive.namelist():
                    if name.endswith(("/model.fits", "/pre_psf.fits")):
                        with fits.open(io.BytesIO(archive.read(name))) as hdus:
                            assert [h.name for h in hdus] == expected, (tier, name)


def test_full_ships_the_whole_lens_folder(release):
    root, out = release
    lenses = lens_members(out / "full")
    for tile, names in lenses.items():
        folder = {p.name for p in (root / "inspect" / "cat" / tile).iterdir()}
        assert {n for n in names if not n.startswith("dataset/")} == folder
    with zipfile.ZipFile(out / "full" / "catalogue_csvs.zip") as archive:
        names = archive.namelist()
    assert "cat_full/catalogue_tables/magnitudes.csv" in names
    assert "cat_full/catalogue_tables/lens_index.csv" not in names
    assert "cat_full/csv/dr1_catalogue_6_master.csv" in names


def test_parts_are_contiguous_and_cover_every_lens_once(release):
    _root, out = release
    for tier in ("vis", "full"):
        rows = list(csv.DictReader(open(out / tier / "parts_manifest.csv")))
        assert [r["tile_id"] for r in rows] == ["T1", "T2", "T3", "T4"]
        parts = [r["part_zip"] for r in rows]
        assert parts == sorted(parts) and set(parts) == {"catalogue_part_01.zip", "catalogue_part_02.zip"}
        for part in sorted((out / tier).glob("catalogue_part_*.zip")):
            tiles = {n.split("/")[1] for n in members(part)}
            assert tiles == {r["tile_id"] for r in rows if r["part_zip"] == part.name}


def test_sha256sums_and_file_manifest(release):
    _root, out = release
    for tier in ("csv", "vis", "full"):
        tier_dir = out / tier
        sums = dict(reversed(line.split("  ", 1)) for line in (tier_dir / "SHA256SUMS").read_text().splitlines())
        on_disk = {p.name for p in tier_dir.iterdir()} - {"SHA256SUMS"}
        assert set(sums) == on_disk
        for name, digest in sums.items():
            assert hashlib.sha256((tier_dir / name).read_bytes()).hexdigest() == digest
        manifest = read_rows(zip_text(tier_dir / "catalogue_csvs.zip", "/file_manifest.csv"))
        listed = {(r["zip"], r["member"]): r for r in manifest}
        for zip_path in tier_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_path) as archive:
                assert archive.testzip() is None
                for name in archive.namelist():
                    if name.split("/", 1)[1] == "file_manifest.csv":
                        continue
                    row = listed.pop((zip_path.name, name))
                    data = archive.read(name)
                    assert int(row["bytes"]) == len(data)
                    assert row["sha256"] == hashlib.sha256(data).hexdigest()
                    info = archive.getinfo(name)
                    stored = name.endswith((".fits", ".png", ".jpg"))
                    assert info.compress_type == (zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
        assert not listed  # nothing listed that is not in a zip


def test_pngs_ship_as_palette_and_fits_unchanged(release, tmp_path):
    from PIL import Image

    root, out = release
    lens = root / "inspect" / "cat" / "T1"
    with zipfile.ZipFile(sorted((out / "full").glob("catalogue_part_*.zip"))[0]) as archive:
        png = archive.read("cat_full/T1/vis_pix_fit.png")
        fits_bytes = archive.read("cat_full/T1/convergence.fits")
        cutout = archive.read("cat_full/T1/dataset/T1.fits")
    assert Image.open(io.BytesIO(png)).mode == "P" and len(png) < (lens / "vis_pix_fit.png").stat().st_size
    assert fits_bytes == (lens / "convergence.fits").read_bytes()
    assert cutout == (root / "dataset" / "dr1_sep1" / "T1" / "T1.fits").read_bytes()
    snap = json.loads((out / "full" / "snapshot.json").read_text())
    assert snap["png_palette"] and "256-colour palette" in zip_text(out / "full" / "catalogue_csvs.zip", "/README.md")
    assert not json.loads((out / "csv" / "snapshot.json").read_text())["png_palette"]

    plain = tmp_path / "plain"
    package_release.main(_args(root, plain, "--tier=vis", "--n_parts=1", "--no_png_palette",
                               f"--excluded={_excluded(root)}"))
    with zipfile.ZipFile(plain / "vis" / "catalogue_part_01.zip") as archive:
        assert archive.read("cat_vis/T1/vis_pix_fit.png") == (lens / "vis_pix_fit.png").read_bytes()
    snap = json.loads((plain / "vis" / "snapshot.json").read_text())
    assert not snap["png_palette"] and "palette" not in zip_text(plain / "vis" / "catalogue_csvs.zip", "/README.md")


def test_missing_file_accounting(release):
    _root, out = release
    for tier in ("vis", "full"):
        snap = json.loads((out / tier / "snapshot.json").read_text())
        rows = list(csv.DictReader(open(out / tier / "missing_files.csv")))
        got = {(r["tile_id"], r["product"]) for r in rows}
        expected = {("T2", "lens_mass.csv"), ("T2", package_release.SOURCE_RECONSTRUCTION),
                    ("T4", "dataset/info.json")}
        if tier == "full":
            expected |= {("T3", n) for n in CATALOGUE_MISSING["T3"]}
        assert got == expected
        assert snap["n_missing_files"] == len(expected)
        assert sum(snap["missing_counts"].values()) == len(expected)
        assert snap["missing_counts"]["lens_mass.csv"] == 1
        assert snap["lens_count"] == 4 and snap["excluded_count"] == 1 and snap["n_manifest_included"] == 5
        assert snap["cutoff_unix"] == pytest.approx(CUTOFF)
        assert snap["catalogue_build_git_revision"] == "abc1234"
        assert snap["manifest_sha256"] == hashlib.sha256((_root / "manifest.csv").read_bytes()).hexdigest()
        issues = zip_text(out / tier / "catalogue_csvs.zip", "/KNOWN_ISSUES.md")
        assert "`lens_mass.csv` | 1 |" in issues and "Excluded tiles" in issues
        excluded = read_rows(zip_text(out / tier / "catalogue_csvs.zip", "/excluded_tiles.csv"))
        assert [r["tile_id"] for r in excluded] == ["T5"]


def test_same_snapshot_stamp_across_tiers(release):
    _root, out = release
    snaps = [json.loads((out / t / "snapshot.json").read_text()) for t in ("csv", "vis", "full")]
    for key in ("cutoff_unix", "manifest_sha256", "git_head", "release_name"):
        assert len({json.dumps(s[key]) for s in snaps}) == 1, key
    assert "fit_sersic.png" not in snaps[1]["products"] and "fit_sersic.png" in snaps[2]["products"]


def test_absent_lens_not_excluded_is_an_error(release, tmp_path):
    root, _out = release
    absent = tmp_path / "absent.csv"
    with pytest.raises(SystemExit, match="no lens folder"):
        package_release.main(_args(root, tmp_path / "rel", "--tier=vis", "--n_parts=2", f"--absent_out={absent}"))
    assert not (tmp_path / "rel").exists()
    assert [r["tile"] for r in csv.DictReader(open(absent))] == ["T5"]


def test_dry_run_writes_nothing(release, tmp_path):
    root, _out = release
    summary = package_release.main(_args(root, tmp_path / "rel", "--tier=full", "--n_parts=2", "--dry_run",
                                         f"--excluded={_excluded(root)}"))
    assert not (tmp_path / "rel").exists()
    assert len(summary["full"]["parts"]) == 2 and summary["full"]["lens_count"] == 4


def test_manifest_must_match_the_csv_snapshot(release, tmp_path):
    root, _out = release
    other = tmp_path / "manifest.csv"
    other.write_text((root / "manifest.csv").read_text() + "TZ,EUCL-DR1-L000098,B,1.9,sep1,False,model_unusable,x\n")
    with pytest.raises(SystemExit, match="sha256"):
        package_release.main(_args(root, tmp_path / "rel", "--tier=csv") + [f"--manifest={other}"])


def test_refuses_to_overwrite_a_packaged_tier(release):
    root, out = release
    with pytest.raises(SystemExit, match="already packaged"):
        package_release.main(_args(root, out, "--tier=csv"))


def test_merge_catalogue_zip_writer_unchanged(tmp_path):
    spec = importlib.util.spec_from_file_location("_merge_under_test", PROJECT_ROOT / "scripts/tools/merge_catalogue.py")
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)
    cat = tmp_path / "mycat"
    (cat / "T1").mkdir(parents=True)
    (cat / "T1" / "model.fits").write_bytes(b"F" * 100)
    (cat / "T1" / "lens_mass.csv").write_text("a,b\n" * 50)
    (cat / "README.md").write_text("x")
    name = merge.write_zip((tmp_path / "p.zip", cat, ["README.md", "T1/lens_mass.csv", "T1/model.fits"]))
    assert name == "p.zip" and not list(tmp_path.glob(".p.zip.tmp"))
    with zipfile.ZipFile(tmp_path / "p.zip") as archive:
        infos = {i.filename: i.compress_type for i in archive.infolist()}
    assert infos == {"mycat/README.md": zipfile.ZIP_DEFLATED, "mycat/T1/lens_mass.csv": zipfile.ZIP_DEFLATED,
                     "mycat/T1/model.fits": zipfile.ZIP_STORED}
    zip_name, digest, tiles, n = merge.verify_zip((tmp_path / "p.zip", "mycat"))
    assert (zip_name, tiles, n) == ("p.zip", ["T1"], 3)
    assert digest == hashlib.sha256((tmp_path / "p.zip").read_bytes()).hexdigest()
    assert merge.part_name(1, 10) == "catalogue_part_01.zip" and merge.part_name(7, 120) == "catalogue_part_007.zip"
