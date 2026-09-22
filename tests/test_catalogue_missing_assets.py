"""Completed fits without optional output must not abort a sample build.

Use real on-disk serialized results and the actual aggregators, without fitting.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from astropy.io import fits

from test_catalogue_latent_columns import (
    SAMPLE,
    _run_producer,
    _write_result,
    pipeline_config,  # noqa: F401
)


@pytest.fixture
def results(pipeline_config):  # noqa: F811 - imported pytest fixture
    import autofit as af
    import autolens as al

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, bulge=al.lp.Sersic)
        ),
        dataset_model=af.Model(al.DatasetModel),
    )
    model.dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(-2.0, 2.0)
    model.dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(-2.0, 2.0)
    paths = {}
    for i, lens in enumerate(("TileAAA", "TileBBB", "TileCCC")):
        for band in ("vis", "nir_h"):
            path = Path(
                _write_result(
                    model,
                    path_prefix=Path(SAMPLE) / lens,
                    name=band,
                    unique_tag="sersic_lens_model",
                    offset=1.0 + i,
                    files={"wcs": {"crval_ra_deg": 150.0 + i}},
                )
            )
            image = path / "image"
            image.mkdir(exist_ok=True)
            for name in ("galaxy_images", "model_galaxy_images"):
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        fits.ImageHDU(np.ones((3, 3)), name="GALAXY_0"),
                        fits.ImageHDU(np.full((3, 3), 2.0), name="GALAXY_1"),
                    ]
                ).writeto(image / f"{name}.fits")
            Image.new("RGB", (120, 120), color=(20 + i, 30, 40)).save(image / "fit.png")
            paths[lens, band] = path
    return pipeline_config, paths


@pytest.mark.parametrize(
    "producer,asset,product",
    [
        ("deblending", "image/galaxy_images.fits", "pre_psf.fits"),
        ("deblending", "image/model_galaxy_images.fits", "model.fits"),
        ("multi_wavelength", "image/fit.png", "fit_multi_wavelength.png"),
    ],
)
def test_missing_image_asset_continues_and_recovers(
    results, tmp_path, monkeypatch, capsys, producer, asset, product
):
    output, paths = results
    missing = paths["TileBBB", "nir_h"] / asset
    original = missing.read_bytes()
    missing.unlink()
    inspect = tmp_path / "inspect"
    _run_producer(producer, monkeypatch, output, inspect)
    for lens in ("TileAAA", "TileCCC"):
        assert (inspect / lens / product).exists()
    assert not (inspect / "TileBBB" / product).exists()
    if producer == "deblending":
        assert not (inspect / "TileBBB" / "pre_psf.fits").exists()
        with fits.open(inspect / "TileCCC" / "model.fits") as hdus:
            assert [h.name for h in hdus][1:] == [
                "VIS_GALAXY_0",
                "VIS_GALAXY_1",
                "NIR_H_GALAXY_0",
                "NIR_H_GALAXY_1",
            ]
    log = capsys.readouterr().out
    assert "built=2" in log and "skipped=1" in log and "errors=0" in log
    assert "lens=TileBBB band=nir_h" in log
    before = (inspect / "TileAAA" / product).read_bytes()
    _run_producer(producer, monkeypatch, output, inspect)
    assert (inspect / "TileAAA" / product).read_bytes() == before
    missing.write_bytes(original)
    _run_producer(producer, monkeypatch, output, inspect)
    assert (inspect / "TileBBB" / product).exists()


@pytest.mark.parametrize(
    "producer,asset",
    [
        ("magnitudes", "files/wcs.json"),
        ("magnitudes", "files/latent/latent_summary.json"),
        ("astrometric_offsets", "files/wcs.json"),
    ],
)
def test_missing_csv_asset_keeps_rows_aligned(
    results, tmp_path, monkeypatch, capsys, producer, asset
):
    output, paths = results
    (paths["TileBBB", "nir_h"] / asset).unlink()
    inspect = tmp_path / "inspect"
    _run_producer(producer, monkeypatch, output, inspect)
    target = inspect / f"{producer}.csv"
    with target.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert ("TileBBB", "nir_h") not in {(r["lens_name"], r["waveband"]) for r in rows}
    assert all(
        float(r["crval_ra_deg"])
        == {"TileAAA": 150, "TileBBB": 151, "TileCCC": 152}[r["lens_name"]]
        for r in rows
    )
    assert "built=5" in capsys.readouterr().out
    before = target.read_bytes()
    _run_producer(producer, monkeypatch, output, inspect)
    assert target.read_bytes() == before


@pytest.mark.parametrize(
    "producer,asset,error",
    [
        ("deblending", "image/galaxy_images.fits", OSError),
        ("multi_wavelength", "image/fit.png", OSError),
        ("magnitudes", "files/wcs.json", json.JSONDecodeError),
        ("astrometric_offsets", "files/wcs.json", json.JSONDecodeError),
    ],
)
def test_corruption_is_fatal(
    results, tmp_path, monkeypatch, capsys, producer, asset, error
):
    output, paths = results
    (paths["TileBBB", "nir_h"] / asset).write_bytes(b"corrupt")
    with pytest.raises(error):
        _run_producer(producer, monkeypatch, output, tmp_path / "inspect")
    assert "errors=1" in capsys.readouterr().out


@pytest.mark.parametrize("producer", ["magnitudes", "astrometric_offsets"])
def test_malformed_wcs_is_not_missing(results, tmp_path, monkeypatch, producer):
    output, paths = results
    (paths["TileBBB", "nir_h"] / "files/wcs.json").write_text("{}")
    with pytest.raises(KeyError):
        _run_producer(producer, monkeypatch, output, tmp_path / "inspect")


def test_failed_output_write_is_fatal(results, tmp_path, monkeypatch, capsys):
    output, _ = results

    def fail(*args, **kwargs):
        raise FileNotFoundError("output destination vanished")

    monkeypatch.setattr(fits.HDUList, "writeto", fail)
    with pytest.raises(FileNotFoundError, match="destination"):
        _run_producer("deblending", monkeypatch, output, tmp_path / "inspect")
    assert "errors=1" in capsys.readouterr().out


def test_all_csv_assets_missing_clears_stale_rows(results, tmp_path, monkeypatch):
    output, paths = results
    inspect = tmp_path / "inspect"
    _run_producer("magnitudes", monkeypatch, output, inspect)
    for path in paths.values():
        (path / "files/wcs.json").unlink()
    _run_producer("magnitudes", monkeypatch, output, inspect)
    with (inspect / "magnitudes.csv").open() as stream:
        assert list(csv.DictReader(stream)) == []
    assert not list(inspect.glob("*/magnitudes.csv"))


def test_complete_eight_band_pair_has_17_hdus(results, tmp_path, monkeypatch):
    import shutil
    import autofit as af

    output, paths = results
    original = paths["TileAAA", "vis"]
    model = af.SearchOutput(original).model
    bands = (
        "vis",
        "decam_g",
        "decam_r",
        "decam_i",
        "decam_z",
        "nir_y",
        "nir_j",
        "nir_h",
    )
    for band in bands[1:-1]:
        path = Path(
            _write_result(
                model,
                path_prefix=Path(SAMPLE) / "TileAAA",
                name=band,
                unique_tag="sersic_lens_model",
                offset=1.0,
            )
        )
        shutil.copytree(original / "image", path / "image", dirs_exist_ok=True)
    inspect = tmp_path / "inspect"
    _run_producer("deblending", monkeypatch, output, inspect)
    for name in ("model.fits", "pre_psf.fits"):
        with fits.open(inspect / "TileAAA" / name) as hdus:
            hdus.verify("exception")
            assert len(hdus) == 17
            assert [h.name for h in hdus][1:] == [
                f"{b.upper()}_GALAXY_{i}" for b in bands for i in (0, 1)
            ]


def test_zip_completed_result_missing_asset(results, tmp_path, monkeypatch):
    import shutil
    import zipfile

    output, paths = results
    path = paths["TileBBB", "nir_h"]
    (path / "image/galaxy_images.fits").unlink()
    with zipfile.ZipFile(path.with_suffix(".zip"), "w") as archive:
        for file in path.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(path))
    shutil.rmtree(path)
    inspect = tmp_path / "inspect"
    _run_producer("deblending", monkeypatch, output, inspect)
    assert not (inspect / "TileBBB" / "pre_psf.fits").exists()
    assert (inspect / "TileCCC" / "model.fits").exists()
    assert not path.exists()  # temporary extraction must not expand the results tree


def test_bundle_reaches_later_stages_with_missing_assets(results, tmp_path):
    import os
    import shutil
    import subprocess
    from test_catalogue_latent_columns import PROJECT_ROOT

    output, paths = results
    for asset in ("image/galaxy_images.fits", "image/fit.png", "files/wcs.json"):
        (paths["TileBBB", "nir_h"] / asset).unlink()
    # Run the real shell orchestration in an isolated mirror, not real science output.
    root = tmp_path / "bundle"
    root.mkdir()
    for folder in ("scripts", "catalogue", "config"):
        shutil.copytree(
            PROJECT_ROOT / folder,
            root / folder,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    (root / "output_sed").symlink_to(output, target_is_directory=True)
    env = {
        **os.environ,
        "OUTPUT_DIR": "output_sed",
        "SED_OUTPUT_DIR": "output_sed",
        "CREATE_ARCHIVE": "0",
    }
    run = subprocess.run(
        ["bash", "scripts/build_inspection_bundle.sh", SAMPLE, "regression"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[10/10]" in run.stdout
    inspect = root / "inspect" / f"{SAMPLE}_regression"
    assert (inspect / "TileCCC" / "model.fits").exists()
    with Image.open(inspect / "TileCCC" / "fit_multi_wavelength.png") as image:
        image.load()
    for name in ("magnitudes", "astrometric_offsets"):
        with (inspect / f"{name}.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 5


def test_inspection_copy_distinguishes_missing_member_and_corrupt_zip(tmp_path):
    import zipfile
    from scripts.tools.build_inspect import extract_zip_member

    archive = tmp_path / "result.zip"
    with zipfile.ZipFile(archive, "w"):
        pass
    assert not extract_zip_member(archive, "image/fit.png", tmp_path / "fit.png")
    archive.write_bytes(b"corrupt")
    with pytest.raises(zipfile.BadZipFile):
        extract_zip_member(archive, "image/fit.png", tmp_path / "fit.png")


@pytest.mark.parametrize(
    "filename,payload,error",
    [
        ("fit.png", b"not a png", OSError),
        ("coolest.json", b"not json", json.JSONDecodeError),
    ],
)
@pytest.mark.parametrize("archived", [False, True])
def test_collector_rejects_corrupt_assets_before_publication(
    tmp_path, filename, payload, error, archived
):
    import zipfile
    from scripts.tools.build_inspect import collect_member

    source = tmp_path / "result"
    source.mkdir()
    (source / filename).write_bytes(payload)
    if archived:
        archive = tmp_path / "result.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.write(source / filename, filename)
        source = archive
    destination = tmp_path / filename
    with pytest.raises(error):
        collect_member(source, filename, destination)
    assert not destination.exists()


def test_collector_validates_existing_complete_products(tmp_path):
    from scripts.tools.build_inspect import process_dataset

    dataset = tmp_path / "TileAAA"
    for stage in ("vis_lp", "vis_pix"):
        (dataset / "initial_lens_model" / stage / "result" / "image").mkdir(
            parents=True
        )
    inspect = tmp_path / "inspect"
    target = inspect / dataset.name
    target.mkdir(parents=True)
    for name in (
        "vis_lp_fit.png",
        "vis_pix_fit.png",
        "vis_lp_image_with_positions.png",
        "rgb.png",
        "segmentation.png",
    ):
        Image.new("RGB", (10, 10)).save(target / name)
    (target / "coolest.json").write_text('{"valid": true}')
    assert process_dataset(dataset, tmp_path / "data", inspect) == "already"
    (target / "coolest.json").write_text("invalid")
    with pytest.raises(json.JSONDecodeError):
        process_dataset(dataset, tmp_path / "data", inspect)


def test_incomplete_png_refresh_removes_old_composite(results, tmp_path, monkeypatch):
    import autofit as af

    output, paths = results
    inspect = tmp_path / "inspect"
    _run_producer("multi_wavelength", monkeypatch, output, inspect)
    target = inspect / "TileAAA/fit_multi_wavelength.png"
    assert target.exists()
    _write_result(
        af.SearchOutput(paths["TileAAA", "vis"]).model,
        path_prefix=Path(SAMPLE) / "TileAAA",
        name="nir_y",
        unique_tag="sersic_lens_model",
        offset=1.0,
    )
    _run_producer("multi_wavelength", monkeypatch, output, inspect)
    assert not target.exists()
    assert (inspect / "TileCCC/fit_multi_wavelength.png").exists()


@pytest.mark.parametrize("matching_result", [False, True])
def test_witt_wynne_refresh_clears_skipped_products(
    results, tmp_path, monkeypatch, matching_result
):
    import sys
    import witt_wynne

    output, _ = results
    inspect = tmp_path / "inspect"
    lens = inspect / "TileAAA"
    lens.mkdir(parents=True)
    for target in (inspect / "witt_wynne.csv", lens / "witt_wynne.csv"):
        target.write_text("lens_name,value\nTileAAA,1\n")
    (lens / "witt_wynne.in").write_text("old projection")
    args = [
        "witt_wynne.py",
        f"--sample={SAMPLE}",
        f"--output_path={output}",
        f"--inspect_dir={inspect}",
    ]
    if matching_result:
        args += ["--unique_tag=sersic_lens_model", "--search_name=vis"]
    monkeypatch.setattr(sys, "argv", args)
    witt_wynne.main()
    assert (inspect / "witt_wynne.csv").read_text().splitlines() == ["lens_name,value"]
    assert not (lens / "witt_wynne.csv").exists()
    assert not (lens / "witt_wynne.in").exists()
