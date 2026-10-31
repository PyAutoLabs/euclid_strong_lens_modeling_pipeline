"""
``scripts/tools/regenerate_coolest.py`` rewrites released COOLEST templates (an
approved JSON change, euclid#115), so its guards are tested end to end on a
bundle built from a known tracer: the regenerated template carries the sky
position angle, the original is snapshotted with its sha256, a rerun changes
nothing and erases no record, and a template whose stored tracer does not
reproduce it is left untouched.
"""
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

al = pytest.importorskip("autolens")
pytest.importorskip("coolest")

from autolens import output_to_json  # noqa: E402
from astropy.io import fits  # noqa: E402

from scripts.tools import sky_frame  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOL = PROJECT_ROOT / "scripts" / "tools" / "regenerate_coolest.py"
LENS = "Tile000000001RA0000000000000DEC0000000000000"
PA_TRUE, Q_TRUE = 30.0, 0.6


def tracer_from():
    phi = 90.0 - PA_TRUE  # native DR1 frame
    mass = al.mp.Isothermal(centre=(-0.3, -0.2), ell_comps=al.convert.ell_comps_from(axis_ratio=Q_TRUE, angle=phi),
                            einstein_radius=1.2)
    shear = al.mp.ExternalShear(gamma_1=0.03, gamma_2=0.04)
    return al.Tracer(galaxies=[al.Galaxy(redshift=0.5, mass=mass), al.Galaxy(redshift=1.0)],
                     fields=[al.MassField(redshift=0.5, shear=shear)] if hasattr(al, "MassField") else None)


def build(tmp_path, corrupt=False):
    tracer = tracer_from()
    work = tmp_path / "work"
    work.mkdir()
    output_to_json(obj=tracer, file_path=work / "tracer.json")
    dataset = SimpleNamespace(shape_native=(100, 100), pixel_scales=(0.1, 0.1))
    stored = al.interop.coolest.to_coolest(galaxies=tracer, file_path=work / "coolest", dataset=dataset,
                                           on_unsupported="skip",
                                           metadata={"pipeline": "euclid_strong_lens_modeling_pipeline",
                                                     "search": "vis_pix", "dataset": LENS})
    stored = Path(stored)
    if corrupt:
        template = json.loads(stored.read_text())
        template["lensing_entities"][0]["mass_model"][0]["parameters"]["theta_E"]["point_estimate"]["value"] += 0.5
        stored.write_text(json.dumps(template))

    search = tmp_path / "output" / LENS / "initial_lens_model" / "vis_pix"
    search.mkdir(parents=True)
    with zipfile.ZipFile(search / "abc.zip", "w") as z:
        z.write(work / "tracer.json", "files/tracer.json")
        z.write(stored, "files/coolest.json")

    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = 150.0, -30.0
    header["CRPIX1"], header["CRPIX2"] = 50.5, 50.5
    header["CD1_1"], header["CD2_2"] = -0.1 / 3600.0, 0.1 / 3600.0
    header["CD1_2"], header["CD2_1"] = 0.0, 0.0
    (tmp_path / "dataset" / LENS).mkdir(parents=True)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((100, 100)), header=header, name="VIS_BGSUB")]).writeto(
        tmp_path / "dataset" / LENS / f"{LENS}.fits")

    (tmp_path / "inspect" / LENS).mkdir(parents=True)
    (tmp_path / "inspect" / LENS / "coolest.json").write_bytes(stored.read_bytes())
    return tmp_path / "inspect" / LENS / "coolest.json"


def run(tmp_path):
    return subprocess.run(
        [sys.executable, str(TOOL), f"--target_dir={tmp_path / 'inspect'}", f"--output_path={tmp_path / 'output'}",
         f"--dataset_path={tmp_path / 'dataset'}"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )


def reports(tmp_path):
    return sorted((tmp_path / "inspect_coolest_regeneration").glob("coolest_regeneration_report_*.csv"))


def mass_value(path, key):
    template = json.loads(Path(path).read_text())
    return template["lensing_entities"][0]["mass_model"][0]["parameters"][key]["point_estimate"]["value"]


def test__regenerates_in_the_sky_frame_snapshots_and_is_idempotent(tmp_path):
    template = build(tmp_path)
    original = template.read_bytes()
    assert mass_value(template, "phi") == pytest.approx(-PA_TRUE, abs=1e-6)  # the mirrored record

    result = run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert mass_value(template, "phi") == pytest.approx(PA_TRUE, abs=1e-6)
    assert mass_value(template, "q") == pytest.approx(Q_TRUE, abs=1e-9)
    assert mass_value(template, "center_y") == pytest.approx(0.3, abs=1e-9)
    assert mass_value(template, "center_x") == pytest.approx(-0.2, abs=1e-9)
    assert json.loads(template.read_text())["meta"]["frame"].startswith("+y north")

    snapshot = tmp_path / "inspect_coolest_regeneration" / "pre_sky_frame"
    assert (snapshot / LENS / "coolest.json").read_bytes() == original
    manifest = (snapshot / "manifest_sha256.txt").read_text()
    assert hashlib.sha256(original).hexdigest() in manifest
    assert not any(p.name.startswith(("_", "pre_")) for p in (tmp_path / "inspect").iterdir())

    regenerated = template.read_bytes()
    assert run(tmp_path).returncode == 0
    assert template.read_bytes() == regenerated
    assert (snapshot / "manifest_sha256.txt").read_text() == manifest
    with open(reports(tmp_path)[-1]) as f:
        assert [r["status"] for r in csv.DictReader(f)] == ["already"]


def test__a_template_its_tracer_does_not_reproduce_is_left_untouched(tmp_path):
    template = build(tmp_path, corrupt=True)
    original = template.read_bytes()
    result = run(tmp_path)
    assert result.returncode == 1
    assert template.read_bytes() == original
    with open(reports(tmp_path)[-1]) as f:
        assert [r["status"] for r in csv.DictReader(f)] == ["input_mismatch"]


def test__sky_pa_matches_the_conversion_rule():
    assert sky_frame.pa_east_of_north(90.0 - PA_TRUE, np.array([[-1, 0], [0, 1]]) * 0.1 / 3600) == pytest.approx(PA_TRUE)
