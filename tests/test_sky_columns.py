"""
The sky-frame catalogue columns (``scripts/tools/csv_snapshot/sky_columns.py``)
are additive: every existing cell must come back byte-identical, the new cells
must follow the DR1 rules (east = -x, north = -y, PA = (90 - phi) mod 180, the
bounds of a negated column swapped), and a rerun must change nothing.
"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "tools" / "csv_snapshot"))

import sky_columns  # noqa: E402

MASTER = "dr1_catalogue_2_master.csv"


def ell_comps(q, phi_deg):
    import math

    f = (1 - q) / (1 + q)
    return f * math.sin(math.radians(2 * phi_deg)), f * math.cos(math.radians(2 * phi_deg))


def write(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")  # as assemble.Table
        w.writeheader()
        w.writerows(rows)


def build(tmp_path):
    e0, e1 = ell_comps(0.7, 60.0)  # native phi 60 -> PA 30
    fields = ["unique_identifier", "mass_centre_0", "mass_centre_0_lower_1_sigma", "mass_centre_0_upper_1_sigma",
              "mass_centre_1", "mass_centre_1_lower_1_sigma", "mass_centre_1_upper_1_sigma",
              "mass_ell_comps_0", "mass_ell_comps_1", "mass_shear_gamma_1", "mass_shear_gamma_2",
              "lens_light_centre_0", "lens_light_centre_1", "lens_light_ell_comps_0", "lens_light_ell_comps_1"]
    rows = [
        dict(unique_identifier="a", mass_centre_0="0.3", mass_centre_0_lower_1_sigma="0.25",
             mass_centre_0_upper_1_sigma="0.36", mass_centre_1="-0.2", mass_centre_1_lower_1_sigma="-0.26",
             mass_centre_1_upper_1_sigma="-0.15", mass_ell_comps_0=repr(e0), mass_ell_comps_1=repr(e1),
             mass_shear_gamma_1="0.0", mass_shear_gamma_2="0.05",  # native shear angle 45 -> PA 45
             lens_light_centre_0="0.1", lens_light_centre_1="0.0", lens_light_ell_comps_0="0.0",
             lens_light_ell_comps_1="0.1"),  # native phi 0 -> PA 90
        dict(unique_identifier="b"),  # a not_modelled placeholder: every model cell blank
    ]
    write(tmp_path / MASTER, fields, rows)
    write(tmp_path / "lens_mass.csv", ["unique_identifier", "centre_0", "centre_1", "ell_comps_0", "ell_comps_1"],
          [dict(unique_identifier="a", centre_0="0.3", centre_1="-0.2", ell_comps_0=repr(e0), ell_comps_1=repr(e1))])
    return fields


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test__values_follow_the_dr1_rules_and_existing_cells_are_untouched(tmp_path):
    fields = build(tmp_path)
    before = read(tmp_path / MASTER)
    raw_before = (tmp_path / MASTER).read_bytes().split(b"\n")
    sky_columns.add_sky_columns(tmp_path, log=lambda *_: None)
    after = read(tmp_path / MASTER)
    raw_after = (tmp_path / MASTER).read_bytes().split(b"\n")

    # every line keeps its original bytes as a prefix: only cells were appended
    assert len(raw_after) == len(raw_before)
    assert all(a.startswith(b) for a, b in zip(raw_after, raw_before))
    assert b"\r" not in (tmp_path / MASTER).read_bytes()

    for b, a in zip(before, after):
        assert {k: a[k] for k in fields} == b
    with open(tmp_path / MASTER, newline="") as f:
        assert next(csv.reader(f))[: len(fields)] == fields

    a = after[0]
    assert float(a["mass_north_offset"]) == pytest.approx(-0.3)
    assert float(a["mass_east_offset"]) == pytest.approx(0.2)
    assert float(a["mass_north_offset_lower_1_sigma"]) == pytest.approx(-0.36)
    assert float(a["mass_north_offset_upper_1_sigma"]) == pytest.approx(-0.25)
    assert float(a["mass_east_offset_lower_1_sigma"]) == pytest.approx(0.15)
    assert float(a["mass_east_offset_upper_1_sigma"]) == pytest.approx(0.26)
    assert float(a["mass_pa_east_of_north"]) == pytest.approx(30.0)
    assert float(a["mass_shear_pa_east_of_north"]) == pytest.approx(45.0)
    assert float(a["lens_light_pa_east_of_north"]) == pytest.approx(90.0)
    assert all(after[1][k] == "" for k in after[1] if k != "unique_identifier")

    sub = read(tmp_path / "lens_mass.csv")[0]
    assert float(sub["pa_east_of_north"]) == pytest.approx(30.0)
    assert float(sub["north_offset"]) == pytest.approx(-0.3)


def test__rerun_is_byte_identical(tmp_path):
    build(tmp_path)
    sky_columns.add_sky_columns(tmp_path, log=lambda *_: None)
    first = (tmp_path / MASTER).read_bytes(), (tmp_path / "lens_mass.csv").read_bytes()
    sky_columns.add_sky_columns(tmp_path, log=lambda *_: None)
    assert ((tmp_path / MASTER).read_bytes(), (tmp_path / "lens_mass.csv").read_bytes()) == first


def test__describe_gives_units_for_every_new_column():
    assert sky_columns.describe("mass_pa_east_of_north")[0] == "deg"
    assert sky_columns.describe("mass_shear_pa_east_of_north")[0] == "deg"
    assert sky_columns.describe("source_light_north_offset_upper_3_sigma")[0] == "arcsec"
    assert "credible-interval" in sky_columns.describe("lens_light_east_offset_lower_1_sigma")[1]
    assert sky_columns.describe("mass_centre_0") is None
