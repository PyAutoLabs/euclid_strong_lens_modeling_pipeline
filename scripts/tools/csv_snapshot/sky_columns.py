"""
Euclid Pipeline: CSV Snapshot Sky-Frame Columns
===============================================

Stage 2b of ``scripts/tools/csv_snapshot/build.py`` (after ``assemble``, before
``package``). The fitted centres, ``ell_comps`` and shear components are in the
pipeline's native frame, which for the north-up DR1 cut-outs has +y south and
+x west (``sky_frame``; ``docs/sky_orientation/phase1.md``). This stage *adds*
sky-frame columns beside them; every existing column is left byte-identical.

For each model prefix ``P`` of the master table (``mass_``, ``lens_light_``,
``source_light_``; the unprefixed columns of ``lens_mass.csv``,
``lens_sersic.csv`` and ``source_sersic.csv``):

- ``P east_offset`` / ``P north_offset`` (arcsec, from the cut-out centre; for
  ``source_light_`` in the source plane): ``east = -centre_1``,
  ``north = -centre_0``, with every credible-interval suffix carried over (a
  lower bound of ``centre_1`` becomes the upper bound of ``east_offset``);
- ``P pa_east_of_north`` (deg, [0, 180)): the position angle of the major axis
  of the ellipse whose components are the posterior-median ``ell_comps`` (and
  of the max-likelihood ones, ``_max_lh``, where present). It is the PA of the
  median components, not a median of PAs, so no credible interval is given;
- ``mass_shear_pa_east_of_north`` (deg): the stretch direction of the external
  shear of the median ``gamma_1`` / ``gamma_2``.

The rules are ``sky_frame``'s for the DR1 cut-out WCS (``CD1_1 = -0.1"/px``,
``CD2_2 = +0.1"/px``, no rotation — identical for all 15,032 cut-outs,
2026-10-31): ``PA = (90 - phi) mod 180`` with ``phi`` the native angle
counter-clockwise from +x.

Idempotent: a table that already has the columns is rewritten with the same
values.
"""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/tools, for sky_frame

import numpy as np  # noqa: E402

import sky_frame  # noqa: E402

DR1_CD = np.array([[-0.1 / 3600.0, 0.0], [0.0, 0.1 / 3600.0]])
INTERVALS = [("_lower_1_sigma", "_upper_1_sigma"), ("_lower_3_sigma", "_upper_3_sigma")]
TABLES = {"lens_mass.csv": "", "lens_sersic.csv": "", "source_sersic.csv": ""}
PREFIXES = ("mass_", "lens_light_", "source_light_")

DESCRIPTIONS = {
    "east_offset": ("arcsec", "offset of the centre east of the cut-out centre (sky frame; = -centre_1)"),
    "north_offset": ("arcsec", "offset of the centre north of the cut-out centre (sky frame; = -centre_0)"),
    "pa_east_of_north": ("deg", "major-axis position angle east of north, [0, 180), of the ellipse with the "
                                "posterior-median ell_comps (PA of the median components, not a median PA)"),
    "pa_east_of_north_max_lh": ("deg", "major-axis position angle east of north, [0, 180), of the "
                                       "max-likelihood ell_comps"),
    "shear_pa_east_of_north": ("deg", "external-shear stretch direction east of north, [0, 180), of the "
                                      "posterior-median gamma_1 / gamma_2"),
}


def _float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value):
    return "" if value is None else repr(float(value))


def pa_from_ell_comps(e0, e1):
    if e0 is None or e1 is None:
        return None
    phi = math.degrees(0.5 * math.atan2(e0, e1))
    return float(sky_frame.pa_east_of_north(phi, DR1_CD))


def shear_pa(g1, g2):
    if g1 is None or g2 is None:
        return None
    phi = math.degrees(0.5 * math.atan2(g2, g1))
    return float(sky_frame.pa_east_of_north(phi, DR1_CD))


def sky_values(row, p):
    """The sky-frame cells of one row for model prefix ``p`` ('' for sub-tables)."""
    out = {}
    get = lambda k: _float(row.get(p + k))  # noqa: E731
    if p + "centre_0" in row:
        out[p + "east_offset"] = _neg(get("centre_1"))
        out[p + "north_offset"] = _neg(get("centre_0"))
        if p + "centre_0_max_lh" in row:
            out[p + "east_offset_max_lh"] = _neg(get("centre_1_max_lh"))
            out[p + "north_offset_max_lh"] = _neg(get("centre_0_max_lh"))
        for lo, hi in INTERVALS:
            if p + "centre_0" + lo in row:
                # negation swaps the bounds
                out[p + "east_offset" + lo] = _neg(get("centre_1" + hi))
                out[p + "east_offset" + hi] = _neg(get("centre_1" + lo))
                out[p + "north_offset" + lo] = _neg(get("centre_0" + hi))
                out[p + "north_offset" + hi] = _neg(get("centre_0" + lo))
    if p + "ell_comps_0" in row:
        out[p + "pa_east_of_north"] = pa_from_ell_comps(get("ell_comps_0"), get("ell_comps_1"))
        if p + "ell_comps_0_max_lh" in row:
            out[p + "pa_east_of_north_max_lh"] = pa_from_ell_comps(get("ell_comps_0_max_lh"),
                                                                   get("ell_comps_1_max_lh"))
    if p + "shear_gamma_1" in row:
        out[p + "shear_pa_east_of_north"] = shear_pa(get("shear_gamma_1"), get("shear_gamma_2"))
    return out


def _neg(value):
    return None if value is None else -value


def add_to_table(path: Path, prefixes):
    """Append (or refresh) the sky columns of one CSV, atomically, keeping the
    file's own line terminator (``assemble.Table`` writes ``\n``) so every
    existing cell and line prefix comes back byte-identical."""
    with open(path, "rb") as stream:
        first = stream.readline()
    terminator = "\r\n" if first.endswith(b"\r\n") else "\n"
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return 0
    new_fields = []
    for p in prefixes:
        for k in sky_values(rows[0], p):
            if k not in fields and k not in new_fields:
                new_fields.append(k)
    for row in rows:
        for p in prefixes:
            row.update({k: _fmt(v) for k, v in sky_values(row, p).items()})
    staged = path.with_suffix(".csv.sky_tmp")
    with open(staged, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + new_fields, lineterminator=terminator)
        writer.writeheader()
        writer.writerows(rows)
    staged.replace(path)
    return len(new_fields)


def add_sky_columns(out_dir, log=print):
    out_dir = Path(out_dir)
    masters = sorted(out_dir.glob("dr1_catalogue_*_master.csv"))
    if len(masters) != 1:
        raise SystemExit(f"expected one master table in {out_dir}, found {[m.name for m in masters]}")
    n = add_to_table(masters[0], PREFIXES)
    log(f"sky: {masters[0].name}: {n} sky-frame columns added")
    for name in TABLES:
        if (out_dir / name).exists():
            n = add_to_table(out_dir / name, ("",))
            log(f"sky: {name}: {n} sky-frame columns added")


def describe(col):
    """``(units, description)`` of a sky-frame column, or ``None``."""
    for p in PREFIXES + ("",):
        if not col.startswith(p):
            continue
        stem = col[len(p):]
        for key, (units, text) in DESCRIPTIONS.items():
            for suffix in ["", "_max_lh"] + [s for pair in INTERVALS for s in pair]:
                if stem == key + suffix:
                    if suffix.startswith("_lower") or suffix.startswith("_upper"):
                        text += f"; {suffix[1:].replace('_', ' ')} credible-interval bound"
                    elif suffix == "_max_lh":
                        text += "; max likelihood"
                    return units, text
    return None
