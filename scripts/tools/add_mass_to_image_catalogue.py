"""
Euclid Pipeline: Einstein Radius For The Slim Image Catalogue
=============================================================

Adds the lens mass model to a slim image catalogue built by
``scripts/tools/build_image_catalogue.py``, so the GUI that browses the PNGs can
also show each lens's Einstein radius.

It reads a master ``lens_mass.csv`` written by ``catalogue/scripts/lens_mass.py``
into a *staging* directory (never into the catalogue directly: ``output/`` holds
lenses the catalogue does not, and ``lens_mass.py`` makes a folder for every lens
it finds), restricts it to the lenses that have a ``Tile*`` folder in the
catalogue, and writes:

- ``<catalogue_dir>/mass.csv`` — the slim table, one row per catalogue lens:
  ``lens_name, einstein_radius, einstein_radius_lower_1sigma,
  einstein_radius_upper_1sigma`` (arcsec). ``einstein_radius`` here is the
  *effective* Einstein radius latent (``effective_einstein_radius`` in
  ``lens_mass.csv``, from the tangential critical curve), not the SIE's own
  ``einstein_radius`` parameter; the 1σ columns are bounds, not ± errors.
- ``<catalogue_dir>/<lens>/mass.csv`` — that lens's one row of the slim table.
- ``<catalogue_dir>/lens_mass.csv`` and ``<catalogue_dir>/<lens>/lens_mass.csv``
  — the full ``lens_mass.csv`` (SIE + shear + effective Einstein radius, median
  / 1σ / 3σ), restricted the same way.

Catalogue lenses with no ``lens_mass`` row (or a blank effective Einstein radius)
are listed on stdout; their per-lens mass files are removed, so a stale row from
an earlier run never survives. Every file is written to a temporary name and
renamed, so it is safe to re-run over the same catalogue at any time.

Usage
-----
    python scripts/tools/add_mass_to_image_catalogue.py \
        --lens_mass=inspect/dr1_sep1_rest_images_mass_staging/lens_mass.csv \
        --catalogue_dir=inspect/dr1_sep1_rest_images
"""

import argparse
import csv
import sys
from pathlib import Path

DATASET_PREFIX = "Tile"

# Slim column name -> lens_mass.csv column it is copied from.
MASS_COLUMNS = {
    "lens_name": "lens_name",
    "einstein_radius": "effective_einstein_radius",
    "einstein_radius_lower_1sigma": "effective_einstein_radius_lower_1_sigma",
    "einstein_radius_upper_1sigma": "effective_einstein_radius_upper_1_sigma",
}


def write_csv(path: Path, fieldnames, rows):
    """Write ``rows`` to ``path`` via a temporary file, then rename over it."""
    staged = path.with_name(f".{path.name}.tmp")
    with open(staged, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    staged.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add mass.csv / lens_mass.csv to a slim image catalogue."
    )
    parser.add_argument(
        "--lens_mass",
        required=True,
        help="Master lens_mass.csv from catalogue/scripts/lens_mass.py (staging dir).",
    )
    parser.add_argument(
        "--catalogue_dir",
        required=True,
        help="Slim image catalogue directory (one Tile* folder per lens).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    master = Path(args.lens_mass)
    catalogue_dir = Path(args.catalogue_dir)

    lenses = sorted(
        p.name
        for p in catalogue_dir.iterdir()
        if p.is_dir() and p.name.startswith(DATASET_PREFIX)
    )

    with open(master, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    missing_cols = [c for c in MASS_COLUMNS.values() if c not in fields]
    if missing_cols:
        print(f"ERROR: {master} lacks columns {missing_cols}", file=sys.stderr)
        return 1

    # One row per lens. lens_mass.py lists each completed vis_pix fit once, so a
    # repeated lens_name means a re-fit left two results; keep the last and say so.
    by_lens = {}
    duplicates = set()
    for row in rows:
        name = row.get("lens_name")
        if name in by_lens:
            duplicates.add(name)
        by_lens[name] = row

    in_catalogue = set(lenses)
    kept = [by_lens[name] for name in lenses if name in by_lens]
    slim = [
        {slim_col: row[src_col] for slim_col, src_col in MASS_COLUMNS.items()}
        for row in kept
    ]
    blank = [r["lens_name"] for r in slim if not r["einstein_radius"]]
    missing = [name for name in lenses if name not in by_lens]

    write_csv(catalogue_dir / "lens_mass.csv", fields, kept)
    write_csv(catalogue_dir / "mass.csv", list(MASS_COLUMNS), slim)

    for row, slim_row in zip(kept, slim):
        lens_dir = catalogue_dir / row["lens_name"]
        write_csv(lens_dir / "lens_mass.csv", fields, [row])
        write_csv(lens_dir / "mass.csv", list(MASS_COLUMNS), [slim_row])

    # A catalogue lens that lost its row (should not happen) must not keep an
    # old per-lens copy from a previous run.
    for name in missing:
        for filename in ("lens_mass.csv", "mass.csv"):
            (catalogue_dir / name / filename).unlink(missing_ok=True)

    print(
        f"lens_mass master: {len(rows)} rows, {len(by_lens)} lenses "
        f"({len(set(by_lens) - in_catalogue)} not in the catalogue, ignored)"
    )
    print(
        f"catalogue: {len(lenses)} lenses; mass rows written {len(kept)}; "
        f"missing from lens_mass {len(missing)}; blank einstein_radius {len(blank)}"
    )
    for label, names in (
        ("MISSING from lens_mass", missing),
        ("BLANK einstein_radius", blank),
        ("DUPLICATE lens_name (kept last)", sorted(duplicates & in_catalogue)),
    ):
        for name in names:
            print(f"  {label}: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
