"""
Euclid Pipeline: Post-Build Catalogue Check
===========================================

Two checks on a merged catalogue ``inspect/<CATALOGUE>/`` (the output of
``hpc/batch_cpu/submit_catalogue_chunked.sh``), both of which the build's own QA
does not catch:

(a) **Dropped magnitude rows.** ``band_inventory.csv`` (QA) says which SED bands
    completed for each lens; ``magnitudes.csv`` should hold one row per completed
    band. A lens with fewer magnitude wavebands than completed bands lost a row
    in the magnitude producer while QA still marks it complete (the
    Tile102015606 case: VIS magnitude row missing, band_inventory fully
    complete). Written to ``magnitude_rows_short.csv`` with the missing
    wavebands.

(b) **Missing model.fits.** Every lens in ``lens_index.csv`` (or, if absent,
    every ``band_inventory.csv`` tile) whose folder ``<CATALOGUE>/<tile>/`` has
    no ``model.fits``. Written to ``no_model_fits.csv`` (with the reason:
    ``no_lens_folder`` or ``no_model_fits``).

A ``summary.txt`` gives the counts. Exit status is 0 whatever is found: this is
a report, not a gate.

Usage
-----
    python scripts/tools/catalogue_post_build_check.py \
        --catalogue_dir=inspect/dr1_catalogue_20261005 \
        --out_dir=inspect/dr1_catalogue_20261005_post_build_check
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def bands(field):
    return {b for b in (field or "").split(";") if b}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--catalogue_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    cat = Path(args.catalogue_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)

    # (a) magnitude rows vs QA-completed bands
    mag_bands = defaultdict(set)
    with open(cat / "magnitudes.csv", newline="") as f:
        for row in csv.DictReader(f):
            mag_bands[row["lens_name"]].add(row["waveband"])

    inventory = []
    with open(cat / "band_inventory.csv", newline="") as f:
        inventory = list(csv.DictReader(f))

    short = []
    for row in inventory:
        tile = row["tile_name"]
        completed = bands(row["completed_bands"])
        have = mag_bands.get(tile, set())
        if len(have) < len(completed):
            short.append({
                "tile_name": tile,
                "completed_band_count": len(completed),
                "magnitude_band_count": len(have),
                "missing_magnitude_bands": ";".join(sorted(completed - have)),
                "qa_fully_complete": row.get("fully_complete", ""),
            })
    with open(out / "magnitude_rows_short.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile_name", "completed_band_count",
                                          "magnitude_band_count",
                                          "missing_magnitude_bands",
                                          "qa_fully_complete"])
        w.writeheader()
        w.writerows(short)

    # (b) lenses lacking model.fits
    index_csv = cat / "lens_index.csv"
    if index_csv.exists():
        with open(index_csv, newline="") as f:
            lenses = [(r["tile_id"], r.get("sample", "")) for r in csv.DictReader(f)]
    else:
        lenses = [(r["tile_name"], "") for r in inventory]
    missing_model = []
    for tile, sample in lenses:
        folder = cat / tile
        if not folder.is_dir():
            missing_model.append({"tile_name": tile, "sample": sample, "reason": "no_lens_folder"})
        elif not (folder / "model.fits").exists():
            missing_model.append({"tile_name": tile, "sample": sample, "reason": "no_model_fits"})
    with open(out / "no_model_fits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tile_name", "sample", "reason"])
        w.writeheader()
        w.writerows(missing_model)

    summary = (
        f"catalogue                         : {cat}\n"
        f"band_inventory lenses             : {len(inventory)}\n"
        f"magnitudes.csv lenses             : {len(mag_bands)}\n"
        f"(a) magnitude rows < QA bands     : {len(short)}"
        f" (of which QA fully_complete=True: "
        f"{sum(1 for s in short if s['qa_fully_complete'] == 'True')})\n"
        f"(b) lenses checked for model.fits : {len(lenses)}\n"
        f"(b) lenses lacking model.fits     : {len(missing_model)}\n"
    )
    (out / "summary.txt").write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
