"""
Euclid Pipeline: SED Chain Completeness Audit
=============================================

For every lens in one or more manifests, compares the wavebands its cut-out
holds (every ``<BAND>_BGSUB`` HDU, exactly as ``scripts/lens_model_waveband.py``
discovers them) against the bands whose ``sersic_lens_model/<band>/`` search in
the SED results tree has completed (a ``<hash>.zip``, or an unzipped directory
with a ``.completed`` marker).

Writes ``sed_completeness.csv`` (one row per lens: expected / completed /
missing bands and a status of ``complete``, ``partial`` or ``none``) and
``sed_missing_lenses.txt`` (every lens whose status is not ``complete``, with
its status and missing bands). ``catalogue/scripts/magnitudes.py`` keeps each
completed band of a partial lens, so this file is how a partial lens is told
apart from a complete one in ``magnitudes.csv``.

Usage
-----
    python scripts/tools/sed_completeness.py --sample=dr1_sep1 \
        --manifest=hpc/run_manifests/positions_priority_250.txt \
        --manifest=hpc/run_manifests/vis_pix_sed_next5000_20260926_part01.txt \
        --out_dir=inspect/dr1_sep1_rest_magnitudes
"""

import argparse
import csv
import sys
from pathlib import Path

from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def expected_bands(fits_path: Path) -> list:
    with fits.open(fits_path, lazy_load_hdus=True) as hdu_list:
        return [
            hdu.name.replace("_BGSUB", "").lower()
            for hdu in hdu_list
            if hdu.name.endswith("_BGSUB")
        ]


def completed_bands(sed_lens_dir: Path) -> list:
    root = sed_lens_dir / "sersic_lens_model"
    if not root.is_dir():
        return []
    done = []
    for band_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(band_dir.glob("*.zip")) or any(band_dir.glob("*/.completed")):
            done.append(band_dir.name)
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--sample", default="dr1_sep1")
    parser.add_argument("--sed_output_path", default="output_sed")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else PROJECT_ROOT / p

    names = []
    for manifest in args.manifest:
        for line in resolve(manifest).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
    names = list(dict.fromkeys(names))

    sed_root = resolve(args.sed_output_path) / args.sample
    dataset_root = PROJECT_ROOT / "dataset" / args.sample
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in names:
        fits_path = dataset_root / name / f"{name}.fits"
        expected = expected_bands(fits_path) if fits_path.exists() else []
        done = completed_bands(sed_root / name)
        missing = [band for band in expected if band not in done]
        if not done:
            status = "none"
        elif missing or not expected:
            status = "partial"
        else:
            status = "complete"
        rows.append(
            {
                "lens_name": name,
                "status": status,
                "n_expected": len(expected),
                "n_completed": len(done),
                "expected_bands": ";".join(expected),
                "completed_bands": ";".join(done),
                "missing_bands": ";".join(missing),
            }
        )

    with open(out_dir / "sed_completeness.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "sed_missing_lenses.txt", "w") as f:
        f.write("# lens_name status missing_bands (lenses whose SED chain is not complete)\n")
        for row in rows:
            if row["status"] != "complete":
                f.write(f"{row['lens_name']} {row['status']} {row['missing_bands'] or '-'}\n")

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"{len(rows)} lenses: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
