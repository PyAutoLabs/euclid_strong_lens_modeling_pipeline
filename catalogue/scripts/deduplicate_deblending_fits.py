"""
Euclid Catalogue: De-duplicate Deblended FITS HDUs
==================================================

Stage 2b of ``scripts/build_inspection_bundle.sh``. Ported from the
``dr1_prelim_grade_ab`` catalogue's ``deduplicate_deblending_fits.py`` (the
``euclid`` project), which produced that release's
``fits_deduplication_validation.csv`` and ``fits_deduplication_summary.json``.

__Why__

``pre_psf.fits`` and ``model.fits`` carry one pair of HDUs per waveband, named
``<BAND>_GALAXY_0`` / ``<BAND>_GALAXY_1``. A lens whose band was re-fitted (a
retry or recovery sweep) has two completed results for that band; collected
naively they give two HDUs with the same EXTNAME and a reader cannot tell which
is current. Stage 2 (``deblending.py``) now keeps only the newest result per
band (``catalogue_util.latest_result_per_lens_band``), so this stage is the
guard: it checks every bundle, rewrites any file that still repeats an EXTNAME
(keeping the first HDU of each name — the band order stage 2 wrote), refuses a
file that carries a ``VIS_PIX`` extension (the deblending bundles hold Sersic
fits only), and records what it saw.

__Outputs__ (in ``--inspect_dir``, or ``--qa_dir`` if given)

- ``fits_deduplication_validation.csv`` — one row per FITS file checked, the
  reference release's columns: ``duplicate_extnames, duplicate_extnames_after,
  duplicate_extnames_removed, fits_name, input_hdus, output_hdus, output_path,
  tile_name, unique_extnames, vis_pix_extnames_after``. ``output_path`` is the
  file's path relative to the bundle root (``<tile>/<fits_name>``), so the CSV
  stays true when chunk bundles are merged into one catalogue.
- ``fits_deduplication_summary.json`` — counts and HDU-count distributions.

Files are rewritten in place only when they actually repeat an EXTNAME.

Usage
-----
    python catalogue/scripts/deduplicate_deblending_fits.py \
        --inspect_dir=inspect/dr1_sep1_rest
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import catalogue_util

FITS_NAMES = ("model.fits", "pre_psf.fits")

VALIDATION_COLUMNS = [
    "duplicate_extnames",
    "duplicate_extnames_after",
    "duplicate_extnames_removed",
    "fits_name",
    "input_hdus",
    "output_hdus",
    "output_path",
    "tile_name",
    "unique_extnames",
    "vis_pix_extnames_after",
]


def deduplicate_fits(path: Path) -> dict:
    """Check one FITS file; rewrite it in place only if an EXTNAME repeats."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        input_hdus = len(hdul)
        kept = [hdul[0].copy()]
        seen = []
        duplicates = []
        for hdu in hdul[1:]:
            extname = hdu.header.get("EXTNAME")
            if extname in seen:
                duplicates.append(str(extname))
                continue
            seen.append(extname)
            kept.append(hdu.copy())

    if duplicates:
        staged = path.with_name(f".{path.name}.dedup.tmp")
        fits.HDUList(kept).writeto(staged, overwrite=True)
        staged.replace(path)

    with fits.open(path, memmap=False) as hdul:
        extnames = [str(hdu.header.get("EXTNAME", "")) for hdu in hdul[1:]]
        output_hdus = len(hdul)
    duplicate_after = len(extnames) - len(set(extnames))
    vis_pix_after = [name for name in extnames if "VIS_PIX" in name]

    return {
        "duplicate_extnames": ";".join(duplicates),
        "duplicate_extnames_after": duplicate_after,
        "duplicate_extnames_removed": len(duplicates),
        "input_hdus": input_hdus,
        "output_hdus": output_hdus,
        "unique_extnames": len(set(seen)),
        "vis_pix_extnames_after": ";".join(vis_pix_after),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Guard pre_psf.fits / model.fits against repeated EXTNAMEs."
    )
    parser.add_argument("--inspect_dir", required=True, metavar="path")
    parser.add_argument(
        "--qa_dir",
        default=None,
        metavar="path",
        help="Where the validation CSV / summary JSON go. Default: --inspect_dir.",
    )
    parser.add_argument(
        "--datasets_file",
        default=None,
        metavar="path",
        help="Only check these lens folders (one name per line).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inspect_dir = Path(args.inspect_dir)
    if not inspect_dir.is_absolute():
        inspect_dir = catalogue_util.PROJECT_ROOT / inspect_dir
    qa_dir = Path(args.qa_dir) if args.qa_dir else inspect_dir
    qa_dir.mkdir(parents=True, exist_ok=True)

    requested = catalogue_util.requested_dataset_names(args.datasets_file)
    if requested is None:
        tiles = sorted(p.name for p in inspect_dir.iterdir() if p.is_dir())
    else:
        tiles = sorted(t for t in set(requested) if (inspect_dir / t).is_dir())

    rows = []
    hdu_counts = Counter()
    removed_counts = Counter()
    for tile in tiles:
        for fits_name in FITS_NAMES:
            path = inspect_dir / tile / fits_name
            if not path.exists():
                continue
            row = deduplicate_fits(path)
            row.update(
                tile_name=tile, fits_name=fits_name, output_path=f"{tile}/{fits_name}"
            )
            if row["duplicate_extnames_after"]:
                raise RuntimeError(f"{path} still repeats an EXTNAME")
            if row["vis_pix_extnames_after"]:
                raise RuntimeError(f"{path} carries VIS_PIX extensions")
            rows.append(row)
            hdu_counts[(fits_name, row["output_hdus"])] += 1
            removed_counts[(fits_name, row["duplicate_extnames_removed"])] += 1

    validation_csv = qa_dir / "fits_deduplication_validation.csv"
    with validation_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=VALIDATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": len(rows),
        "tiles_checked": len(tiles),
        "tiles_with_model_fits": sum(r["fits_name"] == "model.fits" for r in rows),
        "tiles_with_pre_psf_fits": sum(r["fits_name"] == "pre_psf.fits" for r in rows),
        "files_rewritten": sum(bool(r["duplicate_extnames_removed"]) for r in rows),
        "hdu_count_distribution": {
            f"{name}:{count}": n for (name, count), n in sorted(hdu_counts.items())
        },
        "duplicates_removed_distribution": {
            f"{name}:{count}": n for (name, count), n in sorted(removed_counts.items())
        },
        "validation_csv": validation_csv.name,
    }
    (qa_dir / "fits_deduplication_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
