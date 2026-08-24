"""
Move Segmentation FITS to Dataset Folders
==========================================

For each lens folder in ``Q1-Output/`` whose name matches a folder in
``dataset/q1_walmsley_old/``, move all ``.fits`` files into
``dataset/q1_walmsley_old/<lens>/segmentation/``.

Usage (from project root)::

    # Dry-run (no files moved, just shows what would happen):
    python preprocess/move_segmentation_fits.py --sample=q1_walmsley_old --dry-run

    # Move files (skips lenses where the target already has all fits):
    python preprocess/move_segmentation_fits.py --sample=q1_walmsley_old

    # Overwrite / re-move even if target already has the files:
    python preprocess/move_segmentation_fits.py --sample=q1_walmsley_old --force

    # Use a different segmentation output folder:
    python preprocess/move_segmentation_fits.py --sample=q1_walmsley_old --segmentation-name=cosmos_output
"""

import argparse
import shutil
from pathlib import Path


def main(
    sample_name: str, segmentation_name: str, dry_run: bool = False, force: bool = False
) -> None:
    Q1_OUTPUT = Path(segmentation_name) / sample_name
    DATASET_ROOT = Path("dataset") / sample_name
    walmsley_names = {d.name for d in DATASET_ROOT.iterdir() if d.is_dir()}
    q1_names = {d.name for d in Q1_OUTPUT.iterdir() if d.is_dir()}

    matched = sorted(walmsley_names & q1_names)
    print(
        f"Matched lenses: {len(matched)} (of {len(q1_names)} in Q1-Output, {len(walmsley_names)} in q1_walmsley_old)"
    )

    moved_total = skipped_total = 0

    for lens_name in matched:
        src_dir = Q1_OUTPUT / lens_name
        dst_dir = DATASET_ROOT / lens_name / "segmentation"

        fits_files = sorted(src_dir.glob("*.fits"))
        if not fits_files:
            print(f"  [skip] {lens_name}: no .fits in Q1-Output")
            continue

        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)

        moved = []
        skipped = []
        for src in fits_files:
            dst = dst_dir / src.name
            if dst.exists() and not force:
                skipped.append(src.name)
            else:
                if not dry_run:
                    shutil.move(str(src), str(dst))
                moved.append(src.name)

        if moved:
            tag = "[dry-run]" if dry_run else "[moved]"
            print(f"  {tag} {lens_name}: {', '.join(moved)}")
        if skipped:
            print(f"  [skip]  {lens_name}: already present: {', '.join(skipped)}")

        moved_total += len(moved)
        skipped_total += len(skipped)

    action = "Would move" if dry_run else "Moved"
    print(
        f"\n{action} {moved_total} file(s), skipped {skipped_total} already-present file(s)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sample",
        metavar="name",
        required=True,
        help="Sample subdirectory inside dataset/",
    )
    parser.add_argument(
        "--segmentation-name",
        metavar="name",
        default="Segmentation-all",
        help="Segmentation output folder name (default: Segmentation-all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without moving anything",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files in destination"
    )
    args = parser.parse_args()
    main(
        sample_name=args.sample,
        segmentation_name=args.segmentation_name,
        dry_run=args.dry_run,
        force=args.force,
    )
