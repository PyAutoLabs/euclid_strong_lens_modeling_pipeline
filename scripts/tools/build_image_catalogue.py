"""
Euclid Pipeline: Slim Image Catalogue
=====================================

Collects just three PNGs per lens into a flat ``<catalogue_dir>/<tile>/``
layout, so a whole sample can be browsed by eye without the full ten-stage
inspection bundle (CSVs, FITS, COOLEST templates, ...) that
``scripts/build_inspection_bundle.sh`` writes:

- ``subplot_fit_lp.png``  — ``initial_lens_model/vis_lp`` fit subplot
  (``image/fit.png`` in the result zip)
- ``subplot_fit_pix.png`` — ``initial_lens_model/vis_pix`` fit subplot
  (``image/fit.png`` in the result zip)
- ``rgb.png``             — the RGB subplot from the vis_lp result
  (``image/rgb.png``), falling back to the dataset's own ``rgb_0.jpg`` /
  ``rgb_0.png`` thumbnail

Nothing else is written. The collection helpers (``result_source``,
``collect_member``, ``copy_asset``, ``validate_asset``) are reused from
``scripts/tools/build_inspect.py``, so zip members are streamed to atomically
published files and the result archives are never expanded or rewritten.

By default every ``Tile*`` dataset of the sample is considered; ``--datasets``
and/or ``--manifest`` restrict it to named tiles. A lens whose ``vis_lp`` or
``vis_pix`` search has not finished is warned and skipped. It is incremental: a
lens whose three PNGs all exist is validated and reported as ``already``.

Usage
-----
    python scripts/tools/build_image_catalogue.py --sample=dr1_sep1

    python scripts/tools/build_image_catalogue.py \
        --sample=dr1_sep1 \
        --manifest=hpc/run_manifests/image_catalogue_test10.txt \
        --catalogue_dir=inspect/dr1_image_catalogue_test10
"""

import argparse
import os
import sys
import tarfile
import time
from pathlib import Path

# Import the sibling collector by file location, independent of the caller's
# cwd or of any other ``scripts`` package on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_inspect import (  # noqa: E402
    PROJECT_ROOT,
    collect_member,
    copy_asset,
    result_source,
    validate_asset,
)
try:
    import project_paths  # noqa: E402
except ImportError:  # a mirror of scripts/ alone (the inspection bundle) has no project_paths.py
    from types import SimpleNamespace

    project_paths = SimpleNamespace(dataset_root=lambda: PROJECT_ROOT / "dataset")


DEFAULT_SAMPLE = "dr1_sep1"

DATASET_PREFIX = "Tile"

TARGET_NAMES = ("subplot_fit_lp.png", "subplot_fit_pix.png", "rgb.png")


def process_dataset(
    dataset_dir: Path, dataset_main_dir: Path, catalogue_dir: Path
) -> str:
    """
    Collect one lens's three PNGs. Returns ``built``, ``already`` or
    ``skipped``.
    """
    dataset_name = dataset_dir.name

    vis_lp_source = result_source(dataset_dir / "initial_lens_model" / "vis_lp")
    vis_pix_source = result_source(dataset_dir / "initial_lens_model" / "vis_pix")

    if vis_lp_source is None or vis_pix_source is None:
        print(
            f"WARNING {dataset_name}: skipping; missing vis_lp or vis_pix result"
        )
        return "skipped"

    out_dir = catalogue_dir / dataset_name
    targets = {name: out_dir / name for name in TARGET_NAMES}

    if all(path.exists() for path in targets.values()):
        for path in targets.values():
            validate_asset(path)
        return "already"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not collect_member(
        vis_lp_source, "image/fit.png", targets["subplot_fit_lp.png"]
    ):
        print(f"WARNING {dataset_name} stage=vis_lp: missing image/fit.png")
        return "skipped"
    if not collect_member(
        vis_pix_source, "image/fit.png", targets["subplot_fit_pix.png"]
    ):
        print(f"WARNING {dataset_name} stage=vis_pix: missing image/fit.png")
        return "skipped"

    if not collect_member(vis_lp_source, "image/rgb.png", targets["rgb.png"]):
        for extension in (".jpg", ".png", ".jpeg"):
            rgb_fallback = dataset_main_dir / dataset_name / f"rgb_0{extension}"
            if rgb_fallback.exists():
                copy_asset(rgb_fallback, targets["rgb.png"])
                print(f"NOTE {dataset_name}: rgb.png from dataset {rgb_fallback.name}")
                break

    missing = [name for name, path in targets.items() if not path.exists()]
    if missing:
        print(f"WARNING {dataset_name}: incomplete; missing {', '.join(missing)}")
        return "skipped"
    return "built"


def read_names(datasets: str, manifest: str) -> list:
    """Tile names from ``--datasets`` (comma-separated) and ``--manifest``."""
    names = []
    if datasets:
        names += [name.strip() for name in datasets.split(",") if name.strip()]
    if manifest:
        for line in Path(manifest).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
    return list(dict.fromkeys(names))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect the three per-lens catalogue PNGs of a sample."
    )
    parser.add_argument(
        "--sample",
        metavar="name",
        default=DEFAULT_SAMPLE,
        help=(
            "Sample subdirectory inside the results and dataset directories. "
            f"Default: '{DEFAULT_SAMPLE}'."
        ),
    )
    parser.add_argument(
        "--output_path",
        metavar="path",
        default=None,
        help="Root results directory. Default: $PYAUTO_OUTPUT_DIR or 'output'.",
    )
    parser.add_argument(
        "--catalogue_dir",
        metavar="path",
        default=None,
        help="Directory the per-lens folders are written to. Default: inspect/<sample>_images.",
    )
    parser.add_argument(
        "--datasets",
        metavar="names",
        default=None,
        help="Comma-separated tile names to collect. Default: every Tile* dataset.",
    )
    parser.add_argument(
        "--manifest",
        metavar="path",
        default=None,
        help="File of tile names, one per line ('#' comments allowed).",
    )
    parser.add_argument(
        "--tar_to",
        metavar="path",
        default=None,
        help="After building, write an uncompressed tar of the catalogue dir here.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    def resolve(path_str, default):
        path = Path(path_str) if path_str is not None else default
        return path if path.is_absolute() else PROJECT_ROOT / path

    output_path = resolve(
        args.output_path, Path(os.environ.get("PYAUTO_OUTPUT_DIR", "output"))
    )
    catalogue_dir = resolve(args.catalogue_dir, Path("inspect") / f"{args.sample}_images")
    manifest = str(resolve(args.manifest, None)) if args.manifest else None

    results_dir = output_path / args.sample
    dataset_main_dir = project_paths.dataset_root() / args.sample

    if not results_dir.is_dir():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        return 1

    names = read_names(args.datasets, manifest)
    if names:
        dataset_dirs = []
        for name in names:
            if (results_dir / name).is_dir():
                dataset_dirs.append(results_dir / name)
            else:
                print(f"WARNING {name}: skipping; no results directory")
        n_missing_dirs = len(names) - len(dataset_dirs)
    else:
        dataset_dirs = sorted(
            path
            for path in results_dir.iterdir()
            if path.is_dir() and path.name.startswith(DATASET_PREFIX)
        )
        n_missing_dirs = 0

    catalogue_dir.mkdir(parents=True, exist_ok=True)

    counts = {"built": 0, "already": 0, "skipped": n_missing_dirs}
    started = time.time()
    print(f"Scanning {len(dataset_dirs)} datasets in {results_dir}...", flush=True)

    try:
        for dataset_dir in dataset_dirs:
            counts[process_dataset(dataset_dir, dataset_main_dir, catalogue_dir)] += 1
    finally:
        print(f"Done in {time.time() - started:.1f}s", flush=True)
        for name, count in counts.items():
            print(f"  {name}: {count}", flush=True)

    size_bytes = sum(
        path.stat().st_size
        for name in TARGET_NAMES
        for path in catalogue_dir.glob(f"*/{name}")
    )
    n_lenses = sum(1 for path in catalogue_dir.iterdir() if path.is_dir())
    per_lens = size_bytes / n_lenses / 1024 / 1024 if n_lenses else 0.0
    print(
        f"Catalogue: {n_lenses} lenses, {size_bytes / 1024 / 1024:.1f} MB "
        f"({per_lens:.2f} MB/lens) in {catalogue_dir}",
        flush=True,
    )

    if args.tar_to:
        tar_path = Path(args.tar_to)
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print(f"Writing tar -> {tar_path}", flush=True)
        with tarfile.open(tar_path, "w") as tf:
            tf.add(catalogue_dir, arcname=f"inspect/{catalogue_dir.name}")
        size_mb = tar_path.stat().st_size / 1024 / 1024
        print(
            f"Tar done in {time.time() - started:.1f}s ({size_mb:.1f} MB)", flush=True
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
