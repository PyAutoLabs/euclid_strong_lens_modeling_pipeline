"""
Euclid Pipeline: Inspection File Collector
===========================================

Collects the six per-lens PNGs and the two COOLEST templates of the inspection
bundle into a flat ``inspect/<sample>/<dataset_name>/`` layout, so a whole
sample can be reviewed by scrolling one folder instead of clicking through
``output/`` and unpacking a zip per lens.

Six images are collected per lens (two of them best-effort):

- ``vis_lp_fit.png``                  — ``initial_lens_model/vis_lp`` fit subplot
- ``vis_pix_fit.png``                 — ``initial_lens_model/vis_pix`` fit subplot
- ``vis_lp_image_with_positions.png`` — the vis_lp image with the multiple-image
  positions overlaid; best-effort, absent when the lens has no ``positions.json``
- ``rgb.png``                         — the RGB subplot from the vis_lp result,
  falling back to the dataset's own ``rgb_0.jpg`` / ``rgb_0.png`` thumbnail
- ``fit_sersic.png``                  — ``sersic_lens_model/vis`` fit subplot,
  present only once that pipeline has run for the lens
- ``segmentation.png``                — copied from the dataset folder; its
  ultimate producer is ``preprocess/segmentation.py``

Two COOLEST templates are collected alongside them, both best-effort — a result
fitted before the pipeline started writing them simply has none:

- ``coolest.json``                    — ``initial_lens_model/vis_pix``'s
  ``files/coolest.json``, the COOLEST template of the pixelized-source fit
- ``coolest_sersic.json``             — ``sersic_lens_model/vis``'s
  ``files/coolest.json``, present only once that pipeline has run for the lens

This script *collects*, it never re-renders: images come out of the result zip
that PyAutoFit writes when a search finishes, or — when the results have not
been zipped, as in a ``PYAUTO_TEST_MODE`` run — out of the unzipped result
directory's ``image/`` folder. Selected zip members are streamed with
``zipfile`` to staging files; the result archive is never expanded or rewritten.

It is incremental: a lens whose targets all exist is reported as ``already`` and
validated before reuse. A lens whose ``vis_lp`` or ``vis_pix`` search has not
finished is warned and skipped — a normal state of a sample still being fitted.
Missing members produce per-lens warnings and skips. Corrupt ZIP, image or JSON
assets and output-write errors fail the stage; existing complete products are
validated before they count as already present.

Stage 1 of ``scripts/build_inspection_bundle.sh``.

Usage
-----
    python scripts/tools/build_inspect.py --sample=q1_walsmley

    python scripts/tools/build_inspect.py \
        --sample=dr1_prelim_grade_ab \
        --inspect_dir=inspect/dr1_prelim_grade_ab_run250 \
        --tar_to=/scratch/dr1_prelim_grade_ab.tar
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional
from tempfile import TemporaryDirectory

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_SAMPLE = "q1_walsmley"


def latest_zip(directory: Path) -> Optional[Path]:
    """
    The most recently written result zip in ``directory``, or ``None``.
    """
    zips = sorted(
        directory.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return zips[0] if zips else None


def latest_result_dir(directory: Path) -> Optional[Path]:
    """
    The most recently written *unzipped* result directory in ``directory`` that
    has an ``image/`` folder, or ``None``.

    PyAutoFit zips a result when the search finishes; a run that was interrupted,
    or one made in ``PYAUTO_TEST_MODE``, leaves the images in the result
    directory instead. Collecting from both means the bundle can be built off a
    smoke run as well as a production one.
    """
    candidates = [
        result_dir
        for result_dir in directory.iterdir()
        if result_dir.is_dir() and (result_dir / "image").is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def validate_asset(path: Path):
    """Decode copied assets before publishing or accepting an existing product."""
    if path.suffix.lower() == ".json":
        with path.open() as stream:
            json.load(stream)
    else:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()


def copy_asset(source: Path, dest: Path):
    """Validate and publish a file without leaving partially copied output."""
    with TemporaryDirectory(prefix=".asset-", dir=dest.parent) as temporary:
        staged = Path(temporary) / dest.name
        shutil.copyfile(source, staged)
        validate_asset(staged)
        staged.replace(dest)


def extract_zip_member(zip_path: Path, member: str, dest: Path) -> bool:
    """Only an absent archive member is optional; decode/write errors propagate."""
    with zipfile.ZipFile(zip_path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError:
            dest.unlink(missing_ok=True)
            return False
        with TemporaryDirectory(prefix=".asset-", dir=dest.parent) as temporary:
            staged = Path(temporary) / dest.name
            with archive.open(info) as source, staged.open("wb") as target:
                shutil.copyfileobj(source, target)
            validate_asset(staged)
            staged.replace(dest)
    return True


def collect_member(source, member: str, dest: Path) -> bool:
    """Collect one optional generated asset, invalidating an obsolete copy."""
    if source is None:
        dest.unlink(missing_ok=True)
        return False
    if source.is_dir():
        src = source / member
        if not src.exists():
            dest.unlink(missing_ok=True)
            return False
        copy_asset(src, dest)
        return True
    return extract_zip_member(source, member, dest)


def result_source(search_dir: Path):
    """
    The zip or unzipped result directory to collect this search's images from,
    or ``None`` when the search has not produced either.
    """
    if not search_dir.is_dir():
        return None
    return latest_zip(search_dir) or latest_result_dir(search_dir)


def process_dataset(
    dataset_dir: Path, dataset_main_dir: Path, inspect_dir: Path
) -> str:
    """
    Collect one lens's images and COOLEST templates. Returns ``built``,
    ``already``, ``skipped`` or ``error``.
    """
    dataset_name = dataset_dir.name

    vis_lp_source = result_source(dataset_dir / "initial_lens_model" / "vis_lp")
    vis_pix_source = result_source(dataset_dir / "initial_lens_model" / "vis_pix")
    sersic_source = result_source(dataset_dir / "sersic_lens_model" / "vis")

    if vis_lp_source is None or vis_pix_source is None:
        print(
            f"WARNING {dataset_name}: skipping inspection images; missing vis_lp or vis_pix result"
        )
        return "skipped"

    out_dir = inspect_dir / dataset_name
    targets = {
        "vis_lp_fit.png": out_dir / "vis_lp_fit.png",
        "vis_pix_fit.png": out_dir / "vis_pix_fit.png",
        "vis_lp_image_with_positions.png": out_dir / "vis_lp_image_with_positions.png",
        "rgb.png": out_dir / "rgb.png",
        "segmentation.png": out_dir / "segmentation.png",
        "coolest.json": out_dir / "coolest.json",
    }
    # fit_sersic.png and coolest_sersic.json are optional — only present once
    # the sersic pipeline has finished for this lens. Adding them to the targets
    # when the result exists stops the "already" check from skipping a lens that
    # has just gained one.
    if sersic_source is not None:
        targets["fit_sersic.png"] = out_dir / "fit_sersic.png"
        targets["coolest_sersic.json"] = out_dir / "coolest_sersic.json"

    if all(path.exists() for path in targets.values()):
        for path in targets.values():
            validate_asset(path)
        return "already"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not collect_member(vis_lp_source, "image/fit.png", targets["vis_lp_fit.png"]):
        print(f"WARNING {dataset_name} band=vis stage=vis_lp: missing image/fit.png")
        return "skipped"
    if not collect_member(vis_pix_source, "image/fit.png", targets["vis_pix_fit.png"]):
        print(f"WARNING {dataset_name} band=vis stage=vis_pix: missing image/fit.png")
        return "skipped"

    # Best-effort: a lens without positions.json has no positions overlay.
    collect_member(
        vis_lp_source,
        "image/image_with_positions.png",
        targets["vis_lp_image_with_positions.png"],
    )

    if not collect_member(vis_lp_source, "image/rgb.png", targets["rgb.png"]):
        for extension in (".jpg", ".png", ".jpeg"):
            rgb_fallback = dataset_main_dir / dataset_name / f"rgb_0{extension}"
            if rgb_fallback.exists():
                copy_asset(rgb_fallback, targets["rgb.png"])
                break

    segmentation_source = dataset_main_dir / dataset_name / "segmentation.png"
    if segmentation_source.exists():
        copy_asset(segmentation_source, targets["segmentation.png"])

    # Best-effort: results fitted before the pipeline wrote COOLEST templates
    # carry no `files/coolest.json`, and a run whose `coolest` package was
    # missing logged a warning instead of writing one.
    collect_member(vis_pix_source, "files/coolest.json", targets["coolest.json"])

    if sersic_source is not None:
        collect_member(sersic_source, "image/fit.png", targets["fit_sersic.png"])
        collect_member(
            sersic_source, "files/coolest.json", targets["coolest_sersic.json"]
        )

    missing = [name for name, path in targets.items() if not path.exists()]
    if missing:
        print(
            f"WARNING {dataset_name} band=vis: incomplete inspection images; missing {', '.join(missing)}"
        )
        return "skipped"
    return "built"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect the per-lens inspection PNGs of a sample."
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
        help=(
            "Root results directory. Default: $PYAUTO_OUTPUT_DIR or 'output'. "
            "Test-mode runs land in '<output>/test_mode'."
        ),
    )
    parser.add_argument(
        "--inspect_dir",
        metavar="path",
        default=None,
        help="Directory the PNGs are written to. Default: inspect/<sample>.",
    )
    parser.add_argument(
        "--dataset_prefix",
        metavar="str",
        default="",
        help=(
            "Only collect datasets whose directory name starts with this "
            "prefix (the DR1 sample uses 'Tile'). Default: every dataset."
        ),
    )
    parser.add_argument(
        "--tar_to",
        metavar="path",
        default=None,
        help="After building, write an uncompressed tar of the inspect dir here.",
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
    inspect_dir = resolve(args.inspect_dir, Path("inspect") / args.sample)

    results_dir = output_path / args.sample if args.sample else output_path
    dataset_main_dir = (
        PROJECT_ROOT / "dataset" / args.sample
        if args.sample
        else PROJECT_ROOT / "dataset"
    )

    if not results_dir.is_dir():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        return 1

    inspect_dir.mkdir(parents=True, exist_ok=True)

    counts = {"built": 0, "already": 0, "skipped": 0, "error": 0}
    started = time.time()
    dataset_dirs = sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith(args.dataset_prefix)
    )
    print(f"Scanning {len(dataset_dirs)} datasets in {results_dir}...", flush=True)

    try:
        for dataset_dir in dataset_dirs:
            counts[process_dataset(dataset_dir, dataset_main_dir, inspect_dir)] += 1
    except Exception:
        counts["error"] += 1
        raise
    finally:
        print(f"Done in {time.time() - started:.1f}s", flush=True)
        for name, count in counts.items():
            print(f"  {name}: {count}", flush=True)

    if args.tar_to:
        tar_path = Path(args.tar_to)
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        print(f"Writing tar -> {tar_path}", flush=True)
        with tarfile.open(tar_path, "w") as tf:
            tf.add(inspect_dir, arcname=f"inspect/{inspect_dir.name}")
        size_mb = tar_path.stat().st_size / 1024 / 1024
        print(
            f"Tar done in {time.time() - started:.1f}s ({size_mb:.1f} MB)", flush=True
        )

    # Missing optional members skip a lens; corrupt archives and write errors
    # propagate so the wrapper cannot report a structurally broken build as OK.
    return 0


if __name__ == "__main__":
    sys.exit(main())
