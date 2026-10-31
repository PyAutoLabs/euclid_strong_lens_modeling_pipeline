"""
Euclid Pipeline: WCS Package Of The Top-Graded Lenses
=====================================================

Builds a small, shareable package of where the fitted lenses and their lensed
sources sit on the sky, for the ``N`` best lenses of a sample by v5 lens-finding
grade score. Both source models are included: ``vis_lp`` (a parametric MGE
source) and ``vis_pix`` (a pixelized source, whose bright clumps are found by
the percentile rule of ``util.pixelized_source_clumps_from``).

Selection: every tile of the grades CSV (``catalogue/grades/dr1_sep1_consolidated_manifest.csv``,
lens-finding catalogue v5; rows with ``include_in_catalogue=False`` are skipped)
is ranked by ``score``, descending (ties broken by tile name). The v5 score
alone already orders the grades (A >= 2.5 > B >= 1.5 > C), so no separate grade
key is needed. The ranking is walked from the top and the first ``N`` tiles
with a finished ``vis_lp`` *and* ``vis_pix`` search (a ``<hash>.zip`` in the
stage directory) that both carry ``files/wcs.json`` are taken. Every tile passed
over is written to ``skipped.csv`` with its reason. A stage with more than one
zip uses the most recently written one — ``build_inspect.result_source``'s
choice, which ``build_image_catalogue.py`` shares — and is noted in the log.

Written to ``<package_dir>/``:

- ``<tile>/wcs_vis_lp.json`` / ``<tile>/wcs_vis_pix.json`` — the stages'
  ``files/wcs.json`` copied byte-for-byte out of the result zips (PyAutoFit's
  typed JSON: ``{"type": "dict", "arguments": {...}}``, lists as
  ``{"type": "list", "values": [...]}``);
- ``<tile>/subplot_fit_lp.png`` / ``subplot_fit_pix.png`` / ``rgb.png`` — the
  ``image/fit.png`` of each stage and the vis_lp ``image/rgb.png`` (falling
  back to the dataset's ``rgb_0`` thumbnail), taken from the *same* zips as the
  JSONs, so the images and the coordinates always describe one fit;
- ``wcs.csv`` — one row per lens x stage (see ``WCS_COLUMNS``);
- ``source_clumps.csv`` — one row per vis_pix clump x lensed image of that
  clump (see ``CLUMP_COLUMNS``); a clump with no image rows gets one row with
  blank image columns, so every clump is listed;
- ``skipped.csv`` — the tiles passed over on the way to ``N``, and why;
- ``README.txt`` — the file and coordinate conventions, for the recipient.

``--sample`` may be a comma-separated list (``dr1_sep1_rest,dr1_v2_new``): each
grade row is then looked up under its own ``sample`` column's results tree, so
one package covers a merged multi-sample catalogue. ``--n=0`` takes every
qualifying lens. ``--skip_images`` writes only the JSONs and CSVs (the chunked
catalogue merge uses this: the catalogue's lens folders already hold the PNGs).

``--lens_mass`` optionally joins a ``lens_mass.csv`` from
``catalogue/scripts/lens_mass.py`` (a vis_pix scrape) onto ``wcs.csv`` as the
effective Einstein radius median and 1-sigma bounds. ``--zip_to`` writes a
deflated zip with the package folder as its top-level directory.

The result archives are only read, never expanded or rewritten; members are
streamed and validated with ``build_inspect``'s helpers.

Usage
-----
    python scripts/tools/build_wcs_package.py --n=100 \
        --lens_mass=inspect/dr1_sep1_rest_structure_astrometry_staging/lens_mass/lens_mass.csv \
        --package_dir=inspect/dr1_sep1_rest_wcs_top100 \
        --zip_to=inspect/dr1_sep1_rest_wcs_top100.zip
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

# Import the sibling collector by file location, independent of the caller's
# cwd or of any other ``scripts`` package on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_inspect import (  # noqa: E402
    PROJECT_ROOT,
    copy_asset,
    extract_zip_member,
    validate_asset,
)
try:
    import project_paths  # noqa: E402
except ImportError:  # a mirror of scripts/ alone (the inspection bundle) has no project_paths.py
    from types import SimpleNamespace

    project_paths = SimpleNamespace(dataset_root=lambda: PROJECT_ROOT / "dataset")


DEFAULT_SAMPLE = "dr1_sep1"

DEFAULT_GRADES = "catalogue/grades/dr1_sep1_consolidated_manifest.csv"

STAGES = ("vis_lp", "vis_pix")

PNG_NAMES = ("subplot_fit_lp.png", "subplot_fit_pix.png", "rgb.png")

WCS_COLUMNS = [
    "rank",
    "lens_name",
    "unique_identifier",
    "grade",
    "score",
    "stage",
    "result_zip",
    "crpix_x",
    "crpix_y",
    "crval_ra_deg",
    "crval_dec_deg",
    "source_model",
    "source_centre_y_arcsec",
    "source_centre_x_arcsec",
    "n_lensed_images",
    "lensed_source_image_y_arcsec",
    "lensed_source_image_x_arcsec",
    "lensed_source_image_ra_deg",
    "lensed_source_image_dec_deg",
    "source_clump_rule",
    "n_source_clumps",
    "effective_einstein_radius",
    "effective_einstein_radius_lower_1sigma",
    "effective_einstein_radius_upper_1sigma",
]

CLUMP_COLUMNS = [
    "rank",
    "lens_name",
    "clump_index",
    "peak_y_arcsec",
    "peak_x_arcsec",
    "peak_value",
    "mesh_pixels",
    "n_images",
    "image_index",
    "image_y_arcsec",
    "image_x_arcsec",
    "image_ra_deg",
    "image_dec_deg",
]

LIST_KEYS = (
    "lensed_source_image_y_arcsec",
    "lensed_source_image_x_arcsec",
    "lensed_source_image_ra_deg",
    "lensed_source_image_dec_deg",
)


def plain_from(value):
    """PyAutoFit typed JSON (``{"type": "dict"|"list", ...}``) -> plain Python."""
    if isinstance(value, dict):
        if value.get("type") == "dict" and "arguments" in value:
            return {k: plain_from(v) for k, v in value["arguments"].items()}
        if value.get("type") in ("list", "tuple") and "values" in value:
            return [plain_from(v) for v in value["values"]]
        return {k: plain_from(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain_from(v) for v in value]
    return value


def joined(values) -> str:
    """A list as one semicolon-separated CSV cell (full ``repr`` precision)."""
    return ";".join(repr(float(v)) for v in values)


def stage_zip(stage_dir: Path):
    """
    ``(zip, n_zips)`` for a finished stage: the most recently written
    ``<hash>.zip`` (``build_inspect.latest_zip``'s rule), or ``(None, 0)``.
    """
    if not stage_dir.is_dir():
        return None, 0
    zips = sorted(stage_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return (zips[0] if zips else None), len(zips)


def ranked_tiles(grades_csv: Path) -> list:
    """Grade rows with a tile and a score, best ``score`` first, ties by tile name.
    Rows a manifest marks ``include_in_catalogue=False`` are left out."""
    with open(grades_csv, newline="", encoding="utf-8-sig") as f:
        rows = [
            row for row in csv.DictReader(f)
            if row.get("score") and row.get("tile") and row.get("include_in_catalogue", "True") == "True"
        ]
    rows.sort(key=lambda row: (-float(row["score"]), row["tile"]))
    return rows


def read_lens_mass(path: Path) -> dict:
    """``lens_name -> row`` of a ``lens_mass.csv``; a repeated name keeps the last."""
    with open(path, newline="") as f:
        return {row["lens_name"]: row for row in csv.DictReader(f)}


def write_csv(path: Path, fieldnames, rows):
    """Write ``rows`` to ``path`` via a temporary file, then rename over it."""
    staged = path.with_name(f".{path.name}.tmp")
    with open(staged, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    staged.replace(path)


def check_lens(results_dir: Path, tile: str):
    """
    ``(zips, reason)``: the stage zips of a lens that qualifies, or ``None`` and
    why it does not.
    """
    tile_dir = results_dir / tile
    if not tile_dir.is_dir():
        return None, "no results directory"
    zips = {}
    for stage in STAGES:
        zip_path, n_zips = stage_zip(tile_dir / "initial_lens_model" / stage)
        if zip_path is None:
            return None, f"{stage} not finished (no zip)"
        if n_zips > 1:
            print(f"NOTE {tile} stage={stage}: {n_zips} zips; using newest {zip_path.name}")
        with zipfile.ZipFile(zip_path) as archive:
            if "files/wcs.json" not in archive.namelist():
                return None, f"{stage} zip has no files/wcs.json"
        zips[stage] = zip_path
    return zips, None


def build_lens(tile, zips, out_dir: Path, dataset_main_dir: Path, images=True):
    """Collect one lens's two JSONs and three PNGs; returns the decoded JSONs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wcs = {}
    for stage in STAGES:
        dest = out_dir / f"wcs_{stage}.json"
        if not extract_zip_member(zips[stage], "files/wcs.json", dest):
            raise RuntimeError(f"{tile}: files/wcs.json vanished from {zips[stage]}")
        wcs[stage] = plain_from(json.loads(dest.read_text()))
    if not images:
        return wcs

    for stage, name in (("vis_lp", "subplot_fit_lp.png"), ("vis_pix", "subplot_fit_pix.png")):
        if not extract_zip_member(zips[stage], "image/fit.png", out_dir / name):
            raise RuntimeError(f"{tile} stage={stage}: missing image/fit.png")

    if not extract_zip_member(zips["vis_lp"], "image/rgb.png", out_dir / "rgb.png"):
        for extension in (".jpg", ".png", ".jpeg"):
            rgb_fallback = dataset_main_dir / tile / f"rgb_0{extension}"
            if rgb_fallback.exists():
                copy_asset(rgb_fallback, out_dir / "rgb.png")
                print(f"NOTE {tile}: rgb.png from dataset {rgb_fallback.name}")
                break
        else:
            raise RuntimeError(f"{tile}: no rgb.png in vis_lp zip or dataset")
    return wcs


def rows_for_lens(rank, grade_row, zips, wcs, mass_row):
    """The ``wcs.csv`` rows (one per stage) and ``source_clumps.csv`` rows of a lens."""
    tile = grade_row["tile"]
    wcs_rows, clump_rows = [], []
    for stage in STAGES:
        record = wcs[stage]
        row = {
            "rank": rank,
            "lens_name": tile,
            "unique_identifier": grade_row["unique_identifier"],
            "grade": grade_row["grade"],
            "score": grade_row["score"],
            "stage": stage,
            "result_zip": zips[stage].name,
        }
        for key in (
            "crpix_x", "crpix_y", "crval_ra_deg", "crval_dec_deg", "source_model",
            "source_centre_y_arcsec", "source_centre_x_arcsec", "source_clump_rule",
        ):
            row[key] = record.get(key, "")
        if all(key in record for key in LIST_KEYS):
            row["n_lensed_images"] = len(record["lensed_source_image_ra_deg"])
            for key in LIST_KEYS:
                row[key] = joined(record[key])
        if stage == "vis_pix":
            clumps = record.get("source_clumps")
            row["n_source_clumps"] = "" if clumps is None else len(clumps)
            if mass_row:
                row["effective_einstein_radius"] = mass_row.get("effective_einstein_radius", "")
                row["effective_einstein_radius_lower_1sigma"] = mass_row.get(
                    "effective_einstein_radius_lower_1_sigma", ""
                )
                row["effective_einstein_radius_upper_1sigma"] = mass_row.get(
                    "effective_einstein_radius_upper_1_sigma", ""
                )
            for clump_index, clump in enumerate(clumps or []):
                base = {
                    "rank": rank,
                    "lens_name": tile,
                    "clump_index": clump_index,
                    "peak_y_arcsec": clump.get("peak_y_arcsec", ""),
                    "peak_x_arcsec": clump.get("peak_x_arcsec", ""),
                    "peak_value": clump.get("peak_value", ""),
                    "mesh_pixels": clump.get("mesh_pixels", ""),
                }
                images = list(
                    zip(
                        clump.get("image_y_arcsec") or [],
                        clump.get("image_x_arcsec") or [],
                        clump.get("image_ra_deg") or [],
                        clump.get("image_dec_deg") or [],
                    )
                )
                base["n_images"] = len(images)
                if not images:
                    clump_rows.append(base)
                for image_index, (y, x, ra, dec) in enumerate(images):
                    clump_rows.append(
                        {
                            **base,
                            "image_index": image_index,
                            "image_y_arcsec": y,
                            "image_x_arcsec": x,
                            "image_ra_deg": ra,
                            "image_dec_deg": dec,
                        }
                    )
        wcs_rows.append(row)
    return wcs_rows, clump_rows


README = """\
WCS package: top {n} {sample} lenses by v5 lens-finding grade score
{underline}

Built {today} by scripts/tools/build_wcs_package.py (euclid_dr1 project).

Selection
---------
Every tile of {grades} was ranked by score (the v5 SWAP_Score), descending;
the score alone orders the grades (A >= 2.5 > B >= 1.5 > C). Walking down that
ranking, the first {n} tiles with BOTH a finished vis_lp and a finished vis_pix
fit, each carrying files/wcs.json, were taken. Tiles passed over on the way are
listed in skipped.csv with the reason. Rank 1 = best score.
score of the {n} taken: {score_max} (rank 1) down to {score_min} (rank {n}).

The two fits
------------
- vis_lp  : lens light + mass (SIE + external shear) with a PARAMETRIC source
            (multi-Gaussian expansion). The source position is its light-profile centre.
- vis_pix : the same lens model with a PIXELIZED source reconstruction. Bright
            source "clumps" are found by the percentile rule: mesh pixels above
            0.5 x the reconstruction's 99th percentile, split into connected
            groups (source_clump_rule = "percentile"; "brightest_pixel" is the
            failsafe when no group is large enough, "none" = nothing
            reconstructed). The source position is the peak of the brightest
            clump.

Files
-----
<tile>/wcs_vis_lp.json    files/wcs.json of the vis_lp fit, copied verbatim from
<tile>/wcs_vis_pix.json   the result zip (and the same for vis_pix). PyAutoFit
                          typed JSON: dicts are {{"type": "dict", "arguments":
                          {{...}}}}, lists are {{"type": "list", "values": [...]}}.
<tile>/subplot_fit_lp.png   fit subplot of the vis_lp fit (same zip as the JSON)
<tile>/subplot_fit_pix.png  fit subplot of the vis_pix fit
<tile>/rgb.png              RGB colour image of the lens
wcs.csv            one row per lens x fit (2 rows per lens), columns below
source_clumps.csv  one row per vis_pix source clump x lensed image of that clump
skipped.csv        tiles passed over during selection, with reason

wcs.csv columns
---------------
rank, lens_name (tile), unique_identifier (v5 catalogue), grade, score,
stage (vis_lp | vis_pix), result_zip (the fit's result archive name),
crpix_x, crpix_y       1-based WCS pixel of the image-plane origin (0, 0) =
                       the tile centre that all arcsec values are measured from
crval_ra_deg, crval_dec_deg
                       RA/Dec (deg) of the maximum-likelihood LENS LIGHT centre
                       (despite the FITS-style name, not the reference pixel)
source_model           light_profile (vis_lp) | pixelized (vis_pix)
source_centre_y_arcsec, source_centre_x_arcsec
                       source-plane position of the source (see "The two fits")
n_lensed_images        number of multiple images of that source position
lensed_source_image_{{y,x}}_arcsec, lensed_source_image_{{ra,dec}}_deg
                       those images: semicolon-separated lists, one entry per
                       image, same order in all four columns. Solved with the
                       lens equation (al.PointSolver), sub-pixel.
source_clump_rule, n_source_clumps   vis_pix only
effective_einstein_radius[_lower_1sigma|_upper_1sigma]
                       vis_pix rows only, arcsec: the effective Einstein radius
                       (from the tangential critical curve) median and 1-sigma
                       bounds (not +/- errors), from a lens_mass.csv scrape of
                       the vis_pix fits; blank if not available.

source_clumps.csv columns
-------------------------
rank, lens_name, clump_index (0 = brightest), peak_y_arcsec, peak_x_arcsec
(source plane), peak_value, mesh_pixels, n_images, image_index, image_y_arcsec,
image_x_arcsec, image_ra_deg, image_dec_deg. The clump images are read off the
fit's mapper (the brightest model pixel of each image region, i.e. data-pixel
precision), not solved; they can merge two images into one arc or split one
across a critical curve, so they need not match the solver's lensed images in
wcs.csv. A clump with no image has one row with the image columns blank.

Coordinate conventions
----------------------
- arcsec values are (y, x) in the image plane (or source plane for source_* /
  peak_*), relative to the tile centre at pixel (crpix_x, crpix_y).
- IMPORTANT: +y is NOT north. The image array is loaded without a row flip, so
  +y points along increasing FITS row, i.e. SOUTH for a north-up image. Use the
  RA/Dec columns for sky positions; every RA/Dec went through the tile's own
  FITS WCS, so it is right whatever the cut-out orientation.
- RA/Dec are degrees in the frame of the tile's FITS WCS header.
"""


def main() -> int:
    args = parse_args()

    def resolve(path_str, default):
        path = Path(path_str) if path_str is not None else default
        return path if path.is_absolute() else PROJECT_ROOT / path

    output_path = resolve(
        args.output_path, Path(os.environ.get("PYAUTO_OUTPUT_DIR", "output"))
    )
    grades_csv = resolve(args.grades, None)
    package_dir = resolve(
        args.package_dir,
        Path("inspect") / f"{args.sample.replace(',', '_')}_wcs_top{args.n}",
    )
    samples = [name for name in args.sample.split(",") if name]
    lens_mass = read_lens_mass(resolve(args.lens_mass, None)) if args.lens_mass else {}

    for sample in samples:
        if not (output_path / sample).is_dir():
            print(
                f"ERROR: results directory not found: {output_path / sample}",
                file=sys.stderr,
            )
            return 1
    if package_dir.exists():
        print(f"Removing previous package {package_dir}", flush=True)
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    started = time.time()
    taken, skipped = [], []
    for grade_row in ranked_tiles(grades_csv):
        if args.n and len(taken) == args.n:
            break
        tile = grade_row["tile"]
        position = len(taken) + len(skipped) + 1
        sample = grade_row.get("sample") or samples[0]
        if sample not in samples:
            reason = f"sample {sample} (not {args.sample})"
            zips = None
        else:
            zips, reason = check_lens(output_path / sample, tile)
        if zips is None:
            print(f"SKIP {tile} (score rank {position}): {reason}", flush=True)
            skipped.append(
                {
                    "score_rank": position,
                    "lens_name": tile,
                    "grade": grade_row["grade"],
                    "score": grade_row["score"],
                    "sample": grade_row.get("sample", ""),
                    "reason": reason,
                }
            )
            continue
        taken.append((grade_row, zips))

    if args.n and len(taken) < args.n:
        print(f"WARNING: only {len(taken)} qualifying lenses", flush=True)

    wcs_rows, clump_rows = [], []
    for rank, (grade_row, zips) in enumerate(taken, start=1):
        tile = grade_row["tile"]
        dataset_main_dir = project_paths.dataset_root() / (grade_row.get("sample") or samples[0])
        wcs = build_lens(
            tile, zips, package_dir / tile, dataset_main_dir, images=not args.skip_images
        )
        mass_row = lens_mass.get(tile)
        if lens_mass and mass_row is None:
            print(f"NOTE {tile}: no lens_mass row", flush=True)
        lens_wcs, lens_clumps = rows_for_lens(rank, grade_row, zips, wcs, mass_row)
        wcs_rows += lens_wcs
        clump_rows += lens_clumps

    write_csv(package_dir / "wcs.csv", WCS_COLUMNS, wcs_rows)
    write_csv(package_dir / "source_clumps.csv", CLUMP_COLUMNS, clump_rows)
    write_csv(
        package_dir / "skipped.csv",
        ["score_rank", "lens_name", "grade", "score", "sample", "reason"],
        skipped,
    )
    title = f"WCS package: top {args.n} {args.sample} lenses by v5 lens-finding grade score"
    (package_dir / "README.txt").write_text(
        README.format(
            n=len(taken),
            sample=args.sample,
            underline="=" * len(title),
            today=date.today().isoformat(),
            grades=grades_csv.relative_to(PROJECT_ROOT)
            if grades_csv.is_relative_to(PROJECT_ROOT)
            else grades_csv,
            score_max=taken[0][0]["score"] if taken else "",
            score_min=taken[-1][0]["score"] if taken else "",
        )
    )

    validate_package(package_dir, taken, wcs_rows, clump_rows, images=not args.skip_images)
    print(
        f"Built {len(taken)} lenses ({len(skipped)} skipped) in "
        f"{time.time() - started:.1f}s -> {package_dir}",
        flush=True,
    )

    if args.zip_to:
        zip_path = resolve(args.zip_to, None)
        staged = zip_path.with_name(f".{zip_path.name}.tmp")
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package_dir.rglob("*")):
                if path.is_file():
                    archive.write(
                        path, arcname=f"{package_dir.name}/{path.relative_to(package_dir)}"
                    )
        staged.replace(zip_path)
        print(f"Zip -> {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


def validate_package(package_dir: Path, taken, wcs_rows, clump_rows, images=True):
    """Structural checks of the built package; raises on any failure."""
    lens_dirs = [p for p in package_dir.iterdir() if p.is_dir()]
    assert len(lens_dirs) == len(taken), (len(lens_dirs), len(taken))
    n_json = n_png = 0
    for lens_dir in lens_dirs:
        for stage in STAGES:
            validate_asset(lens_dir / f"wcs_{stage}.json")
            n_json += 1
        for name in PNG_NAMES if images else ():
            path = lens_dir / name
            assert path.stat().st_size > 0, path
            validate_asset(path)
            n_png += 1
    assert len(wcs_rows) == len(STAGES) * len(taken), len(wcs_rows)
    n_nan = 0
    for row in wcs_rows:
        for key in ("crval_ra_deg", "crval_dec_deg"):
            if row[key] == "" or math.isnan(float(row[key])):
                n_nan += 1
        for key in ("lensed_source_image_ra_deg", "lensed_source_image_dec_deg"):
            for value in (row.get(key) or "").split(";"):
                if value and math.isnan(float(value)):
                    n_nan += 1
    for row in clump_rows:
        for key in ("image_ra_deg", "image_dec_deg"):
            if row.get(key, "") != "" and math.isnan(float(row[key])):
                n_nan += 1
    assert n_nan == 0, f"{n_nan} NaN / blank RA-Dec values"
    print(
        f"Validated: {len(lens_dirs)} lens folders, {n_json} JSONs, {n_png} PNGs, "
        f"wcs.csv {len(wcs_rows)} rows, source_clumps.csv {len(clump_rows)} rows, "
        "no NaN RA/Dec",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package the wcs.json of the top-N graded lenses (vis_lp + vis_pix)."
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Lenses to take; 0 = all. Default: 100."
    )
    parser.add_argument(
        "--skip_images",
        action="store_true",
        help="Write only the wcs JSONs and CSVs, not the three PNGs per lens.",
    )
    parser.add_argument(
        "--sample",
        metavar="name",
        default=DEFAULT_SAMPLE,
        help=(
            f"Results sample subdirectory, or a comma-separated list of them "
            f"(each grade row read from its own sample). Default: '{DEFAULT_SAMPLE}'."
        ),
    )
    parser.add_argument(
        "--grades",
        metavar="path",
        default=DEFAULT_GRADES,
        help=f"Per-tile grades CSV (tile, grade, score[, sample]). Default: {DEFAULT_GRADES}.",
    )
    parser.add_argument(
        "--output_path",
        metavar="path",
        default=None,
        help="Root results directory. Default: $PYAUTO_OUTPUT_DIR or 'output'.",
    )
    parser.add_argument(
        "--package_dir",
        metavar="path",
        default=None,
        help="Package directory (rebuilt from scratch). Default: inspect/<sample>_wcs_top<n>.",
    )
    parser.add_argument(
        "--lens_mass",
        metavar="path",
        default=None,
        help="Optional lens_mass.csv (catalogue/scripts/lens_mass.py) to join the Einstein radius from.",
    )
    parser.add_argument(
        "--zip_to",
        metavar="path",
        default=None,
        help="After building, write a deflated zip of the package here.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
