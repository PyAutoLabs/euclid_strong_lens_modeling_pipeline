"""
Euclid Pipeline: Merge A Chunked Catalogue Build
================================================

The merge step of the chunked DR1 catalogue build
(``hpc/batch_cpu/submit_catalogue_chunked.sh``). Each array task of
``hpc/batch_cpu/submit_catalogue_chunk`` ran ``scripts/build_inspection_bundle.sh``
over one chunk of the tile list, per sample, into

    <catalogue>/_build/chunk_<k>/<sample>/          (per-lens folders + chunk CSVs)

This script turns those into ONE catalogue:

1. **Lens folders** are moved (same filesystem, so a rename) to
   ``<catalogue>/<tile>/``. Tile names must be unique across samples and
   chunks; a repeat is an error.
2. **Root CSVs** (``lens_mass``, ``lens_sersic``, ``source_sersic``,
   ``magnitudes``, ``witt_wynne``, ``astrometric_offsets``,
   ``fits_deduplication_validation``) are concatenated: one header (every chunk
   must agree; a mismatch is an error), de-duplicated on the lens key
   (``lens_name``, plus ``waveband`` for the per-band tables) keeping the last
   chunk's row, and sorted on that key. ``fits_deduplication_summary.json`` is
   recomputed over the merged validation rows.
3. **QA CSVs** ``band_inventory.csv`` and ``unresolved_manifest.csv`` — ported
   from the ``dr1_prelim_grade_ab`` release's ``scripts/audit_sed_outputs.py``
   (``euclid`` project), same columns — over every requested tile: expected
   bands from the dataset FITS ``*_BGSUB`` HDUs, completed bands from the
   ``output_sed`` ``sersic_lens_model/<band>/*.zip`` results, checkpoints.
4. **Grades**: ``grades.csv`` — one row per catalogue tile with its v2 grade and
   its real ``sample`` — from ``catalogue/grades/tile_grades_dr1_sep1_v2.csv``
   (dr1_sep1_*) and ``hpc/run_manifests/dr1_v2_new_selection_20261001.csv``
   (dr1_v2_new).
5. **WCS**: ``scripts/tools/build_wcs_package.py`` over the whole merged
   sample (``--n=0 --skip_images``, multi-sample), giving ``wcs.csv``,
   ``source_clumps.csv`` and ``wcs_skipped.csv`` at the catalogue root (the
   per-lens ``wcs_vis_lp.json`` / ``wcs_vis_pix.json`` were already collected
   by ``build_inspect.py``).
6. ``lens_index.csv`` (tile -> sample, grade, which products exist, part zip)
   and ``README.md`` (which also lists ``excluded_tiles.csv`` — tiles left out
   of the build and why — when the driver was given ``EXCLUDED_FILE``).
7. With ``--archive=chunked``: the distribution, in ``<catalogue>_dist/``:
   ``catalogue_csvs.zip`` (every root file), ``catalogue_part_NN.zip`` (whole
   lens folders, each part capped at ``--part_max_gb``; or, with ``--n_parts=k``,
   EXACTLY k parts cut contiguously in tile order at near-equal byte totals),
   ``parts_manifest.csv``
   (``tile_id, sample, part_zip``), ``SHA256SUMS``; each zip is re-opened,
   CRC-tested and checked against the manifest before the step succeeds. FITS
   / PNG / JPG members are ZIP_STORED (they barely compress), text is DEFLATED.

Usage
-----
    python scripts/tools/merge_catalogue.py --catalogue_dir=inspect/dr1_catalogue_20261006 \
        --archive=chunked --part_max_gb=10      # or --n_parts=10
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import date, datetime, timezone
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ROOT_CSVS = [
    "lens_mass.csv",
    "lens_sersic.csv",
    "source_sersic.csv",
    "magnitudes.csv",
    "witt_wynne.csv",
    "astrometric_offsets.csv",
    "fits_deduplication_validation.csv",
]

INVENTORY_COLUMNS = [
    "global_index",
    "tile_name",
    "expected_band_count",
    "expected_bands",
    "completed_band_count",
    "completed_bands",
    "missing_band_count",
    "missing_bands",
    "unexpected_completed_bands",
    "output_bands",
    "output_directory",
    "checkpoint_status",
    "checkpoint_count",
    "valid_checkpoint_count",
    "invalid_checkpoint_count",
    "last_attempted_band",
    "fully_complete",
]

GRADE_COLUMNS = [
    "tile",
    "unique_identifier",
    "match_method",
    "v1_grade",
    "v2_grade",
    "v2_score",
    "reid",
    "v1_unique_identifier",
    "sample",
]

STORED_SUFFIXES = {".fits", ".png", ".jpg", ".jpeg", ".zip", ".gz"}

INDEX_PRODUCTS = [
    "vis_lp_fit.png",
    "vis_pix_fit.png",
    "fit_sersic.png",
    "fit_multi_wavelength.png",
    "model.fits",
    "pre_psf.fits",
    "convergence.fits",
    "coolest.json",
    "wcs_vis_pix.json",
    "lens_mass.csv",
    "lens_sersic.csv",
    "source_sersic.csv",
    "magnitudes.csv",
]


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def read_tiles(path: Path):
    """``(sample, tile)`` pairs of a ``tiles.txt`` (``<sample> <tile>`` per line)."""
    pairs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            sample, tile = line.split()[:2]
            pairs.append((sample, tile))
    return pairs


def chunk_bundles(build_dir: Path):
    """Every ``(chunk_dir, sample, bundle_dir)`` the chunk jobs wrote."""
    bundles = []
    for chunk_dir in sorted(build_dir.glob("chunk_[0-9]*")):
        if not chunk_dir.is_dir():
            continue
        for bundle in sorted(p for p in chunk_dir.iterdir() if p.is_dir()):
            bundles.append((chunk_dir, bundle.name, bundle))
    return bundles


# ---------------------------------------------------------------------------
# 1. lens folders
# ---------------------------------------------------------------------------


def move_lens_folders(bundles, catalogue_dir: Path):
    """Move every chunk's lens folders to the catalogue root; tile -> sample."""
    tile_sample = {}
    for _chunk, sample, bundle in bundles:
        for lens_dir in sorted(p for p in bundle.iterdir() if p.is_dir()):
            if lens_dir.name.startswith("."):
                continue
            tile = lens_dir.name
            if tile in tile_sample:
                raise RuntimeError(
                    f"tile {tile} built twice ({tile_sample[tile]} and {sample})"
                )
            tile_sample[tile] = sample
            dest = catalogue_dir / tile
            if dest.exists():
                shutil.rmtree(dest)
            lens_dir.rename(dest)
    # Folders already at the root from an earlier merge of the same build.
    return tile_sample


# ---------------------------------------------------------------------------
# 2. root CSVs
# ---------------------------------------------------------------------------


def merge_csv(name, bundles, catalogue_dir: Path):
    header = None
    rows = {}
    sources = 0
    for _chunk, _sample, bundle in bundles:
        path = bundle / name
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                continue
            if header is None:
                header = list(reader.fieldnames)
            elif list(reader.fieldnames) != header:
                raise RuntimeError(
                    f"{path}: header differs from the other chunks' {name}\n"
                    f"  {reader.fieldnames}\n  vs {header}"
                )
            sources += 1
            for row in reader:
                rows[row_key(name, row)] = row
    if header is None:
        log(f"{name}: no chunk wrote one; skipped")
        return 0
    target = catalogue_dir / name
    staged = target.with_name(f".{name}.tmp")
    with staged.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
    staged.replace(target)
    log(f"{name}: {len(rows)} rows from {sources} chunk bundles")
    return len(rows)


def row_key(name, row):
    if name == "fits_deduplication_validation.csv":
        return (row.get("tile_name", ""), row.get("fits_name", ""))
    return (row.get("lens_name", ""), row.get("waveband", ""))


def dedup_summary(catalogue_dir: Path):
    path = catalogue_dir / "fits_deduplication_validation.csv"
    if not path.exists():
        return
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    hdu = Counter((r["fits_name"], r["output_hdus"]) for r in rows)
    removed = Counter((r["fits_name"], r["duplicate_extnames_removed"]) for r in rows)
    summary = {
        "rows": len(rows),
        "tiles_with_model_fits": sum(r["fits_name"] == "model.fits" for r in rows),
        "tiles_with_pre_psf_fits": sum(r["fits_name"] == "pre_psf.fits" for r in rows),
        "files_rewritten": sum(r["duplicate_extnames_removed"] not in ("", "0") for r in rows),
        "hdu_count_distribution": {f"{k[0]}:{k[1]}": v for k, v in sorted(hdu.items())},
        "duplicates_removed_distribution": {
            f"{k[0]}:{k[1]}": v for k, v in sorted(removed.items())
        },
        "validation_csv": path.name,
    }
    (catalogue_dir / "fits_deduplication_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


# ---------------------------------------------------------------------------
# 3. QA: band inventory (port of the dr1_prelim audit_sed_outputs.py)
# ---------------------------------------------------------------------------


def bands_from_fits(path: Path):
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdus:
        return [
            hdu.name.removesuffix("_BGSUB").lower()
            for hdu in hdus
            if hdu.name and hdu.name.endswith("_BGSUB")
        ]


def checkpoint_is_valid(path: Path):
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as checkpoint:
            list(checkpoint.keys())
        return True
    except (OSError, ValueError, RuntimeError):
        return False


def joined(values):
    return ";".join(sorted(values))


def band_inventory(pairs, sed_output: Path, catalogue_dir: Path):
    rows = []
    for global_index, (sample, tile) in enumerate(sorted(pairs, key=lambda p: p[1])):
        fits_path = PROJECT_ROOT / "dataset" / sample / tile / f"{tile}.fits"
        expected = set(bands_from_fits(fits_path)) if fits_path.exists() else set()
        tile_root = sed_output / sample / tile
        model_root = tile_root / "sersic_lens_model"
        band_dirs = [p for p in model_root.glob("*") if p.is_dir()] if model_root.is_dir() else []
        completed = {d.name for d in band_dirs if any(d.glob("*.zip"))}
        output_bands = {d.name for d in band_dirs}
        missing = expected - completed
        unexpected = completed - expected

        checkpoints = list(
            (tile_root / "initial_lens_model" / "vis_lp").glob(
                "*/files/search_internal/checkpoint.hdf5"
            )
        )
        checkpoints.extend(model_root.glob("*/*/files/search_internal/checkpoint.hdf5"))
        validity = {path: checkpoint_is_valid(path) for path in checkpoints}
        valid = [p for p, ok in validity.items() if ok]
        invalid = [p for p, ok in validity.items() if ok is False]
        if invalid:
            status = "invalid"
        elif valid:
            status = "valid"
        elif checkpoints:
            status = "unchecked"
        else:
            status = "absent"
        last_band = ""
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            parts = latest.parts
            if "sersic_lens_model" in parts:
                last_band = parts[parts.index("sersic_lens_model") + 1]
            elif "vis_lp" in parts:
                last_band = "vis_lp"
        rows.append(
            {
                "global_index": global_index,
                "tile_name": tile,
                "expected_band_count": len(expected),
                "expected_bands": joined(expected),
                "completed_band_count": len(completed),
                "completed_bands": joined(completed),
                "missing_band_count": len(missing),
                "missing_bands": joined(missing),
                "unexpected_completed_bands": joined(unexpected),
                "output_bands": joined(output_bands),
                "output_directory": "present" if tile_root.is_dir() else "absent",
                "checkpoint_status": status,
                "checkpoint_count": len(checkpoints),
                "valid_checkpoint_count": len(valid),
                "invalid_checkpoint_count": len(invalid),
                "last_attempted_band": last_band,
                "fully_complete": bool(expected) and not missing and not unexpected,
            }
        )
    for name, subset in (
        ("band_inventory.csv", rows),
        ("unresolved_manifest.csv", [r for r in rows if not r["fully_complete"]]),
    ):
        with (catalogue_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=INVENTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(subset)
    log(
        f"band_inventory.csv: {len(rows)} tiles, "
        f"{sum(r['fully_complete'] for r in rows)} fully complete; "
        f"unresolved_manifest.csv: {sum(not r['fully_complete'] for r in rows)}"
    )


# ---------------------------------------------------------------------------
# 4. grades
# ---------------------------------------------------------------------------


def write_grades(tile_sample, grades_csvs, v2_new_csvs, catalogue_dir: Path):
    found = {}
    for path in grades_csvs:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                if row["tile"] in tile_sample:
                    found[row["tile"]] = {c: row.get(c, "") for c in GRADE_COLUMNS}
    for path in v2_new_csvs:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                tile = row["tile"]
                if tile in tile_sample and tile not in found:
                    found[tile] = {
                        "tile": tile,
                        "unique_identifier": row.get("unique_identifier", ""),
                        "match_method": row.get("selection", "v2_new_selection"),
                        "v1_grade": "",
                        "v2_grade": row.get("v2_grade", ""),
                        "v2_score": row.get("v2_score", ""),
                        "reid": "",
                        "v1_unique_identifier": "",
                        "sample": "",
                    }
    for tile, row in found.items():
        row["sample"] = tile_sample[tile]
    target = catalogue_dir / "grades.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GRADE_COLUMNS)
        writer.writeheader()
        for tile in sorted(found):
            writer.writerow(found[tile])
    missing = sorted(set(tile_sample) - set(found))
    log(f"grades.csv: {len(found)} tiles graded, {len(missing)} without a grade row")
    return found, missing


# ---------------------------------------------------------------------------
# 5. WCS
# ---------------------------------------------------------------------------


def build_wcs(catalogue_dir: Path, build_dir: Path, samples, output_dir: str):
    package_dir = build_dir / "wcs_package"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "tools" / "build_wcs_package.py"),
        "--n=0",
        "--skip_images",
        f"--sample={','.join(samples)}",
        f"--output_path={output_dir}",
        f"--grades={catalogue_dir / 'grades.csv'}",
        f"--package_dir={package_dir}",
    ]
    if (catalogue_dir / "lens_mass.csv").exists():
        command.append(f"--lens_mass={catalogue_dir / 'lens_mass.csv'}")
    log("wcs: " + " ".join(command))
    with (build_dir / "wcs_package.log").open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
    for source, dest in (
        ("wcs.csv", "wcs.csv"),
        ("source_clumps.csv", "source_clumps.csv"),
        ("skipped.csv", "wcs_skipped.csv"),
    ):
        shutil.copyfile(package_dir / source, catalogue_dir / dest)
    # Per-lens wcs JSONs a lens folder lacks (older build) are filled from the package.
    for lens_pkg in package_dir.iterdir():
        lens_dir = catalogue_dir / lens_pkg.name
        if lens_pkg.is_dir() and lens_dir.is_dir():
            for name in ("wcs_vis_lp.json", "wcs_vis_pix.json"):
                if not (lens_dir / name).exists() and (lens_pkg / name).exists():
                    shutil.copyfile(lens_pkg / name, lens_dir / name)
    with (catalogue_dir / "wcs.csv").open(newline="") as stream:
        n = sum(1 for _ in csv.DictReader(stream))
    log(f"wcs.csv: {n} rows")


# ---------------------------------------------------------------------------
# 6. index + README
# ---------------------------------------------------------------------------


def lens_tiles(catalogue_dir: Path):
    return sorted(
        p.name for p in catalogue_dir.iterdir() if p.is_dir() and not p.name.startswith("_")
    )


def write_index(catalogue_dir, tile_sample, grades, part_of):
    columns = ["tile_id", "sample", "unique_identifier", "v2_grade", "v2_score", "n_files"]
    columns += [f"has_{name.replace('.', '_')}" for name in INDEX_PRODUCTS]
    columns += ["n_source_reconstruction_csv", "part_zip"]
    with (catalogue_dir / "lens_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for tile in lens_tiles(catalogue_dir):
            files = {p.name for p in (catalogue_dir / tile).iterdir() if p.is_file()}
            grade = grades.get(tile, {})
            row = {
                "tile_id": tile,
                "sample": tile_sample.get(tile, ""),
                "unique_identifier": grade.get("unique_identifier", ""),
                "v2_grade": grade.get("v2_grade", ""),
                "v2_score": grade.get("v2_score", ""),
                "n_files": len(files),
                "n_source_reconstruction_csv": sum(
                    n.startswith("vis_pix_source_plane_reconstruction_") for n in files
                ),
                "part_zip": part_of.get(tile, ""),
            }
            for name in INDEX_PRODUCTS:
                row[f"has_{name.replace('.', '_')}"] = name in files
            writer.writerow(row)


def git_revision():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def write_readme(catalogue_dir, tile_sample, samples, counts, archive):
    per_sample = Counter(tile_sample.values())
    lines = [
        f"Euclid DR1 strong-lens catalogue: {catalogue_dir.name}",
        "=" * (36 + len(catalogue_dir.name)),
        "",
        f"Built {date.today().isoformat()} on RAL by the chunked catalogue build "
        f"(hpc/batch_cpu/submit_catalogue_chunked.sh, euclid_dr1 @ {git_revision()}).",
        "",
        "Samples (tile ids are unique across samples):",
        "",
    ]
    lines += [f"- `{s}`: {per_sample.get(s, 0)} lens folders" for s in samples]
    lines += [
        "",
        f"Lens folders: {len(tile_sample)}",
        "",
        "Root files",
        "----------",
        "",
        "| File | Rows | What |",
        "|------|-----:|------|",
    ]
    what = {
        "lens_mass.csv": "vis_pix mass model (SIE + shear), one row per lens",
        "lens_sersic.csv": "lens light Sersic (VIS), one row per lens",
        "source_sersic.csv": "source light Sersic (VIS), one row per lens",
        "magnitudes.csv": "lens / lensed-source fluxes, one row per lens x band",
        "witt_wynne.csv": "Witt-Wynne SIEP projection, one row per lens",
        "astrometric_offsets.csv": "band-to-VIS grid offsets (prior +/-0.5\"), one row per lens x non-VIS band",
        "wcs.csv": "sky positions (lens light, source, lensed images), one row per lens x {vis_lp, vis_pix}",
        "source_clumps.csv": "vis_pix source clumps x their lensed images",
        "wcs_skipped.csv": "lenses with no wcs.csv rows, and why",
        "grades.csv": "v2 lens-finding grade + sample of every lens",
        "lens_index.csv": "per lens: sample, grade, which products exist, distribution part",
        "band_inventory.csv": "QA: expected vs completed SED bands, every requested tile",
        "unresolved_manifest.csv": "QA: the band_inventory rows not fully complete",
        "excluded_tiles.csv": "QA: tiles left out of the build (no finished vis_lp + vis_pix), and why",
        "fits_deduplication_validation.csv": "QA: per pre_psf/model.fits, HDU and duplicate-EXTNAME counts",
        "fits_deduplication_summary.json": "QA: summary of the above",
    }
    for name, description in what.items():
        if name == "excluded_tiles.csv" and name not in counts:
            continue
        lines.append(f"| `{name}` | {counts.get(name, '')} | {description} |")
    lines += [
        "",
        "Per-lens folder",
        "---------------",
        "",
        "PNGs: vis_lp_fit, vis_pix_fit, vis_lp_image_with_positions, rgb, segmentation,",
        "fit_sersic, fit_multi_wavelength. FITS: pre_psf / model (one HDU pair per band,",
        "<BAND>_GALAXY_0 = lens, _1 = lensed source), convergence, potential, deflections.",
        "JSON: coolest, coolest_sersic, wcs_vis_lp, wcs_vis_pix. CSV: this lens's rows of",
        "every root table, plus vis_pix_source_plane_reconstruction_<i>.csv (the",
        "pixelized source reconstruction, one row per source-plane mesh pixel).",
        "witt_wynne.in: isit4or2or1 input.",
        "",
        "Every Sersic product comes from the newest completed result per lens and band.",
    ]
    if archive == "chunked":
        lines += [
            "",
            "Distribution",
            "------------",
            "",
            f"`{catalogue_dir.name}_dist/`: `catalogue_csvs.zip` (all root files),",
            "`catalogue_part_NN.zip` (whole lens folders), `parts_manifest.csv`",
            "(tile_id -> part zip) and `SHA256SUMS`.",
        ]
    (catalogue_dir / "README.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# 7. distribution zips
# ---------------------------------------------------------------------------


def folder_size(path: Path):
    return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def split_parts(tiles, sizes, n_parts: int):
    """Cut ``tiles`` (in order) into EXACTLY ``n_parts`` contiguous, non-empty
    groups whose byte totals are near-equal.

    Boundary ``i`` falls where the running total crosses ``i * total / n_parts``
    (a tile joins the earlier group when its midpoint lies below the target),
    so each group is within one largest-folder of ``total / n_parts``.
    """
    n = len(tiles)
    if n_parts < 1:
        raise ValueError(f"n_parts must be >= 1, got {n_parts}")
    if n < n_parts:
        raise ValueError(f"cannot cut {n} lens folders into {n_parts} non-empty parts")
    total = sum(sizes)
    parts, start = [], 0
    for i in range(1, n_parts):
        target = total * i / n_parts
        cum = sum(sizes[:start]) + sizes[start]
        end = start + 1
        # Leave at least one tile for every later part.
        while end < n - (n_parts - i) and cum + sizes[end] / 2 <= target:
            cum += sizes[end]
            end += 1
        parts.append(list(tiles[start:end]))
        start = end
    parts.append(list(tiles[start:]))
    return parts


def plan_parts(catalogue_dir: Path, cap_bytes: int, n_parts=None):
    """Whole lens folders per part: greedy up to ``cap_bytes``, or, with
    ``n_parts``, exactly that many contiguous near-equal parts."""
    tiles = lens_tiles(catalogue_dir)
    if n_parts:
        sizes = [folder_size(catalogue_dir / tile) for tile in tiles]
        return split_parts(tiles, sizes, n_parts)
    parts, current, size = [], [], 0
    for tile in tiles:
        tile_size = folder_size(catalogue_dir / tile)
        if current and size + tile_size > cap_bytes:
            parts.append(current)
            current, size = [], 0
        current.append(tile)
        size += tile_size
    if current:
        parts.append(current)
    return parts


def compression_for(path: Path):
    return zipfile.ZIP_STORED if path.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_zip(job):
    """Write one zip of ``files`` (absolute) under ``arc_root``; return its name."""
    zip_path, catalogue_dir, relative_files = job
    staged = zip_path.with_name(f".{zip_path.name}.tmp")
    with zipfile.ZipFile(staged, "w", allowZip64=True) as archive:
        for relative in relative_files:
            source = catalogue_dir / relative
            archive.write(
                source,
                arcname=f"{catalogue_dir.name}/{relative}",
                compress_type=compression_for(source),
                compresslevel=6 if compression_for(source) == zipfile.ZIP_DEFLATED else None,
            )
    staged.replace(zip_path)
    return zip_path.name


def verify_zip(job):
    """CRC-test a zip and return ``(name, sha256, tiles, members)``."""
    zip_path, arc_root = job
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"{zip_path}: CRC failure on {bad}")
        names = archive.namelist()
    tiles = sorted(
        {n.split("/")[1] for n in names if n.count("/") >= 2 and n.startswith(f"{arc_root}/")}
    )
    return zip_path.name, sha256(zip_path), tiles, len(names)


def distribute(catalogue_dir: Path, dist_dir: Path, cap_gb: float, workers: int, tile_sample,
               n_parts=None):
    dist_dir.mkdir(parents=True, exist_ok=True)
    parts = plan_parts(catalogue_dir, int(cap_gb * 1024**3), n_parts)
    if n_parts and len(parts) != n_parts:
        raise RuntimeError(f"planned {len(parts)} parts, --n_parts={n_parts}")
    width = max(2, len(str(len(parts))))
    part_of = {}
    jobs = []
    for index, tiles in enumerate(parts, start=1):
        name = f"catalogue_part_{index:0{width}d}.zip"
        files = []
        for tile in tiles:
            part_of[tile] = name
            files += [f"{tile}/{p.name}" for p in sorted((catalogue_dir / tile).iterdir()) if p.is_file()]
        jobs.append((dist_dir / name, catalogue_dir, files))
    rule = f"exactly {n_parts}" if n_parts else f"cap {cap_gb} GB"
    log(f"distribution: {len(parts)} part zips ({rule}), {workers} workers")

    # The index names each tile's part, so it is written before the CSV zip.
    write_index(catalogue_dir, tile_sample, read_grades(catalogue_dir), part_of)
    with (dist_dir / "parts_manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tile_id", "sample", "part_zip"])
        for tile in sorted(part_of):
            writer.writerow([tile, tile_sample.get(tile, ""), part_of[tile]])
    shutil.copyfile(dist_dir / "parts_manifest.csv", catalogue_dir / "parts_manifest.csv")

    root_files = sorted(p.name for p in catalogue_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    jobs.insert(0, (dist_dir / "catalogue_csvs.zip", catalogue_dir, root_files))

    with Pool(max(1, workers)) as pool:
        for name in pool.imap_unordered(write_zip, jobs):
            log(f"  wrote {name}")
        verified = pool.map(verify_zip, [(job[0], catalogue_dir.name) for job in jobs])

    # Manifest coverage: every lens folder in exactly one part, as planned.
    seen = Counter()
    for name, _digest, tiles, _members in verified:
        if name == "catalogue_csvs.zip":
            if tiles:
                raise RuntimeError("catalogue_csvs.zip holds lens folders")
            continue
        for tile in tiles:
            seen[tile] += 1
            if part_of.get(tile) != name:
                raise RuntimeError(f"{tile} found in {name}, manifest says {part_of.get(tile)}")
    lenses = set(lens_tiles(catalogue_dir))
    if set(seen) != lenses or any(v != 1 for v in seen.values()):
        raise RuntimeError(
            f"part zips cover {len(seen)} tiles, catalogue has {len(lenses)}; "
            f"repeats {[t for t, v in seen.items() if v != 1][:5]}"
        )
    with (dist_dir / "SHA256SUMS").open("w") as stream:
        for name, digest, _tiles, _members in sorted(verified):
            stream.write(f"{digest}  {name}\n")
        stream.write(f"{sha256(dist_dir / 'parts_manifest.csv')}  parts_manifest.csv\n")
    for name, _digest, tiles, members in sorted(verified):
        size = (dist_dir / name).stat().st_size / 1024**3
        log(f"  verified {name}: {members} members, {len(tiles)} lenses, {size:.2f} GB")
    return part_of


def read_grades(catalogue_dir):
    path = catalogue_dir / "grades.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        return {row["tile"]: row for row in csv.DictReader(stream)}


# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Merge a chunked catalogue build.")
    parser.add_argument("--catalogue_dir", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--sed_output_dir", default="output_sed")
    parser.add_argument(
        "--grades",
        action="append",
        default=None,
        help="Grade CSVs in the tile_grades schema (repeatable). "
        "Default: catalogue/grades/tile_grades_dr1_sep1_v2.csv.",
    )
    parser.add_argument(
        "--v2_new_selection",
        action="append",
        default=None,
        help="dr1_v2_new selection CSVs (repeatable). "
        "Default: hpc/run_manifests/dr1_v2_new_selection_20261001.csv.",
    )
    parser.add_argument("--archive", default="0", help="'chunked' writes the distribution zips.")
    parser.add_argument("--part_max_gb", type=float, default=10.0)
    parser.add_argument(
        "--n_parts",
        type=int,
        default=0,
        help="Write EXACTLY this many part zips (contiguous, near-equal bytes); "
        "overrides --part_max_gb. 0 = use --part_max_gb.",
    )
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--skip_wcs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    catalogue_dir = Path(args.catalogue_dir)
    if not catalogue_dir.is_absolute():
        catalogue_dir = PROJECT_ROOT / catalogue_dir
    build_dir = catalogue_dir / "_build"
    pairs = read_tiles(build_dir / "tiles.txt")
    samples = sorted({sample for sample, _tile in pairs})

    n_chunks = len(list(build_dir.glob("chunk_[0-9]*.txt")))
    done = sorted(build_dir.glob("chunk_[0-9]*.done"))
    log(f"{len(pairs)} requested tiles over {samples}; {len(done)}/{n_chunks} chunks done")
    if len(done) != n_chunks and not args.allow_partial:
        raise SystemExit("not every chunk finished (no .done marker); --allow_partial to merge anyway")
    tiles_requested = [tile for _s, tile in pairs]
    if len(set(tiles_requested)) != len(tiles_requested):
        raise SystemExit("tiles.txt repeats a tile id")

    bundles = chunk_bundles(build_dir)
    tile_sample = move_lens_folders(bundles, catalogue_dir)
    # Lens folders moved by an earlier (interrupted) merge.
    requested_sample = {tile: sample for sample, tile in pairs}
    for tile in lens_tiles(catalogue_dir):
        tile_sample.setdefault(tile, requested_sample.get(tile, ""))
    log(f"lens folders: {len(tile_sample)}")

    counts = {}
    for name in ROOT_CSVS:
        counts[name] = merge_csv(name, bundles, catalogue_dir)
    dedup_summary(catalogue_dir)

    band_inventory(pairs, PROJECT_ROOT / args.sed_output_dir, catalogue_dir)

    grades = args.grades or [str(PROJECT_ROOT / "catalogue/grades/tile_grades_dr1_sep1_v2.csv")]
    v2_new = args.v2_new_selection or [
        str(PROJECT_ROOT / "hpc/run_manifests/dr1_v2_new_selection_20261001.csv")
    ]
    write_grades(tile_sample, grades, v2_new, catalogue_dir)

    if not args.skip_wcs:
        build_wcs(catalogue_dir, build_dir, samples, args.output_dir)

    for name in ("wcs.csv", "source_clumps.csv", "wcs_skipped.csv", "grades.csv",
                 "band_inventory.csv", "unresolved_manifest.csv", "excluded_tiles.csv"):
        path = catalogue_dir / name
        if path.exists():
            with path.open(newline="") as stream:
                counts[name] = sum(1 for _ in csv.DictReader(stream))
    counts["lens_index.csv"] = len(lens_tiles(catalogue_dir))
    write_readme(catalogue_dir, tile_sample, samples, counts, args.archive)

    if args.archive == "chunked":
        distribute(
            catalogue_dir,
            catalogue_dir.parent / f"{catalogue_dir.name}_dist",
            args.part_max_gb,
            args.workers,
            tile_sample,
            n_parts=args.n_parts or None,
        )
    else:
        write_index(catalogue_dir, tile_sample, read_grades(catalogue_dir), {})
    log(f"merged catalogue: {catalogue_dir} ({len(tile_sample)} lenses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
