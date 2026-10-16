"""
Euclid Pipeline: Package The Catalogue Release Tiers
====================================================

Packages ONE merged catalogue build into three download tiers, all stamped with
the same snapshot cutoff and membership manifest:

- **csv**: the CSV-only snapshot tables (``scripts/tools/csv_snapshot/build.py``).
- **vis**: csv + per lens the VIS products (``VIS_PRODUCTS`` below, plus every
  ``vis_pix_source_plane_reconstruction_<i>.csv``) and the dataset cutout under
  ``<tile>/dataset/``. ``model.fits`` / ``pre_psf.fits`` hold one HDU pair per
  band; in this tier they are rewritten to their PRIMARY + ``VIS_*`` HDUs, so no
  multi-band model product ships. The dataset cutout ``<tile>.fits`` is the
  unchanged modelling input and keeps every delivered band.
- **full**: csv + everything in the per-lens folder (unchanged files) + the
  dataset cutout, plus the merged catalogue's root tables under
  ``catalogue_tables/``.

Inputs: the merged tree of ``scripts/tools/merge_catalogue.py``
(``inspect/<CATALOGUE>/<tile>/``), the dataset tree (``dataset/dr1_sep1/<tile>/``),
the consolidated manifest (its ``include_in_catalogue=True`` rows are the
release membership) and the CSV snapshot directory (``--csv_dir``), whose
``_build/snapshot.json`` supplies the cutoff and must name the same manifest
sha256.

Output, per tier, in ``<out_dir>/<tier>/``:

- ``catalogue_csvs.zip``: ``<root>/README.md``, ``KNOWN_ISSUES.md``,
  ``snapshot.json``, ``file_manifest.csv`` (every member of every zip of the tier
  with bytes and sha256, except itself), ``missing_files.csv``,
  ``parts_manifest.csv`` and ``excluded_tiles.csv`` (vis/full), and the CSV
  snapshot under ``<root>/csv/``. ``<root>`` is ``<release_name>_<tier>``.
- ``catalogue_part_NN.zip`` (vis/full): whole lens folders, contiguous in tile
  order, near-equal bytes with ``--n_parts`` or capped at ``--part_max_gb``.
  FITS/PNG are ZIP_STORED, text DEFLATED (``release_util.py``, shared with
  ``merge_catalogue.py``).
- ``SHA256SUMS`` (``sha256sum -c`` format), ``snapshot.json`` (cutoff, manifest
  sha256, project git HEAD, tier, product list, lens count, missing-file
  counts per product), ``missing_files.csv``, ``parts_manifest.csv``.

Every zip is re-opened and CRC-tested and its member list checked against the
plan before the tier succeeds. A per-lens product may be missing (a lens with no
Sersic fit has no ``fit_sersic.png``); each is counted per product and listed in
``missing_files.csv``. A manifest tile with no lens folder at all is an error
unless it is listed in ``--excluded`` (a CSV with a ``tile`` or ``tile_id``
column); excluded tiles are never packaged. ``--absent_out=<csv>`` writes the
absent tiles for review before they are passed as ``--excluded``.

Usage
-----
    python scripts/tools/package_release.py --catalogue_dir=inspect/dr1_catalogue_20261020 \\
        --dataset_dir=dataset/dr1_sep1 --csv_dir=inspect/dr1_catalogue_20261020_csv \\
        --tier=all --n_parts=10 [--excluded=inspect/dr1_catalogue_20261020/excluded_tiles.csv] [--dry_run]
"""

import argparse
import csv
import io
import json
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_util import (  # noqa: E402
    add_bytes_hashed,
    add_file_hashed,
    part_name,
    plan,
    sha256,
    verify_zip,
    write_sha256sums,
)

sys.path.insert(0, str(Path(__file__).resolve().parent / "csv_snapshot"))
from common import file_sha256, git_revision, included_rows, read_csv, utc_iso  # noqa: E402

TIERS = ("csv", "vis", "full")

VIS_FITS = ["model.fits", "pre_psf.fits", "convergence.fits", "deflections.fits", "potential.fits"]
VIS_PNGS = ["vis_lp_fit.png", "vis_pix_fit.png", "vis_lp_image_with_positions.png", "rgb.png", "segmentation.png"]
VIS_TEXT = [
    "coolest.json", "wcs_vis_lp.json", "wcs_vis_pix.json", "lens_mass.csv", "lens_sersic.csv",
    "source_sersic.csv", "witt_wynne.csv", "witt_wynne.in",
]
VIS_PRODUCTS = VIS_FITS + VIS_PNGS + VIS_TEXT
SOURCE_RECONSTRUCTION = "vis_pix_source_plane_reconstruction_*.csv"
FULL_EXTRA = ["fit_multi_wavelength.png", "fit_sersic.png", "coolest_sersic.json", "magnitudes.csv", "astrometric_offsets.csv"]
FULL_PRODUCTS = VIS_PRODUCTS + FULL_EXTRA
MULTI_BAND_PRODUCTS = set(FULL_EXTRA)
# FITS with one HDU pair per band. The human chose (2026-10-15) to ship them whole in every tier;
# add "model.fits" / "pre_psf.fits" here to reduce the vis tier to PRIMARY + VIS_* HDUs.
VIS_STRIP_FITS: set = set()

# Dataset cutout products, relative to dataset/<sample>/<tile>/; "{tile}" is filled in.
DATASET_EXPECTED = ["{tile}.fits", "info.json", "positions.json", "segmentation/*.fits"]
DATASET_OPTIONAL = ["positions*.json"]

# Merged-catalogue root files that describe the old _dist layout, not this release.
CATALOGUE_ROOT_SKIP = {"lens_index.csv", "parts_manifest.csv"}

FILE_MANIFEST_COLUMNS = ["zip", "member", "bytes", "sha256"]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def resolve(path):
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def tier_products(tier):
    """Expected per-lens product names of a tier (glob patterns included)."""
    if tier == "vis":
        return VIS_PRODUCTS + [SOURCE_RECONSTRUCTION]
    if tier == "full":
        return FULL_PRODUCTS + [SOURCE_RECONSTRUCTION]
    return []


def dataset_products():
    return [f"dataset/{p}" for p in DATASET_EXPECTED]


def read_excluded(path):
    if not path:
        return {}
    rows = read_csv(path)
    out = {}
    for row in rows:
        tile = row.get("tile") or row.get("tile_id")
        if not tile:
            raise SystemExit(f"{path}: needs a tile or tile_id column")
        out[tile] = row.get("reason", "") or row.get("note", "")
    return out


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def plan_tile(tile, tier, catalogue_dir: Path, dataset_dir: Path):
    """``(members, missing)`` for one lens. A member is ``(kind, source, arc,
    size)`` with ``kind`` ``file`` or ``vis_fits`` (rewrite to VIS HDUs) and
    ``arc`` relative to the zip root; ``missing`` lists absent product names."""
    lens_dir = catalogue_dir / tile
    members, missing = [], []
    present = {p.name: p for p in lens_dir.iterdir() if p.is_file() and not p.name.startswith(".")}
    if tier == "full":
        chosen = sorted(present)
        missing += [name for name in FULL_PRODUCTS if name not in present]
    else:
        chosen = [name for name in VIS_PRODUCTS if name in present]
        missing += [name for name in VIS_PRODUCTS if name not in present]
        chosen += sorted(n for n in present if Path(n).match(SOURCE_RECONSTRUCTION))
    if not any(Path(n).match(SOURCE_RECONSTRUCTION) for n in present):
        missing.append(SOURCE_RECONSTRUCTION)
    for name in chosen:
        path = present[name]
        kind = "vis_fits" if tier == "vis" and name in VIS_STRIP_FITS else "file"
        members.append((kind, str(path), f"{tile}/{name}", path.stat().st_size))

    data_dir = dataset_dir / tile
    seen = set()
    for pattern in DATASET_EXPECTED + DATASET_OPTIONAL:
        pattern = pattern.format(tile=tile)
        found = sorted(p for p in data_dir.glob(pattern) if p.is_file()) if data_dir.is_dir() else []
        if not found and pattern in [p.format(tile=tile) for p in DATASET_EXPECTED]:
            missing.append(f"dataset/{pattern.replace(tile, '{tile}')}")
        for path in found:
            relative = path.relative_to(data_dir).as_posix()
            if relative in seen:
                continue
            seen.add(relative)
            members.append(("file", str(path), f"{tile}/dataset/{relative}", path.stat().st_size))
    return members, missing


def plan_tier(tier, tiles, catalogue_dir, dataset_dir):
    """Members and missing products of every lens of a vis / full tier."""
    plans, missing_rows = {}, []
    for tile in tiles:
        members, missing = plan_tile(tile, tier, catalogue_dir, dataset_dir)
        plans[tile] = members
        missing_rows += [(tile, product) for product in missing]
    return plans, missing_rows


# ---------------------------------------------------------------------------
# zip writing
# ---------------------------------------------------------------------------


def vis_only_fits(path):
    """Bytes of ``path`` keeping PRIMARY + ``VIS_*`` HDUs, or ``None`` when the
    file already holds nothing else."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdus:
        keep = [i for i, h in enumerate(hdus) if i == 0 or (h.name or "").upper().startswith("VIS")]
        if len(keep) == len(hdus):
            return None
        buffer = io.BytesIO()
        fits.HDUList([hdus[i].copy() for i in keep]).writeto(buffer)
    return buffer.getvalue()


def write_part(job):
    """Write one part zip; return ``(zip_name, [(member, bytes, sha256)])``."""
    zip_path, arc_root, members = job
    zip_path = Path(zip_path)
    staged = zip_path.with_name(f".{zip_path.name}.tmp")
    rows = []
    with zipfile.ZipFile(staged, "w", allowZip64=True) as archive:
        for kind, source, arc, _size in members:
            arcname = f"{arc_root}/{arc}"
            data = vis_only_fits(source) if kind == "vis_fits" else None
            if data is not None:
                size, digest = add_bytes_hashed(archive, data, arcname, like=Path(source))
            else:
                size, digest = add_file_hashed(archive, Path(source), arcname)
            rows.append((arcname, size, digest))
    staged.replace(zip_path)
    return zip_path.name, rows


def check_zip(zip_path, expected_rows):
    """CRC-test ``zip_path`` and check its members are exactly ``expected_rows``
    (``(member, bytes, sha256)``), sizes included. Returns its sha256."""
    name, digest, names, sizes = verify_zip(zip_path)
    expected = {member: size for member, size, _sha in expected_rows}
    if len(names) != len(set(names)):
        raise RuntimeError(f"{name}: repeated member names")
    if set(names) != set(expected):
        extra = sorted(set(names) - set(expected))[:5]
        absent = sorted(set(expected) - set(names))[:5]
        raise RuntimeError(f"{name}: members differ from the plan (extra {extra}, absent {absent})")
    wrong = [m for m in names if sizes[m] != expected[m]]
    if wrong:
        raise RuntimeError(f"{name}: member sizes differ from the plan: {wrong[:5]}")
    return digest


def _verify_job(job):
    zip_path, rows = job
    return Path(zip_path).name, check_zip(zip_path, rows)


def run_pool(function, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [function(job) for job in jobs]
    with Pool(min(workers, len(jobs))) as pool:
        return pool.map(function, jobs)


# ---------------------------------------------------------------------------
# documents
# ---------------------------------------------------------------------------


def csv_text(header, rows):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def readme_text(tier, root, snapshot, csv_snapshot):
    lines = [
        f"# Euclid DR1 strong-lens catalogue: {root}",
        "",
        f"Tier `{tier}` of release `{snapshot['release_name']}`. Snapshot cutoff "
        f"{snapshot['cutoff_utc']}: every result is the newest completed fit at or before it.",
        f"Membership: the {snapshot['n_manifest_included']:,} `include_in_catalogue=True` rows of "
        f"`{Path(snapshot['manifest']).name}` (sha256 `{snapshot['manifest_sha256'][:16]}...`).",
        "",
        "The csv, vis and full tiers are cut from one build with one cutoff and one manifest, "
        "so a lens has the same values in all three.",
        "",
        "## Contents",
        "",
        "- `csv/`: the catalogue tables. Start with "
        f"`csv/{csv_snapshot.get('master_file', 'README.txt')}` and `csv/README.txt`, which "
        "documents every table and column convention.",
        "- `snapshot.json`: cutoff, manifest sha256, git revision, products, missing-file counts.",
        "- `file_manifest.csv`: every file of every zip of this tier (zip, member, bytes, sha256).",
        "- `KNOWN_ISSUES.md`: issues known at release time.",
    ]
    if tier != "csv":
        lines += [
            "- `parts_manifest.csv`: tile_id, unique_identifier, part zip.",
            "- `missing_files.csv`: per-lens products absent from this tier, and the product name.",
            "",
            f"Per-lens folders: {snapshot['lens_count']:,} in `catalogue_part_NN.zip` "
            f"({len(snapshot['parts'])} parts), each `{root}/<tile_id>/`.",
            "",
            "## Per-lens products",
            "",
        ]
        if tier == "vis":
            lines += [
                "VIS only. FITS: " + ", ".join(VIS_FITS) + " (model.fits and pre_psf.fits keep the "
                "PRIMARY and VIS_* HDUs: VIS_GALAXY_0 = lens, VIS_GALAXY_1 = lensed source).",
                "PNG: " + ", ".join(VIS_PNGS) + ".",
                "Text: " + ", ".join(VIS_TEXT) + ", vis_pix_source_plane_reconstruction_<i>.csv.",
            ]
        else:
            lines += [
                "Everything the catalogue build wrote for the lens: the vis-tier products with "
                "model.fits / pre_psf.fits holding every band (<BAND>_GALAXY_0 = lens, _1 = lensed "
                "source), plus " + ", ".join(FULL_EXTRA) + ".",
                "`catalogue_tables/` in catalogue_csvs.zip holds the merged build's root tables "
                "(magnitudes.csv, witt_wynne.csv, band_inventory.csv and the rest).",
            ]
        lines += [
            "",
            "`<tile_id>/dataset/`: the modelling input cutout, unchanged: `<tile_id>.fits` "
            "(`<BAND>_BGSUB`, `<BAND>_PSF` and `<BAND>_RMS` HDUs for every delivered band), "
            "`segmentation/*.fits`, `positions*.json`, `info.json`."
            + (" The cutout keeps every band in this tier too, since it is the modelling input."
               if tier == "vis" else ""),
        ]
    lines += [
        "",
        "## Verifying a download",
        "",
        "Put every file of the tier in one directory, then",
        "",
        "```",
        "sha256sum -c SHA256SUMS",
        "```",
        "",
        "Every line must print `OK`. Then unzip the parts into one directory; they never share a file.",
        "",
    ]
    return "\n".join(lines)


def known_issues_text(tier, snapshot, known_issues_path, excluded):
    lines = [f"# Known issues: {snapshot['release_name']} ({tier} tier)", ""]
    if known_issues_path:
        lines += [Path(known_issues_path).read_text().rstrip(), ""]
    if tier != "csv":
        lines += ["## Missing per-lens files", ""]
        if snapshot["missing_counts"]:
            lines += [
                "A completed search can lack a fit-dependent product (for example no Sersic fit, "
                "so no fit_sersic.png). Absent files are listed in missing_files.csv.",
                "",
                "| Product | Lenses without it |",
                "|---------|------------------:|",
            ]
            lines += [f"| `{k}` | {v:,} |" for k, v in sorted(snapshot["missing_counts"].items())]
        else:
            lines.append("None: every lens has every product of this tier.")
        lines.append("")
    if excluded:
        lines += [
            "## Excluded tiles",
            "",
            f"Manifest tiles with no per-lens folder in this release: {len(excluded):,} "
            "(excluded_tiles.csv gives the reason). Their rows remain in the csv tables.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# a tier
# ---------------------------------------------------------------------------


def csv_snapshot_info(csv_dir: Path, manifest: Path):
    snap_path = csv_dir / "_build" / "snapshot.json"
    if not snap_path.exists():
        raise SystemExit(f"{snap_path} not found: --csv_dir must be a csv_snapshot/build.py output")
    snap = json.loads(snap_path.read_text())
    digest = file_sha256(manifest)
    if snap.get("manifest_sha256") != digest:
        raise SystemExit(
            f"{csv_dir}: CSV snapshot was built from a manifest with sha256 "
            f"{snap.get('manifest_sha256')}, --manifest has {digest}"
        )
    counts_path = csv_dir / "_build" / "validation_counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    files = sorted(p for p in csv_dir.iterdir() if p.is_file() and not p.name.startswith((".", "_")))
    if not files:
        raise SystemExit(f"{csv_dir}: no tables")
    return snap, counts, files


def catalogue_revision(catalogue_dir: Path):
    env = catalogue_dir / "_build" / "build.env"
    if env and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GIT_REVISION="):
                return line.split("=", 1)[1].strip()
    return ""


def package_tier(tier, args, ctx):
    """Plan, write and verify one tier; return its snapshot dict."""
    tier_dir = ctx["out_dir"] / tier
    root = f"{ctx['release_name']}_{tier}"
    tiles = ctx["tiles"]
    plans, missing_rows = ({}, [])
    if tier != "csv":
        plans, missing_rows = plan_tier(tier, tiles, ctx["catalogue_dir"], ctx["dataset_dir"])
    missing_counts = Counter(product for _tile, product in missing_rows)
    sizes = [sum(m[3] for m in plans[t]) for t in tiles] if plans else []
    parts = []
    if plans:
        parts = plan(tiles, sizes, int(args.part_max_gb * 1024**3), args.n_parts or None)
        if args.n_parts and len(parts) != args.n_parts:
            raise RuntimeError(f"planned {len(parts)} parts, --n_parts={args.n_parts}")
    part_names = [part_name(i, len(parts)) for i in range(1, len(parts) + 1)]
    part_of = {tile: name for name, group in zip(part_names, parts) for tile in group}
    size_of = dict(zip(tiles, sizes))

    snapshot = {
        "release_name": ctx["release_name"],
        "tier": tier,
        "zip_root": root,
        "cutoff_unix": ctx["csv_snapshot"]["cutoff_unix"],
        "cutoff_utc": ctx["csv_snapshot"]["cutoff_utc"],
        "manifest": str(ctx["manifest_rel"]),
        "manifest_sha256": ctx["manifest_sha256"],
        "git_head": ctx["git_head"],
        "catalogue_dir": str(ctx["catalogue_rel"]) if ctx["catalogue_dir"] else "",
        "catalogue_build_git_revision": ctx["catalogue_revision"],
        "dataset_dir": str(ctx["dataset_rel"]) if ctx["dataset_dir"] else "",
        "csv_dir": str(ctx["csv_rel"]),
        "products": tier_products(tier) + (dataset_products() if tier != "csv" else []),
        "multi_band_fits_reduced_to_vis": sorted(VIS_STRIP_FITS) if tier == "vis" else [],
        "n_manifest_included": ctx["n_included"],
        "lens_count": len(tiles) if tier != "csv" else 0,
        "excluded_count": len(ctx["excluded"]),
        "extra_catalogue_folders": ctx["extra_folders"],
        "missing_counts": dict(sorted(missing_counts.items())),
        "n_missing_files": len(missing_rows),
        "lenses_with_missing_files": len({t for t, _p in missing_rows}),
        "parts": [
            {"zip": name, "tiles": len(group), "first_tile": group[0], "last_tile": group[-1],
             "planned_bytes": sum(size_of[t] for t in group)}
            for name, group in zip(part_names, parts)
        ],
        "created_utc": utc_iso(time.time()),
    }
    log(
        f"{tier}: {snapshot['lens_count']} lens folders, {len(parts)} parts, "
        f"{sum(sizes) / 1024**3:.2f} GB planned, {len(missing_rows)} missing files"
    )
    for name, group in zip(part_names, parts):
        log(f"  {name}: {len(group)} lenses, {sum(size_of[t] for t in group) / 1024**3:.2f} GB")
    for product, n in sorted(missing_counts.items()):
        log(f"  missing {product}: {n}")
    if args.dry_run:
        return snapshot

    if (tier_dir / "SHA256SUMS").exists() and not args.force:
        raise SystemExit(f"{tier_dir} is already packaged (SHA256SUMS exists); --force to rewrite")
    if tier_dir.exists():
        shutil.rmtree(tier_dir)
    tier_dir.mkdir(parents=True)

    # Part zips.
    jobs = [
        (str(tier_dir / name), root, [m for tile in group for m in plans[tile]])
        for name, group in zip(part_names, parts)
    ]
    written = dict(run_pool(write_part, jobs, args.workers))
    for name in part_names:
        log(f"  wrote {name}: {len(written[name])} members")
    digests = dict(run_pool(_verify_job, [(str(tier_dir / n), written[n]) for n in part_names], args.workers))
    seen = Counter()
    for name in part_names:
        for tile in {m.split("/")[1] for m, _s, _h in written[name]}:
            seen[tile] += 1
            if part_of.get(tile) != name:
                raise RuntimeError(f"{tile} found in {name}, plan says {part_of.get(tile)}")
    packaged = set(tiles) if plans else set()
    if set(seen) != packaged or any(v != 1 for v in seen.values()):
        raise RuntimeError(f"part zips cover {len(seen)} lenses, the tier has {len(packaged)}")

    # Documents (also written beside the zips).
    uid_of = ctx["uid_of"]
    docs = {}
    if tier != "csv":
        docs["parts_manifest.csv"] = csv_text(
            ["tile_id", "unique_identifier", "part_zip"], [(t, uid_of[t], part_of[t]) for t in tiles]
        )
        docs["missing_files.csv"] = csv_text(
            ["tile_id", "unique_identifier", "product"], [(t, uid_of[t], p) for t, p in missing_rows]
        )
        if ctx["excluded"]:
            docs["excluded_tiles.csv"] = csv_text(
                ["tile_id", "unique_identifier", "reason", "in_catalogue_tree"],
                [(t, uid_of.get(t, ""), r, (ctx["catalogue_dir"] / t).is_dir()) for t, r in sorted(ctx["excluded"].items())],
            )
    docs["snapshot.json"] = json.dumps(snapshot, indent=2) + "\n"
    docs["README.md"] = readme_text(tier, root, snapshot, ctx["csv_counts"])
    docs["KNOWN_ISSUES.md"] = known_issues_text(tier, snapshot, ctx["known_issues"], ctx["excluded"])
    for name in ("snapshot.json", "missing_files.csv", "parts_manifest.csv"):
        if name in docs:
            (tier_dir / name).write_text(docs[name])

    # catalogue_csvs.zip: documents, the CSV snapshot, and (full) the build's root tables.
    files = [(p, f"csv/{p.name}") for p in ctx["csv_files"]]
    if tier == "full":
        files += [
            (p, f"catalogue_tables/{p.name}")
            for p in sorted(ctx["catalogue_dir"].iterdir())
            if p.is_file() and not p.name.startswith(".") and p.name not in CATALOGUE_ROOT_SKIP
        ]
    csv_zip = tier_dir / "catalogue_csvs.zip"
    staged = csv_zip.with_name(f".{csv_zip.name}.tmp")
    csv_rows = []
    with zipfile.ZipFile(staged, "w", allowZip64=True) as archive:
        for name in sorted(docs):
            size, digest = add_bytes_hashed(archive, docs[name].encode(), f"{root}/{name}")
            csv_rows.append((f"{root}/{name}", size, digest))
        for path, arc in files:
            size, digest = add_file_hashed(archive, path, f"{root}/{arc}")
            csv_rows.append((f"{root}/{arc}", size, digest))
        manifest_rows = [("catalogue_csvs.zip", m, s, h) for m, s, h in csv_rows]
        for name in part_names:
            manifest_rows += [(name, m, s, h) for m, s, h in written[name]]
        file_manifest = csv_text(FILE_MANIFEST_COLUMNS, manifest_rows).encode()
        size, digest = add_bytes_hashed(archive, file_manifest, f"{root}/file_manifest.csv")
        csv_rows.append((f"{root}/file_manifest.csv", size, digest))
    staged.replace(csv_zip)
    digests["catalogue_csvs.zip"] = check_zip(csv_zip, csv_rows)
    if any(m.split("/")[1] in part_of for m, _s, _h in csv_rows):
        raise RuntimeError("catalogue_csvs.zip holds lens folders")

    for name in ("snapshot.json", "missing_files.csv", "parts_manifest.csv"):
        if (tier_dir / name).exists():
            digests[name] = sha256(tier_dir / name)
    write_sha256sums(tier_dir / "SHA256SUMS", digests)
    total = sum((tier_dir / n).stat().st_size for n in digests if n.endswith(".zip"))
    log(f"{tier}: verified {len(digests)} files, {total / 1024**3:.2f} GB of zips -> {tier_dir}")
    return snapshot


# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Package the csv / vis / full catalogue release tiers.")
    parser.add_argument("--catalogue_dir", default=None, help="merged catalogue tree (required for vis/full)")
    parser.add_argument("--dataset_dir", default="dataset/dr1_sep1")
    parser.add_argument("--manifest", default="catalogue/grades/dr1_sep1_consolidated_manifest.csv")
    parser.add_argument("--csv_dir", required=True, help="output of scripts/tools/csv_snapshot/build.py")
    parser.add_argument("--tier", action="append", default=None,
                        help="csv, vis, full (repeatable) or all; default all")
    parser.add_argument("--out_dir", default=None, help="default <catalogue_dir>_release")
    parser.add_argument("--release_name", default=None, help="zip root prefix; default the catalogue_dir name")
    parser.add_argument("--excluded", default=None, help="CSV (tile or tile_id column) of tiles allowed to be absent")
    parser.add_argument("--absent_out", default=None,
                        help="write manifest tiles with no lens folder to this CSV (for review, then --excluded)")
    parser.add_argument("--known_issues", default=None, help="markdown prepended to every KNOWN_ISSUES.md")
    parser.add_argument("--part_max_gb", type=float, default=10.0)
    parser.add_argument("--n_parts", type=int, default=0,
                        help="EXACTLY this many part zips per tier (contiguous, near-equal bytes); "
                        "overrides --part_max_gb. 0 = use --part_max_gb.")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    parser.add_argument("--dry_run", action="store_true", help="plan and report; write nothing")
    parser.add_argument("--force", action="store_true", help="rewrite an already packaged tier")
    return parser.parse_args(argv)


def relative(path):
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main(argv=None):
    args = parse_args(argv)
    tiers = []
    for value in args.tier or ["all"]:
        for t in value.split(","):
            tiers += list(TIERS) if t == "all" else [t]
    bad = set(tiers) - set(TIERS)
    if bad:
        raise SystemExit(f"unknown --tier {sorted(bad)}")
    tiers = [t for t in TIERS if t in tiers]

    catalogue_dir = resolve(args.catalogue_dir)
    dataset_dir = resolve(args.dataset_dir)
    manifest = resolve(args.manifest)
    csv_dir = resolve(args.csv_dir)
    if any(t != "csv" for t in tiers):
        if catalogue_dir is None or not catalogue_dir.is_dir():
            raise SystemExit(f"--catalogue_dir {catalogue_dir} is required for the vis / full tiers")
    release_name = args.release_name or (catalogue_dir.name if catalogue_dir else csv_dir.name)
    out_dir = resolve(args.out_dir) if args.out_dir else (
        catalogue_dir.parent / f"{catalogue_dir.name}_release" if catalogue_dir else csv_dir.parent / f"{release_name}_release"
    )

    rows = included_rows(manifest)
    excluded = read_excluded(resolve(args.excluded))
    uid_of = {r["tile"]: r["unique_identifier"] for r in rows}
    unknown_excluded = sorted(set(excluded) - set(uid_of))
    if unknown_excluded:
        log(f"warning: {len(unknown_excluded)} --excluded tiles are not included manifest rows "
            f"(e.g. {unknown_excluded[:3]}); ignored")
    tiles = sorted(t for t in uid_of if t not in excluded)
    extra = 0
    if catalogue_dir is not None and catalogue_dir.is_dir() and any(t != "csv" for t in tiers):
        folders = {p.name for p in catalogue_dir.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))}
        absent = [t for t in tiles if t not in folders]
        if absent and args.absent_out:
            absent_out = resolve(args.absent_out)
            absent_out.write_text(csv_text(
                ["tile", "unique_identifier", "reason"],
                [(t, uid_of[t], f"no lens folder in {catalogue_dir.name}") for t in absent],
            ))
            log(f"wrote the {len(absent)} absent tiles to {absent_out}; review it, then pass it as --excluded")
        if absent:
            raise SystemExit(
                f"{len(absent)} manifest tiles have no lens folder in {catalogue_dir} and are not in "
                f"--excluded: {absent[:5]}"
            )
        present_excluded = sorted(t for t in excluded if t in folders)
        if present_excluded:
            log(f"warning: {len(present_excluded)} --excluded tiles do have a lens folder; not packaged")
        extra = len(folders - set(uid_of))
        if extra:
            log(f"note: {extra} lens folders are not included manifest rows; not packaged")

    csv_snap, csv_counts, csv_files = csv_snapshot_info(csv_dir, manifest)
    ctx = {
        "out_dir": out_dir,
        "release_name": release_name,
        "tiles": tiles,
        "uid_of": uid_of,
        "excluded": {t: r for t, r in excluded.items() if t in uid_of},
        "extra_folders": extra,
        "catalogue_dir": catalogue_dir,
        "catalogue_rel": relative(catalogue_dir) if catalogue_dir else "",
        "catalogue_revision": catalogue_revision(catalogue_dir) if catalogue_dir else "",
        "dataset_dir": dataset_dir,
        "dataset_rel": relative(dataset_dir),
        "manifest_rel": relative(manifest),
        "manifest_sha256": csv_snap["manifest_sha256"],
        "n_included": len(rows),
        "csv_snapshot": csv_snap,
        "csv_counts": csv_counts,
        "csv_files": csv_files,
        "csv_rel": relative(csv_dir),
        "git_head": git_revision(PROJECT_ROOT),
        "known_issues": resolve(args.known_issues),
    }
    log(f"release {release_name}: tiers {tiers}, {len(tiles)} lenses, cutoff {csv_snap['cutoff_utc']}, -> {out_dir}")
    summary = {tier: package_tier(tier, args, ctx) for tier in tiers}
    if args.dry_run:
        log("dry run: nothing written")
    return summary


if __name__ == "__main__":
    main()
    sys.exit(0)
