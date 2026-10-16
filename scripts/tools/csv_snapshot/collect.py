"""
Euclid Pipeline: CSV Snapshot Collector
=======================================

Stage 1 of ``scripts/tools/csv_snapshot/build.py``. Reads completed-result
metadata only: it never extracts images, never refits and never modifies a
result. For every ``include_in_catalogue=True`` tile of the consolidated
manifest it writes one JSON line holding

- ``expected_bands``: the dataset FITS ``*_BGSUB`` HDUs, or the cached band list
  of ``--band_inventory`` for a cutout unchanged since that inventory;
- ``vis_lp`` / ``vis_pix``: the newest result of
  ``output/<sample>/<tile>/initial_lens_model/<stage>/`` at or before the cutoff;
- ``sed``: the same for every band of
  ``output_sed/<sample>/<tile>/sersic_lens_model/<band>/``.

A result is a ``<hash>.zip`` or ``<hash>/`` directory with a ``.completed``
marker; its ``files/samples_summary.json``, ``files/latent/latent_summary.json``
and ``files/wcs.json`` are decoded (the stored ``model`` objects are dropped).
Results whose modification time is after the cutoff are ignored, so a snapshot
is reproducible while fits keep landing. Read errors are recorded per tile and
end up in ``metadata_errors.csv``.

Output: ``<out_dir>/_build/results.jsonl.gz`` (``results_<k>.jsonl.gz`` with
``--shard``) and ``<out_dir>/_build/snapshot.json`` (cutoff, manifest sha256).

Usage
-----
    python scripts/tools/csv_snapshot/collect.py --out_dir=inspect/dr1_catalogue_v1_1_csv \\
        --cutoff=2026-10-20T00:00:00Z --workers=16
"""

import argparse
import gzip
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BUILD_DIR,
    DEFAULT_MANIFEST,
    PROJECT_ROOT,
    SNAPSHOT_JSON,
    file_sha256,
    git_revision,
    included_rows,
    parse_time,
    read_csv,
    utc_iso,
    write_json,
)

RESULT_FILES = [
    ("summary", "files/samples_summary.json"),
    ("latent", "files/latent/latent_summary.json"),
    ("wcs", "files/wcs.json"),
]


def decode(x):
    """Flatten the PyAutoFit JSON encoding (typed tuples / dicts / instances)."""
    if isinstance(x, list):
        return [decode(v) for v in x]
    if not isinstance(x, dict):
        return x
    if x.get("type") in ("tuple", "list"):
        return decode(x["values"])
    if x.get("type") in ("dict", "instance"):
        return decode(x.get("arguments", {}))
    return {k: decode(v) for k, v in x.items()}


class Collector:
    def __init__(self, project_root: Path, sample: str, cutoff: float, band_cache=None,
                 band_cache_before=None):
        self.root = Path(project_root)
        self.sample = sample
        self.cutoff = cutoff
        self.band_cache = band_cache or {}
        self.band_cache_before = band_cache_before

    def result(self, band_dir: Path, errors):
        """Newest completed result in ``band_dir`` at or before the cutoff."""
        if not band_dir.is_dir():
            return None
        candidates = []
        for path in band_dir.iterdir():
            if path.suffix == ".zip" or path.is_dir():
                mtime = path.stat().st_mtime
                if mtime <= self.cutoff:
                    candidates.append((mtime, str(path), path))
        for mtime, _name, path in sorted(candidates, reverse=True):
            try:
                archive = zipfile.ZipFile(path) if path.suffix == ".zip" else None
                try:
                    names = set(archive.namelist()) if archive else None

                    def has(name):
                        return name in names if archive else (path / name).exists()

                    if not has(".completed"):
                        continue
                    record = {
                        "path": str(path.relative_to(self.root)),
                        "mtime": mtime,
                        "id": path.stem,
                        "completed": True,
                    }
                    for key, name in RESULT_FILES:
                        if has(name):
                            raw = archive.read(name) if archive else (path / name).read_bytes()
                            record[key] = decode(json.loads(raw))
                    # The stored model object is unnecessary and can be huge.
                    for key in ("summary", "latent"):
                        if key in record and isinstance(record[key], dict):
                            record[key].pop("model", None)
                    return record
                finally:
                    if archive:
                        archive.close()
            except Exception as error:  # recorded, never fatal
                errors.append({"path": str(path), "error": repr(error)})
        return None

    def expected_bands(self, tile, fits_path: Path, record, errors):
        cached = self.band_cache.get(tile)
        if (
            cached is not None
            and fits_path.exists()
            and self.band_cache_before is not None
            and fits_path.stat().st_mtime < self.band_cache_before
        ):
            record["expected_bands"] = cached
            record["expected_bands_source"] = "band inventory (unchanged cutout)"
            return
        if fits_path.exists():
            try:
                from astropy.io import fits

                with fits.open(fits_path, lazy_load_hdus=True, memmap=True) as hdus:
                    record["expected_bands"] = sorted(
                        h.name[:-6].lower() for h in hdus if h.name.endswith("_BGSUB")
                    )
                record["expected_bands_source"] = "dataset FITS"
            except Exception as error:
                errors.append({"path": str(fits_path), "error": repr(error)})

    def one(self, row):
        tile = row["tile"]
        errors = []
        record = {"manifest": row, "errors": errors, "sed": {}}
        fits_path = self.root / "dataset" / self.sample / tile / f"{tile}.fits"
        record["input_fits"] = fits_path.exists()
        self.expected_bands(tile, fits_path, record, errors)
        for band in ("vis_lp", "vis_pix"):
            record[band] = self.result(
                self.root / "output" / self.sample / tile / "initial_lens_model" / band, errors
            )
        sed = self.root / "output_sed" / self.sample / tile / "sersic_lens_model"
        if sed.is_dir():
            for band_dir in sorted(sed.iterdir()):
                if band_dir.is_dir():
                    found = self.result(band_dir, errors)
                    if found:
                        record["sed"][band_dir.name] = found
        return record


def load_band_cache(path):
    if not path:
        return {}
    return {
        r["tile_name"]: r["expected_bands"].split(";")
        for r in read_csv(path)
        if r.get("expected_bands")
    }


def shard_rows(rows, shard, n_shards):
    """Contiguous slice ``shard`` (1-based) of ``n_shards``."""
    if not n_shards:
        return rows
    if not 1 <= shard <= n_shards:
        raise SystemExit(f"--shard must be in 1..{n_shards}")
    per = -(-len(rows) // n_shards)
    return rows[(shard - 1) * per: shard * per]


def collect(project_root, sample, manifest, cutoff, out_dir, workers=16, band_inventory=None,
            band_inventory_before=None, shard=0, n_shards=0, log=print):
    """Run the collector; return the path of the JSON-lines file written."""
    project_root = Path(project_root)
    out_dir = Path(out_dir)
    build = out_dir / BUILD_DIR
    build.mkdir(parents=True, exist_ok=True)
    rows = included_rows(manifest)
    band_cache = load_band_cache(band_inventory)
    before = None
    if band_inventory:
        before = parse_time(band_inventory_before) if band_inventory_before else Path(band_inventory).stat().st_mtime
    if not n_shards or shard == 1:
        write_json(
            build / SNAPSHOT_JSON,
            {
                "cutoff_unix": cutoff,
                "cutoff_utc": utc_iso(cutoff),
                "started_utc": utc_iso(time.time()),
                "sample": sample,
                "manifest": str(manifest),
                "manifest_sha256": file_sha256(manifest),
                "n_included": len(rows),
                "band_inventory": str(band_inventory) if band_inventory else "",
                "band_inventory_before_unix": before,
                "git_head": git_revision(project_root),
            },
        )
    rows = shard_rows(rows, shard, n_shards)
    collector = Collector(project_root, sample, cutoff, band_cache, before)
    target = build / (f"results_{shard}.jsonl.gz" if n_shards else "results.jsonl.gz")
    staged = target.with_name(f".{target.name}.tmp")
    start = time.time()
    with gzip.open(staged, "wt", compresslevel=3) as out, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, record in enumerate(pool.map(collector.one, rows), 1):
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
            if i % 500 == 0:
                log(f"collect: {i}/{len(rows)} ({time.time() - start:.0f} s)")
    staged.replace(target)
    log(f"collect: {len(rows)} tiles -> {target} ({time.time() - start:.0f} s)")
    return target


def add_arguments(parser):
    parser.add_argument("--project_root", default=str(PROJECT_ROOT))
    parser.add_argument("--sample", default="dr1_sep1")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--cutoff", default=None,
                        help="ISO-8601 (UTC if naive) or unix seconds; results modified later are "
                        "ignored. Default: now.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) * 4,
                        help="reader threads (I/O bound; default 4 x SLURM_CPUS_PER_TASK)")
    parser.add_argument("--band_inventory", default=None,
                        help="band_inventory.csv of an earlier build; its expected_bands are reused "
                        "for cutouts not modified since --band_inventory_before")
    parser.add_argument("--band_inventory_before", default=None,
                        help="ISO or unix; default the --band_inventory file's mtime")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=0)


def resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect completed-result metadata for a CSV snapshot.")
    add_arguments(parser)
    args = parser.parse_args(argv)
    root = Path(args.project_root)
    collect(
        root,
        args.sample,
        resolve(args.manifest, root),
        parse_time(args.cutoff),
        resolve(args.out_dir, root),
        args.workers,
        resolve(args.band_inventory, root) if args.band_inventory else None,
        args.band_inventory_before,
        args.shard,
        args.n_shards,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
