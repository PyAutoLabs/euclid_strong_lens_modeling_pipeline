"""
Euclid Pipeline: Build A CSV-Only Catalogue Snapshot
====================================================

The single entry point of ``scripts/tools/csv_snapshot/``: collect -> assemble
-> package. Writes the metadata-only catalogue tables of every
``include_in_catalogue=True`` tile of the consolidated manifest into
``<out_dir>/``, with the intermediates under ``<out_dir>/_build/``. No fit is
rerun and no image is extracted: it reads three small JSON files per result.

The table set and column semantics are those of the 2026-10-11 release
(``dr1_catalogue_15067_20261011_csv_only``), so a new snapshot diffs against it
file by file (headers, ``master_columns.csv``, row counts in ``file_manifest.csv``).

``scripts/tools/package_release.py --csv_dir=<out_dir>`` puts these tables into
the csv / vis / full release tiers; it checks that the manifest sha256 recorded
in ``<out_dir>/_build/snapshot.json`` matches the manifest it is given.

Usage
-----
    python scripts/tools/csv_snapshot/build.py \\
        --manifest=catalogue/grades/dr1_sep1_consolidated_manifest.csv \\
        --out_dir=inspect/dr1_catalogue_v1_1_csv --cutoff=2026-10-20T00:00:00Z

    # stages separately (e.g. a collect sharded over several jobs, then one assemble)
    python scripts/tools/csv_snapshot/build.py --out_dir=... --stage=collect --shard=1 --n_shards=4
    python scripts/tools/csv_snapshot/build.py --out_dir=... --stage=assemble,package
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assemble  # noqa: E402
import collect  # noqa: E402
import package  # noqa: E402
from common import DEFAULT_OBJECTS, DEFAULT_UNCOVERED, parse_time  # noqa: E402

STAGES = ("collect", "assemble", "package")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a CSV-only catalogue snapshot.")
    collect.add_arguments(parser)
    parser.add_argument("--objects", default=str(DEFAULT_OBJECTS))
    parser.add_argument("--uncovered", default=str(DEFAULT_UNCOVERED))
    parser.add_argument("--force", action="store_true", help="re-assemble over existing tables")
    package.add_arguments(parser)
    parser.add_argument("--stage", default=",".join(STAGES),
                        help="comma-separated subset of collect,assemble,package")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.project_root)
    p = lambda x: collect.resolve(x, root)  # noqa: E731
    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown --stage {sorted(unknown)}")
    out_dir = p(args.out_dir)
    manifest = p(args.manifest)
    if "collect" in stages:
        collect.collect(
            root, args.sample, manifest, parse_time(args.cutoff), out_dir, args.workers,
            p(args.band_inventory) if args.band_inventory else None, args.band_inventory_before,
            args.shard, args.n_shards,
        )
    if "assemble" in stages:
        assemble.assemble(out_dir, manifest, p(args.objects), p(args.uncovered), args.force)
    if "package" in stages:
        package.package(out_dir, manifest, p(args.uncovered), args.parity_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
