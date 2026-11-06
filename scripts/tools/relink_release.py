"""Re-package a release's documents without rewriting its part zips.

Runs ``package_release.py`` unchanged except that ``write_part`` hard-links each part zip from
the earlier build ``SRC_RELEASE`` and returns its rows from that build's ``file_manifest.csv``,
after checking they are exactly the planned members. The packager still CRC- and size-verifies
every part zip and rebuilds every document, manifest and ``SHA256SUMS``. Use it when only the
tier documents change: ``snapshot.json`` paths (2026-11-03) or ``KNOWN_ISSUES.md`` (2026-11-06).

Usage: SRC_RELEASE=<earlier out_dir> python scripts/tools/relink_release.py <package_release args>
"""
import csv
import io
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_release as pr  # noqa: E402

SRC = Path(os.environ["SRC_RELEASE"])


def old_rows(tier):
    archive = zipfile.ZipFile(SRC / tier / "catalogue_csvs.zip")
    name = next(n for n in archive.namelist() if n.count("/") == 1 and n.endswith("/file_manifest.csv"))
    rows = {}
    for r in csv.DictReader(io.TextIOWrapper(archive.open(name), newline="")):
        rows.setdefault(r["zip"], []).append((r["member"], int(r["bytes"]), r["sha256"]))
    return rows


ROWS = {tier: old_rows(tier) for tier in ("csv", "full") if (SRC / tier).is_dir()}


def write_part(job):
    zip_path, arc_root, members = job
    zip_path = Path(zip_path)
    rows = ROWS[zip_path.parent.name][zip_path.name]
    planned = {f"{arc_root}/{arc}" for _kind, _source, arc, _size in members}
    if planned != {m for m, _s, _h in rows}:
        raise RuntimeError(f"{zip_path.name}: earlier build's members differ from the plan")
    os.link(SRC / zip_path.parent.name / zip_path.name, zip_path)
    return zip_path.name, rows


pr.write_part = write_part

if __name__ == "__main__":
    pr.main(sys.argv[1:])
