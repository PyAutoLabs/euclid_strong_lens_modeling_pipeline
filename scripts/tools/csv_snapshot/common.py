"""
Euclid Pipeline: CSV Snapshot Shared Helpers
============================================

Paths, readers and small formatting helpers shared by the stages of
``scripts/tools/csv_snapshot/`` (see ``build.py``). No CLI.

Layout of a snapshot directory ``<out_dir>``::

    <out_dir>/                       the release tables (README.txt, master, ...)
    <out_dir>/_build/snapshot.json   cutoff, manifest sha256, counts
    <out_dir>/_build/results*.jsonl.gz   one collected JSON line per tile
    <out_dir>/_build/validation_counts.json
"""

import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MANIFEST = PROJECT_ROOT / "catalogue" / "grades" / "dr1_sep1_consolidated_manifest.csv"
DEFAULT_OBJECTS = PROJECT_ROOT / "catalogue" / "DR1_full_Object_Catalogue_v5_fits.csv"
DEFAULT_UNCOVERED = PROJECT_ROOT / "catalogue" / "grades" / "dr1_sep1_uncovered_ab.csv"

BUILD_DIR = "_build"
SNAPSHOT_JSON = "snapshot.json"

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def _clean(value):
    return (value or "").replace("\r", "").strip()


def read_csv(path):
    """Rows of a CSV as dicts, tolerating a BOM and CRLF line ends."""
    with open(path, newline="", encoding="utf-8-sig") as stream:
        return [{_clean(k): _clean(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def included_rows(manifest_path):
    """``include_in_catalogue=True`` rows of the consolidated manifest, in file
    order. Each unique_identifier and tile must appear once."""
    rows = [r for r in read_csv(manifest_path) if r["include_in_catalogue"] == "True"]
    uids = [r["unique_identifier"] for r in rows]
    tiles = [r["tile"] for r in rows]
    if len(set(uids)) != len(uids) or len(set(tiles)) != len(tiles):
        raise SystemExit(f"{manifest_path}: included rows repeat a unique_identifier or tile")
    if not rows:
        raise SystemExit(f"{manifest_path}: no include_in_catalogue=True rows")
    return rows


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value):
    """Unix seconds from ``None`` (now), a number, or an ISO-8601 string
    (naive strings are UTC; a trailing ``Z`` is accepted)."""
    if value is None or value == "":
        return time.time()
    try:
        return float(value)
    except ValueError:
        pass
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def utc_iso(unix):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(unix))


def number_word(n):
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def master_name(n):
    return f"dr1_catalogue_{n}_master.csv"


def excluded_name(n):
    return f"excluded_{number_word(n)}_candidates.csv"


def git_revision(path=PROJECT_ROOT):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def read_snapshot(out_dir):
    return json.loads((Path(out_dir) / BUILD_DIR / SNAPSHOT_JSON).read_text())


def write_json(path, payload):
    path = Path(path)
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(json.dumps(payload, indent=2) + "\n")
    staged.replace(path)
