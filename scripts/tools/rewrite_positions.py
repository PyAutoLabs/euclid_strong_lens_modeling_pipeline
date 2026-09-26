"""
Rewrite every tile's ``positions.json`` with the production positions writer.

WHAT THIS DOES
    For every tile it re-runs the writer ``preprocess/segmentation.py`` uses
    (``segmentation.write_positions`` -> ``positions_finder.find_positions_gate``,
    finder version ``positions_finder.FINDER_VERSION``) on the tile's
    ``segmentation/source_flux.fits``, and overwrites the tile's positions:

    1. **backup** -- if ``positions_segmentation_v1.json`` does not exist, the
       current ``positions.json`` is copied to it. An existing backup is never
       overwritten, so a rerun (or a ``--force`` rerun) always keeps the
       original segmentation-era positions;
    2. **rewrite** -- the finder runs unseeded, from the source-flux map, the
       VIS RMS map (its SNR map) and the lens-flux / VIS light centre, and
       writes ``positions.json`` (when it returns any positions) and the
       ``positions_meta.json`` sidecar (always). The sidecar gains one key
       over ``segmentation.py``'s, ``backup`` (the backup's file name, or null
       when the tile had no ``positions.json`` to back up), which also marks
       the tile as rewritten;
    3. **no positions** -- when the finder returns none (status ``n_lt_2``, or
       ``review``), the old ``positions.json`` is removed (the backup holds
       it), so no stale set is left. ``util.load_vis_dataset`` then reads the
       sidecar (``util.positions_likelihood_list_from_meta`` returns
       ``(True, None)`` for fewer than two ``positions_used`` or a null
       threshold) and fits with the positions penalty off.

    A tile whose sidecar is already current (``backup`` key present, finder
    and schema versions current, and ``positions.json`` matching its
    ``positions_sha``, or absent when no positions were written) is skipped
    unless ``--force``. A tile missing an input (``info.json``,
    ``segmentation/source_flux.fits`` or the VIS RMS map) is recorded as an
    error and left untouched; it never stops a batch.

    Each run writes a CSV under the root, ``positions_rewrite.csv``, or
    ``positions_rewrite_part<I>.csv`` for ``--part I --nparts N``, with one
    row per tile: ``tile``, ``action`` (``written`` | ``skipped`` |
    ``error``), ``status``, ``n_positions``, ``threshold``,
    ``review_reason``, ``backup`` (whether this run created it), ``seconds``
    and ``error``. ``--merge`` concatenates the part CSVs into
    ``positions_rewrite.csv`` (equivalently:
    ``head -1 part0.csv > all.csv; tail -qn +2 part*.csv >> all.csv``).

    Rollback: copy ``positions_segmentation_v1.json`` back to
    ``positions.json`` and delete ``positions_meta.json``.

    Imports: numpy / scipy / astropy, plus matplotlib through
    ``preprocess/segmentation.py`` (for its diagnostic PNG, not called here). No
    PyAutoLens import, so it runs as a plain CPU array job (~1 s per tile).

USAGE
    python scripts/tools/rewrite_positions.py --tile dataset/<sample>/<tile>
    python scripts/tools/rewrite_positions.py --root dataset/<sample> --nproc 8
    python scripts/tools/rewrite_positions.py --root dataset/<sample> --part 3 --nparts 20
    python scripts/tools/rewrite_positions.py --root dataset/<sample> --tiles-file tiles.txt
    python scripts/tools/rewrite_positions.py --root dataset/<sample> --merge
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "preprocess"))
import positions_finder  # noqa: E402
import segmentation  # noqa: E402

BACKUP_NAME = "positions_segmentation_v1.json"
POSITIONS_NAME = "positions.json"
CSV_STEM = "positions_rewrite"
CSV_COLUMNS = (
    "tile",
    "action",
    "status",
    "n_positions",
    "threshold",
    "review_reason",
    "backup",
    "seconds",
    "error",
)


def csv_name(part: Optional[int] = None, nparts: int = 1) -> str:
    """``positions_rewrite.csv``, or ``positions_rewrite_part<I>.csv`` for a split run."""
    if part is None or nparts <= 1:
        return f"{CSV_STEM}.csv"
    return f"{CSV_STEM}_part{part}.csv"


def _read_meta(d: Path) -> Optional[Dict]:
    path = d / positions_finder.META_NAME
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def is_current(tile_dir) -> bool:
    """
    True when the tile was already rewritten by this finder version.

    The sidecar carries the ``backup`` key (only this tool writes it), the
    current schema and finder versions and ``method: gate``; its positions agree
    with ``positions.json`` (sha match), or there is no ``positions.json`` when
    it holds no positions.
    """
    d = Path(tile_dir)
    meta = _read_meta(d)
    if meta is None or "backup" not in meta:
        return False
    if (
        meta.get("method") != "gate"
        or meta.get("version") != positions_finder.META_VERSION
        or meta.get("finder_version") != positions_finder.FINDER_VERSION
    ):
        return False
    pos = d / POSITIONS_NAME
    if meta.get("positions_used"):
        return pos.exists() and meta.get(
            "positions_sha"
        ) == positions_finder.positions_sha(d)
    return not pos.exists()


def _row(tile, action, meta=None, backup=False, seconds=None, error="") -> Dict:
    meta = meta or {}
    used = meta.get("positions_used")
    return dict(
        tile=tile,
        action=action,
        status=meta.get("status", ""),
        n_positions="" if used is None else len(used),
        threshold="" if meta.get("threshold") is None else meta["threshold"],
        review_reason=meta.get("review_reason", ""),
        backup=bool(backup),
        seconds="" if seconds is None else round(seconds, 2),
        error=error,
    )


def rewrite_tile(tile_dir, force: bool = False) -> Dict:
    """
    Back up and rewrite one tile's positions; returns its CSV row (a dict).

    Never raises for a bad tile: missing inputs or a finder failure come back as
    ``action == "error"`` with the tile left as it was (the backup, if this call
    made one, stays: it is a verbatim copy).
    """
    d = Path(tile_dir)
    t0 = time.time()
    try:
        if not force and is_current(d):
            return _row(d.name, "skipped", _read_meta(d), seconds=time.time() - t0)
        for need in ("info.json", "segmentation/source_flux.fits", f"{d.name}.fits"):
            if not (d / need).exists():
                raise FileNotFoundError(f"missing {need}")

        pos = d / POSITIONS_NAME
        backup = d / BACKUP_NAME
        made_backup = False
        if not backup.exists() and pos.exists():
            shutil.copy2(pos, backup)
            made_backup = True

        result, meta = segmentation.write_positions(
            d, require_noise=True, verbose=False
        )
        if not result.positions and pos.exists():
            # No positions: the backup holds the old set; leave none behind.
            pos.unlink()
        meta["backup"] = BACKUP_NAME if backup.exists() else None
        positions_finder.write_meta(d, meta)
        return _row(
            d.name, "written", meta, backup=made_backup, seconds=time.time() - t0
        )
    except Exception as exc:  # one broken tile must not stop a batch
        return _row(d.name, "error", seconds=time.time() - t0, error=repr(exc))


def _rewrite_one(args):
    d, force = args
    return rewrite_tile(d, force=force)


def select_tiles(
    root,
    tiles: Optional[Sequence[str]] = None,
    part: Optional[int] = None,
    nparts: int = 1,
) -> List[str]:
    """Sorted tile names under ``root`` (or ``tiles``), then every ``nparts``-th from ``part``."""
    root = Path(root)
    names = (
        sorted(set(tiles))
        if tiles
        else sorted(p.name for p in root.iterdir() if p.is_dir())
    )
    if part is not None and nparts > 1:
        names = [n for i, n in enumerate(names) if i % nparts == part]
    return names


def rewrite_batch(
    root,
    tiles: Optional[Sequence[str]] = None,
    part: Optional[int] = None,
    nparts: int = 1,
    nproc: int = 1,
    force: bool = False,
    out_csv=None,
) -> List[Dict]:
    """
    Rewrite every selected tile under ``root`` and write the run's CSV.

    A tile named in ``tiles`` but absent under ``root`` is an ``error`` row.
    """
    root = Path(root)
    names = select_tiles(root, tiles, part, nparts)
    jobs = [(root / n, force) for n in names]
    if nproc > 1:
        from multiprocessing import Pool

        with Pool(nproc) as pool:
            rows = list(pool.imap(_rewrite_one, jobs, chunksize=1))
    else:
        rows = [_rewrite_one(j) for j in jobs]
    out_csv = Path(out_csv) if out_csv else root / csv_name(part, nparts)
    write_csv(out_csv, rows)
    return rows


def write_csv(path, rows: Sequence[Dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def merge_csvs(root) -> Path:
    """Concatenate ``positions_rewrite_part*.csv`` under ``root`` into ``positions_rewrite.csv``."""
    root = Path(root)
    rows: List[Dict] = []
    for p in sorted(root.glob(f"{CSV_STEM}_part*.csv")):
        with open(p, newline="") as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r: r["tile"])
    return write_csv(root / csv_name(), rows)


def _summary(rows: Sequence[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        key = r["action"] if r["action"] != "written" else f"written:{r['status']}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite positions.json with the production positions writer "
            f"(positions_finder {positions_finder.FINDER_VERSION}, unseeded, from "
            f"source_flux), backing the old file up to {BACKUP_NAME}."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--tile", help="one tile directory")
    src.add_argument("--root", help="a sample directory whose subdirectories are tiles")
    parser.add_argument(
        "--tiles-file", help="with --root: only the tiles named in this file"
    )
    parser.add_argument(
        "--part", type=int, help="with --root: this part (0-based) of --nparts"
    )
    parser.add_argument(
        "--nparts", type=int, default=1, help="number of parts (default 1)"
    )
    parser.add_argument(
        "--nproc", type=int, default=1, help="worker processes (default 1)"
    )
    parser.add_argument(
        "--force", action="store_true", help="rewrite even if the tile is current"
    )
    parser.add_argument(
        "--out-csv", help="CSV path (default <root>/positions_rewrite[_partI].csv)"
    )
    parser.add_argument(
        "--merge", action="store_true", help="with --root: merge the part CSVs and exit"
    )
    args = parser.parse_args(argv)

    if args.tile:
        row = rewrite_tile(args.tile, force=args.force)
        print(json.dumps(row, indent=2))
        return 1 if row["action"] == "error" else 0

    if args.merge:
        path = merge_csvs(args.root)
        print(f"merged into {path}")
        return 0
    if args.part is not None and not 0 <= args.part < args.nparts:
        parser.error("--part must be in [0, --nparts)")

    tiles = Path(args.tiles_file).read_text().split() if args.tiles_file else None
    t0 = time.time()
    rows = rewrite_batch(
        args.root,
        tiles=tiles,
        part=args.part,
        nparts=args.nparts,
        nproc=args.nproc,
        force=args.force,
        out_csv=args.out_csv,
    )
    print(f"rewrote {len(rows)} tiles in {time.time() - t0:.0f} s: {_summary(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
