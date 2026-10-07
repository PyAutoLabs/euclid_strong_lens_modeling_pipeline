"""
Euclid Pipeline: Consolidated dr1_sep1 Manifest
===============================================

Writes the per-tile manifest of the consolidated ``dr1_sep1`` sample (2026-10-07):
every DR1 dataset tile, from the former samples ``dr1_sep1_rest`` (14,032),
``dr1_sep1_top1000`` (1,000) and ``dr1_v2_new`` (987), with the v2 grade and
whether the tile enters the catalogue. The catalogue build reads this manifest.

Inputs (repository root)
------------------------
- ``catalogue/grades/tile_grades_dr1_sep1_v2.csv`` — the 15,032 sep1 tiles and
  their v2 ``unique_identifier``; its ``sample`` column becomes ``origin_sample``.
- ``hpc/run_manifests/dr1_v2_new_selection_20261001.csv`` — the sep28 cut-outs;
  rows whose ``selection`` starts with ``queued`` are the ``dr1_v2_new`` tiles
  (the 69 ``excluded_*`` rows were never extracted). If ``dataset/dr1_v2_new/``
  still exists its tile dirs must equal that list.
- ``DR1_full_Object_Catalogue_v2.csv`` — ``SWAP_Grade``, ``id_str_for_cutout``.

Rules
-----
- v2 grade not A/B -> ``include_in_catalogue=False``, ``reason=grade_c_x``.
- A/B object with one tile -> ``included``.
- A/B object with two or more tiles -> the tile whose name is exactly the
  object's ``id_str_for_cutout`` (``15arcsec_near_RADEC_`` prefix included) is
  ``included``, the others ``duplicate_tile``. No unique exact match is an error.
  The recentred sep28 cut-out wins every one of the 45 pairs; positions are never
  derived from tile names (a prefixed name carries a nearby cut-out centre).

A/B objects with no tile at all go to ``catalogue/grades/dr1_sep1_uncovered_ab.csv``.

Usage
-----
    python scripts/tools/consolidated_manifest.py
"""
import argparse
import collections
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRADES_DIR = PROJECT_ROOT / "catalogue" / "grades"
SELECTION_CSV = PROJECT_ROOT / "hpc" / "run_manifests" / "dr1_v2_new_selection_20261001.csv"

UNCOVERED_NOTES = {
    "EUCL-DR1-L009168": "no FITS made by the lens-finding team (to be sent separately)",
    "EUCL-DR1-L019278": "no FITS made by the lens-finding team (to be sent separately)",
    "EUCL-DR1-L020583": "tile not in the sep1 segmentation delivery (15,032 segmented tiles vs 15,033 FITS)",
}


def _clean(value):
    return (value or "").replace("\r", "").strip()


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{_clean(k): _clean(v) for k, v in row.items()} for row in csv.DictReader(f)]


def build(out_dir: Path):
    objects = {o["Unique_Identifier"]: o for o in _read(PROJECT_ROOT / "DR1_full_Object_Catalogue_v2.csv")}

    tiles = [(g["tile"], g["unique_identifier"], g["sample"])
             for g in _read(GRADES_DIR / "tile_grades_dr1_sep1_v2.csv")]
    queued = [(s["tile"], s["unique_identifier"], "dr1_v2_new")
              for s in _read(SELECTION_CSV) if s["selection"].startswith("queued")]
    v2_dir = PROJECT_ROOT / "dataset" / "dr1_v2_new"
    if v2_dir.is_dir():
        on_disk = {p.name for p in v2_dir.iterdir() if p.is_dir()}
        if on_disk != {t for t, _, _ in queued}:
            raise SystemExit(f"dataset/dr1_v2_new dirs ({len(on_disk)}) differ from the queued selection rows ({len(queued)})")
    tiles += queued
    names = [t for t, _, _ in tiles]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate tile names across samples")

    by_uid = collections.defaultdict(list)
    for tile, uid, _ in tiles:
        by_uid[uid].append(tile)

    rows = []
    for tile, uid, origin in sorted(tiles):
        obj = objects.get(uid)
        grade = obj["SWAP_Grade"] if obj else ""
        if grade not in ("A", "B"):
            include, reason = False, "grade_c_x"
        elif len(by_uid[uid]) == 1:
            include, reason = True, "included"
        else:
            exact = [t for t in by_uid[uid] if t == obj["id_str_for_cutout"]]
            if len(exact) != 1:
                raise SystemExit(f"{uid}: {len(exact)} tiles match id_str_for_cutout exactly among {by_uid[uid]}")
            include = tile == exact[0]
            reason = "included" if include else "duplicate_tile"
        rows.append(dict(tile=tile, unique_identifier=uid, v2_grade=grade,
                         origin_sample=origin, include_in_catalogue=include, reason=reason))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "dr1_sep1_consolidated_manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    uncovered = [o for uid, o in objects.items() if o["SWAP_Grade"] in ("A", "B") and uid not in by_uid]
    with open(out_dir / "dr1_sep1_uncovered_ab.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["unique_identifier", "v2_grade", "v2_score", "RA_obj", "DEC_obj", "id_str_for_cutout", "note"])
        for o in sorted(uncovered, key=lambda o: o["Unique_Identifier"]):
            uid = o["Unique_Identifier"]
            w.writerow([uid, o["SWAP_Grade"], o["SWAP_Score"], o["RA_obj"], o["DEC_obj"],
                        o["id_str_for_cutout"], UNCOVERED_NOTES.get(uid, "")])

    counts = collections.Counter((r["reason"], r["origin_sample"]) for r in rows)
    print(f"{manifest.relative_to(PROJECT_ROOT)}: {len(rows)} tiles")
    for key in sorted(counts):
        print(f"  {key[0]:15s} {key[1]:17s} {counts[key]}")
    included = [r for r in rows if r["include_in_catalogue"]]
    print(f"  included A+B objects: {len({r['unique_identifier'] for r in included})} (rows {len(included)})")
    print(f"  A+B objects with no tile: {len(uncovered)} -> dr1_sep1_uncovered_ab.csv")
    return rows, uncovered


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out_dir", default=str(GRADES_DIR), help="Output directory (default catalogue/grades).")
    build(Path(parser.parse_args().out_dir))


if __name__ == "__main__":
    main()
