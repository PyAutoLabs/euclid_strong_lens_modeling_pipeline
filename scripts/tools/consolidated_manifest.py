"""
Euclid Pipeline: Consolidated dr1_sep1 Manifest
===============================================

Writes the membership manifest of the DR1 catalogue from the final lens-finding
catalogue v5 (Holloway, 19 Oct 2026), as delivered with its modelling-cutout
columns in ``catalogue/DR1_full_Object_Catalogue_v5_fits.csv``. The catalogue
build (``scripts/tools/csv_snapshot/``, ``package_release.py``, the chunked
build's ``MANIFEST=``) reads this manifest.

One row per v5 Grade A/B object (15,072), keyed on the v5 ``Unique_Identifier``.
No object is remodelled: the catalogue is built from the existing fits only, so
the v5 cutouts are not ingested and an object without a usable model stays in
the object list with ``include_in_catalogue=False`` and a reason.

Columns
-------
- ``tile``: the ``dataset/dr1_sep1`` tile its model was fitted on, the file of
  the v5 ``fits_modelled_20261011`` column (``batch:filename``, the 2026-10-11
  snapshot). Blank when the object was never modelled.
- ``unique_identifier``, ``grade``, ``score``: v5 ``Unique_Identifier``,
  ``SWAP_Grade``, ``SWAP_Score``.
- ``cutout_batch``: the batch of the modelled file (``sep1``: the 1 Sep v1
  cutouts; ``sep28``: the 28 Sep top-up; both live in ``dataset/dr1_sep1``).
- ``include_in_catalogue``, ``reason``: ``included``; ``model_unusable`` (a model
  exists but is not of the lens); ``no_model`` (never modelled).
- ``note``: why, for every row that is not plainly included, and identifier
  relabels.

Rules
-----
- The modelled file is the batch:filename the v5 catalogue records; for the 45
  objects with both a sep1 and a recentred sep28 tile that is the sep28 one, as
  in the v2 manifest. ``TILE_ALIASES`` maps the one modelled file name that
  ``dataset/dr1_sep1`` holds under a different name.
- ``MODEL_UNUSABLE`` lists the models that are not of the lens (reasons from the
  lens-finding team's 2026-10-19 notes); everything else with a model is
  included, whatever its completion state. Fits that never finished are
  flagged by the build's completion columns (``vis_pix_completed`` ...) and, in
  the vis/full tiers, by ``EXCLUDED_FILE``, as before.

The not-included rows are also written to
``catalogue/grades/dr1_sep1_uncovered_ab.csv``, which the CSV snapshot ships as
``excluded_<n>_candidates.csv``.

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
OBJECT_CSV = PROJECT_ROOT / "catalogue" / "DR1_full_Object_Catalogue_v5_fits.csv"
DATASET_DIR = PROJECT_ROOT / "dataset" / "dr1_sep1"

BATCHES = ("sep1", "sep28")
MANIFEST_COLUMNS = ["tile", "unique_identifier", "grade", "score", "cutout_batch",
                    "include_in_catalogue", "reason", "note"]
UNCOVERED_COLUMNS = ["unique_identifier", "grade", "score", "RA_obj", "DEC_obj", "id_str_for_cutout",
                     "tile", "reason", "note"]

# v5 modelled file name -> the dataset/dr1_sep1 tile holding it.
TILE_ALIASES = {
    # L020528: the sep1 delivery stored this cutout without the prefix.
    "15arcsec_near_RADEC_Tile102161607RA2267255398997DEC0704364979868":
        "Tile102161607RA2267255398997DEC0704364979868",
}

MODEL_UNUSABLE = {
    "EUCL-DR1-L001796": "model is of the saturated star the old cutout was centred on (fitted centre 5.3\" "
                        "from the v4 position)",
    "EUCL-DR1-L020505": "fitted lens centre 1.6\" from the v5 position (multi-galaxy system)",
    "EUCL-DR1-L002411": "sep1 cutout has an all-zero VIS image (blank wide-plan mosaic at the deep-field edge)",
}

NO_MODEL = {
    "EUCL-DR1-L026833": "new in v3 (split from L002330); only a v5 cutout exists, not modelled",
    "EUCL-DR1-L026835": "new in v5 (split from L023429); only a v5 cutout exists, not modelled",
    "EUCL-DR1-L009168": "MER tile not in the archive; only a v5 cutout (VIS/Y/J/H) exists, not modelled",
    "EUCL-DR1-L019278": "MER tile not in the archive; only a v5 cutout (VIS/Y/J/H + g) exists, not modelled",
    "EUCL-DR1-L020583": "sep1 cutout not in the segmentation delivery, so never modelled",
}

RELABELLED = {
    "EUCL-DR1-L026834": "model filed under the v2 identifier EUCL-DR1-L002330 (split in v3)",
}


def _clean(value):
    return (value or "").replace("\r", "").strip()


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{_clean(k): _clean(v) for k, v in row.items()} for row in csv.DictReader(f)]


def modelled_tile(obj):
    """(batch, dataset tile) of the object's 2026-10-11 model, or ("", "")."""
    value = obj["fits_modelled_20261011"]
    if not value:
        return "", ""
    batch, _, name = value.partition(":")
    if batch not in BATCHES or not name:
        raise SystemExit(f"{obj['Unique_Identifier']}: unexpected fits_modelled_20261011 {value!r}")
    return batch, TILE_ALIASES.get(name, name)


def build(out_dir: Path, objects_csv: Path = OBJECT_CSV, dataset_dir: Path = DATASET_DIR):
    objects = _read(objects_csv)
    uids = [o["Unique_Identifier"] for o in objects]
    if len(set(uids)) != len(uids):
        raise SystemExit(f"{objects_csv}: repeated Unique_Identifier")
    stray = [o["Unique_Identifier"] for o in objects
             if o["fits_modelled_20261011"] and o["SWAP_Grade"] not in ("A", "B")]
    if stray:
        raise SystemExit(f"modelled objects that are not grade A/B in v5: {stray}")

    rows = []
    for obj in objects:
        if obj["SWAP_Grade"] not in ("A", "B"):
            continue
        uid = obj["Unique_Identifier"]
        batch, tile = modelled_tile(obj)
        if not tile:
            include, reason, note = False, "no_model", NO_MODEL.get(uid, "not in the 2026-10-11 model snapshot")
        elif uid in MODEL_UNUSABLE:
            include, reason, note = False, "model_unusable", MODEL_UNUSABLE[uid]
        else:
            include, reason, note = True, "included", RELABELLED.get(uid, "")
        rows.append(dict(tile=tile, unique_identifier=uid, grade=obj["SWAP_Grade"], score=obj["SWAP_Score"],
                         cutout_batch=batch, include_in_catalogue=include, reason=reason, note=note))
    listed = {r["unique_identifier"] for r in rows}
    for key in [*MODEL_UNUSABLE, *NO_MODEL, *RELABELLED]:
        if key not in listed:
            raise SystemExit(f"{key}: listed in this script but not a v5 A/B object")
    tiles = [r["tile"] for r in rows if r["tile"]]
    if len(set(tiles)) != len(tiles):
        raise SystemExit("two objects map to the same modelled tile")
    rows.sort(key=lambda r: (r["tile"] == "", r["tile"], r["unique_identifier"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "dr1_sep1_consolidated_manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, MANIFEST_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    by_uid = {o["Unique_Identifier"]: o for o in objects}
    uncovered = [r for r in rows if not r["include_in_catalogue"]]
    with open(out_dir / "dr1_sep1_uncovered_ab.csv", "w", newline="") as f:
        w = csv.DictWriter(f, UNCOVERED_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in sorted(uncovered, key=lambda r: r["unique_identifier"]):
            o = by_uid[r["unique_identifier"]]
            w.writerow(dict(unique_identifier=r["unique_identifier"], grade=r["grade"], score=r["score"],
                            RA_obj=o["RA_obj"], DEC_obj=o["DEC_obj"], id_str_for_cutout=o["id_str_for_cutout"],
                            tile=r["tile"], reason=r["reason"], note=r["note"]))

    included = [r for r in rows if r["include_in_catalogue"]]
    print(f"{manifest}: {len(rows)} v5 A/B objects {dict(collections.Counter(r['grade'] for r in rows))}")
    for key, n in sorted(collections.Counter((r["reason"], r["cutout_batch"]) for r in rows).items()):
        print(f"  {key[0]:15s} {key[1] or '-':6s} {n}")
    print(f"  included: {len(included)} {dict(collections.Counter(r['grade'] for r in included))}")
    print(f"  not included: {len(uncovered)} -> dr1_sep1_uncovered_ab.csv")
    if dataset_dir.is_dir():
        on_disk = {p.name for p in dataset_dir.iterdir()}
        absent = sorted(r["tile"] for r in included if r["tile"] not in on_disk)
        print(f"  {dataset_dir}: {len(absent)} included tiles not present"
              + (f" (e.g. {absent[0]})" if absent else ""))
    return rows, uncovered


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out_dir", default=str(GRADES_DIR), help="Output directory (default catalogue/grades).")
    parser.add_argument("--objects", default=str(OBJECT_CSV), help="v5 object catalogue with the cutout columns.")
    parser.add_argument("--dataset_dir", default=str(DATASET_DIR),
                        help="Tile directory checked for the included tiles (report only; skipped if absent).")
    args = parser.parse_args()
    build(Path(args.out_dir), Path(args.objects), Path(args.dataset_dir))


if __name__ == "__main__":
    main()
