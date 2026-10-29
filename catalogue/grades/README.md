# DR1 catalogue membership (lens-finding catalogue v5)

The membership manifest of the DR1 lens-model catalogue, built from the final
lens-finding catalogue **v5 (Holloway, 19 Oct 2026)**:
`../DR1_full_Object_Catalogue_v5_fits.csv` (all 26,662 objects, with the
modelling-cutout columns added by N. Lines; see its `_README.txt` and
`../README_for_James_2026-10-19.txt`). Built 2026-10-19.

The 13 v5 cutouts (`fits_to_use` batch `v5`) are to be modelled as tiles
`dataset/dr1_sep1/<stem>_v5`; as of 2026-10-19 evening they are **not ingested**:
the FITS came without `info.json` / `segmentation/` and no run is submitted (see
`wiki/project/2026-10-19-v5-cutouts-blocked-on-segmentation.md`). Nothing else is
remodelled.

## Files

| File | What |
|------|------|
| `dr1_sep1_consolidated_manifest.csv` | One row per v5 Grade A/B object (15,072): `tile, unique_identifier, grade, score, cutout_batch, include_in_catalogue, reason, note, v5_tile, v5_status` |
| `dr1_sep1_uncovered_ab.csv` | The 8 A/B objects with `include_in_catalogue=False`, with position, reason and note; the CSV snapshot ships it as `excluded_eight_candidates.csv` |

`grade` / `score` are the v5 `SWAP_Grade` / `SWAP_Score`; `unique_identifier`
is the v5 `Unique_Identifier`. `tile` is the `dataset/dr1_sep1` tile the model
was fitted on, taken from v5 `fits_modelled_20261011` (`batch:filename`; the
sep1 and sep28 batches both live in `dataset/dr1_sep1`). Where an object has a
sep1 and a recentred sep28 tile (45 objects) the modelled one is sep28, as in
the earlier v2 manifest. One file name is aliased: L020528's modelled
`15arcsec_near_RADEC_Tile102161607...` is held as `Tile102161607...`.

## v5 cutouts: the switch happens at rebuild time

`v5_tile` is the object's v5 tile and `v5_status` says which tile the row uses.
`consolidated_manifest.py` decides it from disk each time it runs, so the switch
is simply re-running it (on RAL, or locally after pulling `output/`) before the
catalogue rebuild:

| objects | before their v5 run finishes | after |
|---|---|---|
| L000144, L001542, L018877, L019987, L019994, L020612 (recentred, existing model stands; `V5_KEEP_UNTIL_DONE`) | old sep1/sep28 tile stays `included`, `v5_status=pending`, note "v5 run pending" | switches to the v5 tile once `output/dr1_sep1/<v5 tile>/initial_lens_model/vis_pix/*.zip` exists (`v5_status=v5`, `cutout_batch=v5`) |
| L026833, L026835, L009168, L019278, L002411, L001796, L020505 (no usable model) | `no_model` / `model_unusable`, `v5_status=not_ingested` while the tile is not in `dataset/dr1_sep1` | `included` on the v5 tile as soon as the tile is ingested (`v5_status=v5`); an unfinished fit is then flagged by the build's completion columns like any other |

A failed v5 run of one of the six never loses its old model: without a v5
`vis_pix` zip the row stays on the old tile. `--dataset_dir` / `--output_dir`
point the check elsewhere.

## Counts (2026-10-19)

| | A | B | total |
|--|--:|--:|--:|
| v5 A/B objects | 6,073 | 8,999 | 15,072 |
| `included` (sep1 14,077, sep28 987) | 6,071 | 8,993 | 15,064 |
| `model_unusable` | 1 | 2 | 3 |
| `no_model` | 1 | 4 | 5 |

- `model_unusable`: L001796 (the model is of the star the old cutout was
  centred on), L020505 (fitted centre 1.6" from the lens), L002411 (all-zero
  VIS in the sep1 cutout).
- `no_model`: L026833, L026835 (new in v3 / v5), L009168, L019278 (MER tile not
  in the archive), L020583 (not in the segmentation delivery; no v5 cutout, stays
  flagged: it needs the same segmentation products as the v5 tiles).
- `v5_status` today: 6 `pending`, 7 `not_ingested`.
- L026834 is `included`: its model was filed under the v2 identifier L002330.

`included` means a model exists, not that every stage finished. Fits that never
finished (133 objects without a `vis_pix` result in the 2026-10-11 snapshot) are
flagged by the build: the CSV tables' completion columns (`vis_pix_completed`
and the rest) and, for the vis/full tiers, `EXCLUDED_FILE` / `excluded_tiles.csv`
with the reason.

Against the v2 manifest (16,019 rows, one per dataset tile): the 907 grade C/X
tiles and 45 duplicate tiles are no longer listed, the 5 never-modelled objects
are, and the included set lost L001796, L002411 and L020505 (15,067 -> 15,064).

## Regenerate

```bash
python scripts/tools/consolidated_manifest.py
# join the grade onto any per-lens CSV (lens_name or tile column) -> *_graded.csv
python scripts/tools/add_grade_column.py inspect/<catalogue>/magnitudes.csv
```

Nothing reads the v2 inputs any more. The v2 matcher
`scripts/tools/grade_manifest.py`, `DR1_full_Object_Catalogue_v2.csv`, the v2
cutout catalogue, the v1 <-> v2 bridge, `tile_grades_dr1_sep1_v2.csv` and the
matcher's inputs were moved on 2026-10-19 to `Science/archive/euclid_dr1_tmp/`
(see its `MOVES.md`).

## Why output/ and output_sed/ stay flat

Results are not split into per-grade directories and must not be moved or
symlinked into them. The `path_prefix` `<sample>/<tile>` is load-bearing: resume
finds a finished search by it, `vis_lp` -> `vis_pix` (and the SED chain) locate
the upstream result by it, the submit scripts build it from the manifest line,
and the aggregator does not follow symlinks. Select by grade with manifests
instead: pass one as the SLURM array list, or as `--manifest` to
`scripts/tools/build_image_catalogue.py` and `scripts/tools/sed_completeness.py`,
and use `add_grade_column.py` on the resulting CSVs.
