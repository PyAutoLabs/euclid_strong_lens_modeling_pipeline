# DR1 catalogue membership (lens-finding catalogue v5)

The membership manifest of the DR1 lens-model catalogue, built from the final
lens-finding catalogue **v5 (Holloway, 19 Oct 2026)**:
`../DR1_full_Object_Catalogue_v5_fits.csv` (all 26,662 objects, with the
modelling-cutout columns added by N. Lines; see its `_README.txt` and
`../README_for_James_2026-10-19.txt`). Built 2026-10-19.

No object was remodelled for v5. The catalogue uses the existing fits only; the
13 v5 cutouts were not ingested into `dataset/`.

## Files

| File | What |
|------|------|
| `dr1_sep1_consolidated_manifest.csv` | One row per v5 Grade A/B object (15,072): `tile, unique_identifier, grade, score, cutout_batch, include_in_catalogue, reason, note` |
| `dr1_sep1_uncovered_ab.csv` | The 8 A/B objects with `include_in_catalogue=False`, with position, reason and note; the CSV snapshot ships it as `excluded_eight_candidates.csv` |

`grade` / `score` are the v5 `SWAP_Grade` / `SWAP_Score`; `unique_identifier`
is the v5 `Unique_Identifier`. `tile` is the `dataset/dr1_sep1` tile the model
was fitted on, taken from v5 `fits_modelled_20261011` (`batch:filename`; the
sep1 and sep28 batches both live in `dataset/dr1_sep1`). Where an object has a
sep1 and a recentred sep28 tile (45 objects) the modelled one is sep28, as in
the earlier v2 manifest. One file name is aliased: L020528's modelled
`15arcsec_near_RADEC_Tile102161607...` is held as `Tile102161607...`.

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
  in the archive), L020583 (not in the segmentation delivery).
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
