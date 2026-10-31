---
date: 2026-10-07
title: All DR1 lenses consolidated into dr1_sep1; ex-top1000 vis_lp and SED gap fill queued
tags: [euclid, dr1, dr1_sep1, consolidation, vis_lp, sed, ral, submitted]
---

# Consolidation into `dr1_sep1` (2026-10-07 evening)

## What was done
- **Code sync first:** local commit `b1fdce6` holds the RAL-only fixes: `merge_catalogue.py` catches
  `RuntimeError`, the chunked build gains `DEPENDENCY=`, plus `catalogue_post_build_check.py` and the 10-05
  SED-fill script. RAL-only scripts and manifests were rsynced down (`--ignore-existing`).
- **Folder move:** run by `hpc/tools/consolidate_dr1_sep1.py --execute --yes` on the RAL login node,
  launched by the human about 19:00, under nohup. It works from a reversible manifest:
  `hpc/run_manifests/consolidate_dr1_sep1_20261007/moves.tsv` (60,444 ops), logged to `moves.log` and
  `execute.out`. The dry run passed with 0 errors; `--revert --yes` undoes it.
  - It unlinks 15,022 symlinks: 1,000 of them dangled to the retired `dr1_sep1_top1000/`.
  - It archives the 10 stale 09-11 real dirs to `archive_dr1_sep1_ten_20260911/`.
  - It moves 14,032 rest + 987 v2_new dataset tiles, and the 15,015 `output/` + 14,345 `output_sed/` tile dirs.
  - It copies the 1,000 ex-top1000 tiles from `archive_top1000_20260925/`, which stays untouched.
  - Side files go to `*/provenance_dr1_sep1/<old_sample>/`.
  - Expected post-state: dataset 16,019, output 15,015, output_sed 14,345 tile dirs.
- **The top1000 tiles needed no reprocessing:** their archived copies already carry the 09-24 flip,
  artefact fix and segmentation redraw, and the 09-25 finder-2.1 positions (job 350761). Only their
  09-18 fits (343480) used old positions, and those stay archived.
- **Sample names retired:** `dr1_sep1_rest`, `dr1_v2_new` and `dr1_sep1_top1000` are no longer sample
  names. They remain only as `origin_sample` in the manifest and in history files.

## Consolidated manifest
`catalogue/grades/dr1_sep1_consolidated_manifest.csv` is produced by `scripts/tools/consolidated_manifest.py`.

| reason | n |
|---|---:|
| rows | 16,019 |
| included A+B objects, one tile each | 15,067 |
| `duplicate_tile` | 45 |
| `grade_c_x` | 907 |

- **Duplicates rule:** keep the tile whose name exactly equals the object's `id_str_for_cutout`. In all
  45 pairs that is the recentred `15arcsec_near_RADEC_*` v2_new cutout, so 34 objects double-listed in
  the released `dr1_catalogue_20261005` get resolved in v1.1.
- **Uncovered A+B** (`dr1_sep1_uncovered_ab.csv`):
  - L009168 and L019278: no FITS made; left unmodelled by agreement.
  - L020583: its tile is not in the sep1 segmentation delivery. Open: get its files, or leave it.
- **Pending v3 catalogue changes:**
  - L002330 is relabelled to L026834.
  - L026833 is not modelled.
  - L002411 has blank FITS and is not rerun.
  - A+B becomes 15,071.

## Jobs (submitted about 21:00)

| job | what | gating |
|---|---|---|
| 397466 | `submit_gate_ex_top1000_copies`: waits for the 997 copies, then checks finder 2.1 | — |
| 397467 | `submit_initial_lens_model_vis_lp_ex_top1000`: 997 tiles (`vis_lp_ex_top1000_tonight_20261007.txt`), `ral`, 4 CPU / 12 GB / 18 h, `%100` | afterok 397466 |
| 397468 | `submit_gate_consolidation_done`: waits for the mover END line, checks old dirs gone and output_sed = 14,345 | — |
| 397469-397474 | `submit_sersic_waveband_sed_fill_20261005`, `SAMPLE=dr1_sep1`, the 10-05 lists (rest partial part01-03, rest none, v2_new partial, v2_new none = 3,276 tiles), 40 at a time in total | afterok 397468 |

At 21:34 the copies were ETA about 22:40, so vis_lp would start then; the mover was ETA about 02:00, so
SED would start then.

## Next (where to pick up)
1. Check `execute.out`, which ends `execute: N done`, and the gate logs; verify the post-state counts.
2. **Moved-fit identity:** grep the first SED task logs for `Fit Already Completed` on the seeded
   vis_lp. If vis_lp is being refit, the identifier is sample-dependent: cancel the SED arrays and
   rethink.
3. Swap in the sep28 copies, then fit these 4 tiles (they were held out of 397467):
   - L023428 `Tile102023006…`, which is staged locally at `dataset/staging_sep28_L023428/`;
   - the 3 top1000 same-name tiles `Tile102156625…`, `Tile102043581…` and `Tile102029287…`.

   Archive L023428's old outputs first.
4. **Without-model reruns** from `for_james_AB_without_model.csv`:
   - 28 top1000 are covered by 397467;
   - 86 rest and 12 v2_new need a sacct/log diagnosis, then a rerun plan for the human (12 need vis_pix only);
   - use the sep28 copy wherever a name is shared.
5. **v1.1:** vis_pix and SED for the ex-top1000; apply the v3 relabel; wire `merge_catalogue.py` and
   the chunked build to read the consolidated manifest and `SAMPLES=dr1_sep1`; rebuild.
