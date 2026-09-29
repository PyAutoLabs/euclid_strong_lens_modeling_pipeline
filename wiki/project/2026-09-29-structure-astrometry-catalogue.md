---
date: 2026-09-29
title: Einstein radius, lens Sersic n/Reff and per-band astrometric offsets added to the dr1_sep1_rest magnitudes bundle
tags: [euclid, dr1, dr1_sep1_rest, lens_mass, lens_sersic, astrometric_offsets, catalogue, ral, ran]
---

# Structure + astrometry tables for the 5,199-lens magnitudes bundle

Added `lens_mass.csv`, `lens_sersic.csv` and `astrometric_offsets.csv` (master +
per-lens copies) to `inspect/dr1_sep1_rest_magnitudes/` (the job-366738 bundle),
plus a flat `dr1_structure_astrometry_summary.csv` + `README_structure_astrometry.txt`.

## Jobs (RAL, `--partition=gpu`, CPU-only; `ral` had 19/28 nodes drained)

Script `hpc/batch_cpu/submit_structure_astrometry` (commit 201166f), one job per STAGE,
each producer writing to `inspect/dr1_sep1_rest_structure_astrometry_staging/<stage>/`
(unrestricted masters kept there) and then merged into the bundle restricted to its
5,199 lens folders. `magnitudes.csv` / `sed_completeness.csv` / `sed_missing_lenses.txt`
checksums unchanged afterwards.

    for s in lens_mass lens_sersic astrometric_offsets; do
      hpc/sync submit cpu submit_structure_astrometry \
        --export=ALL,STAGE=$s,ASTRO_SCRIPT=catalogue/scripts/astrometric_offsets_head.py \
        -J dr1_struct_$s --partition=gpu
    done

| job | stage | reads | wall | master rows (all) | kept in bundle |
|---|---|---|---|---|---|
| 371299 | lens_mass | `output/` initial_lens_model/vis_pix | 1:01:20 | 5,200 | 5,184 lenses |
| 371300 | lens_sersic | `output_sed/` sersic_lens_model/vis | 2:07:33 | 5,201 | 5,199 lenses |
| 371301 | astrometric_offsets | `output_sed/` sersic_lens_model/<band> | 2:13:14 | 31,312 (5,008 lenses) | 31,312 |

`sersic_lens_model/vis` exists only in `output_sed/` for this sample (0 in `output/`).

**Astrometric producer revision.** The local working tree's
`catalogue/scripts/astrometric_offsets.py` is an uncommitted edit that removes the #88
`prior_edge_y/x` columns (and says the prior is +/-0.2" while
`scripts/lens_model_waveband.py` uses 0.5"); RAL held that edited copy. The run used the
committed HEAD version, copied to RAL as `catalogue/scripts/astrometric_offsets_head.py`.
Neither the working-tree file nor the RAL `astrometric_offsets.py` was changed.

## Coverage (of 5,199 folders)

- lens_mass 5,184: the 15 missing have a `vis_pix` result directory that never completed.
  13 lenses have `effective_einstein_radius = 0`.
- lens_sersic 5,199. 571 with n > 9.5, 42 with n < 0.55, 186 with Reff > 10".
- astrometric_offsets 5,008 lenses / 31,312 rows = exactly the non-VIS rows of
  `magnitudes.csv`; the 191 missing lenses have only VIS completed in the SED chain.
  1,035 rows prior-edge flagged. No duplicate lens or lens x band rows; no empty/NaN cells.

Local `inspect/dr1_sep1_rest_magnitudes.zip` rebuilt (flat top-level files only, as before,
now also including the two `*_graded.csv`); previous zip kept as `.zip.bak_20260929`.
