---
date: 2026-09-30
title: Multi-wavelength SED chain submitted for the v2-grade vis_lp batch
tags: [euclid, dr1, dr1_sep1_rest, sed, sersic_waveband, v2grade, ral, submitted]
---

# SED chain for the v2-grade batch (vis_lp 366938 / 367039 / 367141 / 367243 / 367244)

Every lens of the v2-grade vis_lp batch whose vis_lp task had COMPLETED gets the
multi-wavelength SED chain (`scripts/sersic_lens_model_waveband.py`: VIS Sersic plus one
Sersic fit per NIR Y/J/H and DECam g/r/i/z band, JAX on CPU, `PYAUTO_OUTPUT_DIR=output_sed`).
The vis_lp seed zip is copied from `output/` into `output_sed/`, so the chain cache-skips
vis_lp. Tiles without DECam EXT imaging fit VIS + NIR only. There is no dependency on vis_pix.

## Manifests

`hpc/run_manifests/sed_v2grade_20260930_partNN.txt` holds the tiles of
`vis_lp_v2grade_20260928_partNN.txt`, in their original order, whose vis_lp array task
was COMPLETED in `sacct` at ~13:45 on 2026-09-30. The local and RAL vis_lp manifests have
the same md5. There was no overlap to exclude: none of the 4,982 tiles has an
`output_sed/dr1_sep1_rest/<tile>/` directory, and none is in an earlier SED manifest
(`vis_pix_sed_next5000_20260926_part*`, `sed_rest_full_20260923*`,
`positions_priority_250`, `sersic_waveband_first100_ext_20260921`, or the ten-tile list).

Left out (vis_lp not COMPLETED), 18 tiles:

| part | TIMEOUT (vis_lp task) | still RUNNING (vis_lp task) |
|---|---|---|
| 01 | 845 | 322 |
| 02 | 559, 579 | - |
| 03 | 536, 883 | 112, 279, 396 |
| 04 | 675, 729, 734, 774 | 34 |
| 05 | 502, 697 | 18, 182 |

When the 7 RUNNING tiles finish, they can go into a small follow-up SED array.

## Jobs (submitted 2026-09-30 ~13:55)

Script `hpc/batch_cpu/submit_sersic_waveband_v2grade`, run with `PART` set to the part and
submitted with `sbatch` from `hpc/batch_cpu` on RAL. Every job is named `euclid_dr1_sed_v2g`.

| part | vis_lp | SED array | tasks |
|---|---|---|---|
| 01 | 366938 | 374848 | 998 (`--array=0-997%40`) |
| 02 | 367039 | 374849 | 998 (`0-997%40`) |
| 03 | 367141 | 374850 | 995 (`0-994%40`) |
| 04 | 367243 | 374851 | 995 (`0-994%40`) |
| 05 | 367244 | 374852 | 996 (`0-995%40`) |

That is 4,982 tasks in total, with at most 200 running at once. The cap stops the SED
tasks starving the ~2,650 v2g vis_pix tasks that are still pending.

## Resources and the 3 h wall

- 8 CPU / 16 GB / 3 h, `--partition=ral,gpu`. The gpu nodes run CPU-only work without
  `--gres`. MaxRSS in the n5k SED run was ~3 GB.
- n5k SED arrays (356451-356453, 356555, 356556): completed tasks took p50 0.24 h, p90
  0.65 h, p99 2.66 h and max 9.4 h. However, 908 of 4,975 hit TIMEOUT at 6 h. These tasks
  hang in one band's Nautilus search: there is no progress after "Running search with
  JAX vectorization". 45 of 46 re-runs at 18 h timed out again, so a longer wall only
  burns node-hours.
- Expect ~18 % TIMEOUT. Those tiles are left for a later fix of the hang and a rerun.
- RAL `scripts/sersic_lens_model_waveband.py` md5 `9fdee341...` equals the local HEAD.

## Check / pull

    ssh euclid_jump 'squeue -u jnightin -r -h -o "%j %T" | grep sed_v2g | sort | uniq -c'
    ssh euclid_jump 'sacct -j 374848,374849,374850,374851,374852 -X -n -o State | sort | uniq -c'
    ssh euclid_jump 'sacct -j 374848 -X -n -P -o JobID,State,Elapsed,MaxRSS | grep TIMEOUT'
    hpc/sync pull output_sed
