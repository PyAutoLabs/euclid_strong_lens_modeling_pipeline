---
date: 2026-09-30
title: vis_lp submitted for every remaining dr1_sep1_rest lens (3,779, v2 B + C + X)
tags: [euclid, dr1, dr1_sep1_rest, vis_lp, rest, ral, submitted]
---

# vis_lp for the rest of `dr1_sep1_rest` (375700-375703)

After this batch, every usable tile of `dataset/dr1_sep1_rest/` has a vis_lp fit queued
or done.

## Selection

- `dataset/dr1_sep1_rest/` has 14,032 `Tile*` dirs. 10,250 already have vis_lp from
  `hpc/run_manifests/vis_lp_next5000_20260925.txt` (5,000),
  `vis_lp_v2grade_20260928.txt` (5,000) and `positions_priority_250.txt` (250). These
  three do not overlap. The 09-20 rest_01..05 arrays wrote to the retired dr1_full root,
  and next5000 covers all of their tiles.
- That leaves 3,782 tiles. Their v2 grades in `catalogue/grades/tile_grades_dr1_sep1_v2.csv`
  are B 3,204, C 577 and X 1.
- **3 excluded**, all v2 B. Their `info.json` is 0 bytes, locally and on RAL:
  `Tile102013495RA0156979516303DECNEG0570303278775`,
  `Tile102013503RA0227709362222DECNEG0571284574956` and
  `Tile102013521RA0387803645313DECNEG0568111157503`.
- **Kept:** `Tile102012389RA0685761206842DECNEG0584642793881` (B, 2.040). next5000
  dropped it because the local tree has no FITS or info.json. RAL has a complete tile:
  an 872,640 B FITS and a valid info.json. It is task 0 of part01.
- 126 tiles have no `positions.json` (finder status `n_lt_2`). They are kept and fit with
  no positions penalty, as in next5000 and v2grade (those batches ran 109 and 69 such
  tiles).
- RAL check, from one python call on the login node:
  - None of the 3,782 has an `output/dr1_sep1_rest/<tile>/` directory.
  - Every candidate's FITS, info.json and positions.json has the same size locally and
    on RAL, apart from Tile102012389 above.
  - FITS md5 matches on a 1-in-50 sample (75/75).
- Order: `v2_score` descending, ties broken by tile name. The list is
  `hpc/run_manifests/vis_lp_rest_20260930.txt` (3,779). It is split contiguously into
  `..._part01..04.txt`, so part01 holds the best-scored tiles.

| part | array | tasks | grades | v2_score |
|---|---|---|---|---|
| 01 | 375700 | 945 (`0-944%100`) | B 945 | 2.040 -> 1.983 |
| 02 | 375701 | 945 (`0-944%100`) | B 945 | 1.983 -> 1.899 |
| 03 | 375702 | 945 (`0-944%100`) | B 945 | 1.898 -> 1.649 |
| 04 | 375703 | 944 (`0-943%100`) | B 366, C 577, X 1 | 1.648 -> 0.247 |

## Jobs (submitted 2026-09-30 ~18:00)

The scripts are `hpc/batch_cpu/submit_initial_lens_model_vis_lp_rest_p01..p04`, generated
from `..._vis_lp_v2grade_p05`. They were submitted with `sbatch` from `hpc/batch_cpu` on
RAL, the same as `hpc/sync submit cpu`, and the jobs are named
`euclid_dr1_vis_lp_rest_pNN`. They use the same settings as v2grade: 4 CPU, 12 GB, 18 h,
finder-2.1 positions and `%100` per part.

The partition is **`ral,gpu`**. The v2grade vis_lp used `ral` alone. Now 18 of the 28 ral
nodes are drained, and the vis_pix_v2g and SED_v2g arrays already use `ral,gpu`; gpu nodes
run CPU-only work without `--gres`. The first task started on a gpu node within seconds.

At submission, about 1,940 vis_pix_v2g tasks and about 4,800 SED_v2g tasks were PENDING
on the same partitions. Most of this batch will start after those earlier jobs.

The v2grade vis_lp outcome backs these settings. It covers 366938, 367039, 367141,
367243 and 367244, from sacct over all 5,000 tasks:

- 4,982 COMPLETED and 18 TIMEOUT (0.36 %).
- Completed elapsed: p50 2.06 h, p90 3.63 h, p99 5.35 h, max 13.1 h.
- MaxRSS: p50 3.1 GB, p99 3.8 GB, max 4.07 GB. The 12 GB limit has plenty of headroom
  and could drop to 8 GB, but it was left as it was.

md5 check: the four scripts and five manifests on RAL match local.
`scripts/initial_lens_model.py` (`e71b7844...`, which includes the sparse fix 32fbf62),
`util.py`, `activate.sh` and the `config/` yamls on RAL all match local HEAD.

## Check / pull

    ssh euclid_jump 'squeue -u jnightin -r -h -o "%j %T %r" | grep vis_lp_rest | sort | uniq -c'
    ssh euclid_jump 'sacct -j 375700,375701,375702,375703 -X -n -o JobName%30,State | sort | uniq -c'
    hpc/sync pull output

## Next

The next step is the follow-up for this batch: vis_pix (Numba sparse CPU, guarded on
exactly one vis_lp zip), then the SED chain (`submit_sersic_waveband_*`, seeded from
`output/`). Build both from the COMPLETED tasks of 375700-375703, as was done for v2grade.
Do not use `aftercorr`, which does not work here; see
`2026-09-29-vis-pix-v2grade-overnight.md`.
