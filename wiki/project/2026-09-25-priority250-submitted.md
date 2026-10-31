# 2026-09-25 — first-250 vis_lp submitted on the new positions; top-1000 retired

- **Array `350804`** (`hpc/batch_cpu/submit_initial_lens_model_vis_lp_priority250`,
  partition `ral`, `--array=0-249`, 8 CPUs, 64 GB, 18 h): `vis_lp` only on
  `dataset/dr1_sep1_rest/`, tiles read from
  `hpc/run_manifests/positions_priority_250.txt` (md5 `40a8eceb…`; task N =
  line N; tile-ID order). All 250 tasks started at 19:21 BST on the new
  finder-2.1 positions (181 keep, 51 n_lt_2 / no penalty, 9 drop, 6 flag,
  3 review; 3 tiles have no `positions.json`). Outputs: `output/dr1_sep1_rest/`.
- **Probe array `350664`** (indices 0-9 of `rest_01`, old positions): `_6`
  had been resumed on 09-25 and its `output/` folder was deleted at 15:56 while
  it ran; it died on the missing checkpoint at 16:11 (sacct says COMPLETED).
  Nothing was left to cancel or delete; `output/dr1_sep1_rest/` was empty at
  submission.
- **Top-1000 retired** (move, not delete): `dataset/dr1_sep1_top1000` (1.5 GB,
  positions rewritten by 350761) and `output_old/dr1_sep1_top1000` (array
  343480: 971 COMPLETED, 29 TIMEOUT, old positions) now live under
  `/mnt/ral/jnightin/euclid_dr1/archive_top1000_20260925/`. The local
  `submit_*_top1000` scripts and the top-1000 CSV are untouched.
- Next: `hpc/sync jobs`, then `hpc/sync pull output` once 350804 drains
  (`sacct -j 350804 --format=State -X -n | sort | uniq -c`).
