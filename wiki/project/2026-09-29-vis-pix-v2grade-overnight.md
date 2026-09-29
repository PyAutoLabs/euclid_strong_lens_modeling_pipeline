---
date: 2026-09-29
title: vis_pix follow-up of the 5,000-lens v2-grade vis_lp batch submitted overnight
tags: [euclid, dr1, dr1_sep1_rest, vis_pix, v2grade, ral, submitted]
---

# vis_pix for the v2-grade batch (vis_lp 366938 / 367039 / 367141 / 367243 / 367244)

Every lens of the v2-grade vis_lp batch (manifests
`hpc/run_manifests/vis_lp_v2grade_20260928_part01..05.txt`, 1,000 each, task N = line N)
gets a `vis_pix` fit: Numba sparse CPU (`--use_cpu`), the same guard as the n5k arrays
(exactly one completed vis_lp zip in `output/`). None of the 5,000 tiles had a
`vis_pix` directory before submission, and none appears in any earlier vis_pix manifest.

## Jobs (submitted 2026-09-29 ~21:40)

Scripts `hpc/batch_cpu/submit_initial_lens_model_vis_pix_v2grade_p01..p05`, submitted by
`hpc/batch_cpu/submit_vis_pix_v2grade_overnight.sh` (run on RAL):

| part | vis_lp | immediate array (vis_lp COMPLETED) | dependent single-task jobs (afterok on vis_lp task N) |
|---|---|---|---|
| 01 | 366938 | 372104 (881 tasks, %60) | 119, ids 372105-372255 |
| 02 | 367039 | 372256 (908) | 92, ids 372257-372348 |
| 03 | 367141 | 372349 (814) | 186, ids 372350-372537 |
| 04 | 367243 | 372538 (885) | 115, ids 372539-372653 |
| 05 | 367244 | 372654 (844) | 156, ids 372655-372810 |

4,332 tasks eligible at once; 668 wait on `--dependency=afterok:<vis_lp>_N
--kill-on-invalid-dep=yes`, so each starts when its own vis_lp task succeeds (incl. the
~500 requeued after the 09:35 NODE_FAIL) and is cancelled if that task fails/times out.
All jobs are named `euclid_dr1_vis_pix_v2g_pNN`.

**aftercorr does not work here.** The first attempt (one `--array=0-999` per part with
`--dependency=aftercorr:<vis_lp>`, jobs 372037/38/59/70/71) stayed PENDING(Dependency)
for all 5,000 tasks, including tasks whose vis_lp completed after submission; slurmctld
(22.05.9) purges completed task records after MinJobAge=300 s. Cancelled (never started)
and replaced by the split above. A 2-task dummy test passes only because the parent
records are still in memory, so it does not prove aftercorr for long-running arrays.

## Resources

- 8 CPU / 16 GB / 24 h, `--partition=ral,gpu` (gpu nodes run CPU-only work without
  `--gres`; 19/28 ral nodes are admin-drained, the rest busy with vis_lp).
- n5k vis_pix (356390, 356447-356450; after their repack to 16 GB): MaxRSS p50 6.8 GB,
  p99 10.9 GB, max 11.5 GB over 2,004 steps; CPU efficiency ~66 % of 8.
- n5k elapsed (dense route, before the sparse fix): p50 1.2 h, p90 2-9 h, p99 16-17 h,
  47 TIMEOUTs at 18 h (rerun at 36 h as 359160-359163, all done in <= 2.8 h).
- Sparse fix clone `32fbf62` (sparse operator applied after `apply_over_sampling`,
  ~5.4x per evaluation): RAL `scripts/initial_lens_model.py` md5 `e71b7844...` equals
  clone HEAD, so these runs use InversionImagingSparseNumba. 24 h is headroom.

## Check / pull

    ssh euclid_jump 'squeue -u jnightin -r -o "%j %T %r" | grep vis_pix_v2g | sort | uniq -c'
    ssh euclid_jump 'sacct -u jnightin -S 2026-09-29T21:30 -X -n -o JobName%30,State | grep vis_pix_v2g | sort | uniq -c'
    hpc/sync pull output
