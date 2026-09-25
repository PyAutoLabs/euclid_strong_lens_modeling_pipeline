# 2026-09-25 — scale-up: next 5,000 vis_lp + vis_pix and SED follow-ups for the 250

Submitted ~21:45 BST after the human judged the 250 new-positions vis_lp results good.

| Array | Script | Tiles | Notes |
|-------|--------|-------|-------|
| 351085-351089 | `submit_initial_lens_model_vis_lp_next5000_p01..p05` | 5,000 | `hpc/run_manifests/vis_lp_next5000_20260925_part01..05.txt`: tiles 250-5249 of the tile-ID-ordered union of rest_01..15, minus 4 tiles with missing FITS / empty info.json. `--array=0-999%100`, `ral`, 8 CPU / 64 GB / 18 h. Nice=1000 so the 250 follow-ups outrank them. 109 have no positions.json. |
| 351090 | `submit_initial_lens_model_vis_pix_priority250` | 250 | `--dependency=afterany:350804`; Numba sparse CPU (`--use_cpu`), guards on exactly one vis_lp zip in `output/`. |
| 351091 | `submit_sersic_waveband_priority250` | 250 | `--dependency=afterany:351090`, `%60`, 6 h; seeds vis_lp from `output/` (not the retired dr1_full) into `output_sed/`; refuses the 10 tiles without DECam EXT. |

At 21:23 BST 350804 had 240 COMPLETED / 10 RUNNING (tasks 1, 20, 37, 53, 54, 81, 118, 184, 192, 211, all >2 h); 227 new zips pulled locally.
Pull: `hpc/sync pull output` (vis_lp + vis_pix) and `hpc/sync pull output_sed` (SED).

## 21:50 BST — packing fix on the pending 5,000

The 250 finished tasks peaked at 4.1 GB RSS (median 3.4) and used a median 53 %
of their 8 cores, yet each reserved 64 GB, so nodes held 14 tasks (112 of 252
CPUs). `scontrol update JobId=<array> MinMemoryNode=12288 NumCPUs=4 CPUsPerTask=4
ArrayTaskThrottle=400` on 351085-351089 (memory must be given in MB; "12G" is
rejected). 4,628 pending tasks now run at 4 CPU / 12 GB (63 per node, CPU-bound);
the 372 already running keep 8 CPU / 64 GB. Compare wall times of 4-core vs
8-core tasks in `sacct -j 3510{85..89} --format=JobID,AllocCPUS,Elapsed` before
choosing the default for the next batch.
