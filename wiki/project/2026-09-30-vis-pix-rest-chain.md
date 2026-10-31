---
date: 2026-09-30
title: vis_pix chained onto the rest vis_lp batch (3,779 tiles, one job per tile)
tags: [euclid, dr1, dr1_sep1_rest, vis_pix, rest, ral, submitted]
---

# vis_pix for the rest batch (after vis_lp 375700 / 375701 / 375702 / 375703)

Every tile of the rest vis_lp batch (manifests
`hpc/run_manifests/vis_lp_rest_20260930_part01..04.txt`, 945/945/945/944, task N =
line N; see `2026-09-30-vis-lp-rest-submitted.md`) now has a vis_pix job queued. Each one
starts automatically when its own vis_lp task succeeds. This uses the same model as the
v2-grade vis_pix (`2026-09-29-vis-pix-v2grade-overnight.md`): Numba sparse CPU
(`--use_cpu`), and the guard that requires exactly one completed vis_lp zip.

## Limits checked first (RAL, slurmctld 22.05.9)

- `MaxJobCount = 100000`, `MaxArraySize = 1001`, `MinJobAge = 300 s`, `MaxDBDMsgs = 200232`.
- `sacctmgr` lists no per-user association limits for jnightin. QOS `normal` has no
  MaxSubmitJobs, MaxJobs or Grp* limits. Partitions ral and gpu have `QoS=N/A`.
- Before submission there were 473 job records in the queue (10,435 tasks), 446 of them
  mine. Adding about 3,780 records leaves the total below 10 % of MaxJobCount, even after
  the pending vis_lp arrays split into one record per task.
- Decision: **all 3,779 were submitted now, not in waves.**

## Jobs (submitted 2026-09-30 19:58-20:05)

Scripts: `hpc/batch_cpu/submit_initial_lens_model_vis_pix_rest_p01..p04`. Jobs are named
`euclid_dr1_vis_pix_rest_pNN`. Resources are 8 CPU, 16 GB and 24 h on `--partition=ral,gpu`.
Driver: `hpc/batch_cpu/submit_vis_pix_rest_chain.sh`, run once on RAL. Its log is
`~/vis_pix_rest_chain_20260930_run1.log` on RAL.

| part | vis_lp | immediate array (vis_lp COMPLETED) | dependent single-task jobs (`afterok:<vis_lp>_N`) | skipped |
|---|---|---|---|---|
| 01 | 375700 | 376091 (tasks 0-1, %60) | 943, ids 376092-377037 | 0 |
| 02 | 375701 | none | 945, ids 377038-377987 | 0 |
| 03 | 375702 | none | 945, ids 377988-378936 | 0 |
| 04 | 375703 | none | 944, ids 378940-379889 | 0 |

The id ranges include a few other users' jobs, so the counts are the real numbers. Each
dependent job is a single-task array (`--array=N`), so `SLURM_ARRAY_TASK_ID` is the
manifest index. It carries `--dependency=afterok:<vis_lp>_N --kill-on-invalid-dep=yes`, so
it is cancelled if vis_lp task N fails or times out.

Checks right after submission:

- squeue showed 3,777 jobs `PENDING (Dependency)` and 2 `PENDING (Priority)`.
- None was in `DependencyNeverSatisfied`, and none was cancelled.
- `scontrol show job 376500` showed `Dependency=afterok:375700_410(unfulfilled)`,
  16G, 1-00:00:00, gpu,ral.
- A second `--dry-run` reported `already_have_vis_pix = 945 / 945 / 945 / 944` with
  nothing left to submit.

## The driver (idempotent, safe to re-run)

    ssh euclid_jump 'bash /mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu/submit_vis_pix_rest_chain.sh --dry-run'   # print only
    ssh euclid_jump 'bash /mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu/submit_vis_pix_rest_chain.sh'             # submit what is missing
    # optional part filter: ... submit_vis_pix_rest_chain.sh [--dry-run] 02 04

For each manifest index, the driver:

1. **Skips** the index if a vis_pix_rest job for it is queued, running or completed, or
   if the tile already has a vis_pix zip. It lists vis_pix jobs that ended FAILED,
   TIMEOUT, OUT_OF_MEMORY or NODE_FAIL, and does not resubmit them. A CANCELLED vis_pix
   job counts as absent.
2. Puts COMPLETED vis_lp tasks that have exactly one vis_lp zip into one immediate array.
3. Submits an afterok single for each vis_lp task that is still queued.
4. Lists and skips vis_lp tasks that ended FAILED, TIMEOUT or CANCELLED, and COMPLETED
   tasks that have no zip.

It reads SLURM state (squeue and sacct). It touches the filesystem only for the indices
that are candidates for (2), so a re-run never walks thousands of output directories.
Submitting all 3,777 singles took about 7 minutes, roughly 9 sbatch calls per second.

Race handling:

- **At submission.** A vis_lp task can finish and have its record purged (MinJobAge
  300 s) between the snapshot and the sbatch. sbatch then rejects the dependency, and the
  driver submits that index without one. The job's zip guard makes the decision.
- **Later.** A dependent killed while its vis_lp is COMPLETED shows up as CANCELLED. The
  next re-run submits it again as an immediate task.

**When to re-run it:** any time, but it is worth doing once all vis_lp is done. That
catches race casualties, and vis_lp tasks that were requeued after a NODE_FAIL under a
new record.

## Lessons from the v2-grade vis_pix chain (372104-372810, sacct 2026-09-30 ~19:55)

- Of the 668 dependent singles, 481 had COMPLETED, 19 were RUNNING, 152 PENDING,
  18 CANCELLED and 2 FAILED.
- **All 18 CANCELLED were correct.** Every one had its vis_lp end in TIMEOUT. No dependent
  was killed while its vis_lp completed, so there were no race casualties.
- **7 vis_pix failures (5 array tasks and 2 singles) happened because vis_lp ended
  COMPLETED with no zip.** Example: 367141_365 took 18 s. The python died with
  `h5py OSError: bad object header version number`, most likely while resuming a corrupt
  file after the NODE_FAIL requeue. The vis_lp script still exits 0 because its last
  command is `echo`.
  - The vis_pix zip guard caught all seven within 1-2 s.
  - The rest driver checks the zip before putting an index in the immediate array.
  - COMPLETED is not proof of a vis_lp fit.
- The 3,167 completed v2g vis_pix batch steps (sparse route) used:
  - Elapsed: p50 1.28 h, p90 1.88 h, p99 2.51 h, max 18.2 h. Three ran over 12 h.
  - MaxRSS: p50 6.5 GB, p99 6.8 GB.
  - A few MaxRSS readings are over 16 GB, with a maximum of 175 GB. These are artefacts
    of summing across the forked pool, because the tasks completed under a 16 GB cgroup.
  - 16 GB and 24 h were kept.

## Check / pull

    ssh euclid_jump 'squeue -u jnightin -r -h -o "%j %T %r" | grep vis_pix_rest | sort | uniq -c'
    ssh euclid_jump 'sacct -u jnightin -S 2026-09-30T19:55 -X -n -P -o JobName,State | grep vis_pix_rest | cut -d"|" -f2 | cut -d" " -f1 | sort | uniq -c'
    hpc/sync pull output
