---
date: 2026-10-15
title: Moved to active/, remaining-runs audit, reruns submitted, three-tier release packager, frozen venv, clean-up
tags: [euclid, dr1, dr1_sep1, catalogue, release, ral, cleanup, submitted]
---

# 2026-10-15 evening

## Moves
- Local project is now `Science/active/euclid_dr1`; `hpc/sync.conf` has `HPC_BASE=/mnt/ral/jnightin/active`.
- RAL project is `/mnt/ral/jnightin/active/euclid_dr1`; the old path `/mnt/ral/jnightin/euclid_dr1` is a
  symlink to it, kept because about 80 batch scripts and the run manifests hardcode it.
- Mind policy page `policy/hpc_ral.md` records this (commit 443f268c).

## Remaining-runs audit (`inspect/remaining_runs_audit_20261015/`, job 409950)
Over the 15,067 included tiles of `catalogue/grades/dr1_sep1_consolidated_manifest.csv`:
`vis_stages.csv` (vis_lp / vis_pix / sersic_vis completed flags) and `sed_completeness.csv`.

| stage | complete | missing |
|---|---:|---:|
| vis_lp | 14,962 | 105 |
| vis_pix | 14,936 | 131 |
| sersic_vis | 14,919 | 148 |
| SED all bands | 14,551 | 368 partial + 148 none |

The 105 without any VIS fit all have a dataset FITS; 98 had a stale unfinished vis_lp directory (25 Sep to
1 Oct), 7 never ran, 3 lack positions. Grades: 36 A, 69 B.

## Reruns submitted (all `ral`, 23:35)
The 98 stale vis_lp directories were moved to `<tile>/initial_lens_model/vis_lp_stale_20261015/` first.

| job | script / manifest | tiles | gating |
|---|---|---:|---|
| 410758 | `submit_initial_lens_model_vis_lp_rerun_20261015` / `vis_lp_rerun_20261015.txt` | 105 (%105) | — |
| 410759 | `submit_initial_lens_model_vis_pix_rerun_20261015` / `vis_pix_rerun_20261015.txt` | 131 (%40) | afterany 410758 |
| 410760 | `submit_sersic_waveband_sed_fill_20261010` with `MANIFEST=sed_rerun_20261015.txt` | 516 (%40) | afterany 410758 |

Job ids also in `hpc/run_manifests/rerun_20261015_jobs.txt`. The SED script skips tiles already complete
and tiles whose VIS band was given up (`sed_fill_20261009/given_up_bands.txt`), so the singular-matrix
bands are not chased again unless that list is edited.

## Release packager (uncommitted, tests pass: 46)
One merged build, three tiers: `scripts/tools/csv_snapshot/` (the 10-11 CSV collector, parameterised),
`scripts/tools/package_release.py` (csv / vis / full), `scripts/tools/release_util.py`,
`hpc/batch_cpu/submit_release_package`, `MANIFEST=` for `submit_catalogue_chunked.sh`, tests, and the
"Release tiers" section of `catalogue/README.md`, which has the RAL command chain. Decision: `model.fits`
and `pre_psf.fits` ship whole (all bands) in every tier. Not yet run at scale on RAL: do a `DRY=1` and a
small test catalogue first.

## Frozen venv
`/mnt/ral/jnightin/scratch/make_dr1_venv.sh` (nohup, log `make_dr1_venv.log`, ends `DR1_VENV_DONE`) builds
`/mnt/ral/jnightin/PyAuto_dr1`: pinned clones of the 27 Sep library SHAs (`PINNED_SHAS.txt`) plus a fresh
venv from the frozen requirements. Afterwards switch `activate.sh` to `PYAUTO_HPC_BASE=/mnt/ral/jnightin/PyAuto_dr1`.

## Clean-up (moves only)
- RAL: 34 entries moved to `/mnt/ral/jnightin/archive/euclid_dr1_interim_2026-10/` (stale `output_*` trees,
  `archive_pre_rgbseg_20260924`, test and interim `inspect/` flavours); `archive_top1000_20260925` stays.
- Local: `Science/archive/euclid_dr1_interim_2026-10/` receives the same interim `inspect/` flavours, the
  delivered `dr1_remaining_images_20261005.tar`, the 09-24 artefact-fix log set, `hpc/upload_rest`, the
  September SLURM logs, `scratchpad/`, and `dataset/dr1_sep1_rest` + `dr1_v2_new` are consolidated into
  `dataset/dr1_sep1` (log `moves.log`). The 102 GB v1.0 dist stays until v1.1 is verified.

## Next
1. Morning: `squeue`; check 410758 logs for the 105; then v1.1 chain per `catalogue/README.md` after a dry run.
2. Review and commit the packager + manifests + this record (main is never pushed).
3. Switch `activate.sh` to `PyAuto_dr1` once `DR1_VENV_DONE`.
4. Open from 10-07: sep28 swap for the 4 held tiles, v3 relabel L002330→L026834, L020583.
