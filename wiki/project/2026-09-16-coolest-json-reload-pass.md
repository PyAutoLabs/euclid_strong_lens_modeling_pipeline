---
date: 2026-09-16
title: coolest.json reload pass prepared, then run — the nine completed lenses now carry files/coolest.json
tags: [euclid, dr1, dr1_sep1, initial_lens_model, coolest, reload, force_pickle_overwrite, ral, ran]
---

# coolest.json reload pass prepared for the nine completed lenses — and RAN 2026-09-17

**Status: RAN 2026-09-17** (SLURM array **`343367`**, nine tasks, all `COMPLETED` in 29-35 s).
Both preconditions were satisfied first; the flag is back to `false` on RAL and in this clone,
and `config/general.yaml` is byte-identical to its pre-pass state. See **Result** at the foot
of this page. The sections above are the preparation as it stood before the run.

## Context

The pipeline branch `feature/coolest-observation-grid` (PyAutoLabs/PyAutoLens#739, PR pending
on the pipeline) makes `AnalysisImaging.save_results` write a second record beside
`files/wcs.json`: `files/coolest.json`, a COOLEST (COde-independent Organized LEns STandard,
Galan et al. 2023) template of the maximum log likelihood model. The nine completed sep1 fits
predate it, exactly as they predated the source fields in `wcs.json` a few hours earlier. The
same lever applies — reload, do not refit — so this session prepared that pass rather than
running it, because the RAL stack cannot yet write the file (see **Preconditions**).

## What I did

**Merged the feature branch into the clone.** `git merge origin/feature/coolest-observation-grid`
at **`60133f8`** landed conflict-free as merge commit **`36ba572`**: `util.py` (+111,
`coolest_json_from` and the `save_results` call), `tests/test_coolest_output.py` (new, 268
lines), `tests/test_latent_run_level.py` (+66, two slow run-level assertions),
`scripts/tools/build_inspect.py` (+55/-27, collects `coolest.json` and `coolest_sersic.json`),
`scripts/build_inspection_bundle.sh`, `AGENTS.md`, `README.md`, `catalogue/README.md`. The
branch is based on `650dc2d`, already merged here on 2026-09-16, so merging the pipeline's
`main` later will be a no-op for these commits. The clone's seven local science commits are
preserved. **Nothing was pushed** — `origin` here is the pipeline repo.

`python3 -m pytest tests/test_coolest_output.py -q` → **7 passed** in 2.4 s.
`python3 -m pytest -q -m "not slow" -x` → **131 passed, 10 deselected** in 36 s.

**Wrote `hpc/batch_cpu/submit_reload_coolest`**, a copy of `submit_reload_wcs` with only the
header, `-J euclid_dr1_reload_coolest` and the `Purpose` echo changed. Same
`--array=0,2-9` (index 1, `Tile102007299…`, is still excluded — its `vis_lp` never completed),
same `--partition=ral -t 00:30:00 --mem=8gb --cpus-per-task=1`, same log paths, same dataset
list in the same order, same single `python3 scripts/initial_lens_model.py --sample=dr1_sep1
--dataset=$dataset --stage=all --use_cpu --number_of_cores=1`. **No `--seed`, no model or search
change** — the identifier hashes must not move, or the reload starts a new fit instead of
refreshing a record. The header adds the two preconditions below and the back-up-first rule.

**`force_pickle_overwrite` was deliberately left `false`.** A stale `true` in this clone would
make any later real DR1 fit reload an existing result instead of running it.

## Preconditions for the run day

**Both were satisfied on the run day (2026-09-17) and checked, not assumed** — see **Result**.

- [x] **PyAutoLens#739 merged and `HPCPullPyAuto` run.** The record is written by
      `al.interop.coolest.to_coolest(galaxies=, file_path=, dataset=, on_unsupported=,
      metadata=)`. The `dataset=` and `on_unsupported=` keywords are that PR's work and do not
      exist on RAL's `/mnt/ral/jnightin/PyAuto` until the PR merges and the stack is pulled.
- [x] **`coolest` installed in the RAL venv** (0.1.11 confirmed) (`pip install autolens[coolest]`) — being done in
      this session's sibling step.

Both failure modes are **silent**: `coolest_json_from` catches `ImportError` and every
`Exception`, logs a warning and returns `None`, because a record must never fail a fit. A task
that "COMPLETED" with no `coolest.json` is the expected shape of both misses, so the `.out` logs
have to be read, not just the exit codes.

## Run-day recipe

1. Confirm the two preconditions on RAL: `python3 -c "import coolest"` and that
   `al.interop.coolest.to_coolest` accepts `dataset=` / `on_unsupported=`.
2. **Back up** on RAL: `cp -a output output.prezip.bak.<date>`. `paths.restore()` deletes the
   zip before re-zipping, so this copy is the only thing standing between a bad reload and the
   fits.
3. Flip `output.force_pickle_overwrite: true` in `config/general.yaml` (line 30).
4. `hpc/sync push --no-data` — carries the merged `util.py`, the flag and the submit script.
5. `hpc/sync submit cpu submit_reload_coolest` (i.e. `sbatch hpc/batch_cpu/submit_reload_coolest`
   on RAL). Nine tasks, ~15-30 s each.
6. Grep the logs: each `.out` must show `Forcing pickle overwrite` **twice** and
   `Fit Already Completed: skipping non-linear search` **twice** (once per stage), and **no**
   `coolest.json:` warning line.
7. `hpc/sync pull` — but note this clone's `output/` is flat (`output/<Tile>/…`) where RAL's is
   `output/dr1_sep1/<Tile>/…`; pull with an explicit rsync of
   `euclid_jump:…/output/dr1_sep1/` → `output/`, as the wcs pass did, or a nested tree appears
   beside the flat one.
8. Flip `force_pickle_overwrite` back to `false` **in the same session**, on RAL and here.
9. `bash scripts/build_inspection_bundle.sh dr1_sep1`.
10. Confirm `coolest.json` in each of the nine `inspect/dr1_sep1/<Tile>/` folders — and
    `coolest_sersic.json` only where `sersic_lens_model/vis` has run, which is nowhere yet on
    this sample.

## What the template will contain for these fits

- **The full mass model**: `al.mp.Isothermal` + `al.mp.ExternalShear`, both stages. This is
  what a COOLEST consumer wants from DR1, and it exports whole.
- **`meta.skipped_profiles`**: `on_unsupported="skip"` is passed because these models are not
  wholly analytic. The lens light (and, in `vis_lp`, the source light) is an MGE `Basis` of
  Gaussians — 2 × 20, ordered — and the `vis_pix` source is a Delaunay `Pixelization`. COOLEST
  has no profile for either, so raising would mean no template at all for the stages that
  matter; skipping names every omission under `meta.skipped_profiles` instead, so a reader
  cannot mistake the template for the whole model.
- **The observation grid**: `dataset=` stamps the template with the fitted cut-out's own pixel
  grid — **100 × 100 at 0.1"/pixel** for this sample. Without it COOLEST writes an empty
  observation block and the exported model is not reproducible on the data it was fitted to.
- **Metadata**: `pipeline: euclid_strong_lens_modeling_pipeline`, `search` (`paths.name`, e.g.
  `initial_lens_model`) and `dataset` (the tile name).

## Next

- [x] Run the pass once PyAutoLens#739 has merged and `HPCPullPyAuto` has refreshed RAL. **Done 2026-09-17, array `343367`.**
- [x] Read one written `coolest.json` back and check `meta.skipped_profiles` names the MGE
      `Basis` and the Delaunay `Pixelization`, and that the observation block carries 100 × 100
      at 0.1". **Both confirmed on all nine.**
- [ ] Delete the pre-reload backups once the reloaded results are trusted:
      `/mnt/ral/jnightin/euclid_dr1/output.prezip.bak.2026-09-16b` (this pass, 108 MB) and
      `…output.prezip.bak.2026-09-16` (the wcs pass, 109 MB), both still undeleted.

## Result — RAN 2026-09-17 (SLURM array `343367`)

**Preconditions, verified on RAL, not assumed.**

- `coolest` **0.1.11** importable in the RAL venv under
  `PROJECT_PATH=/mnt/ral/jnightin/euclid_dr1`, `PYAUTO_HPC_BASE=/mnt/ral/jnightin/PyAuto`,
  `source $PROJECT_PATH/activate.sh`.
- `/mnt/ral/jnightin/PyAuto/PyAutoLens` at **`223132c0b`** — "Merge pull request #740 from
  PyAutoLabs/feature/coolest-observation-grid", i.e. the `dataset=` / `on_unsupported=`
  observation-grid work is on the cluster stack.
- `squeue -u jnightin` empty; partition `ral` carried 0 pending and 1 running job of anyone's,
  so the script's own `--partition=ral` was kept (no reason to reach for the `gpu`-CPUs
  alternative).

**Backup.** `cp -a output output.prezip.bak.2026-09-16b` on RAL — **108 MB**, ten tile
directories — taken before anything else and kept separate from the wcs pass's
`output.prezip.bak.2026-09-16`. `df -h /mnt/ral` reported **63 T free of 171 T (64% used)**
before and after.

**The pass.** `force_pickle_overwrite: true` (clone commit `3406406`), `hpc/sync push --no-data`,
then verified on RAL: `config/general.yaml` line 30 `true`, `grep -c coolest_json_from util.py`
= **3**, `hpc/batch_cpu/submit_reload_coolest` present. Submitted with
`hpc/sync submit cpu submit_reload_coolest` → **`343367`**. All nine tasks were `R` on
`euclid-ral-compute-10-2` within 10 s and the queue was empty inside 50 s.

| task | tile | state | elapsed |
|---|---|---|---:|
| `343367_0` | `Tile102005065RA0135279431487DECNEG0701599765928` | COMPLETED | 00:00:35 |
| `343367_2` | `Tile102007899RA0631697483678DECNEG0650586413593` | COMPLETED | 00:00:34 |
| `343367_3` | `Tile102007903RA0668824944366DECNEG0648906250124` | COMPLETED | 00:00:31 |
| `343367_4` | `Tile102008165RA0109661927007DECNEG0642904268129` | COMPLETED | 00:00:32 |
| `343367_5` | `Tile102008219RA0727857418392DECNEG0644388277074` | COMPLETED | 00:00:34 |
| `343367_6` | `Tile102008468RA3567393529734DECNEG0647169093056` | COMPLETED | 00:00:31 |
| `343367_7` | `Tile102008475RA0039777627054DECNEG0637715426503` | COMPLETED | 00:00:34 |
| `343367_8` | `Tile102008532RA0683486015030DECNEG0642073552911` | COMPLETED | 00:00:32 |
| `343367_9` | `Tile102008848RA0601371893270DECNEG0634606579714` | COMPLETED | 00:00:29 |

`ExitCode 0:0` on every task and every `.batch` step.

**The logs say the lever fired and the writer did not complain.** Every one of the nine `.out`
files carries `Forcing pickle overwrite` **twice** and `Fit Already Completed` **twice** — one
pair per stage. The only line matching `coolest.json` in any log is the script's own banner,
`Purpose    : reload both stages, write files/coolest.json`; **no warning line was emitted**,
which is the check that matters, because a failed conversion is a warning and never a failed
task. No `.err` file contains a `Traceback`.

**The record is in the archives, not merely unwarned-about.** On RAL,
`unzip -l` over all 18 result zips (9 tiles × `vis_lp`, `vis_pix`) finds `files/coolest.json`
in **18 of 18**; after the pull, the same is true of all 18 zips in this clone.

**Flag off again.** `force_pickle_overwrite: false` (clone commit `6fd17ca`);
`git diff 9803a8f -- config/general.yaml` is empty, so the file is byte-identical to its
pre-pass state. `hpc/sync push --no-data`, and RAL re-checked: line 30 reads `false`.

**Pull and bundle.** Pulled with the explicit rsync the trap calls for —
`rsync -az --partial --exclude=search_internal euclid_jump:…/output/dr1_sep1/ output/` — so the
clone's flat layout is preserved (10 entries under `output/`, no `output/dr1_sep1/`, 52 MB), plus
the SLURM logs into `hpc/batch_cpu/{output,error}/`. `scripts/build_inspection_bundle.sh` was
then run with `SKIP_SED=1 CREATE_ARCHIVE=0`. **All nine** `inspect/dr1_sep1/<Tile>/` folders now
hold a `coolest.json` (~5.5 kB each); **no** folder holds `coolest_sersic.json`, which is
correct — `sersic_lens_model/vis` has not run on this sample, and stages 4 and 5 reported
"no completed sersic_lens_model/vis results" as they should.

### What the templates actually contain

Identical in shape across all nine (the bundle collects the **`vis_pix`** stage):

- `meta.pipeline` = `euclid_strong_lens_modeling_pipeline`, `meta.search` = `vis_pix`,
  `meta.dataset` = the tile name, `meta["coolest.__version__"]` = `0.1.11`.
- **Mass model, whole**: `galaxy_0` (z = 0.5) carries an `SIE`
  (`theta_E`, `q`, `phi`, `center_x`, `center_y`) and a separate `MassField` entity
  `mass_field_0` carries `ExternalShear` (`gamma_ext`, `phi_ext`). `galaxy_1` (z = 1.0) is the
  source and carries no exportable profile.
- `meta.skipped_profiles` names exactly the two COOLEST has no profile for:
  `galaxy_0: Basis(Gaussian x 40)` (the MGE lens light) and
  `galaxy_1: Pixelization(Delaunay)` (the pixelized source). The `vis_lp` template inside the
  zips skips `Basis(Gaussian x 40)` and `Basis(Gaussian x 20)` instead — lens light and MGE
  source light — and exports the same `SIE` + `ExternalShear`.
- **Observation grid, as predicted**: `observation.pixels` `num_pix_x` = `num_pix_y` = **100**,
  `field_of_view_x` = `field_of_view_y` = **[-5.0, 5.0]** arcsec, and
  `instrument.pixel_size` = **0.1**. `exposure_time`, `mag_zero_point`, `noise.type` and
  `pixels.fits_file.path` are `null`, and `instrument.band` / `name` are empty strings — the
  template carries the grid, not the data.
- `cosmology`: `FlatLambdaCDM`, H0 67.74, Om0 0.3075.

### Deviations from the recipe

- **One deviation, in step 9 only.** `scripts/build_inspection_bundle.sh` resolves its results
  tree as `<OUTPUT_DIR>/<sample>`, so against this clone's *flat* `output/` it fails with
  `ERROR: results directory not found: …/output/dr1_sep1`. The bundle was built by pointing
  `OUTPUT_DIR` at a scratch directory holding a single symlink `dr1_sep1 → …/euclid_dr1/output`.
  Nothing in the repo was moved or renamed. This is the same flat-vs-nested mismatch the pull
  trap already records, showing up on the other side of the pipeline — see **Traps** in
  `state.md`.
- Everything else followed the recipe as written.

### Next

- [ ] Delete `/mnt/ral/jnightin/euclid_dr1/output.prezip.bak.2026-09-16b` (108 MB) once the
      reloaded results are trusted — and `…bak.2026-09-16` (109 MB) from the wcs pass with it.
- [ ] The by-eye sheet is still the owed deliverable; `coolest.json` does not change it.
