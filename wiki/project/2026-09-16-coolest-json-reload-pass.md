---
date: 2026-09-16
title: coolest.json reload pass prepared for the nine completed lenses — merged, tested, not run
tags: [euclid, dr1, dr1_sep1, initial_lens_model, coolest, reload, force_pickle_overwrite, ral, prepared]
---

# coolest.json reload pass prepared for the nine completed lenses — merged, tested, not run

**Status: prepared, not run.** Nothing was submitted to RAL and
`config/general.yaml` still carries `output.force_pickle_overwrite: false`.

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

Neither is satisfied yet; both are on RAL, not here.

- [ ] **PyAutoLens#739 merged and `HPCPullPyAuto` run.** The record is written by
      `al.interop.coolest.to_coolest(galaxies=, file_path=, dataset=, on_unsupported=,
      metadata=)`. The `dataset=` and `on_unsupported=` keywords are that PR's work and do not
      exist on RAL's `/mnt/ral/jnightin/PyAuto` until the PR merges and the stack is pulled.
- [ ] **`coolest` installed in the RAL venv** (`pip install autolens[coolest]`) — being done in
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

- [ ] Run the pass once PyAutoLens#739 has merged and `HPCPullPyAuto` has refreshed RAL.
- [ ] Read one written `coolest.json` back and check `meta.skipped_profiles` names the MGE
      `Basis` and the Delaunay `Pixelization`, and that the observation block carries 100 × 100
      at 0.1".
- [ ] Delete the pre-reload backup once the reloaded results are trusted — as with
      `/mnt/ral/jnightin/euclid_dr1/output.prezip.bak.2026-09-16`, still undeleted.
