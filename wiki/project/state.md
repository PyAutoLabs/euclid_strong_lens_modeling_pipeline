---
title: Project state
type: state
last_touched: 2026-09-18
---

# Project state

The **head pointer** for this project — read this first, before any journal entry.
Rewritten every session, never appended.

## Current run — 2026-09-18

**Array `343480` is running the approved top-1000-brightest `vis_lp` CPU model**
on `dataset/dr1_sep1_top1000/`, indices `0-999`, with 8 CPUs, 64 GB and an
18-hour limit per task. The user approved submission after reviewing the bare-field
form: `fields=field`, with `model.fields.shear.gamma_1/2` unchanged. No `vis_pix`
follow-up was submitted and no scientific result or convergence judgment is implied.

The source port is local commit `2430e4e` (pipeline PR #90); RAL Galaxy is
`90e757d3` and Lens is `478213e78`. Actual imports/model construction, contour
values/JIT/vmap/gradients, one local real-input test-mode likelihood, all 11,000
input files and 115 deployed code/configuration hashes passed before submission.
Earlier science commits, configuration, outputs and untracked preparation are
preserved. Full revisions, hashes and commands: [session record](2026-09-18-flat-fields-top1000.md).

### Also in flight — the sep1 remainder upload

The other **14,032** tiles of the sep1 delivery (15,032 total, minus the 1,000
of `343480`) are extracted to `dataset/dr1_sep1_rest/` (154,346 files, 19 GB)
and uploading to RAL in 15 batches by `hpc/upload_rest/upload_rest.sh`, at a
measured 85-105 KB/s -- roughly **33 hours** end to end. Resume it with the same
command; it is idempotent and appends to part-sent tarballs. The 15 matching
submit scripts `hpc/batch_cpu/submit_initial_lens_model_vis_lp_rest_{01..15}`
are written and machine-checked but **nothing has been submitted**. Everything
here is untracked. Full detail: [session record](2026-09-18-sep1-remainder-upload.md).

## Science goal

The following describes the earlier ten-lens reproduction study. The current
top-1000 run was separately authorized on September 18; no new by-eye ruling is
being inferred from that authorization.

The **actual Euclid DR1 run**, on the September-1 segmentation delivery. `euclid_dr1_prelim`
proved that a complete DR1 catalogue can be delivered from
`euclid_strong_lens_modeling_pipeline` alone; this project runs that pipeline on the real
delivery. The first question it answers is narrow and human-judged: **does the sep1 delivery
reproduce the ten prelim lenses by eye?** The same ten lenses are fitted again, from the sep1
cut-outs and sep1 segmentation, on the identical two-stage CPU route, and the resulting fits
are put side by side with the prelim run **342629**. A "no" — a lens that no longer looks
like a lens under the new segmentation — is itself the finding, and it gates any expansion
beyond ten tiles.

## Data on hand

`/mnt/c/Users/Jammy/Workspace/Science/active/euclid_dr1/Segmentation.zip` (14.8 GB, untracked, local only) —
the sep1 delivery. It is an outer zip holding three **STORED** (uncompressed, and therefore
seekable in place) inner zips,
`Segmentation/eclipse_catalogue_fits_sep1-segmentation_batch{1,2,3}.zip`, carrying
**15032** tile folders between them (batch1 3452, batch2 5790, batch3 5790). Each tile folder
is already in the pipeline's dataset layout: `<Tile>.fits`, `<Tile>.png`, `info.json`,
`positions.json`, `segmentation.png`, and `segmentation/{lens,source,artefact}_{flux,binary}.fits`.

The sample in use is **`dataset/dr1_sep1/`** — ten tile folders, extracted from the zip with
`zipfile.ZipFile(outer.open(inner))` (no intermediate 14.8 GB extraction); the extraction
script is kept outside the repo at the session scratchpad path recorded in the birth journal
entry. The dataset tree is **untracked** (`.gitignore` carries `*.fits`), exactly as in
`euclid_dr1_prelim`.

**Selection rule: the prelim ten, and only those.** No new sort or cut was applied — the
sample is defined as the sep1 counterparts of the ten tiles `euclid_dr1_prelim` fitted, so
that the by-eye comparison is like-for-like. Seven of the ten arrive under a *different*
tile name in the sep1 delivery: the cut-out centre moved, so the RA/DEC digits in the name
changed. The mapping below is the identity map for this project; the array index of
`hpc/batch_cpu/submit_initial_lens_model_two_stage` is `idx − 1`.

| idx | prelim tile (342629) | sep1 tile (this project) | offset (") | prelim `mask_radius` | sep1 `mask_radius` | prelim n_pos | sep1 n_pos |
|----:|---|---|---:|---:|---:|---:|---:|
| 1 | `Tile102005065RA0135279431487DECNEG0701599765928` | `Tile102005065RA0135279431487DECNEG0701599765928` | 0.00 | 3.318 | 5.654 | 2 | 4 |
| 2 | `Tile102007299RA0702283866574DECNEG0660415308762` | `Tile102007299RA0702283866574DECNEG0660415308762` | 0.00 | 6.079 | 6.630 | 3 | 4 |
| 3 | `Tile102007899RA0631694872236DECNEG0650584220817` | `Tile102007899RA0631697483678DECNEG0650586413593` | 0.88 | 3.265 | 5.206 | 2 | 4 |
| 4 | `Tile102007903RA0668831429074DECNEG0648901814905` | `Tile102007903RA0668824944366DECNEG0648906250124` | 1.88 | 5.009 | 4.682 | 3 | 3 |
| 5 | `Tile102008165RA0109664211519DECNEG0642902327064` | `Tile102008165RA0109661927007DECNEG0642904268129` | 0.79 | 2.958 | 4.178 | 3 | 4 |
| 6 | `Tile102008219RA0727851454839DECNEG0644382776514` | `Tile102008219RA0727857418392DECNEG0644388277074` | 2.19 | 4.516 | 5.372 | 1 | 2 |
| 7 | `Tile102008468RA3567390985250DECNEG0647172046261` | `Tile102008468RA3567393529734DECNEG0647169093056` | 1.13 | 3.518 | 3.932 | 3 | 4 |
| 8 | `Tile102008475RA0039777627054DECNEG0637715426503` | `Tile102008475RA0039777627054DECNEG0637715426503` | 0.00 | 6.418 | 6.812 | 4 | 4 |
| 9 | `Tile102008532RA0683495100072DECNEG0642073858939` | `Tile102008532RA0683486015030DECNEG0642073552911` | 1.43 | 4.098 | 4.698 | 2 | 3 |
| 10 | `Tile102008848RA0601376380877DECNEG0634605061157` | `Tile102008848RA0601371893270DECNEG0634606579714` | 0.91 | 3.187 | 5.013 | 3 | 2 |

All ten load through `util.load_vis_dataset(<tile>, sample_name="dr1_sep1")`: 100 × 100 at
0.1"/pixel, all six segmentation FITS present and readable on every tile.

**The reference this project is judged against** is the prelim run **342629** —
`/mnt/c/Users/Jammy/Workspace/Science/archive/euclid_dr1_prelim/output/dr1_prelim_grade_ab/` and its
catalogue bundle `inspect/dr1_prelim_grade_ab_342629/`. It lives in the *other* clone and is
not copied here.

## Where we are now

- **The multi-band SED chain has run on the sep1 sample — 8 of 9 tiles done (2026-09-17).**
  SLURM array **`343381`** (`euclid_sersic_waveband`, `--array=0,2-9`, partition `ral`, 8 cpus,
  64 GB, 12 h) ran `hpc/batch_cpu/submit_sersic_waveband` — new in this clone at `2c4bead`, the
  prelim copy of pipeline PR #70's script with `PROJECT_PATH`, `sample=dr1_sep1` and the sep1
  tile list re-pointed. **Eight tasks `COMPLETED` in 6-16 min.** The chain seeded off a **copy**
  of each tile's `initial_lens_model/vis_lp/<hash>.zip` into `output_sed/dr1_sep1/<tile>/…`, and
  every task's `.out` shows `Fit Already Completed` on the seeded hash, so no `vis_lp` was
  refitted. Local `output_sed/` mirrors RAL **nested** (70 zips: 9 seeds, 9
  `sersic_lens_model/vis`, 52 waveband fits). Journal: `2026-09-17-sed-chain-sep1.md`.
- **The ninth task, `343381_8`, is a Nautilus exploration pathology — the decision is the
  human's (diagnosed 15:03 BST 2026-09-17).** `Tile102008532RA0683486015030DECNEG0642073552911`
  has spent 1 h 36 min on its **first** non-VIS band (`nir_y`) where the other tiles took a
  minute a band. Its `checkpoint.hdf5` reads `n_like` **40000**, `explored=False`, **20 shells**,
  the last three holding 15935 / 14596 / 5610 points with `shell_log_l_min` converging on
  **-16037.3719** and shell 19 spanning **1e-3 nats** over a region **1e-7 arcsec** wide — while
  the single best point of the 40000 is **8 nats higher** (logL -16029.2254) and 0.006" away.
  Not fp32 (min logL spacing 2.2e-11), not a prior edge (no point past |0.19"|): a non-smooth
  likelihood with a lone spike on a plateau, so the `f_live` stop criterion is never reached.
  Because the bands run sequentially in one process, the tile's other six bands cannot start and
  the 12 h walltime kills the task at **~01:33 BST 2026-09-18 with no band completed**. Choice:
  **(a)** `scancel 343381_8` and resubmit under a capped search, or **(b)** accept 8 of 9 —
  commands for both in the journal's "To finish `343381_8`".
- **The bundle stands at 52 / 61 rows.** `inspect/dr1_sep1/` carries
  **`astrometric_offsets.csv` (52 rows, 8 lenses, 7 bands)** and **`magnitudes.csv` (61 rows)**,
  and every tile folder gained `fit_multi_wavelength.png`. Under (a) these reach 59 / 68; under
  (b) they stay as they are.
- **Four rows sit at the ±0.2" grid-offset prior edge, and two tile/band pairs repeat prelim's
  edge.** sep1: `Tile102008165…` `nir_h` (x 0.1909) and `nir_j` (x 0.1906),
  `Tile102008219…` `decam_g` (y 0.1938), `Tile102008475…` `decam_i` (x -0.1855) — three of the
  four with a 3σ bound sitting **on** a limit (0.2000 / -0.1999). The prelim bundle `342629` has
  the **same two tile/band pairs** at the same edge (plus `Tile102007903…` `decam_r`), although
  the sep1 cut-out centres moved by 0.79" and 2.19". The prior (two VIS pixels) is smaller than
  one NISP pixel (0.3") and one DECam pixel (~0.26"), so it is clipping a real offset.
- **Two pipeline follow-ups are filed in the Mind (2026-09-17):** the uncapped waveband search
  and the too-narrow grid-offset prior — `PyAutoMind/draft/bug/euclid/sed_chain_waveband_fit_can_spin_forever.md` and
  `PyAutoMind/draft/bug/euclid/datasetmodel_grid_offset_prior_0_2_clips.md` (Mind `36db392b`). Neither is a local fix; nothing in
  this clone was edited.
- **The nine completed sep1 fits now carry `files/coolest.json` (2026-09-17).** The
  `feature/coolest-observation-grid` work is merged in this clone at `60133f8` and on the RAL
  stack (PyAutoLens `223132c0b`, PR #740), `coolest` **0.1.11** is installed in the RAL venv,
  and the same reload lever was fired again: SLURM array **`343367`** (`--array=0,2-9`), nine
  tasks, all `COMPLETED` in **29-35 s**, every `.out` showing `Forcing pickle overwrite` and
  `Fit Already Completed` twice and **no** `coolest.json` warning. `files/coolest.json` is in
  **18 of 18** zips on RAL and in the clone, and all nine `inspect/dr1_sep1/<Tile>/` folders
  carry it after a bundle rebuild (`coolest_sersic.json` nowhere — that stage has not run).
  Each template exports the whole mass model (`SIE` + `ExternalShear`) on a **100 × 100**
  grid spanning **[-5.0, 5.0]"** at **0.1"/pixel**, and names the MGE `Basis(Gaussian x 40)`
  and the Delaunay `Pixelization` under `meta.skipped_profiles`. The flag is back to `false`
  here and on RAL; the pre-reload copy is
  `/mnt/ral/jnightin/active/euclid_dr1/output.prezip.bak.2026-09-16b` (108 MB). Journal:
  `2026-09-16-coolest-json-reload-pass.md`.
- **The nine completed sep1 fits now carry the source position in `files/wcs.json`
  (2026-09-16).** The pipeline's `main` was merged in at **`650dc2d`** (PRs #71-#73 and #79 —
  the lensed source's multiple images, the pixelized-source clumps off the mapper, and the p99
  clump threshold with a brightest-pixel failsafe), and the finished searches were **reloaded,
  not refitted**: `output.force_pickle_overwrite: true` makes an already-complete search skip
  its non-linear search and re-fire `save_results`, which rewrites `wcs.json` and re-zips in
  place. SLURM array **`343358`**, nine tasks, **29-34 s each**. All 18 `<hash>` identifiers are
  unchanged and the flag is back off. Journal:
  `2026-09-16-wcs-json-reload-pass.md`.
- **The project was born on 2026-09-11.** It is a fresh clone of
  `PyAutoLabs/euclid_strong_lens_modeling_pipeline` at `dbbb2a5` (main, carrying PR #68),
  with **no** local history inherited from `euclid_dr1_prelim`.
- **The route is deliberately identical to prelim's**, so that a difference in the fits is a
  difference in the *data*: `hpc/batch_cpu/submit_initial_lens_model_two_stage`, the two-stage
  CPU path (stage 1 `vis_lp` with JAX pinned to CPU, stage 2 `vis_pix` with JAX off and the
  Numba CPU sparse operator in a Nautilus pool), `--partition=ral`, `--array=0-9`, 8 CPUs,
  64 GB, 36 h. Only four things differ from prelim's copy of that script: the job name, the
  `PROJECT_PATH`, `sample=dr1_sep1`, and the seven renamed tiles.
- **`config/visualize/general.yaml` is back on the library default `colormap: autoarray`**
  (prelim ran `magma`). This is the one deliberate config difference, and it makes the
  side-by-side sheet a comparison against the library's own default rendering.
- **The RAL stack was not touched.** Two prelim jobs (`342629_1`, `342629_3`) were live at
  birth; `/mnt/ral/jnightin/PyAuto` was read for provenance only. Library SHAs as at
  submission: PyAutoFit `e354dbb6b` (09-10), PyAutoArray `667deed3` (09-09),
  PyAutoGalaxy `6640a749` (09-10), PyAutoLens `0da06de63` (09-10), PyAutoNerves `0e7163b` (09-08).

## In flight

- **`343381_8` is still running on RAL, and will not finish on its own** — the SED chain on
  `Tile102008532RA0683486015030DECNEG0642073552911`, submitted 13:33 BST on 2026-09-17, past
  **1 h 36 min** on its first waveband (`nir_y`). Diagnosed from `checkpoint.hdf5` as a Nautilus
  exploration loop on a non-smooth likelihood (see "Where we are now"); its 12 h walltime kills
  it at ~01:33 BST on 2026-09-18 with none of its seven bands written. The human chooses
  `scancel` + capped resubmit, or accept 8 of 9. On either outcome re-pull `output_sed/dr1_sep1/`
  and rebuild the bundle — `astrometric_offsets.csv` reaches 59 rows across 9 lenses and
  `magnitudes.csv` 68 only if the tile completes. The other eight tasks of array **`343381`**
  (`--array=0,2-9`) are `COMPLETED`.
- **SLURM array `342650`** (`euclid_dr1_two_stage`, `--array=0-9`) **is finished**: nine of the
  ten tiles completed both stages (18 zips under `output/dr1_sep1/<tile>/initial_lens_model/`),
  and `Tile102007299RA0702283866574DECNEG0660415308762` (array index 1) never completed its
  `vis_lp`. That tile was therefore excluded from the 2026-09-16 reload pass and has no
  `wcs.json` to regenerate.
- **SLURM array `343367`** (`euclid_dr1_reload_coolest`, `--array=0,2-9`) **is finished** —
  nine `COMPLETED` tasks, 29-35 s each, no traceback and no writer warning. It added
  `files/coolest.json` to all 18 zips; the pre-reload copy is kept on RAL at
  `/mnt/ral/jnightin/active/euclid_dr1/output.prezip.bak.2026-09-16b` (108 MB), for the human to delete.
- **SLURM array `343358`** (`euclid_dr1_reload_wcs`, `--array=0,2-9`) **is finished** —
  nine `COMPLETED` tasks, 29-34 s each, no traceback. It rewrote `files/wcs.json` in all 18
  zips; the pre-reload copy is kept on RAL at
  `/mnt/ral/jnightin/active/euclid_dr1/output.prezip.bak.2026-09-16` (109 MB), for the human to delete.

## Open, carried forward

- **The Sersic products are not in the bundle, although the fits exist.**
  `lens_sersic.csv`, `source_sersic.csv`, `fit_sersic.png` and `coolest_sersic.json` are absent
  from `inspect/dr1_sep1/` because stages 1, 5 and 6 of `scripts/build_inspection_bundle.sh`
  read `OUTPUT_DIR`, while the SED chain writes `sersic_lens_model` into `SED_OUTPUT_DIR`. Run
  by hand against `--output_path=output_sed` they give 9 rows each. Whether the bundle script
  should point those three stages at the SED tree is an open question for the pipeline, not a
  local fix — `euclid_dr1_prelim`'s `342629` bundle has the same gap.
- **The results are pulled; the by-eye sheet is still owed.** Local `output/` mirrors RAL
  (18 zips, 52 MB, flat `output/<Tile>/…` layout). What remains is the **per-tile side-by-side
  sheet** — the new `vis_pix` fit PNG next to 342629's for the same lens — and the
  **effective Einstein radius per tile against prelim's**. Then the ruling is the human's, by
  eye: does the sep1 delivery reproduce the ten prelim lenses? `wcs.json` now supplies the
  source position and the image-plane pixels of every clump, so the sheet can carry markers.
- **`Tile102008475RA0039777627054DECNEG0637715426503` is the first candidate non-reproduction.**
  Its `vis_lp` source centre solves to only **one** lensed image (every other record gives 2-4),
  and its `vis_pix` brightest clump sits **1.75"** away — the two stages disagree about where the
  source is, and the light-profile solution is not multiply imaged. It also carries the sample's
  largest `mask_radius` (6.81").
- **The sep1 mask radii are systematically larger.** `info.json` `mask_radius` grew on 9 of 10
  tiles (only idx 4 shrank), by as much as 2.34" (idx 1, 3.32" → 5.65"). The cut-outs are
  100 × 100 at 0.1", so the loader clips the effective radius at 4.95"; several tiles are now
  clipped where prelim was not. Whether that is a segmentation change or a cut-out change is
  unresolved, and it is a live candidate explanation for any fit that moves.
- **The position lists changed too** — 8 of 10 tiles have a different number of multiple-image
  positions (7 gained one, idx 10 lost one). The positions penalty is what pinned three prelim
  tiles, so this is not cosmetic.
- **Nothing beyond the ten tiles is planned yet.** The remaining 15016 delivered tiles wait on
  the by-eye verdict.

## Traps — don't repeat

- **`output/` is flat here but `output_sed/` is nested.** The SED tree is pulled as
  `output_sed/dr1_sep1/<Tile>/…` (matching RAL) because stages 7-9 of
  `scripts/build_inspection_bundle.sh` resolve `<PROJECT_ROOT>/<SED_OUTPUT_DIR>/<sample>`, so
  `SED_OUTPUT_DIR=output_sed` needs no symlink — while `OUTPUT_DIR` still needs the scratch
  `dr1_sep1 →  output` symlink for the flat `output/`. The two trees deliberately differ; do
  not "fix" one to match the other.
- **Band coverage differs per tile.** `Tile102005065…` carries only `nir_y`/`nir_j`/`nir_h`;
  the other eight carry those plus `decam_g/r/i/z`. A short SED task or a short row count is
  the cut-out, not a failure.
- **Never refresh or touch `/mnt/ral/jnightin/PyAuto` while an array is running.** Mid-run
  library drift is what broke prelim's `342301_1/_7/_8`. Prelim's `342629_1/_3` were still
  live when this project was submitted, and share that stack.
- **The sep1 tile names are not the prelim tile names.** Seven of ten changed. Any script that
  joins this project's results to 342629's must go through the mapping table above, not through
  a name match — a name match silently drops seven lenses.
- **`Segmentation.zip` must never be committed or pushed.** It is 14.8 GB, excluded locally via
  `.git/info/exclude` (not `.gitignore`, so the exclusion does not travel with the repo) and
  `hpc/sync push` transfers only `dataset/`, never the repo root's big files.
- **`origin` here is the pipeline repo, not a project remote.** This clone has local commits
  that must never be pushed.
- **A reload deletes the zip before rewriting it.** `paths.restore()` unzips over the result
  and removes the archive, so `cp -a output output.prezip.bak.<date>` on RAL comes *before* any
  `force_pickle_overwrite` pass, and the flag goes back to `false` in the same session.
- **Never pass `--seed` or change a model/search setting in a reload.** The identifier hash is
  what makes the reload land on the existing result; a changed hash means a new fit was started,
  not a record refreshed.
- **`hpc/sync pull` does not match this clone's `output/` layout.** RAL writes
  `output/dr1_sep1/<Tile>/…`, this clone is flat `output/<Tile>/…`; pull with an explicit
  `rsync … euclid_jump:…/output/dr1_sep1/ output/` or a second nested tree appears beside the
  flat one. The same mismatch bites on the way out:
  `scripts/build_inspection_bundle.sh` looks for `<OUTPUT_DIR>/<sample>` and dies with
  `ERROR: results directory not found: …/output/dr1_sep1` against the flat tree. Build it with
  `OUTPUT_DIR` pointing at a scratch directory holding a symlink `dr1_sep1 → …/euclid_dr1/output`
  (the tools accept an absolute `--output_path`); do not restructure `output/`.
- **Read the inner zips in place.** `zipfile.ZipFile(outer.open(inner))` works because the three
  inner zips are STORED; extracting the outer archive first costs 14.8 GB of `/mnt/c` for nothing.
- **Three sep1 tiles ship 9 files, not 11.** `Tile102013495…`, `Tile102013503…` and
  `Tile102013521…` are missing `positions.json` and `segmentation.png` *inside the delivery
  zip itself* (one per inner batch zip) -- this is not an extraction bug, so do not "repair" it
  by re-extracting. All three are in batch 02. Any tooling that assumes 11 files per tile will
  hang or fail on them; count from `hpc/upload_rest/expected_counts.json` instead.
- **The top-1000 CSV is not brightness-ordered.** `eclipse_catalogue_fits_sep1_top1000_brightest.csv`
  is sorted by tile ID and carries no magnitude column, so the top-1000 submit script's claim that
  "the array index is the catalogue's own brightness order" is false. The ranking was applied
  upstream and cannot be reproduced locally; a new cut needs a new CSV from that source.
- **RAL has outbound internet at 61 MB/s; the laptop uploads at ~95 KB/s.** For any future
  delivery, **record the download URL** -- fetching it on RAL is ~640x faster than pushing it
  from here (minutes rather than days). `Segmentation.zip` had no recorded URL, which is why
  the remainder went up the slow way.
- **Never `pkill -f` a driver script here.** The pattern substring-matches the calling shell and
  kills the session's own Bash; it happened twice in the 09-18 upload session. Resolve the exact
  pid from `/proc/<pid>/cmdline` and `kill` that.

- **A Nautilus waveband fit can spin in exploration forever on a non-smooth likelihood**, and
  the `.out` log is **block-buffered** so it looks frozen at the sampler start line with no new
  output for hours. Read `files/search_internal/checkpoint.hdf5` over ssh — `n_like`,
  `explored`, `shell_n` — not the log. A growing checkpoint means "still sampling", not
  "making progress": compare the shell point counts and `shell_log_l_min` against a tile that
  finished.

## Journal index

- `2026-09-17-sed-chain-sep1.md` — the CPU SED chain (array `343381`, `--array=0,2-9`) run
  on the nine seeded sep1 tiles: eight tiles complete in 6-16 min, the bundle rebuilt with
  `astrometric_offsets.csv` (52 rows) and `magnitudes.csv` (61 rows), and the three
  near-prior-edge registration rows.
- `2026-09-16-coolest-json-reload-pass.md` — the coolest.json reload pass prepared (branch
  merged, tests green, `submit_reload_coolest` written) and then **run on 2026-09-17**
  (array `343367`): all 18 zips and all nine inspect folders now carry a COOLEST template.
- `2026-09-16-wcs-json-reload-pass.md` — pipeline `main` merged in at `650dc2d`, and the nine
  completed lenses reloaded (array `343358`) so `files/wcs.json` carries the source centre, the
  lensed multiple images and the pixelized clumps.
- `2026-09-11-birth-and-sep1-ten-lens-rerun.md` — the clone was born from pipeline `dbbb2a5`,
  the ten sep1 tiles were extracted from the nested delivery zip, the colormap went back to
  the library default, and the ten-lens array was submitted to RAL.
