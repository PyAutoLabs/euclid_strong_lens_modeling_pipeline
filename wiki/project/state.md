---
title: Project state
type: state
last_touched: 2026-09-17
---

# Project state

The **head pointer** for this project — read this first, before any journal entry.
Rewritten every session, never appended.

## Science goal

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

`/mnt/c/Users/Jammy/Science/euclid_dr1/Segmentation.zip` (14.8 GB, untracked, local only) —
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
`/mnt/c/Users/Jammy/Science/euclid_dr1_prelim/output/dr1_prelim_grade_ab/` and its
catalogue bundle `inspect/dr1_prelim_grade_ab_342629/`. It lives in the *other* clone and is
not copied here.

## Where we are now

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
  `/mnt/ral/jnightin/euclid_dr1/output.prezip.bak.2026-09-16b` (108 MB). Journal:
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

- **Nothing is running on RAL.** `squeue` is empty for `jnightin`.
- **SLURM array `342650`** (`euclid_dr1_two_stage`, `--array=0-9`) **is finished**: nine of the
  ten tiles completed both stages (18 zips under `output/dr1_sep1/<tile>/initial_lens_model/`),
  and `Tile102007299RA0702283866574DECNEG0660415308762` (array index 1) never completed its
  `vis_lp`. That tile was therefore excluded from the 2026-09-16 reload pass and has no
  `wcs.json` to regenerate.
- **SLURM array `343367`** (`euclid_dr1_reload_coolest`, `--array=0,2-9`) **is finished** —
  nine `COMPLETED` tasks, 29-35 s each, no traceback and no writer warning. It added
  `files/coolest.json` to all 18 zips; the pre-reload copy is kept on RAL at
  `/mnt/ral/jnightin/euclid_dr1/output.prezip.bak.2026-09-16b` (108 MB), for the human to delete.
- **SLURM array `343358`** (`euclid_dr1_reload_wcs`, `--array=0,2-9`) **is finished** —
  nine `COMPLETED` tasks, 29-34 s each, no traceback. It rewrote `files/wcs.json` in all 18
  zips; the pre-reload copy is kept on RAL at
  `/mnt/ral/jnightin/euclid_dr1/output.prezip.bak.2026-09-16` (109 MB), for the human to delete.

## Open, carried forward

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

## Journal index

- `2026-09-16-coolest-json-reload-pass.md` — the coolest.json reload pass prepared (branch
  merged, tests green, `submit_reload_coolest` written) and then **run on 2026-09-17**
  (array `343367`): all 18 zips and all nine inspect folders now carry a COOLEST template.
- `2026-09-16-wcs-json-reload-pass.md` — pipeline `main` merged in at `650dc2d`, and the nine
  completed lenses reloaded (array `343358`) so `files/wcs.json` carries the source centre, the
  lensed multiple images and the pixelized clumps.
- `2026-09-11-birth-and-sep1-ten-lens-rerun.md` — the clone was born from pipeline `dbbb2a5`,
  the ten sep1 tiles were extracted from the nested delivery zip, the colormap went back to
  the library default, and the ten-lens array was submitted to RAL.
