---
date: 2026-09-11
title: euclid_dr1 born from the sep1 segmentation delivery; the ten prelim lenses resubmitted on RAL
tags: [euclid, dr1, dr1_sep1, segmentation, initial_lens_model, two_stage, ral, birth]
---

# euclid_dr1 born from the sep1 segmentation delivery; the ten prelim lenses resubmitted on RAL

## Context

`euclid_dr1_prelim` answered the pipeline question — a complete DR1 catalogue *can* be
delivered from `euclid_strong_lens_modeling_pipeline` alone — and, in doing so, found that the
rerun scatter on `effective_einstein_radius` is larger than the offset it was meant to detect.
The September-1 segmentation delivery (`Segmentation.zip`) then arrived: the same survey tiles,
re-cut and re-segmented. This session births **`euclid_dr1`**, the project that runs the
pipeline on that real delivery, and starts it with the narrowest possible first question:
**does the sep1 delivery reproduce the ten prelim lenses by eye?** Same ten lenses, same route,
new data — so that anything that moves is the data moving.

## What I did

**Born the clone.** `/mnt/c/Users/Jammy/Science/euclid_dr1/` already held
`Segmentation.zip` and nothing else. `git init` in place, `origin` pointed at
`https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline.git`, and
`git checkout -b main origin/main` at **`dbbb2a510989d0fdd985df294c6c75550b21fca1`**
(`dbbb2a5`, 2026-09-10, the merge of PR #68). Nothing was copied from prelim's `.git` — this
clone carries no prelim history. `Segmentation.zip` was excluded through
`.git/info/exclude` rather than `.gitignore`, so the exclusion is local to this working copy
and never travels upstream. `git status` is clean with the 14.8 GB archive present.

**Extracted the ten tiles.** `Segmentation.zip` is an outer zip holding three inner zips,
`Segmentation/eclipse_catalogue_fits_sep1-segmentation_batch{1,2,3}.zip`, each **STORED**
(`compress_type == 0`) and therefore seekable in place. The recipe is

```python
with zipfile.ZipFile(OUTER) as outer:
    with outer.open(inner_name) as fh, zipfile.ZipFile(fh) as inner:
        ...
```

which reads an inner archive without ever materialising the 14.8 GB outer one. The script is
deliberately **not** in the repo — it lives at the session scratchpad path
`/tmp/claude-1000/-home-jammy-Code-PyAutoLabs/f69e5858-371c-4c1d-809f-2bad3a4fc617/scratchpad/extract_tiles.py`
and is reproduced by the snippet above plus the mapping table below. Every member under
`eclipse_catalogue_fits_sep1-segmentation_batchN/<Tile>/` was written to
`dataset/dr1_sep1/<Tile>/` with the batch prefix stripped; `__MACOSX/` resource forks were
skipped, and so was the batch-level `segmentation/<Tile>.png` preview, which sits outside the
tile folder. Result: **10 directories, 11 files each** — `<Tile>.fits`, `<Tile>.png`,
`info.json`, `positions.json`, `segmentation.png`, and six `segmentation/*.fits`. The tree is
untracked (`.gitignore` carries `*.fits`), as in prelim.

The delivery holds **15032** tile folders in total (batch1 3452, batch2 5790, batch3 5790),
counted from the member listings — the figure carried into this project as "about 15000".

**The mapping.** Seven of the ten prelim tiles arrive under a *different* name: the cut-out
centre moved, so the RA/DEC digits in the tile name changed. The delivered names are the ones
kept, and this table is the identity map (array index = idx − 1):

| idx | prelim tile (342629) | sep1 tile | offset (") | prelim `mask_radius` | sep1 `mask_radius` | prelim n_pos | sep1 n_pos |
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

**Validated the datasets through the pipeline's own loader**, not by eye on the FITS:
`util.load_vis_dataset(<tile>, sample_name="dr1_sep1")` from the clone root, for all ten.
All ten load. Every one is 100 × 100 at `pixel_scales=(0.1, 0.1)`, every one has all six
segmentation FITS readable, and every one builds a positions likelihood. Effective
`mask_radius` after the loader's clipping ran 3.45–4.95" (the clip sits at 4.95" for a
100-pixel cut-out); brightest-pixel `dataset_centre` ranged from 0.04" to 1.54" off the
cut-out centre.

**Config.** One line changed: `config/visualize/general.yaml`,
`colormap: magma` → `colormap: autoarray` — back to the library default, with the trailing
comment untouched. Nothing else in `config/` was touched.

**Submit script.** `hpc/batch_cpu/submit_initial_lens_model_two_stage` was copied from prelim
and diffed back against it. Exactly four changes, on 12 lines:

- `#SBATCH -J euclid_initial_lens_model_two_stage` → `#SBATCH -J euclid_dr1_two_stage`
- `export PROJECT_PATH=/mnt/ral/jnightin/euclid_dr1_prelim` → `…/euclid_dr1`
- `sample=dr1_prelim_grade_ab` → `sample=dr1_sep1`
- seven tile names in `datasets=(...)` (idx 3–7, 9, 10; idx 1, 2 and 8 are unchanged because
  those three tiles kept their names)

Everything else is prelim's, byte for byte: `--array=0-9`, `--partition=ral`,
`--cpus-per-task=8`, `--mem=64gb`, `-t 36:00:00`, the mail address, the two-process
`vis_lp` → `vis_pix` structure with per-stage `env`, and `PYAUTO_HPC_BASE=/mnt/ral/jnightin/PyAuto`.

**Plumbing.** `hpc/sync.conf` copied from prelim with `PROJECT_NAME=euclid_dr1`
(`HPC_HOST=euclid_jump`, `HPC_BASE=/mnt/ral/jnightin`); it is gitignored via `hpc/.gitignore`
and `hpc/sync` never pushes it. `activate.sh` gained prelim's three-line
`AUTOLENS_ASSISTANT=/home/jammy/Code/PyAutoLabs/autolens_assistant` export, making it identical
to prelim's. `wiki/project/` was scaffolded with prelim's `README.md`, `_template.md`,
`_state_template.md` and `_profile_template.md` verbatim; `profile.md` was seeded from prelim's
(the user facts are portable, as the template says they should be), and `state.md` and this
entry were written fresh.

**RAL stack provenance**, read-only — the library checkouts under `/mnt/ral/jnightin/PyAuto`
were *not* pulled or changed, because prelim's `342629_1` and `342629_3` were still running
against them:

| repo | SHA | last commit |
|---|---|---|
| PyAutoFit | `e354dbb6b` | 2026-09-10 |
| PyAutoArray | `667deed3` | 2026-09-09 |
| PyAutoGalaxy | `6640a749` | 2026-09-10 |
| PyAutoLens | `0da06de63` | 2026-09-10 |
| PyAutoNerves | `0e7163b` | 2026-09-08 |

**Pushed and submitted.** `hpc/sync push` created `/mnt/ral/jnightin/euclid_dr1` and
transferred the code directories, the root files and `dataset/dr1_sep1/`. Then
`hpc/sync submit cpu submit_initial_lens_model_two_stage` — SLURM array **`342650`**.

## Outcome

`euclid_dr1` exists, on pipeline `dbbb2a5`, with ten validated sep1 datasets and a ten-task
array queued on RAL's `ral` partition. Output will land in
`output/dr1_sep1/<tile>/` for each of the two stages.

Two differences between the deliveries are already visible in the metadata, before a single fit
has run, and both are candidate explanations for anything that moves:

- **`mask_radius` grew on 9 of the 10 tiles** (only idx 4 shrank), by up to 2.34" — idx 1 goes
  3.32" → 5.65". Since the cut-outs are 100 × 100 at 0.1", the loader clips at 4.95", so several
  tiles are now clipped where prelim's were not.
- **The position lists changed on 8 of 10 tiles** — seven gained a multiple image, idx 10 lost
  one. The positions penalty is exactly what pinned three prelim tiles in `vis_lp`, so this is
  not a cosmetic difference.

Nothing was patched to make anything load: all ten datasets were taken as delivered.

## Next

- [ ] When the array completes: `hpc/sync pull`, then a per-tile side-by-side sheet of the new
      `vis_pix` fit PNG against 342629's for the same lens.
- [ ] Report `effective_einstein_radius` per tile against prelim's, remembering prelim's own
      finding that the rerun scatter (median z 15.4) swamps the 1σ bars — the numeric column is
      context for the eye, not the verdict.
- [ ] The human's by-eye ruling: does the sep1 delivery reproduce the ten prelim lenses?
- [ ] If yes, decide what a run beyond ten tiles looks like — the delivery holds 15026.
- [ ] Resolve whether the larger `mask_radius` values are a segmentation change or a cut-out
      change, if any tile's fit moves.
