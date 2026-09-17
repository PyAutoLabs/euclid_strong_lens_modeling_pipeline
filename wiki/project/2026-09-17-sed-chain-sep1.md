---
date: 2026-09-17
title: the multi-band SED chain run on the nine sep1 tiles — astrometric_offsets.csv is now in the bundle
tags: [euclid, dr1, dr1_sep1, sersic_lens_model, waveband, sed, astrometric_offsets, magnitudes, output_sed, ral, ran]
---

# the multi-band SED chain run on the nine sep1 tiles — `astrometric_offsets.csv` is now in the bundle

**SLURM array `343381`** (`euclid_sersic_waveband`, `--array=0,2-9`), submitted 13:33 BST on
2026-09-17. Eight of the nine tasks `COMPLETED` in **6-16 min**; one (`343381_8`,
`Tile102008532RA0683486015030DECNEG0642073552911`) was **still running** at the point this
session stopped watching it, 40 min in, on its first waveband. The bundle was rebuilt on what
had landed and now carries the two SED products that did not exist for this sample before:
**`astrometric_offsets.csv` (52 rows)** and **`magnitudes.csv` (61 rows)**.

## What was run

`hpc/batch_cpu/submit_sersic_waveband` is new in this clone (commit `2c4bead`). It is
`euclid_dr1_prelim`'s copy of pipeline PR #70's script — the CPU SED route verified there on
`342668_9` — with three lines re-pointed: `export PROJECT_PATH=/mnt/ral/jnightin/euclid_dr1`,
`sample=dr1_sep1`, and the ten-tile `datasets=()` list in the **sep1** tile names, in the same
idx order as `submit_initial_lens_model_two_stage`. Everything else is byte-identical:
`--partition=ral`, `-n 1 --cpus-per-task=8`, `--mem=64gb`, `-t 12:00:00`,
`#SBATCH --array=0-9`, `export PYAUTO_OUTPUT_DIR=output_sed`, the JAX-on-CPU guard, and

```
env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=$THREADS ... \
    python3 $PROJECT_PATH/scripts/sersic_lens_model_waveband.py \
        --sample=$sample --dataset=$dataset --number_of_cores=$THREADS
```

The chain is `vis_lp` (short-circuited from a seeded result) → the VIS Sersic fit
(`sersic_lens_model/vis`) → one Sersic fit per non-VIS band, all in one Python process.

### The seed

Only `initial_lens_model/vis_lp` seeds the chain. Nine of the ten sep1 tiles have one from
array `342650`; `Tile102007299RA0702283866574DECNEG0660415308762` (array index 1) has none —
its `vis_lp` never completed — so it was excluded at submit time. On RAL:

```
cd /mnt/ral/jnightin/euclid_dr1
for z in output/dr1_sep1/*/initial_lens_model/vis_lp/*.zip; do
  t=$(echo "$z" | cut -d/ -f3)
  mkdir -p "output_sed/dr1_sep1/$t/initial_lens_model/vis_lp"
  cp -p "$z" "output_sed/dr1_sep1/$t/initial_lens_model/vis_lp/"
done
```

`cp -p`, never `mv`: `AbstractPaths.restore()` extracts `<hash>.zip` and then removes the
archive, so a moved seed destroys the source result. Nine zips landed under `output_sed/` and
the nine under `output/` were still there afterwards.

### The submit

`hpc/sync submit` passes no sbatch arguments, so the narrowed array went over ssh directly:

```
cd /mnt/c/Users/Jammy/Science/euclid_dr1 && hpc/sync push
ssh euclid_jump "cd /mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu && \
    sbatch --array=0,2-9 submit_sersic_waveband"   # -> Submitted batch job 343381
```

All nine tasks went `RUNNING` immediately on `euclid-ral-compute-1`; nothing queued.

## The seeded short-circuit held

`output.343381_0.out`, three minutes in:

```
vis_lp - INFO - The output path of this fit is /mnt/ral/jnightin/euclid_dr1/output_sed/dr1_sep1/
    Tile102005065RA0135279431487DECNEG0701599765928/initial_lens_model/vis_lp/ed9f8190f1cae79f983a1e36b9888f53
vis_lp - INFO - Fit Already Completed: skipping non-linear search.
vis_lp - INFO - Search complete, returning result
vis    - INFO - ... sersic_lens_model/vis/c2dfdfb9e6acefc4a8aea09d4ea238c7
vis    - INFO - Starting new Nautilus non-linear search (no previous samples found).
```

`ed9f8190f1cae79f983a1e36b9888f53` is the seeded zip's own hash directory, so the chain
resolved onto the copied result rather than starting a fresh `vis_lp` fit.

## Per-task outcome

`sacct -j 343381 --format=JobID,State,Elapsed,MaxRSS`:

| task | tile (idx) | state | elapsed | MaxRSS | non-VIS bands fitted |
|---:|---|---|---:|---:|---:|
| `343381_0` | `Tile102005065…` (1) | COMPLETED | 00:06:06 | 2.67 GB | 3 |
| `343381_2` | `Tile102007899…` (3) | COMPLETED | 00:09:50 | 3.24 GB | 7 |
| `343381_3` | `Tile102007903…` (4) | COMPLETED | 00:16:24 | — | 7 |
| `343381_4` | `Tile102008165…` (5) | COMPLETED | 00:09:53 | 3.11 GB | 7 |
| `343381_5` | `Tile102008219…` (6) | COMPLETED | 00:10:33 | 3.17 GB | 7 |
| `343381_6` | `Tile102008468…` (7) | COMPLETED | 00:07:31 | 2.92 GB | 7 |
| `343381_7` | `Tile102008475…` (8) | COMPLETED | 00:09:36 | 3.12 GB | 7 |
| `343381_8` | `Tile102008532…` (9) | **RUNNING** at 00:40 | — | — | 0 so far |
| `343381_9` | `Tile102008848…` (10) | COMPLETED | 00:09:14 | 3.62 GB | 7 |

No task failed, no traceback, no timeout. `343381_8`'s VIS Sersic fit finished in ~4 min; its
first waveband (`nir_y`) was still sampling 30+ min later, where every other tile spent about a
minute per band. It has a 12 h walltime and its `search_internal/checkpoint.hdf5` was still
growing, so it was left to run.

## Band coverage is not uniform across the sample

The non-VIS bands are whatever the tile's cut-out carries, and that differs:

- `Tile102005065…` (idx 1): **`nir_y`, `nir_j`, `nir_h` only** — no DECam bands. Its task was
  the fastest (6 min) for that reason alone.
- The other eight: **`decam_g`, `decam_r`, `decam_i`, `decam_z`, `nir_y`, `nir_j`, `nir_h`** —
  seven each.

## What was pulled

```
rsync -az --partial --exclude=search_internal \
    euclid_jump:/mnt/ral/jnightin/euclid_dr1/output_sed/dr1_sep1/ output_sed/dr1_sep1/
rsync -az euclid_jump:/mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu/output/ hpc/batch_cpu/output/
rsync -az euclid_jump:/mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu/error/  hpc/batch_cpu/error/
```

The SED tree is pulled **nested** — `output_sed/dr1_sep1/<Tile>/…`, matching RAL — because
`scripts/build_inspection_bundle.sh` resolves stages 7-9 as
`<PROJECT_ROOT>/<SED_OUTPUT_DIR>/<sample>`. So `SED_OUTPUT_DIR=output_sed` works with no
symlink. This is the opposite of `output/`, which is flat here and needs the scratch-symlink
workaround. **The two result trees in this clone now have different layouts.**

**70 zips** landed locally: 9 seeds + 9 `sersic_lens_model/vis` + 52 waveband fits. The SLURM
logs are gitignored (`hpc/batch_cpu/output/.gitignore` is `*`), so they stay local.

## The bundle

```
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
CREATE_ARCHIVE=0 OUTPUT_DIR=<scratch>/outB SED_OUTPUT_DIR=output_sed \
    bash scripts/build_inspection_bundle.sh dr1_sep1
```

`<scratch>/outB` holds the single symlink `dr1_sep1 → …/euclid_dr1/output` that stages 1-6 need
against the flat tree. All nine stages ran; stage 1 reported `skipped: 1` — the tile with no
finished `vis_lp`/`vis_pix`, as before.

`inspect/dr1_sep1/` now holds **`astrometric_offsets.csv`**, `magnitudes.csv` and
`lens_mass.csv` at the root, and each of the nine tile folders gained
`fit_multi_wavelength.png` plus its own `magnitudes.csv` (eight of them also
`astrometric_offsets.csv`).

### `astrometric_offsets.csv`

- **52 rows**, **8 distinct lenses**, **7 wavebands**
  (`decam_g`, `decam_r`, `decam_i`, `decam_z`, `nir_y`, `nir_j`, `nir_h`).
- Per lens: 7 rows each, except `Tile102005065…` with **3** (NISP only).
  `Tile102008532…` has **no rows** — its bands had not finished.
- **8** per-lens `astrometric_offsets.csv` files under the tile folders.
- VIS has no row, by construction: it is the frame the offsets are measured against, and the
  producer drops the results with no `dataset_model.grid_offset` — the log line
  `excluded 9 results whose model has no dataset_model.grid_offset` is those nine VIS fits.
- Median |`grid_offset_y`| **0.026"**, median |`grid_offset_x`| **0.029"** — well inside the
  ±0.2" (two VIS pixel) uniform prior, i.e. most bands register to VIS at the few-hundredths
  of an arcsec level.
- **Three rows sit within 0.011" of the prior edge**, which the producer's docstring calls a QA
  flag rather than a measurement:

  | lens | band | `grid_offset_y` | `grid_offset_x` |
  |---|---|---:|---:|
  | `Tile102008165RA0109661927007DECNEG0642904268129` | `nir_h` | 0.1787 | **0.1909** |
  | `Tile102008165RA0109661927007DECNEG0642904268129` | `nir_j` | 0.1571 | **0.1906** |
  | `Tile102008219RA0727857418392DECNEG0644388277074` | `decam_g` | **0.1938** | -0.1234 |

  Both `Tile102008165…` NISP rows push the same way in x; `nir_y` on the same tile does not.

### `magnitudes.csv`

**61 rows** — the 52 waveband rows plus the nine `sersic_lens_model/vis` rows (VIS has a
magnitude but no offset). Nine per-lens copies.

## What the bundle does *not* carry, and why

`lens_sersic.csv`, `source_sersic.csv`, `fit_sersic.png` and `coolest_sersic.json` are **absent**,
although all nine `sersic_lens_model/vis` searches have completed. Stages 1, 5 and 6 of
`build_inspection_bundle.sh` read **`OUTPUT_DIR`**, and the SED chain writes `sersic_lens_model`
into **`SED_OUTPUT_DIR`** (`PYAUTO_OUTPUT_DIR=output_sed`) — so with `OUTPUT_DIR` pointed at the
`initial_lens_model` tree those three producers see no `sersic_lens_model` at all. The same gap
is visible in `euclid_dr1_prelim`'s `inspect/dr1_prelim_grade_ab_342629/`, which likewise has
`astrometric_offsets.csv` and `magnitudes.csv` but no `lens_sersic.csv`.

Run against the SED tree they do produce output — checked into a scratch directory, not into
the bundle:

```
python3 catalogue/scripts/lens_sersic.py   --sample=dr1_sep1 --output_path=output_sed --inspect_dir=<scratch>
python3 catalogue/scripts/source_sersic.py --sample=dr1_sep1 --output_path=output_sed --inspect_dir=<scratch>
```

→ `lens_sersic.csv` **9 rows**, `source_sersic.csv` **9 rows**, one per-lens copy each for all
nine tiles.

## To finish `343381_8`

```
ssh euclid_jump "squeue -u jnightin -j 343381"          # or: hpc/sync jobs
ssh euclid_jump "sacct -j 343381 --format=JobID,State,Elapsed,MaxRSS"
cd /mnt/c/Users/Jammy/Science/euclid_dr1
rsync -az --partial --exclude=search_internal \
    euclid_jump:/mnt/ral/jnightin/euclid_dr1/output_sed/dr1_sep1/ output_sed/dr1_sep1/
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
CREATE_ARCHIVE=0 OUTPUT_DIR=<scratch>/outB SED_OUTPUT_DIR=output_sed \
    bash scripts/build_inspection_bundle.sh dr1_sep1
```

`<scratch>/outB` is any directory holding `dr1_sep1 → /mnt/c/Users/Jammy/Science/euclid_dr1/output`.
On completion `astrometric_offsets.csv` should reach **59 rows** across **9 lenses** and
`magnitudes.csv` **68**.
