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
- **Four rows sit within 0.03" of the prior edge** — the threshold on this table is now
  **0.03"** (|offset| > 0.17"), not the 0.011" first used, which hid the fourth row. The
  producer's docstring calls a row at the edge a QA flag rather than a measurement:

  | lens | band | `grid_offset_y` | `grid_offset_x` | 3σ bound touching ±0.2"? |
  |---|---|---:|---:|---|
  | `Tile102008165RA0109661927007DECNEG0642904268129` | `nir_h` | 0.1787 | **0.1909** | x upper 0.1910, y upper 0.1788 |
  | `Tile102008165RA0109661927007DECNEG0642904268129` | `nir_j` | 0.1571 | **0.1906** | **x upper 0.2000**, **y upper 0.1993** |
  | `Tile102008219RA0727857418392DECNEG0644388277074` | `decam_g` | **0.1938** | -0.1234 | **y upper 0.2000** |
  | `Tile102008475RA0039777627054DECNEG0637715426503` | `decam_i` | 0.1773 | **-0.1855** | y upper 0.1996, x lower -0.1999 |

  Both `Tile102008165…` NISP rows push the same way in x; `nir_y` on the same tile does not.
  Three of the four have a 3σ bound sitting **on** a prior limit (0.2000 / -0.1999), i.e. the
  posterior is truncated, not merely close to the edge.

### Same edge in both deliveries

The prelim bundle
`/mnt/c/Users/Jammy/Science/euclid_dr1_prelim/inspect/dr1_prelim_grade_ab_342629/astrometric_offsets.csv`
has **four** rows past the same 0.17" threshold:

| prelim lens | band | `grid_offset_y` | `grid_offset_x` |
|---|---|---:|---:|
| `Tile102007903RA0668831429074DECNEG0648901814905` | `decam_r` | **-0.1848** | **-0.1802** |
| `Tile102008165RA0109664211519DECNEG0642902327064` | `nir_h` | **0.1835** | **0.1836** |
| `Tile102008165RA0109664211519DECNEG0642902327064` | `nir_j` | 0.1728 | **0.1908** |
| `Tile102008219RA0727851454839DECNEG0644382776514` | `decam_g` | **0.1966** | -0.1735 |

Cross-matched through the mapping table in `state.md`, **the same two tile/band pairs sit at the
same prior edge in both deliveries**: `Tile102008165…` `nir_h` and `nir_j` (x ≈ +0.19 in both),
and `Tile102008219…` `decam_g` (y ≈ +0.19 in both). They are not the same cut-out — the sep1
cut-out centres moved by **0.79"** (idx 5) and **2.19"** (idx 6) — and the segmentation changed,
yet the fitted offset lands on the same limit. That is the signature of a **prior clipping a real
offset larger than 0.2"**, not of a per-delivery fluke.

The ±0.2" uniform prior is two VIS pixels at 0.1"/pixel. It is **smaller than one NISP pixel
(0.3") and smaller than one DECam pixel (~0.26")**, so a genuine one-pixel registration error in
those instruments cannot be represented at all. The prior lives at
`scripts/lens_model_waveband.py:492-497` in the pipeline repo. Filed as a pipeline follow-up in the
Mind: `PyAutoMind/draft/bug/euclid/datasetmodel_grid_offset_prior_0_2_clips.md` (`36db392b`).

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

## `343381_8` diagnosed at 15:03 BST

`343381_8` (`Tile102008532RA0683486015030DECNEG0642073552911`, idx 9) was still `RUNNING` at
**1 h 26 min** elapsed (15:01 BST; 1 h 36 min at 15:12), still on its **first non-VIS band,
`nir_y`**, output dir
`output_sed/dr1_sep1/<tile>/sersic_lens_model/nir_y/337ab77d5369c13c1a92417f9b0dd61c`.

**The `.out` log is not the signal.** It stops at the Nautilus start line (13:38:30) because
python's stdout is block-buffered when redirected — nothing has flushed since. The signal is
`files/search_internal/checkpoint.hdf5`, which was 6.3 MB with mtime 15:02:35 and still growing.
A copy of it was pulled and read:

- Nautilus attrs: `n_dim` **2**, `n_live` **75**, `n_batch` 100, **`n_like` 40000**,
  **`explored=False`** — still in the exploration phase — and **20 shells** built.
- `shell_n` for the last three shells: **15935, 14596, 5610** points, against ~100-370 for the
  earlier shells. `shell_log_l_min` is converging on **-16037.3719**
  (last three: -16037.38033, -16037.37204, -16037.37194) and `shell_log_v` is down to
  **-21.5, -26.3**: the bound volume keeps shrinking while the likelihood floor moves by 1e-2
  and then 1e-4 nats.
- Shell 19's 5610 points span a logL range of **1e-3 nats** (-16037.37288 to -16037.37194) at
  position mean (y, x) = **(0.010338, -0.008069)"** with std **(4.1e-8, 1.3e-6)"** — a region
  ~**1e-7 arcsec** wide.
- **Not an fp32 floor:** all 5610 logL values are distinct, minimum spacing 2.2e-11 — fp64.
- **Not a prior-edge case:** no point of the 40000 has |offset| > 0.19", so the sampler is not
  pinned against the ±0.2" grid-offset prior.
- The single **highest-likelihood point of all 40000** is logL **-16029.2254** at
  (y, x) = **(0.004092, -0.008149)"** — **8 nats above** the shell-19 plateau and 0.006" away in
  y. **Only one** point of 40000 lies within 0.5 nats of it.

**Reading.** The likelihood surface for this band is non-smooth: a lone 8-nat spike sitting on a
plateau that the sampler keeps sub-dividing into ever-thinner shells. Nautilus therefore never
reaches its `f_live` stop criterion and exploration does not terminate. The fast tiles finished
`nir_y` in about a minute at ~1600 likelihood calls and 15 bounds; this one is at 40000 calls and
20 shells.

**Consequence.** The chain runs every band **sequentially in one process**, so the tile's other
six bands (`nir_j`, `nir_h`, `decam_g/r/i/z`) cannot start until `nir_y` ends. At the 12 h
walltime the task will be killed at about **01:33 BST on 2026-09-18 with no band completed** —
one slow band costs the whole tile. A pipeline follow-up to cap the search is filed in the Mind:
`PyAutoMind/draft/bug/euclid/sed_chain_waveband_fit_can_spin_forever.md` (`36db392b`).

## To finish `343381_8`

**It will not finish on its own.** On the evidence above the `nir_y` search is in an exploration
loop with no stop criterion in reach, so the outcome without intervention is the 12 h walltime
kill at ~01:33 BST on 2026-09-18, with zero of the tile's seven bands written. **The choice is
the human's:**

**(a) Cancel and resubmit that tile with a capped / different search.**

```
ssh euclid_jump "scancel 343381_8"
ssh euclid_jump "sacct -j 343381_8 -X --format=JobID,State,Elapsed"
# then resubmit just that array index once the search is capped
ssh euclid_jump "cd /mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu && sbatch --array=8 submit_sersic_waveband"
```

Note the VIS Sersic fit and the seeded `vis_lp` are already written, so a resubmit short-circuits
straight back to the wavebands. Nothing else in array `343381` is affected.

**(b) Accept 8 of 9 and let the walltime kill it.** No command; the task dies at ~01:33 BST and
the bundle stays at 52 / 61 rows with 8 lenses in `astrometric_offsets.csv`.

**On either outcome, pull and rebuild the same way** (this is the unchanged recipe):

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
If the tile's seven bands do complete, `astrometric_offsets.csv` reaches **59 rows** across
**9 lenses** and `magnitudes.csv` **68**; under (b) they stay at **52** and **61**.
