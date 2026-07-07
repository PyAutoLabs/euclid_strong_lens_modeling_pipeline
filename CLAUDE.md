# Euclid Strong Lens Modeling Pipeline — Claude Instructions

This repository provides the Euclid strong lens modeling pipeline, built on **PyAutoLens**. It fits automated lens models (SIE + shear mass, MGE light) to Euclid VIS imaging data. Run `start_here.py` as a black box on any dataset; use the `scripts/` pipelines for more advanced modeling.

## Scientific Context

For the science behind this pipeline — what Euclid is finding, why SIE
+ shear is a reasonable default mass model, what MGE buys, how
multipoles and external shear affect substructure / cosmography
downstream — see the lensing sub-wiki at
[`PyAutoLabs/PyAutoMemory`](https://github.com/PyAutoLabs/PyAutoMemory),
locally at `../PyAutoMemory/lensing_wiki/`. Most directly relevant:
`entities/euclid-q1.md`, `concepts/mass-models.md`,
`concepts/multipoles.md`, `concepts/external-convergence-shear.md`,
`concepts/lens-finding.md`, `entities/slam-pipeline.md`.

---

## Repository Structure

```
start_here.py              # Entry point — initial SIE + MGE lens model
util.py                    # Shared utilities: dataset loading, analysis, arg parsing
activate.sh                # HPC venv activation (sets PYTHONPATH)
config/                    # PyAutoLens YAML configuration files
dataset/                   # Input data: dataset/<sample>/<dataset_name>/
output/                    # Results (generated at runtime, not committed)
scripts/                   # Additional pipelines:
  full_model.py            #   Full SLaM pipeline (source pix + power-law mass)
  lens_model_waveband.py   #   Multi-waveband modeling with fixed lens model
  sersic_lens_model.py     #   Sersic lens+source fits for SED photometry
  mge_lens_only.py         #   MGE lens-light-only subtraction
preprocess/                # Preprocessing tools (all accept --sample):
  segmentation.py          #   Segmentation diagnostics + positions.json
  adjust_binary.py         #   GUI binary mask tuning (--object=artefact|source|lens)
  validation_GUI.py        #   Annotation GUI for segmentation QA
  move_segmentation_fits.py #  Move segmentation FITS into dataset folders
workflow/                  # Post-run analysis: csv_make.py, png_make.py, fits_make.py
tools/                     # GUI utilities (extra galaxies masking, PSF sizing)
hpc/                       # SLURM submit scripts and sync tooling
tests/                     # Test scripts
```

---

## Running Scripts

Scripts are run from the repository root:

```bash
python start_here.py --dataset=102018665_NEG570040238507752998 --iterations_per_quick_update=10000

python scripts/full_model.py --dataset=102018665_NEG570040238507752998 --sample=q1_walsmley
```

All scripts accept `--dataset`, `--sample` (optional), and `--iterations_per_quick_update` (optional) arguments, parsed by `util.parse_fit_args()`.

---

## Test Runs

Set `PYAUTO_TEST_MODE=1` to make all non-linear searches complete almost instantly with trivial samples. Use this to verify the full pipeline executes without errors before submitting to the HPC or running a real fit. Set `PYAUTO_TEST_MODE=2` to additionally skip the sampler step entirely — fastest mode, used by the `/smoke_test` skill.

```bash
# Initial lens model
PYAUTO_TEST_MODE=1 python start_here.py --dataset=102018665_NEG570040238507752998 --sample=q1_walsmley

# Full SLaM pipeline
PYAUTO_TEST_MODE=1 python scripts/full_model.py --dataset=102018665_NEG570040238507752998 --sample=q1_walsmley

# Sersic lens model
PYAUTO_TEST_MODE=1 python scripts/sersic_lens_model.py --dataset=102018665_NEG570040238507752998 --sample=q1_walsmley

# MGE lens-only subtraction
PYAUTO_TEST_MODE=1 python scripts/mge_lens_only.py --dataset=102018665_NEG570040238507752998 --sample=q1_walsmley
```

The example dataset lives at `dataset/q1_walsmley/102018665_NEG570040238507752998/`.

---

## Codex / Sandboxed Runs

When running from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python start_here.py --dataset=102018665_NEG570040238507752998
```

---

## Dataset Layout

```
dataset/
└── <sample_name>/
    └── <dataset_name>/
        ├── <dataset_name>.fits    # Multi-HDU FITS (VIS + NIR/EXT bands)
        ├── info.json              # Metadata (mask_radius, redshifts, etc.)
        ├── mask_extra_galaxies.fits  # Optional noise scaling mask
        ├── rgb_0.png              # RGB thumbnails
        └── rgb_1.png
```

`util.load_vis_dataset()` reads the FITS file, info.json, and mask automatically.

---

## HPC

```
hpc/
├── batch_gpu/              # GPU submit scripts + SLURM logs
│   ├── submit_start_here   # SLURM batch script for start_here.py
│   ├── submit_full_model   # SLURM batch script for full_model.py
│   ├── output/             # SLURM stdout logs
│   └── error/              # SLURM stderr logs
├── batch_cpu/              # CPU submit scripts + SLURM logs
│   ├── submit_start_here
│   ├── output/
│   └── error/
├── sync                    # Bidirectional sync script (local <-> HPC)
└── sync.conf.example       # Template config for sync
```

Setup: `cp hpc/sync.conf.example hpc/sync.conf` and edit with your HPC host and paths. `sync.conf` is gitignored.

Key commands: `hpc/sync push`, `hpc/sync pull`, `hpc/sync submit gpu submit_start_here`, `hpc/sync jobs`.

---

## Line Endings — Always Unix (LF)

All files must use Unix line endings (`\n`). CRLF will break shell scripts on the HPC. After creating or editing files, verify with `file <path>` and convert with `dos2unix` if needed.
