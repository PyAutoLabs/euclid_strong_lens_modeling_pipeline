# Euclid Strong Lens Modeling Pipeline

This repository makes it straightforward to use the Euclid strong lens modeling pipeline on your local machine
or a supercomputer.

The pipeline uses **PyAutoLens** to perform automated strong lens modeling, with this repository making it simple
to run the pipeline as a black-box on Euclid data.

## JAX & GPU

**PyAutoLens** runs significantly faster on GPUs — often **50x or more** compared to CPUs.

This acceleration is achieved through \[**JAX**\](<https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>), which provides GPU and TPU support.

When you install **PyAutoLens** (see instructions below), JAX will also be installed. However, the default installation may not include GPU support.

To ensure GPU acceleration, it is recommended that you install JAX with GPU support **before** installing **PyAutoLens**, by following the official \[JAX installation guide\](<https://jax.readthedocs.io/en/latest/installation.html>).

If you install **PyAutoLens** without a proper GPU setup, a warning will be displayed.

## Running On A Cluster

`hpc/README.md` covers SLURM submission: a one-job GPU route recommended for a small
subset of lenses, and a two-stage CPU route (`vis_lp` under JAX on CPU, then `vis_pix`
with Numba and a process pool) for large samples on many cores, with example scripts
under `hpc/batch_gpu/` and `hpc/batch_cpu/` and the `hpc/sync` transfer/submit tool.

## Getting Started

**PyAutoLens** supports Python 3.12 and later, with **Python 3.13 recommended**.

You first may want to set up a **Python virtual environment** or **conda environment** to install the pipeline
in (see <https://docs.python.org/3/library/venv.html>).

Next, install **PyAutoLens** via pip:

```bash
pip install --upgrade pip
pip install autolens
```

Clone the pipeline repository:

```bash
git clone https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline
cd euclid_strong_lens_modeling_pipeline
```

Run the pipeline with the example dataset:

```bash
python scripts/initial_lens_model.py --sample=q1_walsmley --dataset=102018665_NEG570040238507752998 --iterations_per_quick_update=10000
```

(`start_here.py` in the repository root is a thin shim over this same script and
accepts the same arguments, so older commands and bookmarks keep working.)

The pipeline will run on the example dataset, outputting results to the `output` folder and in the `dataset` folder,
and it can be easily modified to run on your own data.

The pipeline above is parallelized automatically based on the hardware available to Python (GPU or CPU) and
results are output on-the-fly during the model fitting procedure every 10000 iterations, meaning you can watch
the lens model improve over time!

## Overview

The starting point for Euclid strong lens modeling is `scripts/initial_lens_model.py`. It performs
automated lens modeling in around 10 minutes per lens on a GPU, around 20 minutes on an 8 core CPU
(times from the DR1 science runs under their own `config/`; `hpc/README.md` has the times measured with the committed one).
Which to use for a whole sample is a question of scale — see `hpc/README.md`.

This script can be run as a black-box, with key output being generated, including:

- A SIE plus shear lens mass model.
- Deblended images of the lens and source galaxies.
- Lens light and source models using a multi Gaussian Expansion.

Here is an example of the output, which shows the lens and source galaxies debelended and a source reconstruction
in the source-plane:

<img src="https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline/blob/main/sie_fit.png?raw=true" width="900" />

If key output for your science case is not generated, please contact James Nightingale on the Euclid consortium
SLACK so it can be added to the pipeline and become a standard output of the Euclid strong lens modeling pipeline
and therefore data release.

Some scripts used for the DR1 science runs were deliberately left out of this
repository — see
[`AGENTS.md`](AGENTS.md#not-ported--available-in-scienceeuclid) for the list and a
reason for each, and [`docs/drift_report.md`](docs/drift_report.md) for the full
record of what this pipeline inherited from the DR1 runs and why.

## Workflow

After running `scripts/initial_lens_model.py` on many lenses, you will begin to build up a large number of results
in the `output` folder. Eventually, manually inspecting these results will become tedious, and you will require an
efficient workflow to inspect the results and perform scientific analysis.

The `workflow` folder contains example scripts for creating workflows which enable efficient inspection of
large lens modeling results. Workflows are designed by creating .png, .csv and .fits files from the results
in the `output` folder for fast inspection.

## Documentation

The following links are useful for anyone more interested in the **PyAutoLens** software:

- [The PyAutoLens readthedocs](https://pyautolens.readthedocs.io/en/latest): which includes [an overview of PyAutoLens's core features](https://pyautolens.readthedocs.io/en/latest/overview/overview_1_start_here.html), [a new user starting guide](https://pyautolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html) and [an installation guide](https://pyautolens.readthedocs.io/en/latest/installation/overview.html).
- [The introduction Jupyter Notebook on Colab](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.7.9.1/start_here.ipynb): try **PyAutoLens** in a web browser (without installation).
- [The autolens_workspace GitHub repository](https://github.com/PyAutoLabs/autolens_workspace): example scripts and the HowToLens Jupyter notebook lectures.

## The Scripts

Every script below is run from the repository root and takes the same command-line
arguments (see [Command-Line Arguments](#command-line-arguments)):

```bash
python scripts/full_model.py --sample=q1_walsmley --dataset=EUCLJ174517.55+655612.5 --iterations_per_quick_update=50000
```

### Fitting pipelines

| Script | What it fits | Chains off |
|---|---|---|
| `start_here.py` | Thin shim over `scripts/initial_lens_model.py` — kept so older commands keep working. | — |
| `scripts/initial_lens_model.py` | **The entry point.** SIE + shear mass, MGE lens light, MGE source (`vis_lp`), then a pixelized Delaunay source (`vis_pix`). `--stage=vis_lp` stops after `vis_lp`. | — |
| `scripts/sersic_lens_model.py` | Sersic lens and source with the mass model fixed, for accurate SED photometry. | `initial_lens_model` `vis_lp` |
| `scripts/lens_model_waveband.py` | Every non-VIS band (NIR / EXT) with the VIS lens model held fixed. | `initial_lens_model` or `sersic_lens_model` |
| `scripts/sersic_lens_model_waveband.py` | The **SED chain** driver: runs `initial_lens_model --stage=vis_lp`, then `sersic_lens_model`, then `lens_model_waveband` over every band. Run it under its own `PYAUTO_OUTPUT_DIR`. | — (drives the three above) |
| `scripts/mge_lens_only.py` | MGE subtraction of the lens light only, revealing the lensed source quickly. | — |
| `scripts/full_model.py` | The full SLaM chain: MGE source, two Delaunay pixelized-source stages, refined lens light, then a PowerLaw + shear mass model. Uses `al.mesh.Delaunay` with `al.reg.AdaptSplit` (`reg.Adapt` cannot JIT on the Delaunay family) and `pixels=500` in its second stage — the `autolens_workspace` `delaunay.py` example uses 1000, but Euclid VIS cut-outs are small. | — |

**PyAutoLens** also has automated pipelines for modeling group-scale strong lenses, lensed
point sources (e.g. lensed quasars) and double source plane lenses. They were previously
in this repository and were removed (commit `fc43be0`, 2025-11-05) when it was narrowed to
the Euclid VIS chain above; they still live in **PyAutoLens** itself and can be restored
here on request — contact James Nightingale on the Euclid consortium SLACK.

### Diagnostics and catalogue

| Script | What it does |
|---|---|
| `scripts/tools/diagnose_latent.py` | Replays the Euclid latent catalogue on one converged result and prints every latent value, flagging NaN and zero sentinels. Runs no search. |
| `scripts/tools/diagnose_latent_vis_pix.py` | The population version: the same replay over every `vis_pix` result in a sample, reporting per-dataset OK/ERR. |
| `scripts/tools/build_inspect.py` | Collects the inspection bundle's PNGs out of finished result zips. |
| `scripts/build_inspection_bundle.sh` | Runs all seven catalogue stages in order for a sample. |
| `catalogue/` | The producers that turn finished fits into the per-lens inspection bundle and the master CSVs — see [`catalogue/README.md`](catalogue/README.md) for the 13-file to producer table and the run order. |

## Command-Line Arguments

All fitting pipelines share one argument parser (`util.parse_fit_args`):

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | *required* | Dataset subdirectory inside `dataset/<sample>/`. |
| `--sample` | none | Sample subdirectory inside `dataset/`. Omit for a flat `dataset/<name>/` layout. |
| `--iterations_per_quick_update` | `5000` | Sampler iterations between on-the-fly visualisation updates. |
| `--number_of_cores` | `1` | CPU cores for the non-JAX Nautilus searches. |
| `--use_cpu` | off | CPU mode: disables JAX and applies the CPU sparse operator to the pixelized stage. |
| `--stage` | all | Which stage(s) to run: all, vis_lp (MGE fit only) or vis_pix (pixelized source only; requires a completed vis_lp result and stops with an error otherwise). --skip_pix is a deprecated alias of --stage vis_lp. |

`mask_radius` is **not** an argument — it is always read from the dataset's `info.json`.

Two environment variables change where results go:

- `PYAUTO_OUTPUT_DIR` — the results directory, relative to the repository root
  (default `output`). The SED chain uses this to keep its many per-band results
  out of the main tree; `catalogue/scripts/multi_wavelength.py` and
  `catalogue/scripts/magnitudes.py` read that separate tree by name.
- `PYAUTO_TEST_MODE` — `1` makes every search finish almost instantly with
  trivial samples, `2` additionally skips the sampler. Results land under
  `<output>/test_mode/`. Use it to check a script runs before submitting a real
  fit.

## Simulating a lens

`scripts/simulator.py` is this repository's **only** producer of simulated data — no
fitting script auto-simulates a missing dataset, because most users fit real Euclid
imaging and a silent auto-simulate would hide a broken data path. It writes an ordinary
dataset of this pipeline, so every fitting script reads it unchanged, plus a `truth.json`
recording every parameter, per-band flux, aperture flux, magnification and Einstein
radius that went in; the mock it produced and this repository ships is
`dataset/simulated/euclid_dr1_like/` (see [`dataset/README.md`](dataset/README.md) for
every dataset here and how to regenerate it). `python scripts/simulator.py --help` lists
the rest of the options — bands, image shape, pixel scale, mask radius, seed.

```bash
# --from-params: an analytic lens (Isothermal + shear mass, Sersic lens light,
# Sersic source) from the truth values at the top of the script
python scripts/simulator.py --from-params --output-dataset=my_lens

# --from-result: resimulate a fit you have already run. The tracer is rebuilt from
# that result's model.json + maximum-log-likelihood sample; the bands, PSFs,
# zero-points, WCS and noise come from the dataset it was fitted to.
python scripts/simulator.py --from-result \
    --sample=q1_walsmley --dataset=102018665_NEG570040238507752998 \
    --unique_tag=sersic_lens_model --search=vis \
    --output-dataset=102018665_resimulated
```

## Dataset Requirements

A dataset directory is `dataset/<sample>/<name>/` and holds `<name>.fits` (the
multi-HDU cut-out) and `info.json` (`pixel_scale`, `mask_radius`, optionally
`mask_centre`). Everything else is optional and degrades gracefully. Every
dataset this repository ships is listed in [`dataset/README.md`](dataset/README.md),
with what reads it and how to regenerate it.

The FITS **primary** header additionally carries the `WORST_BAND` / `WORST_PSF_*`
input contract that the aperture-flux latents need. It is documented where it is
read, in `util.py` — `load_vis_dataset` for `WORST_BAND` and the two degradation
paths, `psf_fwhm_arcsec_from_primary_header` for the FWHM key order and the `-99`
sentinel.

## Testing and Continuous Integration

```bash
python -m pytest -q -m "not slow"      # 58 tests, ~4 s, JAX-free, no fit — the local default
python -m pytest -q -m slow            # 3 tests, 10-20 s, one real (non-test-mode) fit
python -m pytest -q                    # both
python3 .github/scripts/run_smoke.py   # every script in smoke_tests.txt, under PYAUTO_TEST_MODE
```

Every pull request runs three CI jobs across two workflows, each on the
ubuntu x Python 3.12/3.13 matrix:

| Workflow | Job | What it runs | A red X means |
|---|---|---|---|
| `.github/workflows/smoke_tests.yml` | `smoke` | Every entry of `smoke_tests.txt`, under `PYAUTO_TEST_MODE` | A script broke. |
| `.github/workflows/tests.yml` | `unit` | `pytest -m "not slow"` | A latent value is wrong, a catalogue column drifted from the DR1 reference, or a repository invariant broke. |
| `.github/workflows/tests.yml` | `slow` | `pytest -m slow` | The pipeline stopped writing latents at all. |

Both workflows are thin callers into PyAutoHeart's reusable `smoke-tests.yml`,
which checks out and source-installs the five-library dependency chain. The smoke
job reports **every** failing script rather than stopping at the first, and uploads
its timings as the `smoke-timings-<python-version>` artifact.

A new script is not automatically covered: `tests/test_repo_invariants.py` fails
unless every `*.py` under `scripts/`, `catalogue/scripts/`, `preprocess/`, `tools/`,
`workflow/`, `.github/scripts/` and the repository root is either listed in
`smoke_tests.txt` or excluded in `config/build/no_run.yaml` with a written reason.
[`AGENTS.md`](AGENTS.md#continuous-integration) carries the runner details and the
three invariants CI enforces.

## Visualization

Every 2D figure the pipeline produces uses the **magma** colormap by default.
To change it, edit the `colormap` key of `config/visualize/general.yaml`:

```yaml
colormap: magma   # any matplotlib colormap name, or `autoarray` for the bundled PyAuto colormap
```

That one key covers imaging data, fits, residual maps and inversion
reconstructions. A single figure can be overridden without touching config by
passing `colormap=` to the plot function. See
[`config/visualize/README.md`](config/visualize/README.md) for the details.
