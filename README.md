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
automated lens modeling in around 10 minutes per lens on a GPU, around 20 minutes on an 8 core CPU.

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

## Workflow

After running `scripts/initial_lens_model.py` on many lenses, you will begin to build up a large number of results
in the `output` folder. Eventually, manually inspecting these results will become tedious, and you will require an
efficient workflow to inspect the results and perform scientific analysis.

The `workflow` folder contains example scripts for creating workflows which enable efficient inspection of
large lens modeling results. Workflows are designed by creating .png, .csv and .fits files from the results
in the `output` folder for fast inspection.

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
| `scripts/initial_lens_model.py` | **The entry point.** SIE + shear mass, MGE lens light, MGE source (`vis_lp`), then a pixelized Delaunay source (`vis_pix`). `--skip_pix` stops after `vis_lp`. | — |
| `scripts/sersic_lens_model.py` | Sersic lens and source with the mass model fixed, for accurate SED photometry. | `initial_lens_model` `vis_lp` |
| `scripts/lens_model_waveband.py` | Every non-VIS band (NIR / EXT) with the VIS lens model held fixed. | `initial_lens_model` or `sersic_lens_model` |
| `scripts/sersic_lens_model_waveband.py` | The **SED chain** driver: runs `initial_lens_model --skip_pix`, then `sersic_lens_model`, then `lens_model_waveband` over every band. Run it under its own `PYAUTO_OUTPUT_DIR`. | — (drives the three above) |
| `scripts/mge_lens_only.py` | MGE subtraction of the lens light only, revealing the lensed source quickly. | — |
| `scripts/full_model.py` | The full SLaM chain: MGE source, two Delaunay pixelized-source stages, refined lens light, then a PowerLaw + shear mass model. Uses `al.mesh.Delaunay` with `al.reg.AdaptSplit` (`reg.Adapt` cannot JIT on the Delaunay family) and `pixels=500` in its second stage — the `autolens_workspace` `delaunay.py` example uses 1000, but Euclid VIS cut-outs are small. | — |

### Diagnostics and catalogue

| Script | What it does |
|---|---|
| `scripts/diagnose_latent.py` | Replays the Euclid latent catalogue on one converged result and prints every latent value, flagging NaN and zero sentinels. Runs no search. |
| `scripts/diagnose_latent_vis_pix.py` | The population version: the same replay over every `vis_pix` result in a sample, reporting per-dataset OK/ERR. |
| `scripts/build_inspect.py` | Collects the inspection bundle's PNGs out of finished result zips. |
| `scripts/build_inspection_bundle.sh` | Runs all seven catalogue stages in order for a sample. |
| `catalogue/` | The producers that turn finished fits into the per-lens inspection bundle and the master CSVs — see [`catalogue/README.md`](catalogue/README.md) for the 13-file to producer table and the run order. |

### Simulating a lens

`scripts/simulator.py` is this repository's **only** producer of simulated data — no
fitting script auto-simulates a missing dataset, because most users fit real Euclid
imaging and a silent auto-simulate would hide a broken data path. It writes an
ordinary dataset of this pipeline, so every fitting script reads it unchanged, plus a
`truth.json` recording every parameter, per-band flux, aperture flux, magnification
and Einstein radius that went in.

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

`python scripts/simulator.py --help` lists the rest (bands, image shape, pixel scale,
mask radius, seed). Under `PYAUTO_TEST_MODE` the output goes to
`$PYAUTO_OUTPUT_DIR/simulator/` instead of `dataset/`, so a smoke run can never
overwrite a committed dataset; `--force-dataset-dir` writes to `dataset/` anyway.

The mock this repository ships is `dataset/simulated/euclid_dr1_like/`. See
[`dataset/README.md`](dataset/README.md) for every dataset here, what it is, what
reads it and how to regenerate it.

Some scripts used for the DR1 science runs were deliberately left out of this
repository — see
[`AGENTS.md`](AGENTS.md#not-ported--available-in-scienceeuclid) for the list and a
reason for each, and [`docs/drift_report.md`](docs/drift_report.md) for the full
record of what this pipeline inherited from the DR1 runs and why.

**PyAutoLens** has automated pipelines for modeling group-scale strong lenses, lensed point sources (e.g. lensed quasars)
and double source plane lenses. These will be added to this repository in future releases, but if you are interested
in using these pipelines sooner please contact James Nightingale on the Euclid consortium SLACK.

## Command-Line Arguments

All fitting pipelines share one argument parser (`util.parse_fit_args`):

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | *required* | Dataset subdirectory inside `dataset/<sample>/`. |
| `--sample` | none | Sample subdirectory inside `dataset/`. Omit for a flat `dataset/<name>/` layout. |
| `--iterations_per_quick_update` | `5000` | Sampler iterations between on-the-fly visualisation updates. |
| `--number_of_cores` | `1` | CPU cores for the non-JAX Nautilus searches. |
| `--use_cpu` | off | CPU mode: disables JAX and applies the CPU sparse operator to the pixelized stage. |
| `--skip_pix` | off | Return after the MGE light-profile fit, skipping the pixelized source stage. Used by the Sersic chain, whose source prior cannot be seeded from a pixelization. |

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

## Dataset Requirements

A dataset directory is `dataset/<sample>/<name>/` and holds `<name>.fits` (the
multi-HDU cut-out) and `info.json` (`pixel_scale`, `mask_radius`, optionally
`mask_centre`). Everything else is optional and degrades gracefully. Every
dataset this repository ships is listed in [`dataset/README.md`](dataset/README.md),
with what reads it and how to regenerate it.

### The `WORST_BAND` / `WORST_PSF_*` header contract

The four aperture-flux latent variables (`total_lens_flux_{1,2,3,4}_fwhm_mujy`)
are matched-aperture photometry: the lens image is convolved to the resolution of
the **worst-seeing** band across all MER bands, and fluxes are measured at 1, 2, 3
and 4 times that band's PSF FWHM. Two things in the FITS **primary** header make
that possible, and both are stamped by the upstream Euclid cut-out generator —
neither this pipeline nor PyAutoReduce writes them, so they are an input contract
on the dataset:

- **`WORST_BAND`** names the worst-seeing band (e.g. `DES_G`). Lower-cased it
  indexes the HDU list to find that band's PSF.
- **`WORST_PSF_MER`**, **`WORST_PSF_HDR`**, **`WORST_PSF`** hold that PSF's FWHM
  in arcsec. They are read in that order — the OU-MER measured value first, then
  the cut-out pipeline's own — skipping Euclid's `-99` "not measured" sentinel.

What happens when they are absent:

- **`WORST_BAND` missing, or naming a band not in the cut-out** — a warning is
  printed and the four aperture latents are skipped for that dataset. The fit
  itself is unaffected.
- **`WORST_BAND` present but all three FWHM keys missing or `-99`** — the load
  **raises**. This is deliberate: the aperture radii are multiples of this FWHM,
  so a guessed value would silently corrupt the photometry rather than fail.

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
[`AGENTS.md`](AGENTS.md#continuous-integration) has the full CI section.

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

## Documentation

The following links are useful for anyone more interested in the **PyAutoLens** software:

- [The PyAutoLens readthedocs](https://pyautolens.readthedocs.io/en/latest): which includes [an overview of PyAutoLens's core features](https://pyautolens.readthedocs.io/en/latest/overview/overview_1_start_here.html), [a new user starting guide](https://pyautolens.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html) and [an installation guide](https://pyautolens.readthedocs.io/en/latest/installation/overview.html).
- [The introduction Jupyter Notebook on Colab](https://colab.research.google.com/github/PyAutoLabs/autolens_workspace/blob/2026.7.9.1/start_here.ipynb): try **PyAutoLens** in a web browser (without installation).
- [The autolens_workspace GitHub repository](https://github.com/PyAutoLabs/autolens_workspace): example scripts and the HowToLens Jupyter notebook lectures.
