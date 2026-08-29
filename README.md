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
python start_here.py --sample=q1_walsmley --dataset=102018665_NEG570040238507752998 --iterations_per_quick_update=10000
```

The pipeline will run on the example dataset, outputting results to the `output` folder and in the `dataset` folder,
and it can be easily modified to run on your own data.

The pipeline above is parallelized automatically based on the hardware available to Python (GPU or CPU) and
results are output on-the-fly during the model fitting procedure every 10000 iterations, meaning you can watch
the lens model improve over time!

## Overview

The starting point for Euclid strong lens modeling is found in the `start_here.py` script. It performs
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

After running the `start_here.py` script on many lenses, you will begin to build up a large number of results
in the `output` folder. Eventually, manually inspecting these results will become tedious, and you will require an
efficient workflow to inspect the results and perform scientific analysis.

The `workflow` folder contains example scripts for creating workflows which enable efficient inspection of
large lens modeling results. Workflows are designed by creating .png, .csv and .fits files from the results
in the `output` folder for fast inspection.

## Additional Pipelines

The following additional pipelines are available in the repository in the `scripts` folder:

- `full_model.py`: A full pipeline which models the Source, Light and Mass using advanced featues like a pixelized source reconstruction and mass model more complex than SIE + shear.
- `lens_model_waveband.py`: After modeling the high resolution VIS imaging, model lower resolution NIR / EXT imaging using a fixed lens model.
- `sersic_lens_model.py`: After getting an initial lens model from VIS imaging, perform fits using Sersic lens and source models which give more accurate photometry for SED fitting.
- `mge_lens_only.py`: Multi-Gaussian Expansion (MGE) subtraction of the lens light only, which better reveals the lensed source.

All pipelines are run with the same API as the `start_here.py` script, for example:

```bash
python scripts/full_model.py --dataset=EUCLJ174517.55+655612.5 --iterations_per_quick_update=50000
```

**PyAutoLens** has automated pipelines for modeling group-scale strong lenses, lensed point sources (e.g. lensed quasars)
and double source plane lenses. These will be added to this repository in future releases, but if you are interested
in using these pipelines sooner please contact James Nightingale on the Euclid consortium SLACK.

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
