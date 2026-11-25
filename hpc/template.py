"""
Template: HPC Pipeline Script
=============================

This template lives in `hpc/`, paired with the CPU and GPU batch submit templates
(`hpc/batch_cpu/template`, `hpc/batch_gpu/template`). It provides the standard interface
between those batch scripts and PyAutoLens modeling code: command-line argument parsing,
configuration setup, and dataset loading, so that the same script runs identically on a
local machine and on the HPC.

Copy this file into `scripts/` and rename it for your science case (e.g. `scripts/imaging.py`),
then fill in the `Model, Analysis & Search` section of `fit()` using scripts from
`autolens_workspace/scripts/` as a reference. The batch templates and the campaign stage
entry (`hpc/batch_cpu/stage`, `STAGE_SCRIPT` in a campaign.yaml) run `scripts/<name>.py`,
so the copy belongs in `scripts/`. The HPC interface (`parse_fit_args`, `__main__`,
`--setup`, `--number_of_cores`) must be preserved — the batch scripts depend on it.

The Euclid pipeline's own stage scripts (`scripts/initial_lens_model.py`,
`scripts/sersic_lens_model_waveband.py`) follow the same interface through
`util.parse_fit_args` and load their data with `util.load_vis_dataset`; read them for a
worked example. Each section below is introduced by a triple-quoted `__Section__`
docstring, as in those scripts. Keep that style when you adapt it.

__Contents__

- **Imports:** Import the required libraries.
- **Configuration:** Push the project `config/` and `output/` paths.
- **Dataset:** Load data, noise-map, PSF and metadata from `info.json`.
- **Settings:** Build `SettingsSearch` (output path prefix, unique tag).
- **Model, Analysis & Search:** Science-specific model-fit — fill this in.
- **HPC Interface:** `parse_fit_args()` and `__main__` — leave unchanged.

__HPC Interface (usage)__

GPU batch scripts call:

    python3 scripts/imaging.py --sample=<sample> --dataset=<dataset>

CPU batch scripts call:

    python3 scripts/imaging.py --sample=<sample> --dataset=<dataset> --setup=cpu --number_of_cores=auto

`--setup <auto|setup_id>`, default `auto`, is the compute setup handed to the analysis
(`al.AnalysisImaging(..., setup=...)`): `auto` chooses the designated setup for the
likelihood and the machine when the fit starts (backend, device, operator, precision),
and a setup_id such as `numba_cpu_sparse_fp64` is used as given. It also accepts the
device intents `cpu` and `gpu` — the values `PYAUTO_SETUP` takes — which choose the
designated setup for that hardware class. It replaces `--use_cpu` and the hand-set
`use_jax`.

`--number_of_cores auto`, the default, is resolved by PyAutoFit when the fit starts: one
process for a JAX analysis (JAX parallelises inside the likelihood and never forks), and
the cores the job may use (`SLURM_CPUS_PER_TASK`) for a NumPy/Numba one. An int is
honoured, except that a JAX analysis still runs one process.

Each fit logs what it chose as its `Setup:` and `Parallel:` lines.
"""

"""
__Imports__

Standard library helpers plus the PyAuto* stack: `autofit` for the model and search,
`autolens` for lensing objects, and `autonerves` for configuration.
"""
import json
import argparse
import os
from pathlib import Path

import autofit as af
import autolens as al
from autonerves import conf


def fit(
    dataset_name: str,
    sample_name: str = None,
    iterations_per_quick_update: int = 5000,
    number_of_cores="auto",
    setup: str = "auto",
):
    """
    Fit a lens model to a single dataset.

    This function is called once per dataset, either from ``__main__`` (local)
    or from a SLURM array task (HPC). Add your science-specific model,
    analysis, and search code in the ``Model, Analysis & Search`` section below.

    Parameters
    ----------
    dataset_name
        Name of the dataset subdirectory inside ``dataset/<sample_name>/``.
    sample_name
        Name of the sample subdirectory inside ``dataset/``.
    iterations_per_quick_update
        Nautilus iterations between on-the-fly visualisation updates.
    number_of_cores
        Nautilus worker processes: ``"auto"`` (default) or an int.
    setup
        The compute setup: ``"auto"`` (default), ``"cpu"``, ``"gpu"`` or a setup_id.
        Pass ``analysis_setup_from(setup)`` to the analysis.
    """

    """
    __Configuration__

    Point PyAutoNerves at the project's `config/` YAML files and `output/` directory, so the
    same paths resolve identically on a local machine and on the HPC.
    """
    project_root = Path(__file__).parent.parent

    conf.instance.push(
        new_path=project_root / "config",
        output_path=project_root / "output",
    )

    """
    __Dataset__

    Load the imaging data, noise-map and PSF for this lens. All dataset-specific values
    (`pixel_scale`, `mask_radius`, redshifts, ...) come from the dataset's `info.json` via
    `info.get(key, default)`, so nothing is hard-coded here. Loading and masking are handled
    by `al.Imaging.from_fits` and `al.Mask2D.circular`
    (`PyAutoArray:autoarray/dataset/imaging/dataset.py`, `PyAutoArray:autoarray/mask/mask_2d.py`).
    """
    dataset_path = project_root / "dataset"

    if sample_name is not None:
        dataset_path = dataset_path / sample_name

    dataset_path = dataset_path / dataset_name

    info_path = dataset_path / "info.json"

    with open(info_path) as f:
        info = json.load(f)

    pixel_scale = info.get("pixel_scale", 0.05)
    mask_radius = info.get("mask_radius", 3.5)
    redshift_lens = info.get("redshift_lens", 0.5)
    redshift_source = info.get("redshift_source", 1.0)

    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        psf_path=dataset_path / "psf.fits",
        pixel_scales=pixel_scale,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )

    dataset = dataset.apply_mask(mask=mask)

    """
    __Settings__

    `SettingsSearch` controls the output path prefix (so each dataset writes to its own
    subdirectory) and the unique tag that identifies this model-fit
    (`PyAutoFit:autofit/non_linear/settings.py`).
    """
    settings_search = af.SettingsSearch(
        path_prefix=(
            Path(sample_name) / dataset_name
            if sample_name is not None
            else Path(dataset_name)
        ),
        unique_tag="initial_lens_model",
        info=None,
        session=None,
    )

    """
    __Model, Analysis & Search__

    Fill in the science-specific model-fit here, using scripts from
    `autolens_workspace/scripts/` as a reference. The four steps are:

    1. **Model** — define the lens light, mass and source light with `af.Model` /
       `af.Collection`.
    2. **Analysis** — create
       `al.AnalysisImaging(dataset=dataset, setup=analysis_setup_from(setup))`; the
       resolved setup chooses the backend and attaches any operator it names.
    3. **Search** — create `af.Nautilus` with `number_of_cores` threaded into the
       search via `{**settings_search.search_dict, "number_of_cores": number_of_cores}`;
       PyAutoFit resolves `"auto"` per analysis.
    4. **Fit** — run `search.fit(model=model, analysis=analysis, **settings_search.fit_dict)`.

    Outputs land under `output/<path_prefix>/<name>/<unique_id>/`.
    """
    raise NotImplementedError(
        "Replace this with your science-specific model, analysis, and search steps. "
        "See autolens_workspace/scripts/ for examples."
    )


"""
__HPC Interface__

`parse_fit_args()` reads the command-line arguments shared by every pipeline script and
`__main__` wires them into `fit()`. The CPU and GPU batch scripts depend on this interface —
leave it unchanged.
"""


SETUP_INTENTS = ("auto", "cpu", "gpu")


def setup_arg(value):
    """Validate ``--setup``: ``auto``, ``cpu``, ``gpu`` or a setup_id."""
    text = str(value).strip().lower()
    if text in SETUP_INTENTS:
        return text
    try:
        al.Setup.from_setup_id(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--setup {value!r} is not valid: {e}")
    return text


def number_of_cores_arg(value):
    """Validate ``--number_of_cores``: ``auto`` or a whole number >= 1."""
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if text.isdigit() and int(text) >= 1:
        return int(text)
    raise argparse.ArgumentTypeError(
        f"--number_of_cores {value!r} is not valid: use auto or a whole number >= 1"
    )


def analysis_setup_from(setup):
    """
    The analysis ``setup`` input for a ``--setup`` value.

    The analyses take ``"auto"`` or a setup_id; the device intents ``cpu`` / ``gpu``
    are read by the libraries from ``PYAUTO_SETUP``, so they are exported there and
    ``"auto"`` is returned.
    """
    if setup in ("cpu", "gpu"):
        os.environ["PYAUTO_SETUP"] = setup
        return "auto"
    return setup


def parse_fit_args():
    """
    Parse command-line arguments shared by all pipeline scripts.

    Returns
    -------
    tuple
        (sample_name, dataset_name, iterations_per_quick_update, number_of_cores, setup)
    """
    parser = argparse.ArgumentParser(description="PyAutoLens HPC Pipeline")

    parser.add_argument(
        "--sample",
        metavar="name",
        required=False,
        default=None,
        help="Sample subdirectory inside dataset/ containing the dataset.",
    )
    parser.add_argument(
        "--dataset",
        metavar="name",
        required=True,
        help="Name of the dataset subdirectory inside dataset/<sample>/.",
    )
    parser.add_argument(
        "--iterations_per_quick_update",
        metavar="int",
        required=False,
        default=5000,
        help="Number of sampler iterations between on-the-fly visualisation updates.",
    )
    parser.add_argument(
        "--number_of_cores",
        metavar="auto|int",
        type=number_of_cores_arg,
        required=False,
        default="auto",
        help="Nautilus worker processes: 'auto' (default) or an int.",
    )
    parser.add_argument(
        "--setup",
        metavar="auto|cpu|gpu|setup_id",
        type=setup_arg,
        required=False,
        default="auto",
        help="Compute setup of the analysis (the values PYAUTO_SETUP accepts).",
    )

    args = parser.parse_args()

    return (
        args.sample,
        args.dataset,
        int(args.iterations_per_quick_update),
        args.number_of_cores,
        args.setup,
    )


if __name__ == "__main__":
    sample_name, dataset_name, iterations_per_quick_update, number_of_cores, setup = (
        parse_fit_args()
    )

    fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
        number_of_cores=number_of_cores,
        setup=setup,
    )
