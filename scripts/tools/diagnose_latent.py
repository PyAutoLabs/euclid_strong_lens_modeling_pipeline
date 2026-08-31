"""
Euclid Pipeline: Latent Variable Diagnostic (single result)
============================================================

Human-inspection tool for the Euclid latent catalogue. Run it when a fit's
``latent_summary.json`` is missing, empty, or full of NaNs and you need to see
*which* latent is misbehaving and *why*.

It takes one converged result on disk (``initial_lens_model/vis_lp`` by
default), rebuilds the ``util.AnalysisImaging`` that produced it, and evaluates
the latent catalogue directly on the maximum-log-likelihood sample. No
non-linear search is run — this is a read-only replay of the latent code path,
so it takes seconds rather than hours.

Three passes are made:

1. ``util.LatentEuclid.keys`` / ``util.LatentEuclid.variables`` on the max-L
   sample — every latent is printed with its value, tagged ``<<NaN>>`` or
   ``<<ZERO sentinel>>`` where suspicious. Exceptions are caught and the full
   traceback printed, so a single broken latent does not hide the rest.
2. ``LensCalc.einstein_radius_from`` in isolation — historically the latent most
   likely to raise, and the reason this script exists. The library latent
   (``effective_einstein_radius``) swallows ``ValueError``/``AttributeError``
   into a NaN; here the exception is allowed through so the traceback is visible.
3. ``LensCalc.einstein_radius_list_from`` for per-critical-curve context on that
   failure — a lens with several tangential critical curves sums them, which is
   the usual explanation for a surprisingly large Einstein radius.

The latent catalogue itself is ``util.LatentEuclid``: the config-enabled subset
of the PyAutoLens library latents (see ``config/latent.yaml``) followed by the
four Euclid-only FWHM aperture-flux µJy latents. Phase 2 of the DR1 prep epic
adds unit tests and CI over those latents; this script and its sibling
``diagnose_latent_vis_pix.py`` are the *human* inspection route and stay useful
for one-off triage of a specific tile.

Usage
-----
Defaults target the shipped example dataset::

    python scripts/tools/diagnose_latent.py

Point it at a real run::

    python scripts/tools/diagnose_latent.py \
        --sample=dr1_prelim_grade_ab \
        --dataset=Tile102007899RA0631694872236DECNEG0650584220817 \
        --search=vis_lp

Test-mode results are written under ``<output>/test_mode/``, so inspect a smoke
run with::

    python scripts/tools/diagnose_latent.py --output_path=output/test_mode
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import util


DEFAULT_SAMPLE = "q1_walsmley"
DEFAULT_DATASET = "102018665_NEG570040238507752998"


def parse_args():
    """
    Command-line arguments. Every value the science-tree original hardcoded
    (``SAMPLE`` / ``DATASET`` / ``RESULT_HASH``) is an argument here.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Replay the Euclid latent catalogue on one converged result and "
            "report which latents are NaN or raising."
        )
    )
    parser.add_argument(
        "--dataset",
        metavar="name",
        default=DEFAULT_DATASET,
        help=(
            "Dataset (tile) subdirectory name. Default: the shipped example "
            f"dataset '{DEFAULT_DATASET}'."
        ),
    )
    parser.add_argument(
        "--sample",
        metavar="name",
        default=DEFAULT_SAMPLE,
        help=(
            "Sample subdirectory inside dataset/ and the output dir. Pass an "
            f"empty string for a flat layout. Default: '{DEFAULT_SAMPLE}'."
        ),
    )
    parser.add_argument(
        "--output_path",
        metavar="path",
        default=None,
        help=(
            "Root results directory. Default: $PYAUTO_OUTPUT_DIR or 'output'. "
            "Test-mode runs land in '<output>/test_mode'."
        ),
    )
    parser.add_argument(
        "--unique_tag",
        metavar="name",
        default="initial_lens_model",
        help="Pipeline stage folder holding the result. Default: initial_lens_model.",
    )
    parser.add_argument(
        "--search",
        metavar="name",
        default="vis_lp",
        help="Search name within the stage folder (e.g. vis_lp, vis_pix, vis).",
    )
    parser.add_argument(
        "--result_hash",
        metavar="hash",
        default=None,
        help=(
            "Result hash subdirectory. Default: the most recently modified "
            "hash directory that contains samples_summary.json + model.json."
        ),
    )
    return parser.parse_args()


def resolve_files_path(search_dir: Path, result_hash: str = None) -> Path:
    """
    Locate the ``files/`` directory of a converged result.

    With ``result_hash`` given this is a direct lookup; without it the newest
    hash directory holding both ``samples_summary.json`` and ``model.json`` is
    used, which is what a reader inspecting "the last run" wants.
    """
    if result_hash is not None:
        return search_dir / result_hash / "files"

    if not search_dir.is_dir():
        raise SystemExit(f"ERROR: no such search directory: {search_dir}")

    candidates = [
        hash_dir / "files"
        for hash_dir in sorted(search_dir.iterdir())
        if hash_dir.is_dir()
        and (hash_dir / "files" / "samples_summary.json").exists()
        and (hash_dir / "files" / "model.json").exists()
    ]
    if not candidates:
        raise SystemExit(
            f"ERROR: no converged result (samples_summary.json + model.json) "
            f"under {search_dir}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def print_latents(keys, values):
    """
    Print each latent key/value pair, flagging NaN and exact-zero values —
    the two failure signatures the production latent bug produced.
    """
    import numpy as np

    for key, value in zip(keys, values):
        tag = ""
        try:
            value_float = float(value)
            if np.isnan(value_float):
                tag = " <<NaN>>"
            elif value_float == 0.0:
                tag = " <<ZERO sentinel>>"
        except (TypeError, ValueError):
            pass
        print(f"  {key}: {value}{tag}")


def main():
    args = parse_args()

    from autolens import conf

    project_root = Path(__file__).parent.parent.parent
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else project_root / os.environ.get("PYAUTO_OUTPUT_DIR", "output")
    )
    if not output_path.is_absolute():
        output_path = project_root / output_path

    conf.instance.push(new_path=project_root / "config", output_path=output_path)

    import autolens as al  # noqa: F401  needed so from_dict resolves al.* classes
    from autofit import from_dict

    sample_name = args.sample if args.sample else None

    d = util.load_vis_dataset(args.dataset, sample_name=sample_name)

    search_dir = output_path
    if sample_name is not None:
        search_dir = search_dir / sample_name
    search_dir = search_dir / args.dataset / args.unique_tag / args.search

    files_path = resolve_files_path(search_dir, result_hash=args.result_hash)
    print(f"[diag] result: {files_path}")

    with open(files_path / "samples_summary.json") as f:
        summary = from_dict(json.load(f))
    with open(files_path / "model.json") as f:
        model = from_dict(json.load(f))
    summary.model = model

    max_log_likelihood_sample = summary.max_log_likelihood_sample
    print(f"[diag] max_log_likelihood = " f"{max_log_likelihood_sample.log_likelihood}")
    print(f"[diag] model has {len(model.paths)} priors")

    parameters = max_log_likelihood_sample.parameter_lists_for_model(model)
    print(f"[diag] max-L parameter vector length = {len(parameters)}")

    analysis = util.AnalysisImaging(
        dataset=d.dataset,
        positions_likelihood_list=d.positions_likelihood_list,
        use_jax=False,
        dataset_main_path=d.dataset_main_path,
        title_prefix="VIS",
        plot_rgb=False,
        skip_rgb_plot=True,
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        magzero=d.magzero,
    )

    print("\n[diag] === pass (a) LatentEuclid on the max-L sample ===")
    keys = util.LatentEuclid.keys(analysis)
    print(f"[diag] {len(keys)} latent keys enabled: {keys}")
    try:
        values = util.LatentEuclid.variables(
            analysis=analysis, parameters=parameters, model=model
        )
        print(f"[diag] returned {len(values)} values")
        print_latents(keys, values)
    except Exception:
        print("[diag] LatentEuclid.variables raised:")
        traceback.print_exc()

    # The Einstein-radius latent is the one that historically raised, so it is
    # also exercised on its own — via the same `LensCalc` NumPy path the
    # library latent takes (`autolens.analysis.latent.effective_einstein_radius`),
    # which swallows ValueError/AttributeError into a NaN. Here it is left to
    # raise so the traceback is visible.
    from autogalaxy.operate.lens_calc import LensCalc

    print("\n[diag] === pass (b) einstein_radius_from in isolation ===")
    instance = model.instance_from_vector(vector=parameters)
    fit = analysis.fit_from(instance=instance)
    lens_calc = LensCalc.from_mass_obj(fit.tracer)
    grid = d.dataset.grids.lp
    print(f"[diag] grid shape = {grid.shape}, pixel_scales = {d.dataset.pixel_scales}")
    try:
        print(f"[diag] einstein_radius = {lens_calc.einstein_radius_from(grid=grid)}")
    except Exception:
        print("[diag] einstein_radius_from raised:")
        traceback.print_exc()

    print("\n[diag] === pass (b') einstein_radius_list_from for context ===")
    try:
        print(
            f"[diag] einstein_radius_list = "
            f"{lens_calc.einstein_radius_list_from(grid=grid)}"
        )
    except Exception:
        print("[diag] einstein_radius_list_from raised:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
