"""
Euclid Pipeline: Latent Variable Diagnostic (population sweep)
==============================================================

The population version of ``scripts/tools/diagnose_latent.py``. Where that script
interrogates one result in depth, this one sweeps the first N converged results
of a sample and prints the latent catalogue for each, so you can tell at a
glance whether a latent failure is *universal* (every tile — a code or config
problem) or *tile-specific* (a data or convergence problem).

It defaults to the ``initial_lens_model/vis_pix`` stage, which is where the
production latent failure was first seen: the pixelized fit replaces
``galaxies.source.bulge``, so any latent that reaches for a source light
profile behaves differently there than in ``vis_lp``.

No non-linear search is run. Each result's ``samples_summary.json`` +
``model.json`` are read from disk, the ``util.AnalysisImaging`` is rebuilt from
the dataset, and ``util.LatentEuclid.variables`` is evaluated on the
maximum-log-likelihood sample. Values are tagged ``<<NaN>>`` or
``<<ZERO sentinel>>``; a tile whose evaluation raises is reported with its
exception type (and, with ``--traceback``, the full traceback) and the sweep
continues, so one bad tile never truncates the population picture.

__Known limitation of the vis_pix stage__

A ``vis_pix`` model carries a ``Delaunay`` pixelization whose *image-plane mesh
grid* is not stored in ``model.json`` — ``scripts/initial_lens_model.py`` builds
it at run time from the preceding ``vis_lp`` result (Hilbert image mesh plus
circle-edge points, then hands it to the analysis as ``AdaptImages``).
Rebuilding the analysis from the dataset alone therefore cannot reconstruct the
inversion, and the tile is reported as::

    MeshException: The mesh `Delaunay` was not given an image-plane mesh grid …

That is a real, reproducible property of the stage rather than a latent bug, and
it is what a universal-failure sweep looks like. For a latent readout that
evaluates end to end, sweep the light-profile stage instead::

    python scripts/tools/diagnose_latent_vis_pix.py --search=vis_lp

The library latents that do not depend on the source reconstruction
(``magnification``, ``effective_einstein_radius``, the lens fluxes) are
identical between the two stages, because ``vis_pix`` holds the lens light as an
instance from ``vis_lp``.

Phase 2 of the DR1 prep epic puts unit tests and CI on the latent catalogue;
this script is the human-inspection route that stays useful for triaging a
real sample after a production run.

Usage
-----
Defaults target the shipped example dataset's sample::

    python scripts/tools/diagnose_latent_vis_pix.py

Sweep a real sample::

    python scripts/tools/diagnose_latent_vis_pix.py --sample=dr1_prelim_grade_ab --limit=5

Inspect a test-mode smoke run::

    python scripts/tools/diagnose_latent_vis_pix.py --output_path=output/test_mode
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


def parse_args():
    """
    Command-line arguments. The science-tree original hardcoded ``SAMPLE``,
    the stage, and a limit of 5; all three are arguments here.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Euclid latent catalogue across several converged "
            "results of a sample to characterise a latent failure pattern."
        )
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
        help="Pipeline stage folder holding the results. Default: initial_lens_model.",
    )
    parser.add_argument(
        "--search",
        metavar="name",
        default="vis_pix",
        help="Search name within the stage folder. Default: vis_pix.",
    )
    parser.add_argument(
        "--limit",
        metavar="int",
        type=int,
        default=5,
        help="Maximum number of datasets to inspect. Default: 5.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        default=False,
        help="Print the full traceback for a tile whose latents raise.",
    )
    return parser.parse_args()


def find_results(sample_root: Path, unique_tag: str, search: str, limit: int):
    """
    Return up to ``limit`` ``(dataset_name, result_hash)`` pairs for datasets
    under ``sample_root`` whose ``<unique_tag>/<search>`` stage has a result
    carrying both ``samples_summary.json`` and ``model.json``.
    """
    if not sample_root.is_dir():
        raise SystemExit(f"ERROR: no such results directory: {sample_root}")

    results = []
    for dataset_dir in sorted(sample_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        search_dir = dataset_dir / unique_tag / search
        if not search_dir.is_dir():
            continue
        for hash_dir in sorted(search_dir.iterdir()):
            files = hash_dir / "files"
            if (files / "samples_summary.json").exists() and (
                files / "model.json"
            ).exists():
                results.append((dataset_dir.name, hash_dir.name))
                break
        if len(results) >= limit:
            break
    return results


def diagnose(dataset_name, result_hash, sample_root, sample_name, unique_tag, search):
    """
    Evaluate the latent catalogue for one dataset. Returns ``("OK", values)``
    or ``("ERR", message)`` — never raises, so one bad tile cannot end the
    sweep.
    """
    from autofit import from_dict

    files_path = (
        sample_root / dataset_name / unique_tag / search / result_hash / "files"
    )

    with open(files_path / "samples_summary.json") as f:
        summary = from_dict(json.load(f))
    with open(files_path / "model.json") as f:
        model = from_dict(json.load(f))
    summary.model = model
    parameters = summary.max_log_likelihood_sample.parameter_lists_for_model(model)

    d = util.load_vis_dataset(dataset_name, sample_name=sample_name)

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

    keys = util.LatentEuclid.keys(analysis)

    try:
        values = util.LatentEuclid.variables(
            analysis=analysis, parameters=parameters, model=model
        )
        return "OK", (keys, values)
    except Exception as e:
        return "ERR", (f"{type(e).__name__}: {e}", traceback.format_exc())


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

    import numpy as np
    import autolens as al  # noqa: F401  needed so from_dict resolves al.* classes

    sample_name = args.sample if args.sample else None
    sample_root = output_path / sample_name if sample_name else output_path

    results = find_results(
        sample_root=sample_root,
        unique_tag=args.unique_tag,
        search=args.search,
        limit=args.limit,
    )
    print(
        f"[diag] inspecting {len(results)} {args.unique_tag}/{args.search} "
        f"results under {sample_root}\n"
    )

    n_ok = 0
    n_err = 0

    for dataset_name, result_hash in results:
        print(f"=== {dataset_name} ({result_hash}) ===")
        status, payload = diagnose(
            dataset_name=dataset_name,
            result_hash=result_hash,
            sample_root=sample_root,
            sample_name=sample_name,
            unique_tag=args.unique_tag,
            search=args.search,
        )
        if status == "ERR":
            n_err += 1
            message, formatted_traceback = payload
            print(f"  ERROR: {message}")
            if "MeshException" in message:
                print(
                    "  NOTE: the image-plane mesh grid of a vis_pix pixelization "
                    "is built at run time from the vis_lp result and is not stored "
                    "in model.json, so this stage cannot be rebuilt standalone. "
                    "Re-run with --search=vis_lp for an end-to-end latent readout."
                )
            if args.traceback:
                print(formatted_traceback)
            print()
            continue
        n_ok += 1
        keys, values = payload
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
        print()

    print(
        f"[diag] {n_ok} result(s) evaluated their latents, {n_err} raised "
        f"({args.unique_tag}/{args.search})."
    )


if __name__ == "__main__":
    main()
