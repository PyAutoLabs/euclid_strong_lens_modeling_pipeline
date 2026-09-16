"""
Euclid Catalogue: Lens Mass Maps FITS
=====================================

New here? Read ``start_here.py`` for the concepts (the model, the output layout,
what a fit writes) and ``catalogue/README.md`` for the run order.

Produces the three per-lens mass-map FITS files of the DR1 catalogue:

- ``convergence.fits``  — the lens mass model's convergence (surface density in
  units of the critical density).
- ``potential.fits``    — its lensing potential.
- ``deflections.fits``  — its deflection angles, two HDUs: ``DEFLECTIONS_Y``
  then ``DEFLECTIONS_X``.

These are the mass model made lookable-at: the same SIE + shear the numbers in
``lens_mass.csv`` describe, written as images a viewer can open in DS9 or a
downstream analysis can re-project, rather than as parameters someone else has
to re-evaluate a profile from.

__Collected, Never Recomputed__

Nothing here fits, evaluates a profile or loads a dataset. Every finished search
already wrote ``image/tracer.fits`` inside its result zip — the
``visualize.plots.tracer.fits_tracer`` output of
``PyAutoLens/autolens/lens/plot/tracer_plots.py::fits_tracer``, with extensions
``MASK``, ``CONVERGENCE``, ``POTENTIAL``, ``DEFLECTIONS_Y``, ``DEFLECTIONS_X``.
This producer selects HDUs out of that file and writes them back out split three
ways, the same idea as ``deblending.py``'s ``pre_psf.fits`` / ``model.fits``. A
lens's posterior is read-only to it: the aggregator opens each zip with
``unzip_temporary=True``, which extracts to a temporary directory that is
removed again, so no result tree is modified.

__The PyAutoFit Version This Needs__

The HDUs are collected with ``af.AggregateFITS.extract_fits``, the same call
``deblending.py`` makes. It requires PyAutoFit at or after the fix in PyAutoFit
PR ``feature/catalogue-mass-maps-fits``: earlier versions resolved the
``FITSTracer`` name ``"tracer"`` through ``SearchOutput.value``, which searches
JSONs first and so returned the ``files/tracer.json`` ``Tracer`` object instead
of ``image/tracer.fits``.

__The Grid The Maps Live On__

``fits_tracer`` evaluates the maps on the **zoomed** mask grid with a one-pixel
buffer (``aa.Zoom2D(mask=grid.mask)`` → ``mask_2d_from(buffer=1)``), not on the
cut-out's own grid. Their shape is therefore *not* the shape of the data or of
``model.fits`` — for the 100x100 DR1 cut-outs it is the zoom of the circular
mask plus a pixel. The header carries that zoomed mask's ``header_dict`` (pixel
scale and origin), which is what any re-projection must use; do not assume the
dataset's WCS applies pixel for pixel.

The step is idempotent: a lens whose three FITS files already exist is skipped,
so the bundle can be rebuilt as more fits land without redoing finished work.
The check is existence only — it does not compare timestamps — so after
*re-fitting* a lens, delete its three FITS files (or its whole folder) to force
them to be rebuilt.

Stage 3 of ``scripts/build_inspection_bundle.sh``.

__What It Needs Upstream, And Where It Looks__

The HDUs come from the FITS a finished ``initial_lens_model`` search wrote. The
default ``--search_name`` is ``vis_pix``, the pixelized-source stage whose mass
model is the one ``lens_mass.csv`` reports, so the maps in the bundle and the
parameters in the CSV describe the same tracer. ``vis_lp`` fitted the same lens
with a light-profile source and has its own ``tracer.fits``; pass
``--search_name=vis_lp`` to bundle that one instead. Only one search's maps are
written, so no ``extname_prefix_list`` is needed and the extension names are the
plain ``CONVERGENCE`` / ``POTENTIAL`` / ``DEFLECTIONS_Y`` / ``DEFLECTIONS_X``
the fit wrote.

A ``PYAUTO_TEST_MODE`` run writes no ``tracer.fits`` at all —
``autonerves.test_mode.skip_fit_output()`` gates the visualisation — so this
stage skips every lens of a test-mode tree, the same way the PNG stages do.

Usage
-----
    python catalogue/scripts/lens_mass_maps.py --sample=q1_walsmley

    python catalogue/scripts/lens_mass_maps.py \
        --sample=dr1_sep1 \
        --inspect_dir=inspect/dr1_sep1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import catalogue_util


def mass_map_products():
    """
    The file-and-HDU contract of this stage: output filename → the
    ``al.agg.fits_tracer`` members extracted into it, in extension order.

    Returned as a fresh ``dict`` per call (insertion-ordered, so it is also the
    order the files are written in), which is the whole of what stage 3
    produces. Keeping it here rather than inline in ``main`` means the contract
    can be asserted without an aggregator, a results tree or a fit —
    ``tests/test_lens_mass_maps.py`` does exactly that.

    ``autolens`` is imported inside the function, like every other library
    import in this producer, so importing this module stays free.
    """
    import autolens as al

    return {
        "convergence.fits": [al.agg.fits_tracer.convergence],
        "potential.fits": [al.agg.fits_tracer.potential],
        "deflections.fits": [
            al.agg.fits_tracer.deflections_y,
            al.agg.fits_tracer.deflections_x,
        ],
    }


def mass_maps_exist(output_dataset_path: Path, filename_list) -> bool:
    """
    Whether every mass-map file of this lens is already present, which is the
    skip at the top of the loop.

    Existence only, deliberately: see the module docstring on re-fitting.
    """
    return all(
        (Path(output_dataset_path) / filename).exists() for filename in filename_list
    )


def write_mass_maps(agg_fits, output_dataset_path: Path, products=None):
    """
    Write one FITS file per entry of ``products`` out of ``agg_fits``.

    Parameters
    ----------
    agg_fits
        Anything exposing ``extract_fits(hdus=...)`` — in production the
        ``af.AggregateFITS`` built around one lens's aggregator, in the tests a
        stub returning synthetic HDUs, which is why the writer takes the object
        rather than building it.
    output_dataset_path
        This lens's folder inside the inspection bundle; created if absent.
    products
        The filename → HDU-member mapping; ``mass_map_products()`` by default.

    Returns
    -------
    The paths written, in the order they were written.

    ``overwrite=True`` matters only when some of the three exist and the rest do
    not: the skip above would not have fired, and the survivors are replaced
    rather than erroring. No ``extname_prefix_list`` is passed — one search per
    file, so the extension names the fit wrote are already unambiguous.
    """
    products = mass_map_products() if products is None else products

    output_dataset_path = Path(output_dataset_path)
    output_dataset_path.mkdir(parents=True, exist_ok=True)

    written = []
    for filename, hdus in products.items():
        hdu_list = agg_fits.extract_fits(hdus=hdus)
        target = output_dataset_path / filename
        hdu_list.writeto(target, overwrite=True)
        written.append(target)

    return written


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write per-lens convergence.fits, potential.fits and "
            "deflections.fits mass maps for a sample."
        )
    )
    catalogue_util.add_common_arguments(parser)
    parser.add_argument(
        "--unique_tag",
        metavar="name",
        default="initial_lens_model",
        help="Pipeline stage the results were written under.",
    )
    parser.add_argument(
        "--search_name",
        metavar="name",
        default="vis_pix",
        help=(
            "Keep only fits whose search.name matches. Default: 'vis_pix', the "
            "stage whose mass model lens_mass.csv reports."
        ),
    )
    return parser.parse_args()


def main():
    """
    __One Aggregator Per Lens, And The Skip__

    Like ``deblending.py``, this producer loops over lens directories and builds
    an aggregator scoped to each: the output is one set of FITS files per lens,
    so the aggregator handed to ``AggregateFITS`` must contain that lens and
    nothing else.

    Failures are per lens too. A lens with no completed search under the tag and
    search name makes ``af.AggregateFITS`` raise ``ValueError`` ("The aggregator
    is empty."), and one whose fit wrote no ``tracer.fits`` raises
    ``FileNotFoundError`` at extraction; both are caught, reported and skipped so
    the rest of the sample still gets bundled — that is what lets a sample still
    being fitted produce a partial bundle rather than a broken one.
    """
    args = parse_args()
    output_path, inspect_path = catalogue_util.resolve_paths(args)

    import autofit as af
    from autofit.aggregator.aggregator import Aggregator

    products = mass_map_products()

    sample_root = catalogue_util.sample_root_from(output_path, args.sample)
    dataset_name_list = catalogue_util.dataset_names_from(sample_root)

    for dataset_name in dataset_name_list:

        output_dataset_path = inspect_path / dataset_name

        # Idempotency: skip a lens whose three mass maps are already present.
        if mass_maps_exist(output_dataset_path, products):
            continue

        print(dataset_name)

        agg = Aggregator.from_directory(
            directory=sample_root / dataset_name,
            completed_only=True,
            unzip_temporary=True,
        )

        agg_query = agg.query(agg.unique_tag == args.unique_tag)
        if args.search_name is not None:
            agg_query = agg_query.query(agg_query.search.name == args.search_name)

        try:
            agg_fits = af.AggregateFITS(aggregator=agg_query)
            write_mass_maps(agg_fits, output_dataset_path, products=products)
        except (ValueError, FileNotFoundError) as e:
            print(f"skipping {dataset_name}: {e}")
            continue


if __name__ == "__main__":
    main()
