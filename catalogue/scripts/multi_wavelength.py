"""
Euclid Catalogue: Multi-wavelength Fit PNG
===========================================

New here? Read ``start_here.py`` for the concepts (the output layout, what the
fit subplots show) and ``catalogue/README.md`` for the run order.

Produces ``fit_multi_wavelength.png`` — one image per lens showing the fit in
every waveband on a single grid, so a reader can judge the multi-band model at a
glance instead of opening one subplot per band.

Four panels are cropped out of each waveband's ``subplot_fit.png``:

- the data,
- the lens-light model,
- the lens-light-subtracted image,
- the source model image.

They are laid out one **column** per waveband, ordered blue → red left to right
by ``WAVEBAND_ORDER``, and one row per panel — so reading across a row compares
the same panel in every band, which is what makes a band that failed obvious.

``--image_set=fit_x1_plane`` instead reads the single-plane subplot (data, model
data, lens-light-subtracted image, normalized residual map), which is what a
band fitted without a source model produces.

The SED chain is run with ``PYAUTO_OUTPUT_DIR=output_sed``, so this producer
reads ``output_sed`` by default rather than the main ``output``.

Stage 6 of ``scripts/build_inspection_bundle.sh``.

__What It Reads, And What It Does Not Do__

Nothing here re-renders a fit. The panels are cropped straight out of the
``subplot_fit.png`` that each band's search already wrote, so a band whose fit
was run with visualization suppressed — ``PYAUTO_FAST_PLOTS=1``, or a test-mode
run that skipped post-fit visualization — has no panels to contribute. That is
the same constraint ``scripts/tools/build_inspect.py`` works under at stage 1,
and ``catalogue/README.md`` describes it under "Building a bundle off a
test-mode run".

The upstream results are the SED chain's: ``scripts/sersic_lens_model_waveband.py``
fits every band under the ``sersic_lens_model`` tag with the searches named after
the bands, which is what makes one lens's bands siblings the aggregator can
gather. ``magnitudes.py`` turns the same results into numbers; this producer
turns them into a picture.

Usage
-----
    python catalogue/scripts/multi_wavelength.py --sample=q1_walsmley

    python catalogue/scripts/multi_wavelength.py \
        --sample=dr1_prelim_grade_ab \
        --inspect_dir=inspect/dr1_prelim_grade_ab_run250 \
        --output_path=output_sed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import catalogue_util


"""
__The Waveband Order__

The list below is the file's one piece of Euclid domain knowledge: the order the
composed image's columns appear in. It is written down rather than derived
because band *names* carry no wavelength — nothing about the string ``des_i``
says it sits between ``des_r`` and ``des_z``.

VIS leads, being the band the lens model was fitted on and the one every other
band's Sersic is seeded from. Then the ground-based optical surveys by filter
rather than grouped by survey, so a ``g`` from DES sits beside a ``g`` from
Pan-STARRS; then the three Euclid NIR bands. The inline comments give each
filter's approximate pivot wavelength, which is what the ordering is really
sorting on.

``deblending.py`` keeps its own, much shorter list for the same purpose. The two
are deliberately not shared: that one orders FITS extensions for the bands a
deblending bundle contains, this one orders image columns for every band an SED
run may have fitted, and a band belonging in one is not a reason to add it to the
other.
"""
# Waveband ordering for the composed image's columns: VIS first, then the ground-based
# optical surveys blue → red, then the Euclid NIR bands. Comments give the
# approximate pivot wavelength of each filter.
WAVEBAND_ORDER = [
    "vis",
    "vis_sersic",
    "cfis_u",  # ~< 400-450 nm — bluest (UV/blue)
    "des_g",  # ~475 nm
    "decam_g",
    "hsc_g",
    "panstarrs_g",  # ~481 nm
    "wishes_g",  # g-band, close to des_g
    "des_r",  # ~620-650 nm
    "decam_r",
    "hsc_r",
    "hsc_r2",
    "panstarrs_r",  # ~617 nm
    "cfis_r",  # r-band, close to des_r
    "des_i",  # ~750-780 nm
    "decam_i",
    "hsc_i",
    "hsc_i2",
    "panstarrs_i",  # ~752 nm
    "des_z",  # ~920-950 nm
    "decam_z",
    "hsc_z",
    "panstarrs_z",  # ~866 nm
    "wishes_z",  # z-band, close to des_z
    "panstarrs_y",  # ~962 nm
    "nir_h",  # near-IR H
    "nir_j",  # near-IR J
    "nir_y",  # near-IR Y
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write a per-lens multi-wavelength fit PNG collecting every "
            "waveband's fit subplot into one image."
        )
    )
    catalogue_util.add_common_arguments(parser, default_output_path="output_sed")
    parser.add_argument(
        "--unique_tag",
        metavar="name",
        default="sersic_lens_model",
        help="Pipeline stage the multi-band results were written under.",
    )
    parser.add_argument(
        "--image_set",
        choices=("lens_model", "fit_x1_plane"),
        default="lens_model",
        help=(
            "Which subplot panels to collect. 'lens_model' (default) reads "
            "subplot_fit; 'fit_x1_plane' reads subplot_fit_x1_plane."
        ),
    )
    return parser.parse_args()


def main():
    """
    __One Aggregator Per Lens__

    The CSV producers build a single aggregator over the whole sample and let it
    emit one row per result. This producer cannot: its output is one composed
    image per lens, and ``AggregateImages`` composes across everything in the
    aggregator it is given. So the loop below walks the sample's lens
    directories via ``catalogue_util.dataset_names_from`` and builds a fresh
    aggregator scoped to ``<sample_root>/<lens>``, whose contents are exactly
    that lens's band searches.

    That also makes the stage fail softly per lens rather than per sample: a lens
    with nothing to compose is caught, reported and skipped, and the rest of the
    sample still gets its images.

    Unlike ``deblending.py``, there is no already-built check here — every lens's
    PNG is rebuilt on each run, which is what makes a re-run pick up bands that
    have landed since the last one.
    """
    args = parse_args()
    output_path, inspect_path = catalogue_util.resolve_paths(args)

    import autofit as af
    import autolens as al
    from autofit.aggregator.aggregator import Aggregator

    sample_root = catalogue_util.sample_root_from(output_path, args.sample)
    dataset_name_list = catalogue_util.dataset_names_from(sample_root)

    for dataset_name in dataset_name_list:

        print(dataset_name)

        agg = Aggregator.from_directory(
            directory=sample_root / dataset_name, completed_only=True
        )

        agg_query = agg.query(agg.unique_tag == args.unique_tag)

        """
        __Ordering The Bands__

        The aggregator returns a lens's searches in whatever order it found them
        on disk, which has nothing to do with wavelength: band names sort by
        survey and letter, so ``nir_h`` lands before ``vis`` and the NIR bands
        are scattered through the optical ones.

        Sorting the aggregator's ``search_outputs`` in place by each band's index
        in ``WAVEBAND_ORDER`` fixes the column order of the composed image, so
        wavelength increases left to right and an SED trend can be read off the
        picture. The key returns ``len(WAVEBAND_ORDER)`` for a name not in the
        list, so an unrecognised band sorts to the end rather than raising or
        being dropped — a new survey filter appears in the image, just at the
        right-hand edge, until it is added to the list above.
        """
        # Reorder by WAVEBAND_ORDER; unknown wavebands sort to the end.
        agg_query.search_outputs = sorted(
            agg_query.search_outputs,
            key=lambda search_output: (
                WAVEBAND_ORDER.index(search_output.name)
                if search_output.name in WAVEBAND_ORDER
                else len(WAVEBAND_ORDER)
            ),
        )

        try:
            agg_image = af.AggregateImages(aggregator=agg_query)
        except ValueError as e:
            print(f"skipping {dataset_name}: {e}")
            continue

        """
        __Which Four Panels__

        ``al.agg.subplot_fit.*`` are coordinates into the subplot PNG the search
        wrote, not quantities to be recomputed: each names one panel of that
        grid, and ``extract_image`` crops it out.

        The default set tells the multi-band story in four steps — the data, the
        lens-light model, the data with that lens light subtracted, and the
        source model image — so a reader can see, band by band, whether the
        deblending worked and whether there is source flux left where the arcs
        are.

        ``--image_set=fit_x1_plane`` reads ``subplot_fit_x1_plane`` instead,
        which is the subplot a band fitted with no source model produces. Its
        panels are different (model data and a normalized residual map in place
        of the source model image), so the choice is per run rather than
        automatic: pick the one matching how the bands were fitted.
        """
        if args.image_set == "fit_x1_plane":
            subplots = [
                al.agg.subplot_fit_x1_plane.data,
                al.agg.subplot_fit_x1_plane.model_data,
                al.agg.subplot_fit_x1_plane.lens_light_subtracted_image,
                al.agg.subplot_fit_x1_plane.normalized_residual_map,
            ]
        else:
            subplots = [
                al.agg.subplot_fit.data,
                al.agg.subplot_fit.lens_light_model,
                al.agg.subplot_fit.lens_light_subtracted_image,
                al.agg.subplot_fit.source_model_image,
            ]

        """
        __Composing And Writing__

        ``extract_image`` lays the crops out as a matrix: one row per result,
        four panels across. ``transpose=True`` flips that, giving one column per
        waveband and one row per panel kind — bands increasing in wavelength left
        to right, the same panel repeated across each row. Panels from bands
        whose subplots differ in size are normalised to a common panel size
        before compositing, so the grid lines up.

        The image is written into the lens's own folder in the inspect directory,
        beside that lens's PNGs, FITS and CSV rows. There is no master version of
        this product: unlike the CSVs, it is per-lens by nature, so nothing is
        split afterwards.
        """
        image = agg_image.extract_image(subplots=subplots, transpose=True)

        output_dataset_path = inspect_path / dataset_name
        output_dataset_path.mkdir(parents=True, exist_ok=True)
        image.save(output_dataset_path / "fit_multi_wavelength.png")


if __name__ == "__main__":
    main()
