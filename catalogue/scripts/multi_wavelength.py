"""
Euclid Catalogue: Multi-wavelength Fit PNG
===========================================

Produces ``fit_multi_wavelength.png`` — one image per lens showing the fit in
every waveband on a single grid, so a reader can judge the multi-band model at a
glance instead of opening one subplot per band.

Four panels are taken from each waveband's ``subplot_fit.png`` and stacked, one
row per band, ordered blue → red by ``WAVEBAND_ORDER``:

- the data,
- the lens-light model,
- the lens-light-subtracted image,
- the source model image.

``--image_set=fit_x1_plane`` instead reads the single-plane subplot (data, model
data, lens-light-subtracted image, normalized residual map), which is what a
band fitted without a source model produces.

The SED chain is run with ``PYAUTO_OUTPUT_DIR=output_sed``, so this producer
reads ``output_sed`` by default rather than the main ``output``.

Stage 6 of ``scripts/build_inspection_bundle.sh``.

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


# Waveband ordering for the stacked rows: VIS first, then the ground-based
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

        image = agg_image.extract_image(subplots=subplots, transpose=True)

        output_dataset_path = inspect_path / dataset_name
        output_dataset_path.mkdir(parents=True, exist_ok=True)
        image.save(output_dataset_path / "fit_multi_wavelength.png")


if __name__ == "__main__":
    main()
