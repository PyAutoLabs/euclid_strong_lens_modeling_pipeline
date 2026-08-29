"""
Euclid Catalogue: Deblended Galaxy Images FITS
===============================================

Produces the two per-lens FITS bundles of the DR1 catalogue:

- ``pre_psf.fits``  — the model lens-light and lensed-source images *before*
  PSF convolution, one HDU per waveband per component.
- ``model.fits``    — the same images *after* PSF convolution, i.e. what the
  model predicts the data to look like.

These are the deblending products: they separate the lens galaxy's light from
the lensed source's light in every band that was fitted, which is what
downstream photometry and SED fitting need and what a viewer opens in DS9 to
judge the fit by eye.

The HDUs are collected with ``af.AggregateFITS.extract_fits`` across the
waveband searches of one lens, and are ordered by ``WAVEBAND_ORDER`` (blue →
red, VIS first) so the extension order is physically meaningful rather than
alphabetical. Unknown wavebands sort to the end rather than being dropped.

The step is idempotent: a lens whose two FITS files already exist is skipped, so
the bundle can be rebuilt as more fits land without redoing finished work.

Stage 2 of ``scripts/build_inspection_bundle.sh``.

Usage
-----
    python catalogue/scripts/deblending.py --sample=q1_walsmley

    python catalogue/scripts/deblending.py \
        --sample=dr1_prelim_grade_ab \
        --inspect_dir=inspect/dr1_prelim_grade_ab_run250
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import catalogue_util


# Waveband ordering for the FITS extensions: VIS first, then the ground-based
# optical bands blue → red, then the Euclid NIR bands.
WAVEBAND_ORDER = [
    "vis",
    "decam_g",
    "decam_r",
    "decam_i",
    "decam_z",
    "nir_y",
    "nir_j",
    "nir_h",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Write per-lens pre_psf.fits and model.fits deblending bundles "
            "for a sample."
        )
    )
    catalogue_util.add_common_arguments(parser)
    parser.add_argument(
        "--unique_tag",
        metavar="name",
        default="sersic_lens_model",
        help="Pipeline stage the results were written under.",
    )
    parser.add_argument(
        "--search_name",
        metavar="name",
        default=None,
        help=(
            "If set, keep only fits whose search.name matches (e.g. vis_pix). "
            "Default: every waveband under the stage."
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

        # Idempotency: skip a lens whose two FITS bundles are already present.
        target_pre_psf = inspect_path / dataset_name / "pre_psf.fits"
        target_model = inspect_path / dataset_name / "model.fits"
        if target_pre_psf.exists() and target_model.exists():
            continue

        print(dataset_name)

        agg = Aggregator.from_directory(
            directory=sample_root / dataset_name, completed_only=True
        )

        agg_query = agg.query(agg.unique_tag == args.unique_tag)
        if args.search_name is not None:
            agg_query = agg_query.query(agg_query.search.name == args.search_name)

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
            agg_fits = af.AggregateFITS(aggregator=agg_query)
        except ValueError as e:
            print(f"skipping {dataset_name}: {e}")
            continue

        waveband_list = [search.name for search in agg_query.values("search")]

        if args.unique_tag == "galaxy_sersic_model":
            # A lens-light-only fit has no lensed source to deblend.
            pre_psf_hdus = [al.agg.fits_galaxy_images.lens_light_image]
            model_hdus = [al.agg.fits_model_galaxy_images.lens_light_image]
        else:
            pre_psf_hdus = [
                al.agg.fits_galaxy_images.lens_light_image,
                al.agg.fits_galaxy_images.lensed_source_image,
            ]
            model_hdus = [
                al.agg.fits_model_galaxy_images.lens_light_image,
                al.agg.fits_model_galaxy_images.lensed_source_image,
            ]

        output_dataset_path = inspect_path / dataset_name
        output_dataset_path.mkdir(parents=True, exist_ok=True)

        hdu_list = agg_fits.extract_fits(
            hdus=pre_psf_hdus, extname_prefix_list=waveband_list
        )
        hdu_list.writeto(output_dataset_path / "pre_psf.fits", overwrite=True)

        hdu_list = agg_fits.extract_fits(
            hdus=model_hdus, extname_prefix_list=waveband_list
        )
        hdu_list.writeto(output_dataset_path / "model.fits", overwrite=True)


if __name__ == "__main__":
    main()
