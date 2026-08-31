"""
Euclid Catalogue: Deblended Galaxy Images FITS
===============================================

New here? Read ``start_here.py`` for the concepts (the model, the output layout,
what a fit writes) and ``catalogue/README.md`` for the run order.

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

__Why Two Files, Not One__

``pre_psf.fits`` and ``model.fits`` hold the same components and differ only by
the PSF, and each answers a different question.

``model.fits`` is the model as the telescope would have seen it — the profiles
convolved with that band's PSF. It is what you overlay on the data, because it
lives in the data's own resolution: a residual is only meaningful against the
convolved model.

``pre_psf.fits`` is the model before the instrument touched it, which is the
astrophysical object rather than the observation. That is the one to measure
sizes and fluxes from, and the one to compare across bands whose PSFs differ —
the difference between the two files *is* the seeing, so keeping both means a
reader never has to guess which they are looking at.

The pair is also what makes "deblending" the right word for this stage. The
lens's light and the lensed source's light overlap in every band, and nothing
short of a model separates them; these files are that separation written out per
band, which is what downstream photometry and SED fitting consume and what a
viewer opens in DS9 to judge a fit by eye.

__What It Needs Upstream, And Where It Looks__

The HDUs come from the FITS a finished ``sersic_lens_model`` search wrote — so a
lens needs that fit before it can be bundled, which is the run-order dependency
``catalogue/README.md`` records for stage 2.

This producer takes *every* search under the tag in whichever results tree it is
pointed at, with no default filter on the search name. Run against the default
``output`` it bundles the VIS Sersic fit that ``scripts/sersic_lens_model.py``
wrote; pointed at ``output_sed``, where the SED chain put one search per band, it
bundles every band. ``--search_name`` narrows to a single named search when only
one is wanted.

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


"""
__The Waveband Order__

FITS extensions have no intrinsic order beyond the one they are written in, and
whatever order the aggregator happened to find the searches in is not it. The
list below fixes that: VIS first, then the DECam optical bands blue → red, then
the three Euclid NIR bands, so walking the extensions of ``pre_psf.fits`` walks
them in increasing wavelength and a per-band flux read straight off the file is
already in SED order.

``multi_wavelength.py`` keeps its own, much longer list for the same purpose over
image columns. The two are deliberately separate — this one names the bands a
deblending bundle is expected to contain, that one every band an SED run may have
fitted — and a band missing from either is not dropped: the sort key sends an
unrecognised name to the end.
"""
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
    """
    __One Aggregator Per Lens, And The Skip__

    Like ``multi_wavelength.py``, and unlike the CSV producers, this one loops
    over lens directories and builds an aggregator scoped to each. It has to:
    the output is one FITS file per lens, gathering that lens's bands into a
    single HDU list, so the aggregator handed to ``AggregateFITS`` must contain
    that lens and nothing else.

    The skip at the top of the loop is what makes stage 2 cheap to re-run. A lens
    whose two bundles already exist is passed over before any result is opened,
    so rebuilding a sample as more fits land costs roughly the lenses that are
    new. Note the check is existence only — it does not compare timestamps — so
    after *re-fitting* a lens, delete its two FITS files (or its whole folder) to
    force them to be rebuilt.

    Failures are per lens too: a lens with no completed search under the tag
    makes ``AggregateFITS`` raise ``ValueError``, which is caught, reported and
    skipped so the rest of the sample still gets bundled.
    """
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

        """
        __Which HDUs, And The Lens-Only Branch__

        ``al.agg.fits_galaxy_images`` and ``al.agg.fits_model_galaxy_images``
        name the same two components — the lens light and the lensed source — in
        two different files the search wrote: the galaxy images and the model
        galaxy images. Selecting from the first gives the pre-PSF bundle and from
        the second the post-PSF one, which is the whole of the difference between
        the two output files.

        The branch drops the lensed-source HDU when the tag is
        ``galaxy_sersic_model``, a lens-light-only fit with no lensed source to
        separate out. No script in this repository writes that tag — the non-lens
        galaxy pipeline is recorded as out of scope in ``docs/drift_report.md`` —
        so the branch is reachable only by passing ``--unique_tag`` explicitly at
        a results tree produced elsewhere. It is kept because the alternative is
        a confusing failure on results that are otherwise perfectly readable.
        """
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

        """
        __Extension Names, And Writing The Two Bundles__

        ``extract_fits`` walks the aggregator in the order set above and copies
        the requested HDUs out of each search's FITS into one list, opening an
        empty primary HDU first as FITS requires.

        ``extname_prefix_list`` is what keeps the result readable. Every search
        calls its extensions the same thing, so without a prefix a lens fitted in
        eight bands would produce sixteen identically named extensions. Passing
        the waveband list — built from the same reordered aggregator, so it lines
        up index for index — prefixes each with its band, upper-cased, giving
        names like ``VIS_GALAXY_0`` and ``NIR_H_GALAXY_1``. Extension name alone
        then says which band and which component an image is.

        Both files are written with ``overwrite=True``, which matters only when
        one of the pair exists and the other does not: the skip above would not
        have fired, and the surviving file is replaced rather than erroring.
        """
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
