"""
Euclid Catalogue: Source Light Sersic CSV
==========================================

New here? Read ``start_here.py`` for the concepts (the MGE, the output layout,
linear light profiles) and ``catalogue/README.md`` for the run order.

Produces ``source_sersic.csv`` — the source-light half of the DR1 catalogue.

Identical in shape to ``lens_sersic.py``, but reads the *source* galaxy's
Sersic profile out of the same ``sersic_lens_model/vis`` results: its
``centre_0``, ``centre_1``, ``ell_comps_0``, ``ell_comps_1``,
``effective_radius`` and ``sersic_index``.

These are the source's *unlensed* (source-plane) parameters, so the centre is
in source-plane arcsec relative to the mask centre and is not directly
comparable to the lens-light centre in ``lens_sersic.csv``.

Intensity is not a column — ``lp_linear.Sersic`` solves it linearly, so it never
enters the non-linear samples. Each column comes in five flavours: median,
lower/upper 1σ, lower/upper 3σ. Only lenses with a ``.completed`` marker are
listed, and the master CSV is split into one row per lens folder.

Stage 5 of ``scripts/build_inspection_bundle.sh``.

__The Other Half Of One Fit__

Stages 4 and 5 scrape the *same* results. ``scripts/sersic_lens_model.py`` fits
one ``lp_linear.Sersic`` to the lens and one to the source in a single search, so
``sersic_lens_model/vis`` holds both galaxies; ``lens_sersic.py`` walks down
``galaxies.lens.bulge`` and this file walks down ``galaxies.source.bulge``. They
are two producers rather than one because the bundle wants two tables, not
because there are two fits — which also means either can be re-run alone, and
that a lens present in one CSV is present in the other.

Read ``lens_sersic.py`` for why the pipeline fits a Sersic at all after the MGE,
and ``lens_mass.py`` for the shared machinery — path resolution, the aggregator,
``completed_only``, the ``lens_name`` column, the five value flavours and the
per-lens split. The sections below cover only what is particular to the source.

Usage
-----
    python catalogue/scripts/source_sersic.py --sample=q1_walsmley

    python catalogue/scripts/source_sersic.py \
        --sample=dr1_prelim_grade_ab \
        --inspect_dir=inspect/dr1_prelim_grade_ab_run250
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import catalogue_util


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write the source-light Sersic master CSV for a sample."
    )
    catalogue_util.add_common_arguments(parser)
    parser.add_argument(
        "--search_name",
        metavar="name",
        default="vis",
        help="search.name to filter on within sersic_lens_model. Default: vis.",
    )
    parser.add_argument(
        "--unique_tag",
        metavar="name",
        default="sersic_lens_model",
        help="Pipeline stage the results were written under.",
    )
    return parser.parse_args()


def main():
    """
    __Query__

    ``sersic_lens_model`` / ``vis`` — the same tag and search name
    ``lens_sersic.py`` uses, because it is the same fit. A lens with no finished
    search under that tag is filtered out by ``completed_only=True`` and is
    absent from both CSVs.
    """
    args = parse_args()
    output_path, inspect_path = catalogue_util.resolve_paths(args)

    import autofit as af
    from autofit.aggregator.aggregator import Aggregator
    from autofit.aggregator.summary.aggregate_csv.column import ValueType
    import autolens as al  # noqa: F401  required for unpickling result types

    sample_root = catalogue_util.sample_root_from(output_path, args.sample)
    if not sample_root.is_dir():
        print(f"no sample directory at {sample_root}; nothing to do")
        return

    agg = Aggregator.from_directory(directory=sample_root, completed_only=True)

    agg_query = agg.query(agg.unique_tag == args.unique_tag)
    agg_query = agg_query.query(agg_query.search.name == args.search_name)

    agg_csv = af.AggregateCSV(aggregator=agg_query)

    """
    __Row Identity__

    ``lens_name`` is still the lens's dataset directory name, not the source's —
    a source has no catalogue identity of its own, and the row is the source *of*
    that lens. It is what joins this table to ``lens_mass.csv`` and
    ``lens_sersic.csv``, and what the per-lens split groups on.
    """
    lens_name_list = [
        search.path_prefix.parts[-1] for search in agg_query.values("search")
    ]
    agg_csv.add_label_column(name="lens_name", values=lens_name_list)

    value_types_all = [
        ValueType.Median,
        ValueType.ValuesAt1Sigma,
        ValueType.ValuesAt3Sigma,
    ]

    """
    __These Are Source-Plane Numbers__

    The six columns are the same six as ``lens_sersic.csv`` — centre, elliptical
    components, effective radius, Sersic index — and ``intensity`` is again
    absent because ``lp_linear`` solves it linearly rather than sampling it.

    What differs is the plane they live in. A lensed source is fitted by placing
    the profile in the *source* plane and tracing it through the mass model, so
    every number here describes the source as it would look unlensed: the
    effective radius is the intrinsic size, not the extent of the arcs, and the
    ellipticity is the intrinsic shape, not the shape the lensing produced.

    Two consequences. Column for column these values are not comparable with
    ``lens_sersic.csv``'s — in particular the two centres are positions in
    different planes, and the source's is not an offset from the lens on the sky.
    And any *observed* quantity has to be combined with the lensing: the
    magnification that relates the two is a latent variable, carried per band in
    ``magnitudes.csv``.

    The source centre priors were seeded from the source's MGE centre in the
    upstream ``vis_lp`` fit and then left free, so the value here is fitted
    rather than inherited.
    """
    # Source Sersic free parameters. Intensity omitted — lp_linear solves it.
    source_args = [
        ("galaxies.source.bulge.centre.centre_0", "centre_0"),
        ("galaxies.source.bulge.centre.centre_1", "centre_1"),
        ("galaxies.source.bulge.ell_comps.ell_comps_0", "ell_comps_0"),
        ("galaxies.source.bulge.ell_comps.ell_comps_1", "ell_comps_1"),
        ("galaxies.source.bulge.effective_radius", "effective_radius"),
        ("galaxies.source.bulge.sersic_index", "sersic_index"),
    ]
    for argument, name in source_args:
        agg_csv.add_variable(argument=argument, name=name, value_types=value_types_all)

    """
    __Saving, And The Per-Lens Split__

    Master ``source_sersic.csv`` at the root of the inspect directory, one row
    per lens; then ``write_per_tile_csv`` drops each lens's row into
    ``<inspect_dir>/<lens_name>/source_sersic.csv``, giving that folder its own
    copy alongside the lens-light and mass rows.
    """
    out_csv = inspect_path / "source_sersic.csv"
    agg_csv.save(path=out_csv)
    print(f"wrote {out_csv} ({len(lens_name_list)} rows)")

    written = catalogue_util.write_per_tile_csv(
        master_csv=out_csv, inspect_path=inspect_path, filename="source_sersic.csv"
    )
    print(f"wrote {written} per-lens source_sersic.csv files")


if __name__ == "__main__":
    main()
