"""
Euclid Catalogue: Source Light Sersic CSV
==========================================

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

    lens_name_list = [
        search.path_prefix.parts[-1] for search in agg_query.values("search")
    ]
    agg_csv.add_label_column(name="lens_name", values=lens_name_list)

    value_types_all = [
        ValueType.Median,
        ValueType.ValuesAt1Sigma,
        ValueType.ValuesAt3Sigma,
    ]

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

    out_csv = inspect_path / "source_sersic.csv"
    agg_csv.save(path=out_csv)
    print(f"wrote {out_csv} ({len(lens_name_list)} rows)")

    written = catalogue_util.write_per_tile_csv(
        master_csv=out_csv, inspect_path=inspect_path, filename="source_sersic.csv"
    )
    print(f"wrote {written} per-lens source_sersic.csv files")


if __name__ == "__main__":
    main()
