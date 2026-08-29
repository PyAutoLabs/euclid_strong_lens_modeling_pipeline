"""
Euclid Catalogue: Lens Mass CSV
================================

Produces ``lens_mass.csv`` — the mass-model half of the DR1 catalogue.

Scrapes the ``initial_lens_model/vis_pix`` results of a whole sample and emits
one master CSV listing each lens once with all 7 free parameters of the
Isothermal (SIE) + ExternalShear mass model, plus the
``effective_einstein_radius`` latent variable. Each column comes in five
flavours: median, lower/upper 1σ and lower/upper 3σ (PyAutoFit's
``ValueType.Median`` / ``ValuesAt1Sigma`` / ``ValuesAt3Sigma``).

Only lenses whose ``vis_pix`` search wrote a ``.completed`` marker are listed —
PyAutoFit's ``completed_only=True`` filter handles that, so a sample still being
fitted produces a partial but never a corrupt catalogue.

The master CSV is also split into one row per lens, dropped in that lens's own
folder inside the inspect directory, so each folder is self-contained.

Stage 3 of ``scripts/build_inspection_bundle.sh``.

Usage
-----
    python catalogue/scripts/lens_mass.py --sample=q1_walsmley

    python catalogue/scripts/lens_mass.py \
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
        description="Write the lens mass-model master CSV for a sample."
    )
    catalogue_util.add_common_arguments(parser)
    parser.add_argument(
        "--search_name",
        metavar="name",
        default="vis_pix",
        help="search.name to filter on within initial_lens_model. Default: vis_pix.",
    )
    parser.add_argument(
        "--unique_tag",
        metavar="name",
        default="initial_lens_model",
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

    # One aggregator over the whole sample. completed_only filters out lenses
    # whose search has not produced a `.completed` marker.
    agg = Aggregator.from_directory(directory=sample_root, completed_only=True)

    agg_query = agg.query(agg.unique_tag == args.unique_tag)
    agg_query = agg_query.query(agg_query.search.name == args.search_name)

    agg_csv = af.AggregateCSV(aggregator=agg_query)

    # lens_name column: the last part of the search's path_prefix, i.e. the
    # dataset (tile) directory name.
    lens_name_list = [
        search.path_prefix.parts[-1] for search in agg_query.values("search")
    ]
    agg_csv.add_label_column(name="lens_name", values=lens_name_list)

    value_types_all = [
        ValueType.Median,
        ValueType.ValuesAt1Sigma,
        ValueType.ValuesAt3Sigma,
    ]

    # SIE mass + ExternalShear: all 7 free parameters with median + 1σ + 3σ.
    mass_args = [
        ("galaxies.lens.mass.centre.centre_0", "centre_0"),
        ("galaxies.lens.mass.centre.centre_1", "centre_1"),
        ("galaxies.lens.mass.ell_comps.ell_comps_0", "ell_comps_0"),
        ("galaxies.lens.mass.ell_comps.ell_comps_1", "ell_comps_1"),
        ("galaxies.lens.mass.einstein_radius", "einstein_radius"),
        ("galaxies.lens.shear.gamma_1", "shear_gamma_1"),
        ("galaxies.lens.shear.gamma_2", "shear_gamma_2"),
    ]
    for argument, name in mass_args:
        agg_csv.add_variable(argument=argument, name=name, value_types=value_types_all)

    # Latent: effective Einstein radius, computed by `util.LatentEuclid` on each
    # sample. `add_variable` pulls from latent_summary for a "latent.<name>"
    # argument.
    agg_csv.add_variable(
        argument="latent.effective_einstein_radius",
        name="effective_einstein_radius",
        value_types=value_types_all,
    )

    out_csv = inspect_path / "lens_mass.csv"
    agg_csv.save(path=out_csv)
    print(f"wrote {out_csv} ({len(lens_name_list)} rows)")

    written = catalogue_util.write_per_tile_csv(
        master_csv=out_csv, inspect_path=inspect_path, filename="lens_mass.csv"
    )
    print(f"wrote {written} per-lens lens_mass.csv files")


if __name__ == "__main__":
    main()
