"""
Euclid Catalogue: Lens Light Sersic CSV
========================================

New here? Read ``start_here.py`` for the concepts (the MGE, the output layout,
linear light profiles) and ``catalogue/README.md`` for the run order.

Produces ``lens_sersic.csv`` — the lens-light half of the DR1 catalogue.

Scrapes the ``sersic_lens_model/vis`` results of a whole sample (written by
``scripts/sersic_lens_model.py``) and emits one master CSV listing each lens
once with the 6 free parameters of its Sersic light profile:
``centre_0``, ``centre_1``, ``ell_comps_0``, ``ell_comps_1``,
``effective_radius``, ``sersic_index``.

Intensity is *not* a column: the profile is an ``lp_linear.Sersic`` whose
intensity is solved by linear algebra at every likelihood evaluation, so it
never enters the non-linear samples.

Each column comes in five flavours: median, lower/upper 1σ, lower/upper 3σ.
Only lenses whose search wrote a ``.completed`` marker are listed. The master
CSV is also split into one row per lens, dropped in that lens's own folder.

Stage 4 of ``scripts/build_inspection_bundle.sh``.

__Why A Sersic, After The MGE__

This is a different fit from the one ``lens_mass.py`` scrapes, and the reason is
worth carrying into any table built from it.

``initial_lens_model`` models the lens light as a Multi-Gaussian Expansion —
tens of Gaussians whose linear amplitudes are solved at every likelihood
evaluation. As "__Multi Gaussian Expansion__" in ``start_here.py`` explains,
that is what makes automated lens modeling of a thousand cut-outs possible. What
it does not give you is a *number to publish*: an effective radius and a Sersic
index are properties of a Sersic profile, and a basis of Gaussians simply has
neither.

So ``scripts/sersic_lens_model.py`` runs a second fit that swaps the MGE for a
single ``lp_linear.Sersic`` per galaxy and enters the mass and shear as
*instances* of the ``initial_lens_model`` ``vis_lp`` result, so the whole
parameter space of that search is light. This producer is the scraper for the
lens half of it; ``source_sersic.py`` is the scraper for the source half of the
same results. Because the mass was frozen there rather than fitted, this CSV
carries no mass columns — those live in ``lens_mass.csv``, scraped from the
``vis_pix`` search that did fit the mass, and joined on ``lens_name``.

__Shared Machinery__

The path resolution, the aggregator, ``completed_only``, the ``lens_name``
label column, the five value flavours and the per-lens split all work exactly as
in ``catalogue/scripts/lens_mass.py``, which explains each of them in full;
the sections below cover only what differs.

Usage
-----
    python catalogue/scripts/lens_sersic.py --sample=q1_walsmley

    python catalogue/scripts/lens_sersic.py \
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
        description="Write the lens-light Sersic master CSV for a sample."
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
    __Query: A Different Pipeline Tag__

    The only structural difference from ``lens_mass.py`` is which fit is
    selected. There the tag was ``initial_lens_model`` and the search
    ``vis_pix``; here it is ``sersic_lens_model`` / ``vis``, which is the
    separate results tree ``scripts/sersic_lens_model.py`` writes beside — not
    inside — the initial fit's.

    That is the run-order dependency ``catalogue/README.md`` records for stage 4:
    a lens with no finished ``sersic_lens_model/vis`` search simply does not
    appear in ``lens_sersic.csv``, because ``completed_only=True`` filtered it
    out one line below. Everything else on the way to the CSV — path resolution,
    the deferred heavy imports, the aggregator — is as ``lens_mass.py`` describes
    it.
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
    __Row Identity And Value Flavours__

    Both as in ``lens_mass.py``: ``lens_name`` is the dataset directory name,
    built by iterating the same aggregator so the list pairs row-for-row with the
    results, and it is the key the per-lens split groups on. The three value
    types give five columns per variable — median, then lower/upper 1σ and 3σ —
    read off the stored posterior rather than propagated from a best fit.
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
    __The Six Lens Sersic Columns, And The Missing Seventh__

    A Sersic has seven parameters. Six of them are here, reached at
    ``galaxies.lens.bulge`` — the two centre components, the two elliptical
    components, the effective radius and the Sersic index.

    The seventh, ``intensity``, is deliberately absent, and its absence is a
    property of the fit rather than an omission in this file. The profile is an
    ``lp_linear.Sersic``: its intensity is solved by linear algebra at every
    likelihood evaluation instead of being sampled, so there is no ``intensity``
    entry in the samples for ``add_variable`` to find and no posterior to
    summarise. If you need lens brightness, take it from the flux latents in
    ``magnitudes.csv``, which are calibrated to µJy and measured per band.

    ``bulge`` is the model component's name, not a morphological claim: the
    single Sersic here stands for the whole lens galaxy's light, and nothing
    fitted a separate disc.
    """
    # Lens Sersic free parameters. Intensity omitted — lp_linear solves it.
    lens_args = [
        ("galaxies.lens.bulge.centre.centre_0", "centre_0"),
        ("galaxies.lens.bulge.centre.centre_1", "centre_1"),
        ("galaxies.lens.bulge.ell_comps.ell_comps_0", "ell_comps_0"),
        ("galaxies.lens.bulge.ell_comps.ell_comps_1", "ell_comps_1"),
        ("galaxies.lens.bulge.effective_radius", "effective_radius"),
        ("galaxies.lens.bulge.sersic_index", "sersic_index"),
    ]
    for argument, name in lens_args:
        agg_csv.add_variable(argument=argument, name=name, value_types=value_types_all)

    """
    __Saving, And The Per-Lens Split__

    The master ``lens_sersic.csv`` goes to the root of the inspect directory, one
    row per lens across the sample; ``write_per_tile_csv`` then drops each lens's
    row into ``<inspect_dir>/<lens_name>/lens_sersic.csv`` so that folder stays
    self-contained. Same contract as ``lens_mass.py``, same helper.
    """
    out_csv = inspect_path / "lens_sersic.csv"
    agg_csv.save(path=out_csv)
    print(f"wrote {out_csv} ({len(lens_name_list)} rows)")

    written = catalogue_util.write_per_tile_csv(
        master_csv=out_csv, inspect_path=inspect_path, filename="lens_sersic.csv"
    )
    print(f"wrote {written} per-lens lens_sersic.csv files")


if __name__ == "__main__":
    main()
