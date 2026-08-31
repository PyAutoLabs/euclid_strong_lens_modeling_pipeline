"""
Euclid Catalogue: Lens Mass CSV
================================

New here? Read ``start_here.py`` for the concepts (the model, the output layout,
latent variables) and ``catalogue/README.md`` for the run order.

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

__Its Tutorial Twin__

``workflow/example/csv/lens_mass.py`` builds the same table this file does — the
mass model parameters of every VIS fit, through the same ``AggregateCSV`` API —
written as a flat top-to-bottom tutorial that explains each call as it makes it,
and saved under its own name into ``workflow/csv/``. Read it if you want the API
explained rather than applied.

This is the production version of that scrape: the same calls wrapped in
argument parsing, sample-wide path resolution, the full set of sigma columns and
the per-lens split, so the bundle builder can drive it. Where the two do the same
thing the sections below point at it rather than repeat it.

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
    """
    __Paths, Config And Imports__

    ``catalogue_util.resolve_paths`` turns the three common arguments into an
    absolute results directory and an absolute inspect directory, resolving
    relative paths against the project root so the producer behaves the same run
    from the repo root or from ``catalogue/``. It also pushes this pipeline's
    ``config/`` onto the autoconf stack, so the notation and label configuration
    used to read the fits is the one they were written under.

    ``autofit`` and ``autolens`` are imported inside ``main`` rather than at the
    top of the file, so ``--help`` — which never reaches here — does not pay for
    them. ``autolens`` itself is unused by name; it is imported because the
    aggregator unpickles PyAutoLens result objects and needs those classes
    importable.
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

    """
    __Aggregator__

    The aggregator walks a directory of finished fits and hands back one result
    per search it finds. Pointing it at ``<output>/<sample>`` rather than at
    ``output`` scopes the whole run to one sample, so lenses from other samples
    never leak into the catalogue.

    ``completed_only=True`` drops every search that has not written a
    ``.completed`` marker. This is the whole of the "partial but never corrupt"
    guarantee in ``catalogue/README.md``: a lens still being fitted is absent
    from the CSV rather than present with half-written numbers.
    """
    # One aggregator over the whole sample. completed_only filters out lenses
    # whose search has not produced a `.completed` marker.
    agg = Aggregator.from_directory(directory=sample_root, completed_only=True)

    """
    __Query: Pipeline Stage And Search__

    Every result under a lens is tagged by the pipeline that wrote it
    (``unique_tag``) and named by the search within it (``search.name``) — the
    ``output/<sample>/<dataset>/<pipeline>/<search>/<hash>/`` layout described
    under "__Reading The Output__" in ``start_here.py``. Two chained queries
    therefore pick out exactly one search per lens: the ``vis_pix`` search of
    ``initial_lens_model``, which is the pixelized-source stage and so the one
    holding the final mass model. Both strings are arguments, so the same script
    scrapes ``vis_lp`` or another pipeline's stage without editing.

    Note the second query is chained off ``agg_query``, narrowing the already
    narrowed set. The tutorial twin re-queries the unfiltered ``agg`` at this
    point; here the results must satisfy both conditions at once, because a lens
    may well have a ``vis`` search under a different pipeline.
    """
    agg_query = agg.query(agg.unique_tag == args.unique_tag)
    agg_query = agg_query.query(agg_query.search.name == args.search_name)

    agg_csv = af.AggregateCSV(aggregator=agg_query)

    """
    __The lens_name Column__

    ``add_label_column`` takes a plain list rather than something read out of the
    model, and pairs it row-for-row with the aggregator's results — so the list
    has to be built by iterating that same aggregator, in that same order, which
    is what the comprehension below does.

    The value is the last part of the search's ``path_prefix``, i.e. the dataset
    directory name, which for DR1 is the tile-derived lens name. It is the only
    column that identifies which lens a row belongs to, and it is the key
    ``catalogue_util.write_per_tile_csv`` groups on at the end of this file, so
    it is written first and never omitted.
    """
    # lens_name column: the last part of the search's path_prefix, i.e. the
    # dataset (tile) directory name.
    lens_name_list = [
        search.path_prefix.parts[-1] for search in agg_query.values("search")
    ]
    agg_csv.add_label_column(name="lens_name", values=lens_name_list)

    """
    __Five Value Flavours Per Variable__

    Every variable added below is written five times over: the median of its
    posterior under the bare column name, then the values at lower and upper 1σ
    and at lower and upper 3σ in four suffixed columns.

    Nothing is propagated here. The sampler already explored the posterior, so
    each of those five numbers is read straight off the stored samples, which is
    why the errors can be asymmetric and why a parameter pinned against a prior
    bound shows it. To get a symmetric ± error, subtract the median from the
    sigma values yourself; the CSV deliberately stores the bounds rather than a
    single collapsed uncertainty.

    A column whose value is missing from a fit's samples is written empty rather
    than dropped, so the header is the same width for every sample.
    """
    value_types_all = [
        ValueType.Median,
        ValueType.ValuesAt1Sigma,
        ValueType.ValuesAt3Sigma,
    ]

    """
    __The Mass Model Columns__

    The ``argument`` strings are model paths: the same dotted route through the
    model that ``model.paths`` prints, from the galaxy collection down to the
    scalar. Tuple parameters such as ``centre`` are reached one component at a
    time, hence ``centre.centre_0`` and ``centre.centre_1``.

    Seven of them, which is every free parameter of the ``vis_pix`` mass model:
    the Isothermal (SIE) ellipsoid's centre, elliptical components and Einstein
    radius, and the two components of the external shear field. ``name``
    shortens each header — the CSV says ``einstein_radius``, not
    ``galaxies_lens_mass_einstein_radius`` — and those short names are the DR1
    catalogue's column names.

    The shear components are ``gamma_1`` and ``gamma_2``, the two parameters of
    PyAutoLens's ``ExternalShear``.
    """
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

    """
    __The Effective Einstein Radius Latent__

    The eighth column is not a model parameter. The effective Einstein radius is
    a *latent variable*: derived from the model rather than sampled by it, and —
    the point of the machinery — recomputed on every sample, so it arrives with a
    full posterior instead of one number carried over from a best fit. That is
    why it takes the same five value flavours as everything above.

    The ``latent.`` prefix is what selects it. ``add_variable`` reads a plain
    ``galaxies...`` argument out of the samples and a ``latent.<name>`` argument
    out of the latent summary the fit wrote beside them. The name has to match a
    key of ``util.LatentEuclid``, whose catalogue "__Latent Variables__" in
    ``start_here.py`` describes and ``config/latent.yaml`` enables —
    ``effective_einstein_radius`` is one of the library latents that file turns
    on.

    Two consequences worth knowing before you read a CSV. A fit run in test mode
    writes no latent summary at all, so these columns are present but blank; and
    the effective Einstein radius is computed from the fitted tracer's tangential
    critical curve, which is not the same quantity as the SIE's own
    ``einstein_radius`` parameter three columns to its left.
    """
    # Latent: effective Einstein radius, computed by `util.LatentEuclid` on each
    # sample. `add_variable` pulls from latent_summary for a "latent.<name>"
    # argument.
    agg_csv.add_variable(
        argument="latent.effective_einstein_radius",
        name="effective_einstein_radius",
        value_types=value_types_all,
    )

    """
    __Saving, And The Per-Lens Split__

    ``save`` writes the master CSV at the root of the inspect directory: one row
    per lens, spanning the whole sample, which is the table a population study
    reads.

    ``write_per_tile_csv`` then re-reads it and drops each lens's own rows into
    ``<inspect_dir>/<lens_name>/lens_mass.csv``. That is what makes a single lens
    folder self-contained and shippable on its own — the copy travels with that
    lens's PNGs and FITS rather than the reader having to carry the master table
    around. The two files are written from the same rows, so they cannot drift.

    The printed row and file counts differ when a lens contributed more than one
    row, which for this producer means a duplicate result; ``magnitudes.py`` is
    the producer where the two counts legitimately differ, one row per waveband.
    """
    out_csv = inspect_path / "lens_mass.csv"
    agg_csv.save(path=out_csv)
    print(f"wrote {out_csv} ({len(lens_name_list)} rows)")

    written = catalogue_util.write_per_tile_csv(
        master_csv=out_csv, inspect_path=inspect_path, filename="lens_mass.csv"
    )
    print(f"wrote {written} per-lens lens_mass.csv files")


if __name__ == "__main__":
    main()
