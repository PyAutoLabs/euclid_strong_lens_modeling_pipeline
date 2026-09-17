"""
Euclid Catalogue: Multi-band Astrometric Offsets CSV
=====================================================

New here? Read ``start_here.py`` for the concepts (latent variables, the FITS
header contract, the output layout) and ``catalogue/README.md`` for the run
order.

Produces ``astrometric_offsets.csv`` — the per-band registration table: how far
each non-VIS band's sky is from the VIS astrometric frame, with errors.

Scrapes the same multi-band ``sersic_lens_model`` results ``magnitudes.py``
does (one search per waveband, written by
``scripts/sersic_lens_model_waveband.py``) and emits one master CSV with a row
per ``(lens, waveband)`` carrying the two fitted ``DatasetModel`` offsets:

- ``grid_offset_y`` — ``dataset_model.grid_offset.grid_offset_0``
- ``grid_offset_x`` — ``dataset_model.grid_offset.grid_offset_1``

plus the ``lens_name``, ``waveband`` and ``crval_ra_deg`` label columns. Each
offset comes in four flavours: max-log-likelihood, median, ±1σ and ±3σ.

__What The Offset Means__

``grid_offset`` is ``(y, x)`` in arcsec and is subtracted from this band's grids
before the VIS model is evaluated (``autoarray/fit/fit_dataset.py``'s
``FitDataset.grids`` → ``Grid2D.subtracted_and_rotated_from``), so a positive
``grid_offset_y`` means this band's sky lies +y arcsec from the VIS astrometric
frame.

These are the *only* two free parameters of a band fit: the VIS Sersic lens
model is frozen and only its intensity is re-solved, so the posterior in this
table is a direct measurement of NISP/EXT-to-VIS registration for that lens.

``scripts/lens_model_waveband.py`` gives both a uniform prior of ±0.2" (two VIS
pixels at 0.1"/pixel), so a value pinned at the edge of that range means the
band's misregistration exceeds what the fit is allowed to model — a QA flag on
the row, not a measurement.

__Why VIS Has No Row__

The VIS Sersic fit under the same tag has no ``dataset_model`` at all: it is the
frame the offsets are measured against, and its offset is zero by construction.
``af.AggregateCSV`` writes a blank cell rather than raising for an argument it
cannot find, and takes its header from the *first* row only, so VIS results are
excluded from the aggregator before the CSV is built rather than blanked in it.

The SED chain is run with ``PYAUTO_OUTPUT_DIR=output_sed``, so this producer
reads ``output_sed`` by default rather than the main ``output``. Re-runs of a
waveband leave more than one result per ``(lens, stage, band)``; the newest is
kept and the rest dropped, by ``magnitudes.latest_result_per_lens_band``.

Stage 9 of ``scripts/build_inspection_bundle.sh``.

__Shared Machinery__

Path resolution, ``completed_only``, ``add_label_column``, ``add_variable`` and
the per-lens split all work as ``catalogue/scripts/lens_mass.py`` describes them
in full, and the multi-band query and de-duplication are
``catalogue/scripts/magnitudes.py``'s, imported rather than copied. The sections
below cover only what is particular to the offsets.

Usage
-----
    python catalogue/scripts/astrometric_offsets.py --sample=q1_walsmley

    python catalogue/scripts/astrometric_offsets.py \
        --sample=dr1_prelim_grade_ab \
        --inspect_dir=inspect/dr1_prelim_grade_ab_run250 \
        --output_path=output_sed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import catalogue_util
import magnitudes

# The model path of the fitted (y, x) offset, as a `model.paths` entry. The two
# `add_variable` calls in `main()` spell the same path as the dotted string
# `Column.path` splits on; it is written out there rather than derived from this
# constant so the column specs stay readable in the source (and to the syntax
# tree `tests/test_catalogue_parity.py` reads them out of).
GRID_OFFSET_Y_PATH = ("dataset_model", "grid_offset", "grid_offset_0")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write the multi-band astrometric offsets master CSV for a sample."
    )
    catalogue_util.add_common_arguments(parser, default_output_path="output_sed")
    parser.add_argument(
        "--unique_tag",
        metavar="name",
        default="sersic_lens_model",
        help="Pipeline stage the multi-band results were written under.",
    )
    return parser.parse_args()


def has_grid_offset(result) -> bool:
    """
    Whether this result's model carries the fitted ``(y, x)`` grid offset.

    Membership in ``model.paths`` is the test, not
    ``model.object_for_path(("dataset_model", ...))``: the VIS result's model has
    no ``dataset_model`` attribute at all, so the lookup raises ``AttributeError``
    rather than returning ``None``. ``model`` reads the result's
    ``files/model.json``, which is cheap and already needed downstream.

    Any failure to read the model is a "no": a result that cannot say what it
    fitted must not be handed to ``add_variable``, which would write blank cells
    for it rather than raise.
    """
    try:
        return GRID_OFFSET_Y_PATH in set(result.model.paths)
    except Exception:
        return False


def with_grid_offset(aggregator):
    """
    Drop the results whose model has no ``dataset_model.grid_offset``.

    ``Aggregator.query`` has predicates for the search's own metadata, not for
    the shape of its model, so this is a plain Python filter over the queried
    results. It matters more than it looks: ``AggregateCSV.fieldnames`` comes
    from the *first* row, so a VIS result arriving first would write a header
    with no offset columns at all, and one arriving later would silently
    contribute a row of blanks.

    The survivors are wrapped back into an ``Aggregator`` — the same move
    ``magnitudes.latest_result_per_lens_band`` makes — so everything downstream
    treats the filtered set as an ordinary aggregator, and the label columns
    below index the rows this aggregator actually yields.
    """
    from autofit.aggregator.aggregator import Aggregator

    kept = [result for result in aggregator if has_grid_offset(result)]

    excluded = len(aggregator) - len(kept)
    if excluded:
        print(
            f"excluded {excluded} results whose model has no "
            "dataset_model.grid_offset"
        )

    return Aggregator(
        search_outputs=kept,
        grid_search_outputs=aggregator.grid_search_outputs,
    )


def main():
    """
    __Query: Every Band Under One Tag, Minus VIS__

    The query is ``magnitudes.py``'s — one ``unique_tag``, no ``--search_name``,
    because the search name here is the waveband and keeping all of them is the
    point — followed by two filters.

    De-duplication runs *first*, on the full queried set: it selects the newest
    result per ``(lens, stage, band)``, and a VIS duplicate competing only with
    other VIS results cannot displace a band's winner. Running it after the
    offset filter would give the same survivors but ask it to reason about a set
    it did not scan.

    Then ``with_grid_offset`` drops VIS. ``AggregateCSV`` raises ``ValueError``
    when handed an empty aggregator, which is the ordinary case for a sample
    whose SED chain has not run yet *and* for a tree holding only VIS fits. That
    is caught and reported rather than raised: stage 9 of the bundle builder
    should leave a partial bundle alone, not abort it.
    """
    args = parse_args()
    output_path, inspect_path = catalogue_util.resolve_paths(args)

    import autofit as af
    from autofit.aggregator.aggregator import Aggregator
    import autolens as al  # noqa: F401  required for unpickling result types

    sample_root = catalogue_util.sample_root_from(output_path, args.sample)
    if not sample_root.is_dir():
        print(f"no sample directory at {sample_root}; nothing to do")
        return

    agg = Aggregator.from_directory(directory=sample_root, completed_only=True, unzip_temporary=True)
    agg_query = agg.query(agg.unique_tag == args.unique_tag)
    agg_query = magnitudes.latest_result_per_lens_band(agg_query, sample_root=sample_root)
    agg_query = with_grid_offset(agg_query)

    try:
        agg_csv = af.AggregateCSV(aggregator=agg_query)
    except ValueError as e:
        print(f"no completed {args.unique_tag} results with a grid offset: {e}")
        return

    """
    __Three Label Columns, Built After The Filter__

    The same three ``magnitudes.csv`` carries: ``lens_name`` (the dataset
    directory, which the per-lens split groups on), ``waveband``
    (``search.name``, the column that makes a row identifiable at all) and
    ``crval_ra_deg`` (the RA of the fitted maximum-likelihood lens light centre,
    read out of the ``wcs.json`` ``util.AnalysisImaging`` writes beside each
    result).

    They are read off ``agg_query`` *after* ``with_grid_offset``, not before:
    ``LabelColumn`` matches its values to rows by row number, so a list built
    from the unfiltered aggregator would shift every label by the number of VIS
    results that preceded it and quietly mislabel the table.
    """
    lens_name_list = [
        search.path_prefix.parts[-1] for search in agg_query.values("search")
    ]
    waveband_list = [search.name for search in agg_query.values("search")]
    crval_ra_deg_list = [
        wcs_dict["crval_ra_deg"] for wcs_dict in agg_query.values("wcs")
    ]

    agg_csv.add_label_column(name="lens_name", values=lens_name_list)
    agg_csv.add_label_column(name="waveband", values=waveband_list)
    agg_csv.add_label_column(name="crval_ra_deg", values=crval_ra_deg_list)

    """
    __Six Columns Per Offset__

    Four value types, and so six columns per offset: the maximum-likelihood
    value, the median, and the lower/upper bounds at 1σ and 3σ. The
    max-likelihood value is carried for the same reason ``magnitudes.csv``
    carries it — the gap between it and the median is itself the warning that a
    band's posterior is not well behaved, which for a two-parameter fit with a
    hard uniform prior usually means the offset has run into the prior edge.
    """
    value_types = (
        af.ValueType.MaxLogLikelihood,
        af.ValueType.Median,
        af.ValueType.ValuesAt3Sigma,
        af.ValueType.ValuesAt1Sigma,
    )

    """
    __The Two Model Parameters__

    Unlike ``magnitudes.py``, neither variable here is a latent: both are
    ordinary model parameters, so the argument is the model path
    ``add_variable`` resolves against the samples summary — ``grid_offset_0``
    and ``grid_offset_1``, the names ``al.DatasetModel``'s ``grid_offset``
    gives its two components.

    ``name`` renames them to ``grid_offset_y`` / ``grid_offset_x`` in the CSV,
    because ``0`` and ``1`` say nothing to a reader of the catalogue and the
    axis order is the part that is easy to get wrong.
    """
    agg_csv.add_variable(
        argument="dataset_model.grid_offset.grid_offset_0",
        name="grid_offset_y",
        value_types=value_types,
    )
    agg_csv.add_variable(
        argument="dataset_model.grid_offset.grid_offset_1",
        name="grid_offset_x",
        value_types=value_types,
    )

    """
    __Saving, And The Per-Lens Split__

    As in ``magnitudes.py``, the two printed counts legitimately differ: the
    master CSV has a row per ``(lens, waveband)``, while ``write_per_tile_csv``
    returns the number of *lenses* it wrote a file for. It groups on
    ``lens_name``, so a lens's whole set of band rows lands together in
    ``<inspect_dir>/<lens_name>/astrometric_offsets.csv`` — that lens's own
    registration table, beside its ``magnitudes.csv``.
    """
    out_csv = inspect_path / "astrometric_offsets.csv"
    agg_csv.save(path=out_csv)
    print(f"wrote {out_csv} ({len(lens_name_list)} rows)")

    written = catalogue_util.write_per_tile_csv(
        master_csv=out_csv, inspect_path=inspect_path, filename="astrometric_offsets.csv"
    )
    print(f"wrote {written} per-lens astrometric_offsets.csv files")


if __name__ == "__main__":
    main()
