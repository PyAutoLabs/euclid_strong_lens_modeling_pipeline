"""
Euclid Catalogue: Multi-band Magnitudes CSV
============================================

New here? Read ``start_here.py`` for the concepts (latent variables, the FITS
header contract, the output layout) and ``catalogue/README.md`` for the run
order.

Produces ``magnitudes.csv`` — the photometry table the downstream photometric
redshift / SED fitting works from.

Scrapes the multi-band ``sersic_lens_model`` results (one search per waveband,
written by ``scripts/sersic_lens_model_waveband.py``) and emits one master CSV
with a row per ``(lens, waveband)`` carrying the Euclid latent flux catalogue:

- ``lens_flux`` and the four FWHM aperture fluxes ``lens_flux_1/2/3/4_fwhm``
- ``lensed_source_flux``, ``source_flux``
- ``magnification``

plus the ``lens_name``, ``waveband`` and ``crval_ra_deg`` label columns.
Each variable comes in four flavours: max-log-likelihood, median, ±1σ and ±3σ.

All the flux quantities are µJy: they come from the ``*_mujy`` latents declared
by ``util.LatentEuclid`` and enabled in ``config/latent.yaml``, which is why the
fit must have been run by this pipeline (``util.AnalysisImaging``) with
``magzero`` in its info dict.

The SED chain is run with ``PYAUTO_OUTPUT_DIR=output_sed``, so this producer
reads ``output_sed`` by default rather than the main ``output``. Re-runs of a
waveband leave more than one result per ``(lens, stage, band)``; the newest is
kept and the rest dropped.

Stage 7 of ``scripts/build_inspection_bundle.sh``.

__Where These Results Come From__

Stages 1-5 read one search per lens. This one reads a whole chain of them.

``scripts/sersic_lens_model_waveband.py`` is the SED chain: it fits the VIS
Sersic, then hands that result to ``fit_waveband``, which fits every remaining
band of the cut-out under the same ``sersic_lens_model`` tag, with the Sersic's
shape fixed and its intensity re-solved against each band's own data. Each band
is a search named after the band, so a lens's results end up as
``<sample>/<lens>/sersic_lens_model/<band>/<hash>/`` — siblings under one tag,
which is exactly the shape this producer walks.

The chain is run with ``PYAUTO_OUTPUT_DIR=output_sed`` so the per-band folders do
not bloat the main tree, and that is why ``--output_path`` defaults to
``output_sed`` here rather than ``output``. When ``output_sed/<sample>/`` does
not exist, ``scripts/build_inspection_bundle.sh`` skips stages 6 and 7 rather
than failing.

__Shared Machinery__

Path resolution, ``completed_only``, ``add_label_column``, ``add_variable`` and
the per-lens split all work as ``catalogue/scripts/lens_mass.py`` describes them
in full. The sections below cover what is particular to multi-band photometry.

Usage
-----
    python catalogue/scripts/magnitudes.py --sample=q1_walsmley

    python catalogue/scripts/magnitudes.py \
        --sample=dr1_prelim_grade_ab \
        --inspect_dir=inspect/dr1_prelim_grade_ab_run250 \
        --output_path=output_sed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import catalogue_util


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write the multi-band magnitudes master CSV for a sample."
    )
    catalogue_util.add_common_arguments(parser, default_output_path="output_sed")
    parser.add_argument(
        "--unique_tag",
        metavar="name",
        default="sersic_lens_model",
        help="Pipeline stage the multi-band results were written under.",
    )
    return parser.parse_args()


def latest_result_per_lens_band(aggregator, sample_root: Path):
    """
    Keep only the newest completed result for each ``(lens, stage, band)``.

    __Why Duplicates Happen__

    A result folder is named for the model and configuration that produced it, so
    re-running a band with anything changed writes a *second* completed folder
    beside the first rather than overwriting it. That is the right behaviour for
    a fitting pipeline — nothing is silently destroyed — and the wrong behaviour
    for a catalogue, where it would put two rows of photometry for the same lens
    and band into ``magnitudes.csv`` with nothing to say which is current.

    Over a long DR1 campaign, where bands get re-run as calibration or priors
    improve, this is the difference between a usable table and one that has to be
    de-duplicated by hand downstream.

    __How The Key Is Built__

    The result's directory is taken relative to the sample root, which makes its
    parts ``<lens>/<stage>/<band>/<hash>/...``; the first three are the key and
    the fourth is what varies between duplicates. A path with fewer than four
    parts is not a per-band result folder and is skipped.

    Recency is the zipped result's timestamp where one exists — PyAutoFit writes
    the zip when the search finishes, so it dates the *completion* rather than
    any later touch of the directory — and the directory's own timestamp
    otherwise, which is the unzipped case (an interrupted run, or a test-mode
    one). The full directory string breaks ties, so the selection is
    deterministic rather than dependent on iteration order.

    The survivors are wrapped back into an ``Aggregator``, so everything
    downstream of this call treats the de-duplicated set as an ordinary
    aggregator; the grid-search outputs are carried across untouched.
    """
    from autofit.aggregator.aggregator import Aggregator

    selected = {}
    for result in aggregator:
        relative = result.directory.relative_to(sample_root)
        if len(relative.parts) < 4:
            continue
        lens_name, stage, band = relative.parts[:3]
        key = (lens_name, stage, band)
        zip_path = Path(f"{result.directory}.zip")
        timestamp = (
            zip_path.stat().st_mtime
            if zip_path.exists()
            else result.directory.stat().st_mtime
        )
        candidate = (timestamp, str(result.directory), result)
        if key not in selected or candidate[:2] > selected[key][:2]:
            selected[key] = candidate

    deduplicated = [candidate[2] for candidate in selected.values()]
    removed = len(aggregator) - len(deduplicated)
    if removed:
        print(f"selected newest result per lens/band; removed {removed} duplicates")

    return Aggregator(
        search_outputs=deduplicated,
        grid_search_outputs=aggregator.grid_search_outputs,
    )


def main():
    """
    __Query: Every Band Under One Tag__

    Note what is *missing* from ``parse_args`` above: there is no
    ``--search_name``. ``lens_mass.py`` and the two Sersic producers narrow to
    one named search per lens; this producer deliberately does not, because the
    search name here is the waveband and keeping all of them is the entire point.

    So there is a single query, on ``unique_tag``, and it returns every band's
    search for every lens in the sample — which is why the master CSV has one row
    per ``(lens, waveband)`` rather than one row per lens.

    ``AggregateCSV`` raises ``ValueError`` when handed an empty aggregator, which
    is the ordinary case for a sample whose SED chain has not run yet. That is
    caught and reported rather than raised: stage 7 of the bundle builder should
    leave a partial bundle alone, not abort it.
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

    agg = Aggregator.from_directory(directory=sample_root, completed_only=True)
    agg_query = agg.query(agg.unique_tag == args.unique_tag)
    agg_query = latest_result_per_lens_band(agg_query, sample_root=sample_root)

    try:
        agg_csv = af.AggregateCSV(aggregator=agg_query)
    except ValueError as e:
        print(f"no completed {args.unique_tag} results: {e}")
        return

    """
    __Three Label Columns__

    ``lens_name`` is the dataset directory name, as everywhere else, and is what
    the per-lens split groups on — here gathering the several band rows of one
    lens into that lens's folder.

    ``waveband`` is ``search.name``, which the SED chain set to the band. It is
    the column that makes a row identifiable at all: without it the rows of one
    lens are indistinguishable, and an SED fit has no way to attach a flux to a
    filter.

    ``crval_ra_deg`` comes from the ``wcs.json`` this pipeline's
    ``util.AnalysisImaging`` writes beside each result, and reads it back through
    the aggregator like any other result file. Despite the FITS-style name it is
    not the cut-out's reference pixel: it is the right ascension of the fitted
    maximum-likelihood lens light centre, which is what lets a catalogue row be
    matched back to a position on the sky rather than to a filename.
    """
    # Label columns: lens_name (the dataset dir), waveband (the search name,
    # e.g. vis / nir_h) and the WCS RA centre for cross-referencing.
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
    __Six Columns Per Variable__

    Four value types, and so six columns for every variable below: the
    maximum-likelihood value, the median, and the lower/upper bounds at 1σ and
    3σ. This is the one master CSV that carries the max-likelihood value as well
    as the median — ``lens_mass.csv`` and the two Sersic tables take the same
    three posterior types without it.

    The extra column is there because photometry is what gets fitted downstream.
    An SED or photometric-redshift fit wants the flux of the single best model
    alongside the marginalised one, and the gap between the two is itself a
    useful warning that a band's posterior is not well behaved.
    """
    value_types = (
        af.ValueType.MaxLogLikelihood,
        af.ValueType.Median,
        af.ValueType.ValuesAt3Sigma,
        af.ValueType.ValuesAt1Sigma,
    )

    """
    __The Flux Latents, And Why They Are In µJy__

    None of the eight variables below is a model parameter. Each is a latent
    variable: recomputed from the model on every sample by ``util.LatentEuclid``,
    so each arrives with a posterior rather than as a number derived once from a
    best fit. The ``latent.`` prefix is what tells ``add_variable`` to read the
    latent summary rather than the samples.

    They come in two groups, which is why the argument names look inconsistent.
    The first are library latents that ``config/latent.yaml`` enables — the total
    lens, lensed-source and source fluxes, and the magnification. The four
    ``*_fwhm_mujy`` names are Euclid-only, listed in
    ``util.LatentEuclid.APERTURE_LATENT_KEYS`` rather than in the library,
    because they need PSF arguments that only this pipeline supplies. ``name``
    shortens every header to the DR1 catalogue's form, so the CSV says
    ``lens_flux`` where the argument says ``latent.total_lens_flux_mujy``.

    The units matter more than anything else here. A fitted flux is in the
    image's own units, which differ from band to band and are useless for an SED.
    The ``_mujy`` latents convert through the ``MAGZERO`` zero-point that
    ``util.load_vis_dataset`` reads out of the FITS header and passes into the
    fit's info dict, giving microJansky — a physical unit comparable across every
    instrument in the table. A fit run without ``magzero`` cannot produce these,
    which is why ``catalogue/README.md`` lists it as a requirement rather than a
    detail.

    The four aperture fluxes go one step further: they are matched-aperture
    photometry, the model lens image convolved to the resolution of the
    worst-seeing band recorded for the cut-out and summed within 1, 2, 3 and 4
    times *that* band's PSF FWHM. Measuring every band through the same effective
    aperture is what makes the per-band fluxes comparable at all;
    "__The FITS Header Contract__" in ``start_here.py`` covers where that
    worst-band PSF comes from.

    Two ways a value column comes back empty, neither of them a fault in this
    script. A test-mode fit writes no latent summary at all, so every value
    column is blank while the header stays the right width. And a fit whose
    analysis carried no worst-band PSF produces NaN for the four aperture latents
    alone — they are then dropped from the written latent summary, so those four
    columns come out blank while the fluxes and magnification beside them are
    still good.
    """
    # Latent names are those of `util.LatentEuclid`: the library µJy flux
    # latents enabled in config/latent.yaml, then the four Euclid-only FWHM
    # aperture-flux latents. Column names keep the shorter DR1 catalogue form.
    latent_args = [
        ("latent.total_lens_flux_mujy", "lens_flux"),
        ("latent.total_lens_flux_1_fwhm_mujy", "lens_flux_1_fwhm"),
        ("latent.total_lens_flux_2_fwhm_mujy", "lens_flux_2_fwhm"),
        ("latent.total_lens_flux_3_fwhm_mujy", "lens_flux_3_fwhm"),
        ("latent.total_lens_flux_4_fwhm_mujy", "lens_flux_4_fwhm"),
        ("latent.total_lensed_source_flux_mujy", "lensed_source_flux"),
        ("latent.total_source_flux_mujy", "source_flux"),
        ("latent.magnification", "magnification"),
    ]
    for argument, name in latent_args:
        agg_csv.add_variable(argument=argument, name=name, value_types=value_types)

    """
    __Saving, And The Per-Lens Split__

    This is the one producer where the two printed counts legitimately differ.
    The master ``magnitudes.csv`` has a row per ``(lens, waveband)``, so the row
    count is roughly the lens count times the number of bands fitted, while
    ``write_per_tile_csv`` returns the number of *lenses* it wrote a file for.

    ``write_per_tile_csv`` handles that without being told: it groups on
    ``lens_name``, so a lens's whole set of band rows lands together in
    ``<inspect_dir>/<lens_name>/magnitudes.csv`` — that lens's own small SED
    table, self-contained beside its ``fit_multi_wavelength.png``.
    """
    out_csv = inspect_path / "magnitudes.csv"
    agg_csv.save(path=out_csv)
    print(f"wrote {out_csv} ({len(lens_name_list)} rows)")

    written = catalogue_util.write_per_tile_csv(
        master_csv=out_csv, inspect_path=inspect_path, filename="magnitudes.csv"
    )
    print(f"wrote {written} per-lens magnitudes.csv files")


if __name__ == "__main__":
    main()
