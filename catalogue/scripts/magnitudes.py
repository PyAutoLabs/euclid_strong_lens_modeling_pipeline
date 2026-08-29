"""
Euclid Catalogue: Multi-band Magnitudes CSV
============================================

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

    Re-running a waveband writes a second result directory beside the first;
    without this the master CSV gains duplicate rows for the same photometry.
    Recency is taken from the zipped result where one exists (the zip is written
    when the search finishes) and from the directory otherwise.
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

    value_types = (
        af.ValueType.MaxLogLikelihood,
        af.ValueType.Median,
        af.ValueType.ValuesAt3Sigma,
        af.ValueType.ValuesAt1Sigma,
    )

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

    out_csv = inspect_path / "magnitudes.csv"
    agg_csv.save(path=out_csv)
    print(f"wrote {out_csv} ({len(lens_name_list)} rows)")

    written = catalogue_util.write_per_tile_csv(
        master_csv=out_csv, inspect_path=inspect_path, filename="magnitudes.csv"
    )
    print(f"wrote {written} per-lens magnitudes.csv files")


if __name__ == "__main__":
    main()
