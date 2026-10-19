"""
Euclid Pipeline: CSV Snapshot Packager
======================================

Stage 3 of ``scripts/tools/csv_snapshot/build.py``: documents and validates the
tables ``assemble.py`` wrote into ``<out_dir>``.

- Checks: one master row per included manifest object, unique identifiers and
  tiles, grades equal to the manifest's, parent A/B objects = included +
  excluded, ``sample_manifest.csv`` row count. Every count comes from the
  manifest and the object catalogue, never a literal.
- Writes ``master_columns.csv`` (column, source table, description, units),
  ``README.txt``, ``validation.csv`` and ``file_manifest.csv`` (file, bytes,
  sha256, data rows), as in the 2026-10-11 release.

The optional ``--parity_json`` (``numeric_comparisons``, ``matched_result_rows``,
``difference_count``; optional ``candidates``, ``tolerance``) records an
extraction parity check against an earlier snapshot in README.txt and
validation.csv; without it no parity claim is made.

Usage
-----
    python scripts/tools/csv_snapshot/package.py --out_dir=inspect/dr1_catalogue_v1_1_csv
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BUILD_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_UNCOVERED,
    PROJECT_ROOT,
    included_rows,
    number_word,
    read_snapshot,
)

STATUS_TEXT = {
    "vis_lp_completed": "Completed initial parametric-source fit exists.",
    "vis_pix_completed": "Completed Delaunay-source fit exists; this does not certify scientific reliability.",
    "vis_pix_summary_available": "Stored Delaunay sampled-parameter posterior summary exists.",
    "vis_pix_latents_available": "Stored Delaunay derived-quantity posterior summary exists.",
    "sersic_vis_completed": "Completed VIS Sersic lens/source fit exists.",
    "sersic_vis_summary_available": "Stored VIS Sersic sampled-parameter posterior summary exists.",
    "expected_bands": "Semicolon-separated input BGSUB bands, from the existing band inventory where available, otherwise read from FITS.",
    "completed_bands": "Semicolon-separated bands with a completed Sersic/SED result.",
    "photometry_bands": "Bands with finite lens flux, lensed source flux, source flux and magnification medians.",
    "missing_bands": "Expected bands lacking a completed Sersic/SED result at snapshot cutoff.",
    "missing_photometry_bands": "Expected bands lacking one or more of the four principal photometry medians.",
    "expected_band_count": "Number of expected bands.",
    "completed_band_count": "Number of bands with completed Sersic/SED results.",
    "photometry_band_count": "Number of bands with all four principal photometry medians.",
    "sed_fully_complete": "All expected bands have a completed fit; outputs may still be missing.",
    "photometry_fully_available": "All expected bands have the four principal photometry medians.",
    "metadata_error_count": "Number of input/result metadata reading errors; see metadata_errors.csv.",
}


def describe(col):
    units = ""
    source = "sample_manifest.csv"
    desc = col
    if col in STATUS_TEXT:
        source = "result_provenance.csv and input band inventory"
        desc = STATUS_TEXT[col]
    elif col in ["RA_obj", "DEC_obj"]:
        source = "lens-finding catalogue v5"
        units = "deg"
        desc = "Original lens-finding object sky coordinate; not inferred from the cutout filename."
    elif col in ["grade", "score"]:
        source = "lens-finding catalogue v5"
        desc = "Lens-finding SWAP grade/score; not a model-success classification."
    elif col == "unique_identifier":
        desc = "Unique v5 object identifier; primary catalogue key."
    elif col == "tile_id":
        desc = "Canonical modelling cutout selected by the consolidated manifest."
    elif col == "cutout_batch":
        desc = "Delivery batch of the modelled cutout (sep1: 1 Sep 2026; sep28: 28 Sep 2026 top-up)."
    else:
        if col.startswith("mass_"):
            source, stem = "lens_mass.csv", col[5:]
            desc = "Delaunay-source SIE+external-shear result: " + stem
        elif col.startswith("lens_light_"):
            source, stem = "lens_sersic.csv", col[11:]
            desc = "VIS single-Sersic foreground light result: " + stem
        elif col.startswith("source_light_"):
            source, stem = "source_sersic.csv", col[13:]
            desc = "VIS single-Sersic source light result: " + stem
        else:
            source, stem = "photometry.csv", col
            desc = "Band-specific Sersic/SED result: " + stem
        if "flux" in stem:
            units = "microJy"
        elif any(v in stem for v in ["centre_", "einstein_radius", "effective_radius"]):
            units = "arcsec"
        else:
            units = "dimensionless"
        if "_lower_" in stem:
            desc += "; lower credible-interval bound, not an error magnitude."
        elif "_upper_" in stem:
            desc += "; upper credible-interval bound, not an error magnitude."
        else:
            desc += "; posterior median."
    return dict(column=col, source_table=source, description=desc, units=units)


def readme_text(snap, stats, n, master_file, excluded_file, excluded_rows, parity):
    n_excl = len(excluded_rows)
    if stats.get("expected_bands_sources", {}).get("band inventory (unchanged cutout)"):
        bands_text = (
            "Expected bands use the previous input-FITS band inventory where available,\n"
            "with FITS headers read for remaining cutouts."
        )
    else:
        bands_text = "Expected bands are read from the input FITS headers."
    exclusions = "\n".join(
        f"- {r['unique_identifier']}" + (f" ({r['reason']})" if r.get("reason") else "") + f": {r.get('note', '')}"
        for r in excluded_rows
    )
    if parity:
        lead = f"A {parity['candidates']:,}-candidate extraction check" if parity.get("candidates") else "An extraction check"
        tol = f" to tolerance {parity['tolerance']}" if parity.get("tolerance") is not None else ""
        verdict = "identical" if not parity.get("difference_count") else f"with {parity['difference_count']:,} differences"
        parity_text = (
            f"{lead} found\n{parity['numeric_comparisons']:,} numerical comparisons across "
            f"{parity['matched_result_rows']:,} matching old/new result rows {verdict}{tol}.\n"
            "This tests CSV extraction, not model validity. "
        )
    else:
        parity_text = ""
    title = "Euclid DR1: CSV-only collaborator snapshot"
    return f"""{title}
{'=' * len(title)}

Snapshot cutoff: {snap['cutoff_utc']}
Membership: exactly {n:,} distinct Grade A/B objects of the lens-finding catalogue
v5 (Holloway, 19 Oct 2026), from the consolidated manifest: the {stats['parent_ab_count']:,}
v5 A/B objects minus {number_word(n_excl)} without a usable model (listed below).
This is an interim modelling snapshot, not a fully completed or scientifically
validated model catalogue. No candidate is removed for a missing fit or band.

Start with {master_file}: one row per candidate. Blank cells mean
unavailable/non-finite measurements, never zero. Semicolons separate list values.
master_columns.csv defines every master column. unique_identifier is the v5
object key; tile_id is the cutout its model was fitted on (sample_manifest.csv).
No object was remodelled for v5. Objects whose model was fitted on an earlier
cutout than the v5 one centred on the final position (the fits stand):
{stats.get('superseded_cutout_rows', 0):,}.

Coverage at this snapshot
-------------------------
- Initial parametric-source (vis_lp) fits: {stats.get('vis_lp_completed', 0):,}.
- Delaunay-source (vis_pix) fits: {stats.get('vis_pix_completed', 0):,}.
- VIS single-Sersic lens/source fits: {stats.get('sersic_vis_completed', 0):,}.
- Every expected band has a completed SED fit: {stats.get('sed_fully_complete', 0):,}.
- Every expected band has all four main photometry medians: {stats.get('photometry_fully_available', 0):,}.
- Metadata read errors: {stats.get('metadata_errors', 0):,} (see metadata_errors.csv).

The completion and photometry flags deliberately differ: a completed search may
lack a latent summary or have non-finite derived quantities. Neither flag is a
visual-inspection success label. grade is a lens-finding grade, not a judgement
of model reliability. Do not treat a blank multiband measurement as a dropout.

Tables and numerical conventions
-------------------------------
- lens_mass.csv: SIE mass and external-shear parameters from vis_pix, effective
  Einstein radius and magnification from that fit's stored derived summaries.
  The effective Einstein radius is derived from the critical curve and differs
  from the SIE einstein_radius parameter.
- lens_sersic.csv / source_sersic.csv: six VIS single-Sersic light parameters.
  effective_radius is the profile's stored half-light-radius parameter (arcsec);
  it is not converted to a circularised radius. Source sizes are source-plane
  sizes. The Sersic fit fixes the lens mass from the initial parametric fit.
- photometry.csv: one row per available candidate/band derived summary. All
  fluxes, including aperture fluxes, are in MICROJANSKYS, not magnitudes.
  lens_flux is foreground light; lensed_source_flux is the observed lensed
  source; source_flux is reconstructed source-plane flux. Magnification is
  dimensionless. Conditional SED-fit uncertainties do not propagate uncertainty
  in the frozen mass model. Fluxes follow the pipeline's modelling masks.
- astrometric_offsets.csv: fitted (y,x) band grid offsets in arcsec relative to
  VIS, subtracted from the band grid before evaluating its model. No prior-edge
  quality flags are inferred in this snapshot.
- wcs.csv: stored fit astrometry for each stage/band. crval_ra_deg/dec_deg are
  the fitted maximum-likelihood lens-light sky centre, not FITS CRVAL headers;
  crpix_x/y refer to the 1-based WCS pixel of the image-plane origin. Columns
  ending _arcsec are angular offsets; _deg are degrees. Other stored fields
  retain their original names. Array cells use semicolons.
- source_clumps.csv: stored clump locations and associated image positions;
  zero-based clump_index is local to the fit, not a cross-band object match.
- result_provenance.csv: selected result IDs, relative paths and timestamps.
  Results are selected independently per candidate/stage/band by newest
  modification time at or before the cutoff, requiring a .completed marker.
  These timestamps record stored files (copies can change modification times).
  Fits completing after the cutoff are reserved for a future snapshot.

An unsuffixed scientific value is a posterior median. _max_lh is the stored
maximum-likelihood value. _lower_1_sigma / _upper_1_sigma and corresponding
3_sigma columns are interval BOUNDS, not +/- errors. The master carries medians
and 1-sigma bounds; the component tables additionally carry maximum-likelihood
values and 3-sigma bounds. Centres use PyAutoLens (y,x) = (centre_0,centre_1).
Ellipticity components, shear components and Sersic indices are dimensionless.
{bands_text} No new likelihood evaluations
or lens fits were performed for this export.
Witt-Wynne approximations and images/FITS are not regenerated or included here.

Excluded parent-catalogue candidates ({n_excl})
{'-' * (38 + len(str(n_excl)))}
{exclusions}

These objects have no lens model (never modelled) or a model that is not of the
lens; they are not failed fits or multiband losses. {excluded_file} supplies
positions, grade, the reason and the recorded notes.

Validation: membership and uniqueness were checked against the full v5 A/B
parent catalogue and selected manifest. {parity_text}File sizes,
SHA-256 hashes and table row counts are supplied in file_manifest.csv.
"""


def package(out_dir, manifest, uncovered_csv, parity_json=None, log=print):
    out_dir = Path(out_dir)
    stats = json.loads((out_dir / BUILD_DIR / "validation_counts.json").read_text())
    snap = read_snapshot(out_dir)
    manifest_rows = included_rows(manifest)
    n = len(manifest_rows)
    master_file = stats["master_file"]
    excluded_file = stats["excluded_file"]
    with open(out_dir / master_file, newline="") as stream:
        rows = list(csv.DictReader(stream))
    uids = {r["unique_identifier"] for r in rows}
    if not (len(rows) == n == len(uids) == len({r["tile_id"] for r in rows})):
        raise SystemExit(f"{master_file}: {len(rows)} rows / {len(uids)} ids, manifest includes {n}")
    if uids != {r["unique_identifier"] for r in manifest_rows}:
        raise SystemExit(f"{master_file}: identifiers differ from the manifest's included rows")
    grades = {r["grade"] for r in rows}
    if grades != {r["grade"] for r in manifest_rows}:
        raise SystemExit(f"{master_file}: grades {sorted(grades)} differ from the manifest's")
    with open(out_dir / excluded_file, newline="") as stream:
        excluded_rows = list(csv.DictReader(stream))
    excluded = {r["unique_identifier"] for r in excluded_rows}
    if excluded & uids or len(uids) + len(excluded) != stats["parent_ab_count"]:
        raise SystemExit("included + excluded objects do not make up the parent A/B catalogue")
    with open(out_dir / "sample_manifest.csv", newline="") as stream:
        if sum(1 for _ in csv.DictReader(stream)) != n:
            raise SystemExit("sample_manifest.csv row count differs from the manifest")
    parity = json.loads(Path(parity_json).read_text()) if parity_json else None

    with open(out_dir / "master_columns.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, ["column", "source_table", "description", "units"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(describe(col) for col in rows[0])
    (out_dir / "README.txt").write_text(
        readme_text(snap, stats, n, master_file, excluded_file, excluded_rows, parity)
    )
    with open(out_dir / "validation.csv", "w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["check", "value"])
        for k, v in stats.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in (
                "n_included", "parent_ab_count", "excluded_parent_count", "superseded_cutout_rows"
            ):
                writer.writerow([k, v])
        writer.writerow(["unique_identifiers", len(uids)])
        writer.writerow(["parent_ab_count", stats["parent_ab_count"]])
        writer.writerow(["excluded_parent_count", len(excluded)])
        writer.writerow(["snapshot_cutoff_utc", snap["cutoff_utc"]])
        if parity:
            writer.writerow(["parity_comparisons", parity["numeric_comparisons"]])
            writer.writerow(["parity_differences", parity.get("difference_count", 0)])
    with open(out_dir / "file_manifest.csv", "w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["file", "bytes", "sha256", "data_rows"])
        for path in sorted(out_dir.iterdir()):
            if not path.is_file() or path.name == "file_manifest.csv" or path.name.startswith("."):
                continue
            count = ""
            if path.suffix == ".csv":
                with open(path, newline="") as s:
                    count = sum(1 for _ in csv.reader(s)) - 1
            writer.writerow([path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), count])
    log(f"package: {out_dir} ({n} candidates, cutoff {snap['cutoff_utc']})")


def add_arguments(parser):
    parser.add_argument("--parity_json", default=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Document and validate a CSV snapshot.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--uncovered", default=str(DEFAULT_UNCOVERED))
    add_arguments(parser)
    args = parser.parse_args(argv)
    p = lambda x: Path(x) if Path(x).is_absolute() else PROJECT_ROOT / x  # noqa: E731
    package(p(args.out_dir), p(args.manifest), p(args.uncovered), args.parity_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
