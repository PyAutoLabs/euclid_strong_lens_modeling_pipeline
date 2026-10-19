"""
Euclid Pipeline: CSV Snapshot Assembler
=======================================

Stage 2 of ``scripts/tools/csv_snapshot/build.py``. Turns the collected JSON
lines (``<out_dir>/_build/results*.jsonl.gz``) into the CSV tables of a
CSV-only snapshot, from stored posterior summaries only (no refitting):

- ``dr1_catalogue_<N>_master.csv``: one row per included candidate (N = the
  manifest's ``include_in_catalogue=True`` count): identity, v5 grade/score and
  sky position, completion flags, posterior medians and 1-sigma bounds of the
  mass, VIS Sersic light and per-band photometry.
- ``lens_mass.csv``, ``lens_sersic.csv``, ``source_sersic.csv``,
  ``photometry.csv``, ``astrometric_offsets.csv``, ``wcs.csv``,
  ``source_clumps.csv``, ``result_provenance.csv``, ``metadata_errors.csv``.
- ``sample_manifest.csv`` (the included manifest rows) and
  ``excluded_<n>_candidates.csv`` (the parent A/B objects not included, with
  the reason: ``catalogue/grades/dr1_sep1_uncovered_ab.csv``).

Column names and value conventions are exactly those of the 2026-10-11
``dr1_catalogue_15067_20261011_csv_only`` release. Counts go to
``<out_dir>/_build/validation_counts.json`` for ``package.py``.

Usage
-----
    python scripts/tools/csv_snapshot/assemble.py --out_dir=inspect/dr1_catalogue_v1_1_csv
"""

import argparse
import csv
import datetime
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BUILD_DIR,
    DEFAULT_MANIFEST,
    DEFAULT_OBJECTS,
    DEFAULT_UNCOVERED,
    PROJECT_ROOT,
    excluded_name,
    included_rows,
    master_name,
    read_csv,
    write_json,
)

SUFFIXES = ["", "_max_lh", "_lower_1_sigma", "_upper_1_sigma", "_lower_3_sigma", "_upper_3_sigma"]
MASS = {
    k: "galaxies.lens.mass." + p
    for k, p in [
        ("centre_0", "centre.centre_0"),
        ("centre_1", "centre.centre_1"),
        ("ell_comps_0", "ell_comps.ell_comps_0"),
        ("ell_comps_1", "ell_comps.ell_comps_1"),
        ("einstein_radius", "einstein_radius"),
    ]
}
MASS.update(
    shear_gamma_1="fields.shear.gamma_1",
    shear_gamma_2="fields.shear.gamma_2",
    effective_einstein_radius="effective_einstein_radius",
    magnification="magnification",
)
LIGHT = dict(
    centre_0="centre.centre_0",
    centre_1="centre.centre_1",
    ell_comps_0="ell_comps.ell_comps_0",
    ell_comps_1="ell_comps.ell_comps_1",
    effective_radius="effective_radius",
    sersic_index="sersic_index",
)
FLUX = dict(
    lens_flux="total_lens_flux_mujy",
    lensed_source_flux="total_lensed_source_flux_mujy",
    source_flux="total_source_flux_mujy",
    magnification="magnification",
)
FLUX.update({f"lens_flux_{i}_fwhm": f"total_lens_flux_{i}_fwhm_mujy" for i in range(1, 5)})
OFFSET = {
    "grid_offset_y": "dataset_model.grid_offset.grid_offset_0",
    "grid_offset_x": "dataset_model.grid_offset.grid_offset_1",
}
BASE = ["unique_identifier", "tile_id", "grade", "score", "RA_obj", "DEC_obj", "cutout_batch"]
STATUS = [
    "vis_lp_completed", "vis_pix_completed", "vis_pix_summary_available",
    "vis_pix_latents_available", "sersic_vis_completed", "sersic_vis_summary_available",
    "expected_bands", "completed_bands", "photometry_bands", "missing_bands",
    "missing_photometry_bands", "expected_band_count", "completed_band_count",
    "photometry_band_count", "sed_fully_complete", "photometry_fully_available",
    "metadata_error_count",
]
ID = ["unique_identifier", "tile_id", "id"]
MASTER_DROP = ("_max_lh", "_lower_3_sigma", "_upper_3_sigma")

TABLE_FILES = {
    "mass": "lens_mass.csv",
    "lens": "lens_sersic.csv",
    "source": "source_sersic.csv",
    "flux": "photometry.csv",
    "offset": "astrometric_offsets.csv",
    "wcs": "wcs.csv",
    "clump": "source_clumps.csv",
    "provenance": "result_provenance.csv",
    "error": "metadata_errors.csv",
}


def clean(v):
    if v is None:
        return ""
    if isinstance(v, float) and not math.isfinite(v):
        return ""
    if isinstance(v, (list, tuple)):
        return ";".join(str(clean(x)) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, separators=(",", ":"))
    return v


def cols(mapping):
    return [k + s for k in mapping for s in SUFFIXES]


def extract(r, mapping):
    """Median, max-likelihood and 1/3-sigma bounds of every ``mapping`` path
    found in the result's sampled summary, else its latent summary."""
    out = {}
    if not r:
        return out
    for label, path in mapping.items():
        for summary in [r.get("summary", {}), r.get("latent", {})]:
            median = summary.get("median_pdf_sample", {}).get("kwargs", {})
            key = path
            if key not in median and key.startswith("fields.shear."):
                key = key.replace("fields.shear.", "galaxies.lens.shear.")
            if key not in median:
                continue
            out[label] = clean(median[key])
            out[label + "_max_lh"] = clean(
                summary.get("max_log_likelihood_sample", {}).get("kwargs", {}).get(key)
            )
            for sigma in (1, 3):
                bounds = summary.get(f"values_at_sigma_{sigma}", {}).get(key)
                if bounds is None:
                    errors = summary.get(f"errors_at_sigma_{sigma}", {}).get(key)
                    if errors is not None:
                        bounds = [median[key] - errors[0], median[key] + errors[1]]
                if bounds is not None:
                    out[f"{label}_lower_{sigma}_sigma"] = clean(bounds[0])
                    out[f"{label}_upper_{sigma}_sigma"] = clean(bounds[1])
    return out


class Table:
    def __init__(self, path, fields):
        self.path = Path(path)
        self.file = open(self.path, "w", newline="")
        self.w = csv.DictWriter(self.file, fields, lineterminator="\n")
        self.w.writeheader()
        self.n = 0

    def row(self, row):
        self.w.writerow({k: clean(v) for k, v in row.items()})
        self.n += 1

    def close(self):
        self.file.close()


def result_files(out_dir: Path):
    build = Path(out_dir) / BUILD_DIR
    whole = build / "results.jsonl.gz"
    shards = sorted(build.glob("results_*.jsonl.gz"), key=lambda p: int(p.stem.split("_")[1].split(".")[0]))
    if whole.exists() and shards:
        raise SystemExit(f"{build}: both results.jsonl.gz and shard files exist; remove one set")
    paths = [whole] if whole.exists() else shards
    if not paths:
        raise SystemExit(f"{build}: no collected results (run collect.py)")
    return paths


def records(out_dir):
    for path in result_files(out_dir):
        with gzip.open(path, "rt") as stream:
            for line in stream:
                yield json.loads(line)


def assemble(out_dir, manifest, objects_csv, uncovered_csv, force=False, log=print):
    out_dir = Path(out_dir)
    manifest_rows = included_rows(manifest)
    manifest_by_uid = {r["unique_identifier"]: r for r in manifest_rows}
    n_included = len(manifest_rows)
    master_file = out_dir / master_name(n_included)
    if master_file.exists() and not force:
        raise SystemExit(f"{master_file} exists; a snapshot directory is written once (--force to redo)")

    # Discover actual band, WCS and clump columns in one streaming pass.
    bands, wcskeys, clumpkeys, n = set(), set(), set(), 0
    for d in records(out_dir):
        n += 1
        bands.update(d.get("expected_bands", []))
        bands.update(d["sed"])
        for r in [d.get("vis_lp"), d.get("vis_pix"), *d["sed"].values()]:
            if not r:
                continue
            w = r.get("wcs", {})
            wcskeys.update(k for k in w if k != "source_clumps")
            for c in w.get("source_clumps", []):
                clumpkeys.update(c)
    if n != n_included:
        raise SystemExit(f"collected {n} records, manifest includes {n_included}")
    bands, wcskeys, clumpkeys = sorted(bands), sorted(wcskeys), sorted(clumpkeys)

    masterfields = BASE + STATUS + [
        prefix + k
        for prefix, mapping in [("mass_", MASS), ("lens_light_", LIGHT), ("source_light_", LIGHT)]
        for k in cols(mapping)
        if not k.endswith(MASTER_DROP)
    ]
    masterfields += [
        band + "_" + k
        for band in bands
        for k in cols({k: v for k, v in FLUX.items() if "_fwhm" not in k})
        if not k.endswith(MASTER_DROP)
    ]
    tables = {
        "master": Table(master_file, masterfields),
        "mass": Table(out_dir / TABLE_FILES["mass"], ID + cols(MASS)),
        "lens": Table(out_dir / TABLE_FILES["lens"], ID + cols(LIGHT)),
        "source": Table(out_dir / TABLE_FILES["source"], ID + cols(LIGHT)),
        "flux": Table(out_dir / TABLE_FILES["flux"], ID + ["waveband", "crval_ra_deg", "crval_dec_deg"] + cols(FLUX)),
        "offset": Table(out_dir / TABLE_FILES["offset"], ID + ["waveband"] + cols(OFFSET)),
        "wcs": Table(out_dir / TABLE_FILES["wcs"], ID + ["stage", "waveband"] + wcskeys),
        "clump": Table(out_dir / TABLE_FILES["clump"], ID + ["stage", "waveband", "clump_index"] + clumpkeys),
        "provenance": Table(
            out_dir / TABLE_FILES["provenance"],
            ID + ["stage", "waveband", "result_path", "result_mtime_utc", "summary_available",
                  "latents_available", "wcs_available"],
        ),
        "error": Table(out_dir / TABLE_FILES["error"], ["unique_identifier", "tile_id", "path", "error"]),
    }
    objects = {
        r["Unique_Identifier"]: r
        for r in read_csv(objects_csv)
        if r.get("Unique_Identifier") in manifest_by_uid or r.get("SWAP_Grade") in ("A", "B")
    }
    stats = Counter()
    seen = set()
    expected_sources = Counter()
    for d in records(out_dir):
        m = d["manifest"]
        uid = m["unique_identifier"]
        tile = m["tile"]
        obj = objects.get(uid)
        if uid in seen:
            raise SystemExit(f"{uid} collected twice")
        if uid not in manifest_by_uid or manifest_by_uid[uid]["tile"] != tile:
            raise SystemExit(f"{uid}/{tile}: not an included manifest row")
        if obj is None or obj["SWAP_Grade"] != m["grade"]:
            raise SystemExit(f"{uid}: missing from the object catalogue or grade differs")
        seen.add(uid)
        master = dict(
            unique_identifier=uid, tile_id=tile, grade=m["grade"], score=obj["SWAP_Score"],
            RA_obj=obj["RA_obj"], DEC_obj=obj["DEC_obj"], cutout_batch=m["cutout_batch"],
        )
        expected = set(d.get("expected_bands", []))
        expected_sources[d.get("expected_bands_source", "none")] += 1
        complete = set(d["sed"])
        photometry = {
            b
            for b, r in d["sed"].items()
            if all(extract(r, FLUX).get(k, "") != "" for k in ("lens_flux", "lensed_source_flux", "source_flux", "magnification"))
        }
        pix = d.get("vis_pix")
        lp = d.get("vis_lp")
        vis = d["sed"].get("vis")
        status = dict(
            vis_lp_completed=bool(lp),
            vis_pix_completed=bool(pix),
            vis_pix_summary_available=bool(pix and pix.get("summary")),
            vis_pix_latents_available=bool(pix and pix.get("latent")),
            sersic_vis_completed=bool(vis),
            sersic_vis_summary_available=bool(vis and vis.get("summary")),
            expected_bands=sorted(expected),
            completed_bands=sorted(complete),
            photometry_bands=sorted(photometry),
            missing_bands=sorted(expected - complete),
            missing_photometry_bands=sorted(expected - photometry),
            expected_band_count=len(expected),
            completed_band_count=len(complete),
            photometry_band_count=len(photometry),
            sed_fully_complete=bool(expected and expected <= complete),
            photometry_fully_available=bool(expected and expected <= photometry),
            metadata_error_count=len(d["errors"]),
        )
        master.update(status)
        stats["rows"] += 1
        stats["grade_" + m["grade"]] += 1
        for k, v in status.items():
            if isinstance(v, bool):
                stats[k] += int(v)
        stats["metadata_errors"] += len(d["errors"])
        stats["input_fits_missing"] += not d["input_fits"]
        for kind, r, mapping, prefix in [
            ("mass", pix, MASS, "mass_"),
            ("lens", vis, {k: "galaxies.lens.bulge." + v for k, v in LIGHT.items()}, "lens_light_"),
            ("source", vis, {k: "galaxies.source.bulge." + v for k, v in LIGHT.items()}, "source_light_"),
        ]:
            values = extract(r, mapping)
            if values:
                tables[kind].row(dict(unique_identifier=uid, tile_id=tile, id=r["id"], **values))
                master.update({prefix + k: v for k, v in values.items() if prefix + k in masterfields})
        results = [("initial_lens_model", "vis_lp", lp), ("initial_lens_model", "vis_pix", pix)]
        results += [("sersic_lens_model", b, r) for b, r in d["sed"].items()]
        for stage, band, r in results:
            if not r:
                continue
            ids = dict(unique_identifier=uid, tile_id=tile, id=r["id"])
            tables["provenance"].row(
                dict(
                    ids, stage=stage, waveband=band, result_path=r["path"],
                    result_mtime_utc=datetime.datetime.fromtimestamp(r["mtime"], datetime.timezone.utc).isoformat(),
                    summary_available=bool(r.get("summary")),
                    latents_available=bool(r.get("latent")),
                    wcs_available=bool(r.get("wcs")),
                )
            )
            w = r.get("wcs", {})
            if w:
                tables["wcs"].row(dict(ids, stage=stage, waveband=band, **{k: v for k, v in w.items() if k != "source_clumps"}))
                for i, c in enumerate(w.get("source_clumps", [])):
                    tables["clump"].row(dict(ids, stage=stage, waveband=band, clump_index=i, **c))
            if stage == "sersic_lens_model":
                values = extract(r, FLUX)
                if values:
                    tables["flux"].row(
                        dict(ids, waveband=band, crval_ra_deg=w.get("crval_ra_deg", ""),
                             crval_dec_deg=w.get("crval_dec_deg", ""), **values)
                    )
                    master.update({band + "_" + k: v for k, v in values.items() if band + "_" + k in masterfields})
                values = extract(r, OFFSET)
                if values:
                    tables["offset"].row(dict(ids, waveband=band, **values))
        for e in d["errors"]:
            tables["error"].row(dict(unique_identifier=uid, tile_id=tile, **e))
        tables["master"].row(master)
    for table in tables.values():
        table.close()
    if seen != set(manifest_by_uid):
        raise SystemExit(f"collected {len(seen)} of the {len(manifest_by_uid)} included objects")

    # Parent A/B objects not included: no model, or a model that is not of the lens.
    uncovered = read_csv(uncovered_csv)
    parent_ab = {uid for uid, o in objects.items() if o.get("SWAP_Grade") in ("A", "B")}
    excluded = {r["unique_identifier"] for r in uncovered}
    if parent_ab - set(manifest_by_uid) != excluded:
        raise SystemExit(
            f"parent A/B objects minus included ({len(parent_ab - set(manifest_by_uid))}) "
            f"differ from {uncovered_csv} ({len(excluded)})"
        )
    with open(out_dir / excluded_name(len(uncovered)), "w", newline="") as stream:
        writer = csv.DictWriter(stream, list(uncovered[0]) if uncovered else ["unique_identifier"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(uncovered)
    with open(out_dir / "sample_manifest.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = dict(stats)
    report["table_rows"] = {k: t.n for k, t in tables.items()}
    report["bands"] = bands
    report["n_included"] = n_included
    report["parent_ab_count"] = len(parent_ab)
    report["excluded_parent_count"] = len(excluded)
    report["superseded_cutout_rows"] = sum(
        bool(o.get("fits_to_use")) and o.get("fits_modelled_20261011", "") not in ("", o["fits_to_use"])
        for uid, o in objects.items() if uid in manifest_by_uid
    )
    report["expected_bands_sources"] = dict(expected_sources)
    report["master_file"] = master_file.name
    report["excluded_file"] = excluded_name(len(uncovered))
    write_json(out_dir / BUILD_DIR / "validation_counts.json", report)
    log(json.dumps(report, indent=2))
    return report


def add_arguments(parser):
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--objects", default=str(DEFAULT_OBJECTS),
                        help="v5 object catalogue (Unique_Identifier, SWAP_Grade, SWAP_Score, RA_obj, DEC_obj, "
                        "fits_to_use, fits_modelled_20261011)")
    parser.add_argument("--uncovered", default=str(DEFAULT_UNCOVERED),
                        help="parent A/B objects not included (catalogue/grades/dr1_sep1_uncovered_ab.csv)")
    parser.add_argument("--force", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Assemble CSV snapshot tables from collected metadata.")
    parser.add_argument("--out_dir", required=True)
    add_arguments(parser)
    args = parser.parse_args(argv)
    root = PROJECT_ROOT
    p = lambda x: Path(x) if Path(x).is_absolute() else root / x  # noqa: E731
    assemble(p(args.out_dir), p(args.manifest), p(args.objects), p(args.uncovered), args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
