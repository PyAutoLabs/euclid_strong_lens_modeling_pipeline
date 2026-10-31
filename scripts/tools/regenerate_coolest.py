"""
Euclid Pipeline: Regenerate The COOLEST Templates In The Sky Frame
=================================================================

The ``coolest.json`` / ``coolest_sersic.json`` templates of an inspection
bundle or catalogue are copied out of each fit's ``files/coolest.json``. Fits
made before 2026-10-31 exported the tracer in the native model frame, which for
the north-up DR1 cut-outs has +y south (``sky_frame``), while COOLEST assumes +y
north: their ``phi`` is the negative of the sky position angle and their
``center_y`` has the wrong sign (``docs/sky_orientation/phase1.md``).

This tool rewrites each template from the fit's stored maximum log likelihood
tracer (``files/tracer.json``), mirrored into the +y-north frame by
``sky_frame.plus_y_north_from`` through the cut-out's own WCS, with the
original template's metadata kept. Nothing is refit and the result zips are
never touched; only the templates in ``--target_dir`` change.

Approved JSON change (human, 2026-10-31, euclid#115). Every run therefore:

1. **verifies the input** — the tracer exported *without* the mirror must
   reproduce the stored template's parameters exactly; a lens where it does not
   is left unchanged and reported (``status=input_mismatch``);
2. **snapshots** every original it replaces into
   ``<report_dir>/pre_sky_frame/<lens>/`` with a sha256 manifest;
3. **reports** a per-file diff (``<report_dir>/coolest_regeneration_report.csv``):
   each oriented parameter before and after, the sha256 of both files, and status
   (one timestamped report per run; the snapshot manifest is append-only).

``<report_dir>`` defaults to ``<target_dir>_coolest_regeneration/``, a sibling of
the bundle, never inside it: the bundle and catalogue stages treat every
sub-directory of ``<target_dir>`` as a lens.

A template already written in the sky frame (``meta.frame`` set by the
post-2026-10-31 pipeline) is left as it is (``status=already``).

Usage
-----
    python scripts/tools/regenerate_coolest.py \
        --target_dir=inspect/dr1_sep1 \
        --output_path=output/dr1_sep1 --sersic_output_path=output_sed/dr1_sep1 \
        --dataset_path=dataset/dr1_sep1 [--dry_run]

    # the bundle layout (stage 1b of scripts/build_inspection_bundle.sh)
    python scripts/tools/regenerate_coolest.py --target_dir=inspect/dr1_sep1 \
        --sample=dr1_sep1 --output_path=output --sersic_output_path=output_sed
"""
import argparse
import csv
import glob
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

import sky_frame  # noqa: E402

SNAPSHOT_DIR = "pre_sky_frame"
REPORT_NAME = "coolest_regeneration_report.csv"
FRAME = "+y north, +x west (sky_frame.plus_y_north_from)"

# template name -> (search tree flag, search sub-path)
TEMPLATES = {
    "coolest.json": ("output", Path("initial_lens_model") / "vis_pix"),
    "coolest_sersic.json": ("sersic", Path("sersic_lens_model") / "vis"),
}

ORIENTED = ("phi", "center_x", "center_y", "phi_ext")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_source(search_dir: Path):
    """The newest result zip of a search, or its newest unzipped result dir."""
    if not search_dir.is_dir():
        return None
    zips = sorted(search_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if zips:
        return zips[-1]
    dirs = sorted([p for p in search_dir.iterdir() if (p / "files" / "tracer.json").exists()],
                  key=lambda p: p.stat().st_mtime)
    return dirs[-1] if dirs else None


def read_member(source: Path, member: str) -> bytes:
    if source.is_dir():
        return (source / member).read_bytes()
    with zipfile.ZipFile(source) as archive:
        return archive.read(member)


def cutout_cd(dataset_dir: Path):
    """CD matrix of the lens's cut-out (its first celestial image HDU)."""
    paths = [p for p in glob.glob(str(dataset_dir / "*.fits")) if "segmentation" not in p]
    if not paths:
        raise FileNotFoundError(f"no cut-out FITS in {dataset_dir}")
    with fits.open(paths[0]) as hdul:
        for hdu in hdul:
            if hdu.data is not None and "CTYPE1" in hdu.header:
                return sky_frame.cd_matrix_from(hdu.header)
    raise ValueError(f"no celestial WCS in {paths[0]}")


def oriented_values(template: dict) -> dict:
    """Every oriented point estimate of a template, keyed entity/profile/param."""
    values = {}
    for entity in template.get("lensing_entities", []):
        for kind in ("mass_model", "light_model"):
            for profile in entity.get(kind, []) or []:
                for name, param in (profile.get("parameters") or {}).items():
                    if name in ORIENTED and isinstance(param, dict):
                        value = (param.get("point_estimate") or {}).get("value")
                        if value is not None:
                            values[f"{entity.get('name')}/{profile.get('type')}/{name}"] = float(value)
    return values


def all_values(template: dict) -> dict:
    """Every point estimate, for the input check."""
    values = {}
    for entity in template.get("lensing_entities", []):
        for kind in ("mass_model", "light_model"):
            for i, profile in enumerate(entity.get(kind, []) or []):
                for name, param in (profile.get("parameters") or {}).items():
                    if isinstance(param, dict):
                        value = (param.get("point_estimate") or {}).get("value")
                        if isinstance(value, (int, float)):
                            values[f"{entity.get('name')}/{kind}/{i}/{name}"] = float(value)
    return values


def export(tracer, shape, metadata, mirror):
    import autolens as al

    if mirror:
        tracer = sky_frame.plus_y_north_from(tracer)
    dataset = SimpleNamespace(shape_native=tuple(shape), pixel_scales=(sky_frame.DR1_PIXEL_SCALE,) * 2)
    with tempfile.TemporaryDirectory() as tmp:
        path = al.interop.coolest.to_coolest(
            galaxies=tracer, file_path=Path(tmp) / "coolest", dataset=dataset,
            on_unsupported="skip", metadata=dict(metadata),
        )
        with open(path) as f:
            return json.load(f)


def regenerate_one(template_path: Path, source: Path, dataset_dir: Path, snapshot_dir: Path, dry_run: bool):
    from autonerves.dictable import from_json

    original = json.loads(template_path.read_text())
    meta = original.get("meta", {})
    row = dict(file=str(template_path), source=str(source), sha256_before=sha256(template_path))
    if "frame" in meta:
        return dict(row, status="already")

    with tempfile.TemporaryDirectory() as tmp:
        tracer_path = Path(tmp) / "tracer.json"
        tracer_path.write_bytes(read_member(source, "files/tracer.json"))
        tracer = from_json(str(tracer_path))
    pixels = original.get("observation", {}).get("pixels", {})
    data_shape = (int(pixels.get("num_pix_y") or 100), int(pixels.get("num_pix_x") or 100))

    keep = {k: v for k, v in meta.items()
            if k not in ("skipped_profiles", "coolest.__version__", "generated_by")}
    native = export(tracer, data_shape, keep, mirror=False)
    before, check = all_values(original), all_values(native)
    if set(before) != set(check) or any(abs(before[k] - check[k]) > 1e-9 for k in before):
        return dict(row, status="input_mismatch")

    try:
        cd = cutout_cd(dataset_dir)
    except FileNotFoundError:
        # A missing input is a per-lens skip, as in every bundle stage: the
        # template stays as it was and the report says why.
        return dict(row, status="no_cutout")
    mirrored = sky_frame.native_is_mirrored(cd)
    keep["frame"] = FRAME
    new = export(tracer, data_shape, keep, mirror=mirrored)

    b, a = oriented_values(original), oriented_values(new)
    for key in sorted(b):
        row[f"before:{key}"] = b[key]
        row[f"after:{key}"] = a.get(key)
    if dry_run:
        return dict(row, status="would_regenerate")

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, snapshot_dir / template_path.name)
    staged = template_path.with_suffix(".json.tmp")
    staged.write_text(json.dumps(new, indent=4))
    staged.replace(template_path)
    return dict(row, status="regenerated", sha256_after=sha256(template_path))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target_dir", required=True, help="inspect/ or catalogue dir of <lens>/ folders")
    p.add_argument("--output_path", required=True, help="results tree holding <lens>/initial_lens_model")
    p.add_argument("--sersic_output_path", default=None, help="results tree holding <lens>/sersic_lens_model")
    p.add_argument("--dataset_path", default=None,
                   help="dataset tree holding <lens>/<lens>.fits (default: project_paths.dataset_root())")
    p.add_argument("--sample", default=None,
                   help="sample subdirectory appended to --output_path, --sersic_output_path and "
                        "--dataset_path (the build_inspection_bundle.sh layout)")
    p.add_argument("--report_dir", default=None,
                   help="where the snapshot and report go (default: <target_dir>_coolest_regeneration)")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    target = Path(a.target_dir)
    if not target.is_dir():
        print(f"{target}: no such directory; nothing to regenerate")
        return 0
    if a.dataset_path is None:
        try:
            import project_paths

            a.dataset_path = str(project_paths.dataset_root())
        except ImportError:  # a mirror of scripts/ alone (the inspection bundle) has no project_paths.py
            a.dataset_path = str(PROJECT_ROOT / "dataset")
    report_dir = Path(a.report_dir) if a.report_dir else target.with_name(target.name + "_coolest_regeneration")
    report_dir.mkdir(parents=True, exist_ok=True)
    trees = {"output": Path(a.output_path), "sersic": Path(a.sersic_output_path or a.output_path)}
    dataset_tree = Path(a.dataset_path)
    if a.sample:
        trees = {k: v / a.sample for k, v in trees.items()}
        dataset_tree = dataset_tree / a.sample
    rows = []
    for lens_dir in sorted(d for d in target.iterdir() if d.is_dir()):
        for name, (tree, sub) in TEMPLATES.items():
            template = lens_dir / name
            if not template.exists():
                continue
            source = latest_source(trees[tree] / lens_dir.name / sub)
            if source is None:
                rows.append(dict(file=str(template), status="no_result"))
                continue
            try:
                rows.append(regenerate_one(template, source, dataset_tree / lens_dir.name,
                                           report_dir / SNAPSHOT_DIR / lens_dir.name, a.dry_run))
            except Exception as e:  # noqa: BLE001 — report and continue
                rows.append(dict(file=str(template), status=f"error:{type(e).__name__}: {e}"))

    stamp = time.strftime("%Y%m%dT%H%M%S")
    suffix = f"_{stamp}" + ("_dry_run" if a.dry_run else "")
    report = report_dir / REPORT_NAME.replace(".csv", f"{suffix}.csv")
    fields = ["file", "status", "source", "sha256_before", "sha256_after"]
    fields += sorted({k for r in rows for k in r} - set(fields))
    with open(report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if not a.dry_run and (report_dir / SNAPSHOT_DIR).exists():
        # Append-only: a rerun reports `already` and must not erase the record
        # of what an earlier run replaced.
        with open(report_dir / SNAPSHOT_DIR / "manifest_sha256.txt", "a") as f:
            for r in rows:
                if r.get("status") == "regenerated":
                    f.write(f"{r['sha256_before']}  {r['file']}\n")

    counts = {}
    for r in rows:
        counts[r["status"].split(":")[0]] = counts.get(r["status"].split(":")[0], 0) + 1
    print(f"{len(rows)} templates -> {report}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    return 1 if any(k in counts for k in ("error", "input_mismatch")) else 0


if __name__ == "__main__":
    sys.exit(main())
