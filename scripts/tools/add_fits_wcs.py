"""
Euclid Pipeline: Add The Cut-Out WCS To The Model FITS
======================================================

The per-lens model FITS of a bundle (``model.fits``, ``pre_psf.fits`` and the
mass maps ``convergence.fits`` / ``potential.fits`` / ``deflections.fits``) are
written by PyAutoLens with only its own grid keywords (``PIXSCAY`` /
``PIXSCAX`` / ``ORIGINY`` / ``ORIGINX``): no celestial WCS, so they are not
self-describing on the sky. This stage adds one to every image HDU, in place,
header only — the data arrays are never read into a new array or rewritten.

The arrays are in the cut-out's own pixel order (no row flip at load,
``sky_frame``), so a product on the full cut-out grid takes the cut-out band's
WCS unchanged. A product on a zoomed grid (the mass maps: the mask's zoom plus
a one-pixel buffer, e.g. 102x102) takes the same WCS with ``CRPIX`` shifted by
the zoom offset, computed from its ``ORIGINY`` / ``ORIGINX`` (arcsec) and shape:
array pixel ``(r, c)`` of an ``(n, m)`` product sits at cut-out pixel
``(r + oy, c + ox)`` with ``oy = (N - n) / 2 - ORIGINY / ps`` and
``ox = (M - m) / 2 + ORIGINX / ps`` for an ``(N, M)`` cut-out.

The band is the EXTNAME prefix (``VIS_GALAXY_0`` -> ``VIS_BGSUB``); an EXTNAME
without a known band prefix (the mass maps) uses VIS. Two cards record the
frame: ``SKYFRAME`` (native +y is south, +x west for this WCS) and, for
deflection maps, ``DEFLNOTE`` (``DEFLECTIONS_Y`` positive points south).

Idempotent: an HDU that already has ``CTYPE1`` is left alone. Source-plane
products are not touched (they are not on the sky).

Usage
-----
    python scripts/tools/add_fits_wcs.py --target_dir=inspect/dr1_sep1 \
        --dataset_path=dataset/dr1_sep1 [--dry_run]
    python scripts/tools/add_fits_wcs.py --target_dir=inspect/dr1_sep1 --sample=dr1_sep1
"""
import argparse
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

import sky_frame  # noqa: E402

PRODUCTS = ("model.fits", "pre_psf.fits", "convergence.fits", "potential.fits", "deflections.fits")
SKYFRAME = "pipeline +y south, +x west"


def cutout_headers(dataset_dir: Path) -> dict:
    """``{band: header}`` of the cut-out's ``<BAND>_BGSUB`` HDUs."""
    paths = [p for p in glob.glob(str(dataset_dir / "*.fits")) if "segmentation" not in p]
    if not paths:
        raise FileNotFoundError(f"no cut-out FITS in {dataset_dir}")
    headers = {}
    with fits.open(paths[0]) as hdul:
        for hdu in hdul:
            name = hdu.name or ""
            if name.endswith("_BGSUB") and hdu.data is not None:
                headers[name[: -len("_BGSUB")]] = (hdu.header.copy(), hdu.data.shape)
    return headers


def band_of(extname: str, bands) -> str:
    for band in sorted(bands, key=len, reverse=True):
        if extname.upper().startswith(band + "_"):
            return band
    return "VIS"


def sky_wcs_header(cutout_header, cutout_shape, shape, pixel_scale, origin_yx):
    """The celestial WCS header of a product of ``shape`` on the cut-out grid."""
    n, m = shape
    big_n, big_m = cutout_shape
    oy = (big_n - n) / 2.0 - origin_yx[0] / pixel_scale
    ox = (big_m - m) / 2.0 + origin_yx[1] / pixel_scale
    wcs = WCS(cutout_header).celestial.deepcopy()
    wcs.wcs.crpix = [wcs.wcs.crpix[0] - ox, wcs.wcs.crpix[1] - oy]
    return wcs.to_header()


def add_wcs(path: Path, headers: dict, dry_run=False) -> int:
    """Add the WCS to every eligible HDU of one file; return how many changed."""
    changed = 0
    with fits.open(path, mode="readonly" if dry_run else "update") as hdul:
        for hdu in hdul:
            if hdu.data is None or hdu.data.ndim != 2 or "CTYPE1" in hdu.header:
                continue
            if "PIXSCAY" not in hdu.header:
                continue
            band = band_of(hdu.name or "", headers)
            if band not in headers:
                raise KeyError(f"{path.name}[{hdu.name}]: no {band}_BGSUB HDU in the cut-out")
            cutout_header, cutout_shape = headers[band]
            if abs(float(hdu.header["PIXSCAY"]) - abs(sky_frame.cd_matrix_from(cutout_header)[1, 1]) * 3600) > 1e-6:
                raise ValueError(f"{path.name}[{hdu.name}]: pixel scale differs from the {band} cut-out")
            new = sky_wcs_header(
                cutout_header, cutout_shape, hdu.data.shape, float(hdu.header["PIXSCAY"]),
                (float(hdu.header.get("ORIGINY", 0.0)), float(hdu.header.get("ORIGINX", 0.0))),
            )
            if not dry_run:
                hdu.header.update(new)
                hdu.header["SKYFRAME"] = SKYFRAME
                if (hdu.name or "").upper().startswith("DEFLECTIONS"):
                    hdu.header["DEFLNOTE"] = "DEFLECTIONS_Y > 0 points south"
            changed += 1
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target_dir", required=True, help="inspect/ or catalogue dir of <lens>/ folders")
    p.add_argument("--dataset_path", default=None, help="dataset tree (default: project_paths.dataset_root())")
    p.add_argument("--sample", default=None, help="sample subdirectory appended to --dataset_path")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    target = Path(a.target_dir)
    if not target.is_dir():
        print(f"{target}: no such directory; nothing to do")
        return 0
    if a.dataset_path is None:
        try:
            import project_paths

            a.dataset_path = str(project_paths.dataset_root())
        except ImportError:  # a mirror of scripts/ alone (the inspection bundle) has no project_paths.py
            a.dataset_path = str(PROJECT_ROOT / "dataset")
    dataset_tree = Path(a.dataset_path) / (a.sample or "")

    counts = {"hdus": 0, "files": 0, "lenses": 0, "skipped": 0, "errors": 0}
    for lens_dir in sorted(d for d in target.iterdir() if d.is_dir()):
        files = [lens_dir / name for name in PRODUCTS if (lens_dir / name).exists()]
        if not files:
            continue
        try:
            try:
                headers = cutout_headers(dataset_tree / lens_dir.name)
            except FileNotFoundError as e:
                # A missing input is a per-lens skip, as in every bundle stage;
                # corrupt or inconsistent inputs still fail the stage.
                counts["skipped"] += 1
                print(f"WARNING {lens_dir.name}: no WCS added; {e}")
                continue
            for path in files:
                n = add_wcs(path, headers, a.dry_run)
                counts["hdus"] += n
                counts["files"] += bool(n)
            counts["lenses"] += 1
        except Exception as e:  # noqa: BLE001 — report and continue
            counts["errors"] += 1
            print(f"ERROR {lens_dir.name}: {type(e).__name__}: {e}")
    verb = "would add" if a.dry_run else "added"
    print(f"{verb} WCS to {counts['hdus']} HDUs in {counts['files']} files of {counts['lenses']} lenses; "
          f"{counts['skipped']} skipped (no cut-out); {counts['errors']} errors")
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
