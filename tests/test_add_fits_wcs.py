"""
``scripts/tools/add_fits_wcs.py`` adds the cut-out WCS to the model FITS of a
bundle, header only. A product on the cut-out grid and one on a zoomed grid
(the mass maps) must both map each array pixel to the sky position of the same
arcsec point on the cut-out, data must be untouched, and a rerun must change
nothing.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOL = PROJECT_ROOT / "scripts" / "tools" / "add_fits_wcs.py"
LENS = "Tile000000001RA0000000000000DEC0000000000000"
PS = 0.1


def cutout_header():
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRVAL1"], h["CRVAL2"] = 150.0, -30.0
    h["CRPIX1"], h["CRPIX2"] = -6518.0, 3160.0
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -PS / 3600, 0.0, 0.0, PS / 3600
    return h


def product_hdu(name, shape, origin=(0.0, 0.0)):
    h = fits.Header()
    h["PIXSCAY"], h["PIXSCAX"], h["ORIGINY"], h["ORIGINX"] = PS, PS, origin[0], origin[1]
    return fits.ImageHDU(np.arange(np.prod(shape), dtype=float).reshape(shape), header=h, name=name)


def build(tmp_path):
    (tmp_path / "dataset" / LENS).mkdir(parents=True)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((100, 100)), header=cutout_header(), name="VIS_BGSUB"),
                  fits.ImageHDU(np.zeros((100, 100)), header=cutout_header(), name="NIR_H_BGSUB")]).writeto(
        tmp_path / "dataset" / LENS / f"{LENS}.fits")
    lens = tmp_path / "inspect" / LENS
    lens.mkdir(parents=True)
    fits.HDUList([fits.PrimaryHDU(), product_hdu("VIS_GALAXY_0", (100, 100)),
                  product_hdu("NIR_H_GALAXY_0", (100, 100))]).writeto(lens / "model.fits")
    fits.HDUList([fits.PrimaryHDU(), product_hdu("DEFLECTIONS_Y", (102, 102)),
                  product_hdu("DEFLECTIONS_X", (102, 102))]).writeto(lens / "deflections.fits")
    fits.HDUList([fits.PrimaryHDU(), product_hdu("CONVERGENCE", (40, 30), origin=(0.35, -0.2))]).writeto(
        lens / "convergence.fits")
    return lens


def run(tmp_path):
    return subprocess.run([sys.executable, str(TOOL), f"--target_dir={tmp_path / 'inspect'}",
                           f"--dataset_path={tmp_path / 'dataset'}"], capture_output=True, text=True)


def test__every_product_pixel_lands_on_its_cutout_sky_position_and_rerun_is_noop(tmp_path):
    lens = build(tmp_path)
    data_before = {p.name: [h.data.copy() for h in fits.open(p) if h.data is not None] for p in lens.iterdir()}
    result = run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    cut = WCS(cutout_header())
    for name, ext in [("model.fits", "VIS_GALAXY_0"), ("model.fits", "NIR_H_GALAXY_0"),
                      ("deflections.fits", "DEFLECTIONS_Y"), ("convergence.fits", "CONVERGENCE")]:
        hdu = fits.open(lens / name)[ext]
        wcs = WCS(hdu.header)
        n, m = hdu.data.shape
        oy, ox = hdu.header["ORIGINY"], hdu.header["ORIGINX"]
        for r, c in [(0, 0), (n - 1, m - 1), (n // 3, 2 * m // 3)]:
            y = oy + (n / 2 - 0.5 - r) * PS
            x = ox + (c - m / 2 + 0.5) * PS
            row, col = 50 - 0.5 - y / PS, x / PS + 50 - 0.5
            np.testing.assert_allclose(wcs.wcs_pix2world(c, r, 0), cut.wcs_pix2world(col, row, 0), atol=1e-10)
        assert hdu.header["SKYFRAME"].startswith("pipeline +y south")
    assert "south" in fits.getheader(lens / "deflections.fits", "DEFLECTIONS_Y")["DEFLNOTE"]

    for p in lens.iterdir():
        after = [h.data for h in fits.open(p) if h.data is not None]
        for a, b in zip(after, data_before[p.name]):
            np.testing.assert_array_equal(a, b)

    headers = {p.name: [dict(h.header) for h in fits.open(p)] for p in lens.iterdir()}
    assert run(tmp_path).returncode == 0
    assert {p.name: [dict(h.header) for h in fits.open(p)] for p in lens.iterdir()} == headers
