"""
Phase-1 Witness (2) for the sky-orientation research: a simulated galaxy with a
known PA east of north, drawn directly in sky coordinates (no PyAutoGalaxy),
written to a north-up FITS with a DR1-style WCS (CD1_1 < 0, CD2_2 > 0), loaded
the DR1 way and fitted by least squares (no non-linear search). The fitted
native-frame ell_comps and centre, converted by
``scripts/research/sky_orientation/conversions.py``, must recover the true PA
and RA / Dec.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research" / "sky_orientation"))
import conversions as cv  # noqa: E402

PS = 0.1
SHAPE = (60, 60)
RA0, DEC0 = 150.0, -30.0


def dr1_header(cd2_2=PS / 3600.0, rot_deg=0.0):
    cd = np.array([[-PS / 3600.0, 0.0], [0.0, cd2_2]])
    r = np.radians(rot_deg)
    cd = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]) @ cd
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRVAL1"], h["CRVAL2"] = RA0, DEC0
    h["CRPIX1"], h["CRPIX2"] = SHAPE[1] / 2 + 0.5, SHAPE[0] / 2 + 0.5
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]
    return h


def sky_ellipse(header, pa_deg, q, east0, north0, sigma=0.4):
    """An elliptical Gaussian drawn on the FITS pixel grid through astropy."""
    wcs = WCS(header)
    rows, cols = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    ra, dec = wcs.wcs_pix2world(cols + 1, rows + 1, 1)
    east = (ra - RA0) * np.cos(np.radians(DEC0)) * 3600.0 - east0
    north = (dec - DEC0) * 3600.0 - north0
    pa = np.radians(pa_deg)
    major = east * np.sin(pa) + north * np.cos(pa)
    minor = east * np.cos(pa) - north * np.sin(pa)
    return np.exp(-0.5 * (major ** 2 + (minor / q) ** 2) / sigma ** 2)


def fit_native(path):
    """Load like the pipeline (no row flip) and fit a Gaussian by least squares."""
    al = pytest.importorskip("autolens")
    from scipy.optimize import least_squares

    data = al.Array2D.from_fits(file_path=path, hdu=0, pixel_scales=PS)
    grid = al.Grid2D.uniform(shape_native=SHAPE, pixel_scales=PS)
    target = np.asarray(data.native)

    def model(p):
        g = al.lp.Gaussian(centre=(p[0], p[1]), ell_comps=(p[2], p[3]), intensity=p[4], sigma=p[5])
        return np.asarray(g.image_2d_from(grid=grid).native)

    res = least_squares(
        lambda p: (model(p) - target).ravel(),
        x0=[0.0, 0.0, 0.0, 0.0, 1.0, 0.5],
        bounds=([-1, -1, -0.9, -0.9, 0, 0.05], [1, 1, 0.9, 0.9, 10, 2]),
    )
    return res.x


@pytest.mark.parametrize("pa_true", [10.0, 30.0, 75.0, 120.0, 160.0])
def test__simulated_north_up_fits__recovers_pa_and_radec(tmp_path, pa_true):
    header = dr1_header()
    east0, north0 = 0.35, -0.2
    path = tmp_path / "sim.fits"
    fits.PrimaryHDU(sky_ellipse(header, pa_true, 0.6, east0, north0), header=header).writeto(path)

    cy, cx, e0, e1, _, _ = fit_native(path)
    cd = cv.cd_matrix_from(header)

    # Native frame for DR1: +y south, +x west.
    assert cy == pytest.approx(-north0, abs=0.01)
    assert cx == pytest.approx(-east0, abs=0.01)

    q, phi = cv.q_phi_from_ell_comps((e0, e1))
    assert q == pytest.approx(0.6, abs=0.02)
    assert cv.pa_difference(cv.pa_from_native_angle(phi, cd, PS), pa_true) < 0.5
    assert cv.pa_difference((90.0 - phi) % 180.0, pa_true) < 0.5

    east, north = cv.east_north_from(cy, cx, cd, PS)
    ra = RA0 + east / 3600.0 / np.cos(np.radians(DEC0))
    dec = DEC0 + north / 3600.0
    ra_w, dec_w = WCS(header).wcs_pix2world(
        SHAPE[1] / 2 + 0.5 - east0 / PS, SHAPE[0] / 2 + 0.5 + north0 / PS, 1
    )
    assert 3600 * abs((ra - ra_w) * np.cos(np.radians(DEC0))) < 0.01
    assert 3600 * abs(dec - dec_w) < 0.01


def test__plus_y_north_components__dr1_sign_rules():
    cd = cv.cd_matrix_from(dr1_header())
    e0, e1 = cv.plus_y_north_ell_comps((0.2, 0.1), cd, PS)
    assert (e0, e1) == pytest.approx((-0.2, 0.1))
    g1, g2 = cv.plus_y_north_shear(0.05, 0.03, cd, PS)
    assert (g1, g2) == pytest.approx((0.05, -0.03))
    assert cv.is_mirrored(cd)


@pytest.mark.parametrize("rot", [0.0, 25.0, -60.0])
def test__pa_round_trip_for_rotated_and_unmirrored_wcs(rot):
    for cd2_2 in (PS / 3600.0, -PS / 3600.0):
        cd = cv.cd_matrix_from(dr1_header(cd2_2=cd2_2, rot_deg=rot))
        for pa in (0.0, 33.0, 90.0, 147.0):
            phi = cv.native_angle_from_pa(pa, cd, PS)
            assert cv.pa_difference(cv.pa_from_native_angle(phi, cd, PS), pa) < 1e-9


def test__moments_pa_matches_drawn_ellipse():
    header = dr1_header()
    image = sky_ellipse(header, 40.0, 0.5, 0.0, 0.0)
    pa, q = cv.moments_pa(image, cv.cd_matrix_from(header), PS)
    assert cv.pa_difference(pa, 40.0) < 0.5
    assert q == pytest.approx(0.5, abs=0.03)
