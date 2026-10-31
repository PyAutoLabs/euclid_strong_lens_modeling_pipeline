"""
Native-frame -> sky-frame conversions for oriented DR1 quantities (research, phase 1).

PyAutoArray loads a FITS array with no row flip and treats array row 0 as the
top of its arcsec grid, so for an array of ``shape`` with pixel scale ``ps``:

    y = (ny / 2 - 0.5 - row) * ps        x = (col - nx / 2 + 0.5) * ps

with ``row``/``col`` 0-based array indices, and array row ``r`` is FITS row
``r + 1``. A FITS CD matrix maps pixel offsets to intermediate world
coordinates (east, north) in degrees:

    east  = CD1_1 * dcol + CD1_2 * drow
    north = CD2_1 * dcol + CD2_2 * drow

Every function here goes through that linear map, so it is right for any CD
matrix (rotation or parity). For the DR1 cutouts (CD1_1 = -ps, CD2_2 = +ps, no
rotation) it reduces to east = -x, north = -y: native +y is south, +x is west.

Angles: PyAutoGalaxy measures an ellipse / shear angle ``phi`` in degrees
counter-clockwise from +x (``autogalaxy/convert.py``):

    ell_comps = (f sin 2phi, f cos 2phi),  f = (1 - q) / (1 + q)
    gamma_1 = g cos 2phi,  gamma_2 = g sin 2phi

A position angle (PA) here is east of north, in [0, 180).

No library source is changed; nothing here is used by the pipeline.
"""
import numpy as np


def cd_matrix_from(header):
    """The 2x2 CD matrix (deg / pixel) of a celestial FITS header."""
    if "CD1_1" in header:
        return np.array(
            [
                [header["CD1_1"], header.get("CD1_2", 0.0)],
                [header.get("CD2_1", 0.0), header["CD2_2"]],
            ]
        )
    pc = np.array(
        [
            [header.get("PC1_1", 1.0), header.get("PC1_2", 0.0)],
            [header.get("PC2_1", 0.0), header.get("PC2_2", 1.0)],
        ]
    )
    return pc * np.array([[header["CDELT1"]], [header["CDELT2"]]])


def east_north_from(y, x, cd, pixel_scale):
    """
    Sky offsets (east, north) in arcsec of a native-frame (y, x) arcsec offset.

    A native offset (dy, dx) is a pixel offset drow = -dy / ps, dcol = dx / ps.
    """
    dcol = np.asarray(x) / pixel_scale
    drow = -np.asarray(y) / pixel_scale
    east = 3600.0 * (cd[0, 0] * dcol + cd[0, 1] * drow)
    north = 3600.0 * (cd[1, 0] * dcol + cd[1, 1] * drow)
    return east, north


def pa_from_native_angle(phi_deg, cd, pixel_scale):
    """
    PA east of north, in [0, 180), of an axis at native angle ``phi_deg``
    (counter-clockwise from +x). DR1: PA = (90 - phi) mod 180.
    """
    phi = np.radians(phi_deg)
    east, north = east_north_from(np.sin(phi), np.cos(phi), cd, pixel_scale)
    return np.degrees(np.arctan2(east, north)) % 180.0


def native_angle_from_pa(pa_deg, cd, pixel_scale):
    """Inverse of ``pa_from_native_angle``: native angle in [0, 180)."""
    pa = np.radians(pa_deg)
    east, north = np.sin(pa), np.cos(pa)
    inv = np.linalg.inv(cd * 3600.0 / pixel_scale)
    dcol, drow = inv @ np.array([east, north])
    x, y = dcol, -drow
    return np.degrees(np.arctan2(y, x)) % 180.0


def is_mirrored(cd):
    """True when the native (x, y) -> (east, north) map flips handedness."""
    m = np.array([[cd[0, 0], -cd[0, 1]], [cd[1, 0], -cd[1, 1]]])
    return bool(np.linalg.det(m) > 0)


def q_phi_from_ell_comps(ell_comps):
    """Axis ratio and native angle (deg) from PyAutoGalaxy ell_comps."""
    e0, e1 = ell_comps
    f = np.hypot(e0, e1)
    q = (1.0 - f) / (1.0 + f)
    phi = np.degrees(0.5 * np.arctan2(e0, e1))
    return q, phi


def phi_from_shear(gamma_1, gamma_2):
    """Shear magnitude and native angle (deg) from PyAutoGalaxy gamma_1/2."""
    return np.hypot(gamma_1, gamma_2), np.degrees(0.5 * np.arctan2(gamma_2, gamma_1))


def plus_y_north_ell_comps(ell_comps, cd, pixel_scale):
    """
    The ell_comps the same ellipse would have in a frame with +y north and +x
    west (the frame a row-flipped load would give, i.e. a FITS-standard
    north-up image displayed with origin=lower). DR1: (-e0, +e1).
    """
    q, phi = q_phi_from_ell_comps(ell_comps)
    pa = pa_from_native_angle(phi, cd, pixel_scale)
    phi_n = np.radians(pa + 90.0)  # +y north, +x west: phi' = PA + 90
    f = (1.0 - q) / (1.0 + q)
    return f * np.sin(2 * phi_n), f * np.cos(2 * phi_n)


def plus_y_north_shear(gamma_1, gamma_2, cd, pixel_scale):
    """As ``plus_y_north_ell_comps`` for shear. DR1: (+gamma_1, -gamma_2)."""
    g, phi = phi_from_shear(gamma_1, gamma_2)
    pa = pa_from_native_angle(phi, cd, pixel_scale)
    phi_n = np.radians(pa + 90.0)
    return g * np.cos(2 * phi_n), g * np.sin(2 * phi_n)


def moments_pa(image, cd, pixel_scale, centre_yx=(0.0, 0.0), radius=None):
    """
    Pixel route, independent of the angle algebra above: the flux-weighted
    second-moment major-axis PA (east of north) and axis ratio of a native-frame
    array, read through the CD matrix. Negative pixels get zero weight.
    """
    image = np.asarray(image, dtype=float)
    ny, nx = image.shape
    rows, cols = np.mgrid[0:ny, 0:nx]
    y = (ny / 2 - 0.5 - rows) * pixel_scale - centre_yx[0]
    x = (cols - nx / 2 + 0.5) * pixel_scale - centre_yx[1]
    east, north = east_north_from(y, x, cd, pixel_scale)
    w = np.clip(image, 0.0, None)
    if radius is not None:
        w = w * (np.hypot(y, x) < radius)
    w = w / w.sum()
    me = (w * east).sum()
    mn = (w * north).sum()
    see = (w * (east - me) ** 2).sum()
    snn = (w * (north - mn) ** 2).sum()
    sen = (w * (east - me) * (north - mn)).sum()
    pa = np.degrees(0.5 * np.arctan2(2 * sen, snn - see)) % 180.0
    lam = np.linalg.eigvalsh(np.array([[see, sen], [sen, snn]]))
    return pa, float(np.sqrt(lam[0] / lam[1]))


def pa_difference(a, b):
    """Smallest difference between two axis PAs (mod 180), in degrees."""
    d = (np.asarray(a) - np.asarray(b)) % 180.0
    return np.minimum(d, 180.0 - d)
