"""
Sky-frame conversions for the DR1 products.

The pipeline loads every cut-out FITS with no row flip (``autonerves.fitsable``),
so for the north-up DR1 cut-outs (``CD1_1 < 0``, ``CD2_2 > 0``, no rotation) the
model frame is a vertical mirror of the sky: **+y is south, +x is west**. Every
fit is self-consistent in that frame and lensing is parity invariant, so no fit
is wrong — but an oriented quantity read off a fit is the mirror image of its
sky value (``docs/sky_orientation/phase1.md``).

This module holds the two conversions the release needs:

- ``plus_y_north_from``: a copy of a fitted ``Tracer`` (or any profile / galaxy
  / ``MassField``) with every oriented parameter mirrored into the frame with
  +y north and +x west — the frame PyAutoLens's COOLEST interop and the COOLEST
  standard assume. Exporting that copy gives a correct COOLEST template for DR1;
  exporting the fitted tracer as-is gives one with ``phi = -PA`` and the wrong
  sign of ``center_y``.
- ``pa_east_of_north`` / ``east_north_from``: sky position angles and offsets of
  native-frame values, for catalogue columns.

The mirror ``y -> -y`` maps a native angle ``phi`` (counter-clockwise from +x)
to ``-phi``. With PyAutoGalaxy's definitions that is:

- ``centre (y, x) -> (-y, x)``
- ``ell_comps (f sin 2phi, f cos 2phi) -> (-e0, e1)``
- ``ExternalShear (gamma_1, gamma_2) -> (gamma_1, -gamma_2)``
- ``multipole_comps (k sin m phi, k cos m phi) -> (-c0, c1)``

Scalar parameters (Einstein radii, slopes, radii, intensities, Sersic indices,
redshifts) are unchanged.
"""
import copy

import numpy as np

# The DR1 cut-out pixel scale, arcsec per pixel.
DR1_PIXEL_SCALE = 0.1

# Attribute names that carry an orientation, and how each mirrors.
_MIRROR_PAIR_SIN_FIRST = ("ell_comps", "multipole_comps")


def native_is_mirrored(cd) -> bool:
    """
    Whether the native model frame of a cut-out with CD matrix ``cd`` (deg / px,
    rows not flipped at load) is the vertical mirror of the +y-north, +x-west
    frame — True for every DR1 cut-out.

    Only unrotated WCSs are handled: the release has no other kind (all 15,032
    ``dr1_sep1`` cut-outs scanned 2026-10-31), and a rotated one raises rather
    than being silently mislabelled.
    """
    cd = np.asarray(cd, dtype=float)
    if abs(cd[0, 1]) > 1e-12 or abs(cd[1, 0]) > 1e-12:
        raise ValueError(f"rotated WCS not supported by the DR1 sky frame: CD = {cd.tolist()}")
    if cd[0, 0] >= 0:
        raise ValueError(f"east-right WCS (CD1_1 >= 0) not supported: CD = {cd.tolist()}")
    return bool(cd[1, 1] > 0)


def _mirror_in_place(obj, seen):
    if id(obj) in seen:
        return
    seen.add(id(obj))

    if isinstance(obj, (list, tuple)):
        for item in obj:
            _mirror_in_place(item, seen)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            _mirror_in_place(item, seen)
        return
    if not hasattr(obj, "__dict__") or isinstance(obj, (np.ndarray, type)):
        return

    for name, value in list(vars(obj).items()):
        if name == "centre" and value is not None and len(value) == 2:
            setattr(obj, name, (-float(value[0]), float(value[1])))
        elif name in _MIRROR_PAIR_SIN_FIRST and value is not None:
            setattr(obj, name, (-float(value[0]), float(value[1])))
        elif name == "gamma_2" and np.isscalar(value):
            setattr(obj, name, -float(value))
        elif ("angle" in name or name.startswith("phi")) and np.isscalar(value):
            raise ValueError(
                f"{type(obj).__name__}.{name} is an explicit angle the sky-frame mirror "
                "does not know how to convert"
            )
        elif name == "cosmology":
            continue
        else:
            _mirror_in_place(value, seen)


def plus_y_north_from(obj):
    """
    A deep copy of ``obj`` (a ``Tracer``, ``Galaxy``, profile or ``MassField``)
    with every oriented parameter mirrored ``y -> -y`` (module docstring). The
    input is untouched. Applying it twice returns the original parameters.
    """
    mirrored = copy.deepcopy(obj)
    _mirror_in_place(mirrored, set())
    return mirrored


def east_north_from(y, x, cd, pixel_scale=DR1_PIXEL_SCALE):
    """
    Sky offsets (east, north) in arcsec of a native-frame (y, x) arcsec offset,
    through the CD matrix (deg / px). A native offset is a pixel offset
    ``drow = -y / ps``, ``dcol = x / ps``; DR1 gives east = -x, north = -y.
    """
    cd = np.asarray(cd, dtype=float)
    dcol = np.asarray(x) / pixel_scale
    drow = -np.asarray(y) / pixel_scale
    east = 3600.0 * (cd[0, 0] * dcol + cd[0, 1] * drow)
    north = 3600.0 * (cd[1, 0] * dcol + cd[1, 1] * drow)
    return east, north


def pa_east_of_north(phi_deg, cd, pixel_scale=DR1_PIXEL_SCALE):
    """
    Position angle east of north, in [0, 180), of an axis at native angle
    ``phi_deg`` (counter-clockwise from +x). DR1: ``(90 - phi) mod 180``.
    """
    phi = np.radians(phi_deg)
    east, north = east_north_from(np.sin(phi), np.cos(phi), cd, pixel_scale)
    return np.degrees(np.arctan2(east, north)) % 180.0


def cd_matrix_from(header):
    """The 2x2 CD matrix (deg / px) of a celestial FITS header."""
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
