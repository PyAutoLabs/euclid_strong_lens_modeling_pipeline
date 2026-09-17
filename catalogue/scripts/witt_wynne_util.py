"""
Witt-Wynne (SIEP) Solver and Projection
=======================================

New here? Read `start_here.py` for the pipeline entry point and
`catalogue/README.md` for the catalogue run order.

The singular isothermal elliptical *potential* (SIEP) is the one strong-lens
model whose lens equation reduces to a quartic with a closed-form solution.
Given a source position it returns, without any iteration, the number of images
(4 / 3 / 2 / 1 -- 3 only for flattened lenses, ``e`` >~ 0.36, when the source
crosses the pseudo-caustic), their positions, their signed magnifications and
their time lags.
That speed is what makes it usable inside a transient broker: when a supernova
alert lands near a known quad, the question "is this a fourth image or a
foreground star?" has to be answered before the object fades.

This module holds the two halves the catalogue producer needs:

1. The **solver** -- a pure-numpy port of Paul Schechter's `isit4or2or1`
   (`solve_quartic` ... `time_lags_days`), plus the `.in` / CSV writers that
   feed the compiled C++.
2. The **projection** -- `witt_wynne_from_tracer` (caustic-matched, the default)
   and `witt_wynne_vector_sum` (Schechter's literal ellipticity-plus-shear
   prescription), which turn a **PyAutoLens** ``Tracer`` into the nine numbers
   the `.in` file carries.

The projection throws away everything the SIEP cannot represent: source shape,
non-isothermality, secondary perturbers, and the components of ellipticity and
shear perpendicular to the direction of their sum.

__Attribution__

The solver ported here is `isit4or2or1` v1.0 (Schechter, Lu & Hernandez),
archived at Zenodo, DOI 10.5281/zenodo.20086659, and released under
**CC-BY-4.0**. This port is a derivative work: it follows the structure and the
conventions of `SIEP_CLI.v1.0.cpp`, and the same CC-BY-4.0 attribution applies
to it. Please cite the Zenodo record and the papers below if you use it.

- Schechter, Lu & Hernandez 2026 -- `isit4or2or1` v1.0, SN 2025wny and the LSST
  alert protocol (Zenodo DOI 10.5281/zenodo.20086659, CC-BY-4.0).
- Witt 1996, ApJ 472, L1 -- the hyperbola on which the image positions lie.
- Wynne & Schechter 2018, arXiv:1808.06151 -- the ellipse that intersects it.
- Schechter & Wynne 2019, arXiv:1901.08517 -- the resulting quartic.
- Falor & Schechter 2022, arXiv:2205.06269 -- the asymptotically circular lens
  equation (ACLE) and the 4/2/1 root count used here.

__Provenance__

This file is the **canonical fixed copy** of the reusable span
(`solve_quartic` ... `isit_csv_row`) of the **autolens_workspace** guide
`scripts/guides/misc/witt_wynne.py` (autolens_workspace#552). The pipeline must
not import the workspace, so the code is duplicated on purpose; the guide is
being corrected from this file, not the other way round. The fixes applied here
relative to the shipped guide, all from the independent numerical review of
2026-09-17, are listed under `__Review Fixes__` below.

__Conventions__

The C++ follows Keeton's `gravlens` conventions, and this port reproduces them
exactly:

- Coordinates are ``(x, y)`` in arcseconds and position angle ``phi`` is degrees
  East of North. **PyAutoLens** grids are ordered ``(y, x)``, so the only change
  of variables needed is the swap of the tuple order. **The two are mixed in
  this module's own signatures**: ``centre`` is ``(x, y)`` (the solver's order,
  because it is written straight into the `.in` file) while ``source_centre`` is
  ``(y, x)`` (**PyAutoLens**'s order, because it comes from a tracer). Every
  function below states which it takes.
- A position angle ``phi`` maps to a **PyAutoLens** angle (degrees
  counter-clockwise from the positive x-axis) as ``angle = phi - 90``, and back
  as ``PA_isit = (angle_ccw + 90) mod 180``. Getting this backwards reflects
  every predicted image about the potential's minor axis.
- ``e`` is the ellipticity of the *potential*, ``e = 1 - q_psi``, not of the
  density. It is not **PyAutoLens**'s ``ell_comps`` magnitude ``(1 - q)/(1 + q)``,
  which describes the density. To first order in ellipticity the density-to-
  potential map is ``e ~ (1 - q) / 3``.
- In **PyAutoGalaxy**'s convention a mass ellipticity at angle ``theta``
  elongates the tangential caustic *along* ``theta``, whereas an external shear
  at ``theta_gamma`` elongates it at ``theta_gamma + 90``. The shear therefore
  enters the 2-theta vector sum **with a minus sign**, as ``-(gamma_1, gamma_2)``.
- The astroid fitted by ``ellipticity_from_caustic`` is centred on
  ``mass_list[0].centre`` -- the first admissible lens-plane mass profile --
  while ``b`` and the caustic it is fitted to come from the whole tracer. With a
  strong secondary perturber the caustic's centroid is displaced from that
  centre, so the fitted astroid sits slightly off the caustic it is matched to.
- Distances are angular diameter distances in ``h^-1 Mpc``, with
  ``D_H = 3000 h^-1 Mpc``.
- The original hardcodes ``h = 0.7`` in its ``TIMECONSTANT`` while taking
  distances in ``h^-1 Mpc``, so its lags are on an ``h = 0.7`` scale whatever
  cosmology produced the distances. This port exposes ``h`` as an argument
  defaulting to ``0.7`` so the shipped regression is exact.
- The time constant keeps the C++'s own rounded literals -- ``D_H = 3000``,
  ``9.78e9 / h`` yr and ``365.0`` days per year. Together they make the lags
  **0.12 % low** against an exact ``(1 + z_l) D_l D_s / (c D_ls)``. That is kept
  deliberately, so the lags are like-for-like with `isit4or2or1`'s own output;
  it is negligible against the 5-13 % the projection itself costs.

__Review Fixes__

1. ``find_intersections`` filters wrong-branch quartic roots by their
   lens-equation residual, so the **1-image verdict can actually fire**. The
   test needs ``e``, which is why this function takes it (the branch is not
   determined by ``(p, q)`` alone).
2. ``images_from_source`` returns a **NaN row** for the quartic's degeneracies
   (``min(|p|, |q|) < 1e-6`` -- source on a potential axis or at the centre --
   ``e`` outside ``(0, 1)``, a non-positive ``b``, no surviving root, or any
   non-finite output). The verdict for such a row is the sentinel
   ``N_IMAGES_SENTINEL = -1``, read with ``n_images_from``. It never raises and
   never returns a silent finite number.
3. ``_mass_and_shear_from`` picks the mass profile **by class** (the
   Isothermal / PowerLaw family, spherical variants included), never a light
   profile and never an ``lp_basis.Basis`` -- which subclasses ``MassProfile``,
   so the unfixed guide picked the MGE lens light as the lens mass on exactly
   this pipeline's model shape, an 84 degree position-angle error. With no
   admissible mass profile, or no tangential caustic (a sub-critical lens), the
   projections return a sentinel ``WittWynne`` (``valid=False``, all-NaN
   parameters, ``reason`` set) instead of raising.
4. ``witt_wynne_vector_sum`` returns the same sentinel when the ellipticity and
   the shear cancel (``e < 1e-3``), which the unfixed guide followed with image
   positions of ~1e7 arcsec.

__Usage__

.. code-block:: python

    model = witt_wynne_from_tracer(
        tracer=tracer, grid=grid, source_centre=(y, x)   # PyAutoLens order
    )
    if model.valid:
        x_image, y_image, magnification, angle = model.images()
        verdict = n_images_from(x_image)                 # 4, 2, 1 or -1
        lags = model.lags(x_image=x_image, y_image=y_image)
        write_isit_input(path=inspect_path / "witt_wynne.in", model=model)
        row = isit_csv_row(name=lens_name, model=model)
    else:
        print(f"skipping {lens_name}: {model.reason}")

The module is pure numpy at import time: ``autolens`` / ``autogalaxy`` are
imported inside the functions that need them, so the solver and the writers cost
nothing to import.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "D_H",
    "ISIT_CSV_HEADER",
    "N_IMAGES_SENTINEL",
    "P_Q_MINIMUM",
    "WittWynne",
    "caustic_semi_axes_from",
    "ellipticity_from_caustic",
    "find_intersections",
    "images_from_source",
    "isit_csv_row",
    "n_images_from",
    "siep_astroid_semi_axes",
    "solve_quartic",
    "source_from_image",
    "time_lags_days",
    "to_detector",
    "to_registered",
    "witt_wynne_from_tracer",
    "witt_wynne_vector_sum",
    "write_isit_input",
]


"""
__The Quartic__
"""


def solve_quartic(a: float, b: float, c: float, d: float, e: float) -> List[complex]:
    """
    The four (generally complex) roots of ``a x^4 + b x^3 + c x^2 + d x + e``,
    from the closed-form general quartic formula.

    A direct transcription of the C++ so that the roots come back in the same
    order, which is what fixes the A/B/C/D image labelling of the reference
    output. ``numpy.roots`` gives the same set in a different order.
    """
    a, b, c, d, e = (complex(v) for v in (a, b, c, d, e))

    p = (8.0 * a * c - 3.0 * b * b) / (8.0 * a * a)
    q = (b**3 - 4.0 * a * b * c + 8.0 * a * a * d) / (8.0 * a**3)

    delta_0 = c * c - 3.0 * b * d + 12.0 * a * e
    delta_1 = (
        2.0 * c**3
        - 9.0 * b * c * d
        + 27.0 * b * b * e
        + 27.0 * a * d * d
        - 72.0 * a * c * e
    )

    big_q = (0.5 * (delta_1 + np.sqrt(delta_1 * delta_1 - 4.0 * delta_0**3))) ** (
        1.0 / 3.0
    )
    s = 0.5 * np.sqrt(-2.0 / 3.0 * p + (big_q + delta_0 / big_q) / (3.0 * a))

    k_1 = np.sqrt(-4.0 * s * s - 2.0 * p + q / s)
    k_2 = np.sqrt(-4.0 * s * s - 2.0 * p - q / s)

    return [
        -0.25 * b / a - s + 0.5 * k_1,
        -0.25 * b / a - s - 0.5 * k_1,
        -0.25 * b / a + s + 0.5 * k_2,
        -0.25 * b / a + s - 0.5 * k_2,
    ]


"""
__The Asymptotically Circular Lens Equation__
"""

# Below this, a source lies on a potential axis (or at the centre) to within the
# quartic's ability to tell, and `images_from_source` returns the sentinel row.
P_Q_MINIMUM = 1e-6

# The verdict returned for a sentinel (all-NaN) row.
N_IMAGES_SENTINEL = -1

# A wrong-branch quartic root misses the lens equation by order unity in units
# of `b` (measured minimum 1.49 over the review's grid). A root that misses it
# by more than the tolerance but by much less than this is neither: the quartic
# is ill-conditioned, which happens in a narrow band just outside the on-axis
# screen of `P_Q_MINIMUM`. `find_intersections` then returns nothing at all, so
# the verdict is the sentinel rather than an image short.
AMBIGUOUS_RESIDUAL = 1e-2


def find_intersections(
    p: float,
    q: float,
    e: float,
    threshold: float = 1e-5,
    residual_threshold: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The scaled image positions ``(x_hat, y_hat)`` of the ACLE.

    Witt's hyperbola ``(x - p)(y - q) = p q`` and the unit circle
    ``x^2 + y^2 = 1`` intersect in the image positions of the scaled problem.
    Eliminating ``y`` gives

        x^4 - 2 p x^3 + (p^2 + q^2 - 1) x^2 + 2 p x - p^2 = 0,   y = q x / (x - p)

    A root counts as real when its imaginary part is below ``threshold``
    (Falor & Schechter 2022, section 2.3).

    **Not every real root is an image.** The map back to the image plane takes
    the square root of ``G = sqrt(x^2 + y^2 / (1 - e)^2)``, and half the real
    roots can sit on its wrong branch -- the recovered position then solves the
    lens equation of a *negative* mass. Such a root is dropped here: the
    recovered image must satisfy the SIEP lens equation to
    ``residual_threshold * max(1, p^2 + q^2)`` in units of ``b``, equivalently
    ``sign(x_hat) = sign(x)`` and ``sign(y_hat) = sign(y)``. Without this the
    solver returns only ever 4 or 2 images and the "1" of `isit4or2or1` never
    fires; with it, the far-source 1-image verdict is exact.

    A root that misses the lens equation by more than the tolerance but by far
    less than a wrong-branch root's order unity means the quartic is
    ill-conditioned rather than the root unphysical. **No roots are returned at
    all** in that case, so ``images_from_source`` reports the degeneracy
    sentinel instead of a verdict one image short.

    The two branches are separated by fifteen orders of magnitude, so the
    threshold is not a tuning knob: measured over the review's grid out to 8x
    the caustic, a genuine root's residual is ``< 1.3e-9`` and a wrong-branch
    root's is ``~2``. It is scaled by ``p^2 + q^2`` only because the quartic's
    own conditioning degrades with the coefficients: the residual of a genuine
    root grows as roughly ``8e-11 (p^2 + q^2)``, so a fixed ``1e-10`` would
    start discarding real images of far sources.

    The branch test needs the potential ellipticity ``e`` -- it is *not* fixed
    by ``(p, q)`` alone, because the 2-image / 1-image boundary moves with ``e``
    (a circular lens has none: it images every source twice only inside its
    Einstein radius). This is the one signature that differs from the shipped
    guide's.

    The ACLE degenerates when ``q = 0`` (source exactly on the potential's major
    axis) or ``p = 0`` (minor axis): the quartic acquires a double root and
    ``y = q x / (x - p)`` is indeterminate. Both are screened by
    ``images_from_source`` before this function is reached.
    """
    roots = solve_quartic(1.0, -2.0 * p, p * p + q * q - 1.0, 2.0 * p, -p * p)

    scale = e * (2.0 - e)
    axis_ratio_squared = (1.0 - e) ** 2
    tolerance = min(residual_threshold * max(1.0, p * p + q * q), AMBIGUOUS_RESIDUAL)

    x_list, y_list = [], []

    for root in roots:
        if abs(root.imag) >= threshold:
            continue

        x_hat = float(root.real)
        if x_hat == p:
            continue

        y_hat = q * x_hat / (x_hat - p)
        if not (np.isfinite(x_hat) and np.isfinite(y_hat)):
            continue

        # The recovered image position in the registered frame, in units of b:
        # x = b x_hat + x_s and y = b y_hat / (1 - e) + y_s.
        x_over_b = x_hat + p * scale / axis_ratio_squared
        y_over_b = (y_hat - q * scale) / (1.0 - e)

        g = np.sqrt(x_over_b**2 + y_over_b**2 / axis_ratio_squared)
        if not np.isfinite(g) or g <= 0.0:
            continue

        # The lens equation, divided by b: x_hat = x / G and
        # y_hat = y / ((1 - e) G), which only the physical branch satisfies.
        residual = max(
            abs(x_hat - x_over_b / g),
            abs(y_hat - y_over_b / ((1.0 - e) * g)),
        )

        if not np.isfinite(residual):
            continue

        if residual > tolerance:
            if residual < AMBIGUOUS_RESIDUAL:
                # Neither a solution nor the wrong branch: the quartic is too
                # ill-conditioned here to say which, so the whole solve is
                # reported degenerate rather than silently short of an image.
                return np.asarray([], dtype=float), np.asarray([], dtype=float)
            continue

        x_list.append(x_hat)
        y_list.append(y_hat)

    return np.asarray(x_list, dtype=float), np.asarray(y_list, dtype=float)


"""
__Registered Coordinates__

"Registered" coordinates are centred on the potential with the x-axis along its
major axis. The rotation is a proper one (determinant +1), so handedness is
preserved and the inverse is the transpose.
"""


def to_registered(
    x: np.ndarray, y: np.ndarray, pa_deg: float, centre: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """``centre`` is ``(x, y)``, the solver's order."""
    phi = np.radians(pa_deg)
    sin, cos = np.sin(phi), np.cos(phi)
    return (
        sin * (x - centre[0]) - cos * (y - centre[1]),
        cos * (x - centre[0]) + sin * (y - centre[1]),
    )


def to_detector(
    x: np.ndarray, y: np.ndarray, pa_deg: float, centre: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """``centre`` is ``(x, y)``, the solver's order."""
    phi = np.radians(pa_deg)
    sin, cos = np.sin(phi), np.cos(phi)
    return sin * x + cos * y + centre[0], -cos * x + sin * y + centre[1]


"""
__Images From a Source__
"""


def _nan_row() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The sentinel returned for a degenerate solve: one all-NaN image row, whose
    verdict through ``n_images_from`` is ``N_IMAGES_SENTINEL`` (-1).

    The shape is the same 4-tuple of arrays a successful solve returns, so a
    caller that forgets to check still gets NaN out rather than a number.
    """
    nan = np.array([np.nan])
    return nan.copy(), nan.copy(), nan.copy(), nan.copy()


def n_images_from(x_image: np.ndarray, y_image: Optional[np.ndarray] = None) -> int:
    """
    The 4 / 3 / 2 / 1 verdict of an ``images_from_source`` result, or
    ``N_IMAGES_SENTINEL`` (-1) when the solve was degenerate.

    A sentinel row is all-NaN, so it can never be confused with a genuine
    1-image verdict, whose position is finite.
    """
    x_image = np.asarray(x_image, dtype=float)

    if x_image.size == 0 or not np.all(np.isfinite(x_image)):
        return N_IMAGES_SENTINEL

    if y_image is not None:
        y_image = np.asarray(y_image, dtype=float)
        if y_image.size != x_image.size or not np.all(np.isfinite(y_image)):
            return N_IMAGES_SENTINEL

    return int(x_image.size)


def _is_solvable(e: float, b: float, x_s: float, y_s: float, centre) -> bool:
    """Whether the SIEP parameters are inside the solver's domain at all."""
    values = [float(e), float(b), float(x_s), float(y_s), *[float(c) for c in centre]]

    if not all(np.isfinite(values)):
        return False

    return 0.0 < float(e) < 1.0 and float(b) > 0.0


def images_from_source(
    x_s: float,
    y_s: float,
    e: float,
    b: float,
    pa_deg: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    threshold: float = 1e-5,
    residual_threshold: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The images of a source at ``(x_s, y_s)``, in the solver's ``(x, y)`` order.

    The source position is scaled into the ACLE parameters

        p = x'_s (1 - e)^2 / (b e (2 - e)),    q = -y'_s (1 - e) / (b e (2 - e))

    solved, and mapped back with ``x = b x_hat + x'_s`` and
    ``y = b y_hat / (1 - e) + y'_s``. The signed magnification follows from
    inverting ``mu^-1 = I - H(psi)`` (Falor & Schechter 2022, equation F1), and
    ``angle`` is the position angle of the eigenvector of the smaller
    eigenvalue, i.e. the direction along which the image is stretched.

    ``centre`` is the potential's centre as ``(x, y)`` -- the solver's order,
    *not* **PyAutoLens**'s ``(y, x)``.

    Returns ``(x_image, y_image, magnification, angle_deg)``, four arrays of the
    same length; the number of images is that length, read with
    ``n_images_from``. **The solve is degenerate** -- and all four arrays come
    back as a single NaN, verdict ``N_IMAGES_SENTINEL`` (-1) -- when

    - ``min(|p|, |q|) < P_Q_MINIMUM`` (1e-6): the source lies on a potential
      axis or at the lens centre, where the quartic has a double root and
      returns finite, plausible, *wrong* positions;
    - ``e`` is outside ``(0, 1)``, ``b`` is not positive, or any input is
      non-finite;
    - no quartic root survives the lens-equation filter;
    - any returned position, magnification or angle is non-finite (a source
      exactly on a fold or cusp).

    It never raises.
    """
    if not _is_solvable(e=e, b=b, x_s=x_s, y_s=y_s, centre=centre):
        return _nan_row()

    e, b = float(e), float(b)
    phi = np.radians(pa_deg)

    x_reg_s, y_reg_s = to_registered(x_s, y_s, pa_deg=pa_deg, centre=centre)

    with np.errstate(all="ignore"):
        p = x_reg_s * (1.0 - e) ** 2 / (b * (2.0 - e) * e)
        q = -y_reg_s * (1.0 - e) / (b * (2.0 - e) * e)

        if not (np.isfinite(p) and np.isfinite(q)):
            return _nan_row()

        if min(abs(float(p)), abs(float(q))) < P_Q_MINIMUM:
            return _nan_row()

        x_hat, y_hat = find_intersections(
            p=p,
            q=q,
            e=e,
            threshold=threshold,
            residual_threshold=residual_threshold,
        )

        if x_hat.size == 0:
            return _nan_row()

        x = b * x_hat + x_reg_s
        y = b * y_hat / (1.0 - e) + y_reg_s

        x_image, y_image = to_detector(x, y, pa_deg=pa_deg, centre=centre)

        m = (1.0 - e) ** 2 * (x * x + y * y / (1.0 - e) ** 2) ** 1.5
        magnification = m / (m - b * (x * x + y * y))

        numerator = 2.0 * b * (1.0 - e) ** 2 * x * y
        denominator = b * (1.0 - e) ** 2 * (x - y) * (x + y) - np.sqrt(
            b * b * (1.0 - e) ** 4 * (x * x + y * y) ** 2
        )
        angle = np.arctan(numerator / denominator) + phi - np.pi / 2.0
        angle = angle - np.floor((angle + np.pi / 2.0) / np.pi) * np.pi
        angle = np.degrees(angle)

    for values in (x_image, y_image, magnification, angle):
        if not np.all(np.isfinite(values)):
            return _nan_row()

    return x_image, y_image, magnification, angle


"""
__Source From an Image__

The inverse problem: given one image position, where is the source? Intersecting
Wynne's ellipse with Witt's hyperbola at the known image gives a closed form, so
a single detected image of a candidate transient fixes the source and hence the
other three images.
"""


def source_from_image(
    x_i: float,
    y_i: float,
    e: float,
    b: float,
    pa_deg: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """
    The source position ``(x, y)`` of an image at ``(x_i, y_i)``.

    ``centre`` is ``(x, y)``, the solver's order. Returns ``(nan, nan)`` rather
    than raising when ``e`` is outside ``(0, 1)``, ``b`` is not positive or an
    input is non-finite; the image plane's own degeneracy (an image exactly on
    the potential's major axis, ``y_reg = 0``) also returns ``(nan, nan)``, from
    the ``0 / 0`` in the last term.
    """
    if not _is_solvable(e=e, b=b, x_s=x_i, y_s=y_i, centre=centre):
        return float("nan"), float("nan")

    x_reg, y_reg = to_registered(x_i, y_i, pa_deg=pa_deg, centre=centre)

    with np.errstate(all="ignore"):
        x_hat = x_reg / b
        y_hat = y_reg / b

        y_s = y_hat - y_hat / (1.0 - e) * np.sqrt(
            1.0 / ((1.0 - e) ** 2 * x_hat * x_hat + y_hat * y_hat)
        )
        x_s = x_hat * e * (2.0 - e) + (1.0 - e) ** 2 * x_hat * y_s / y_hat

    return to_detector(x_s * b, y_s * b, pa_deg=pa_deg, centre=centre)


"""
__Time Lags__

The Fermat potential of the SIEP gives the arrival-time surface directly. Delays
are referred to the leading image, so the earliest arrival has a lag of zero and
every other lag is positive and in days.
"""

D_H = 3000.0  # Hubble distance in h^-1 Mpc (the C++'s own rounded literal).


def time_lags_days(
    x_image: np.ndarray,
    y_image: np.ndarray,
    x_s: float,
    y_s: float,
    e: float,
    b: float,
    pa_deg: float,
    d_ol: float,
    d_ls: float,
    z_lens: float,
    z_source: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    h: float = 0.7,
) -> np.ndarray:
    """
    Time lags in days, referred to the leading image.

    ``d_ol`` and ``d_ls`` are angular diameter distances in ``h^-1 Mpc``; the
    effective distance uses comoving distances ``chi = D (1 + z)``. ``centre``
    is ``(x, y)``, the solver's order.

    The time constant keeps the C++'s rounded literals -- ``D_H = 3000``,
    ``9.78e9 / h`` yr and ``365.0`` d/yr -- which together run **0.12 % low**
    against the exact ``(1 + z_l) D_l D_s / (c D_ls)``. That is deliberate: it
    makes these lags directly comparable with `isit4or2or1`'s. The original
    fixes ``h = 0.7`` here regardless of the distances it was given, so ``h``
    defaults to ``0.7``; pass the tracer's own ``h`` to compare against
    ``tracer.time_delays_from``, whose lags are a factor ``h / 0.7`` different.
    """
    time_constant = 9.78e9 * (1.0 / h) * 365.0  # 1/H0 in days.

    chi_ol = (d_ol / D_H) * (1.0 + z_lens)
    chi_ls = (d_ls / D_H) * (1.0 + z_source)
    d_eff = chi_ol * (chi_ol + chi_ls) / chi_ls

    arcsec_to_radians = np.pi / (3600.0 * 180.0)

    x_reg_s, y_reg_s = to_registered(x_s, y_s, pa_deg=pa_deg, centre=centre)
    x_reg_s, y_reg_s = x_reg_s * arcsec_to_radians, y_reg_s * arcsec_to_radians

    x_reg, y_reg = to_registered(x_image, y_image, pa_deg=pa_deg, centre=centre)
    x_reg, y_reg = x_reg * arcsec_to_radians, y_reg * arcsec_to_radians

    b_radians = b * arcsec_to_radians

    with np.errstate(all="ignore"):
        shapiro = -d_eff * b_radians * np.sqrt(x_reg**2 + y_reg**2 / (1.0 - e) ** 2)
        geometric = 0.5 * d_eff * ((x_reg - x_reg_s) ** 2 + (y_reg - y_reg_s) ** 2)

        delays = (shapiro + geometric) * time_constant

        return delays - delays.min()


"""
__The SIEP Astroid__

The 4/2/1 boundary is the astroid ``|p|^(2/3) + |q|^(2/3) = 1``, which in the
source plane has semi-axes

    a_long  = b e (2 - e) / (1 - e)^2      along the potential's major axis
    a_short = b e (2 - e) / (1 - e)        perpendicular to it

so the astroid is always elongated *along* the major axis, with axis ratio
``1 / (1 - e)``. Two numbers, one free parameter ``e`` once ``b`` is fixed: the
caustic-matched projection picks the ``e`` that matches both as well as it can.
"""


def siep_astroid_semi_axes(e: float, b: float) -> Tuple[float, float]:
    scale = b * e * (2.0 - e)
    return scale / (1.0 - e) ** 2, scale / (1.0 - e)


def ellipticity_from_caustic(b: float, a_long: float, a_short: float) -> float:
    """
    The potential ellipticity whose astroid best matches the two caustic
    semi-axes, by bounded least squares on their relative residuals.

    The fit is bounded to ``(1e-6, 0.9)``; at ``q <= 0.35`` the true caustic is
    too elongated for any astroid and the residuals reach -18 % / +16 %.
    """
    from scipy.optimize import minimize_scalar

    def cost(e):
        long_axis, short_axis = siep_astroid_semi_axes(e=e, b=b)
        return (long_axis / a_long - 1.0) ** 2 + (short_axis / a_short - 1.0) ** 2

    return float(minimize_scalar(cost, bounds=(1e-6, 0.9), method="bounded").x)


"""
__The Model Container__
"""


@dataclass(frozen=True)
class WittWynne:
    """
    The nine numbers the `.in` file needs, in the solver's own ``(x, y)``
    convention, plus the ``h`` the distances were computed with.

    ``valid`` is False on a **sentinel** model -- one the projection could not
    produce (no admissible mass profile, no tangential caustic, an ellipticity
    that cancelled) -- whose ``e``, ``b``, ``pa_deg`` and ``centre`` are NaN and
    whose ``reason`` says why. A valid model may also carry a ``reason``: it
    records anything the projection had to choose, such as which of two
    admissible mass profiles was used.
    """

    centre: Tuple[float, float]  # (x, y) of the potential, arcsec.
    source: Tuple[float, float]  # (x, y) of the source, arcsec.
    e: float  # Potential ellipticity, 1 - q_psi.
    b: float  # Einstein radius, arcsec.
    pa_deg: float  # Position angle of the major axis, degrees East of North.
    d_ol: float  # Angular diameter distance to the lens, h^-1 Mpc.
    d_ls: float  # Angular diameter distance lens to source, h^-1 Mpc.
    z_lens: float
    z_source: float
    h: float
    valid: bool = True
    reason: str = ""

    def zeroed(self) -> "WittWynne":
        """
        The same model with the potential at the origin, so no sky coordinates
        are divulged. The solver is translation invariant, so every predicted
        position and lag is unchanged.
        """
        return replace(
            self,
            centre=(0.0, 0.0),
            source=(self.source[0] - self.centre[0], self.source[1] - self.centre[1]),
        )

    def images(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(x_image, y_image, magnification, angle_deg)``; a NaN row on a
        sentinel model, since its ``e`` is NaN."""
        return images_from_source(
            x_s=self.source[0],
            y_s=self.source[1],
            e=self.e,
            b=self.b,
            pa_deg=self.pa_deg,
            centre=self.centre,
        )

    def n_images(self) -> int:
        """The 4 / 3 / 2 / 1 verdict, or ``N_IMAGES_SENTINEL`` (-1); 3 fires only
        for flattened lenses (``e`` >~ 0.36) whose source crosses the
        pseudo-caustic."""
        x_image, y_image, _, _ = self.images()
        return n_images_from(x_image, y_image)

    def lags(self, x_image: np.ndarray, y_image: np.ndarray) -> np.ndarray:
        return time_lags_days(
            x_image=x_image,
            y_image=y_image,
            x_s=self.source[0],
            y_s=self.source[1],
            e=self.e,
            b=self.b,
            pa_deg=self.pa_deg,
            d_ol=self.d_ol,
            d_ls=self.d_ls,
            z_lens=self.z_lens,
            z_source=self.z_source,
            centre=self.centre,
            h=self.h,
        )


"""
__Projecting a PyAutoLens Model__
"""


def caustic_semi_axes_from(
    caustic: np.ndarray, centre_yx: Tuple[float, float]
) -> Tuple[float, float, float]:
    """
    Semi-axes and long-axis orientation of a tangential caustic.

    The furthest point from the centre is a major cusp, which fixes the long
    axis; the short semi-axis is then the largest excursion perpendicular to it.
    ``caustic`` and ``centre_yx`` are both in **PyAutoLens** ``(y, x)`` order.
    Returns ``(a_long, a_short, angle_deg)`` with ``angle_deg``
    counter-clockwise from the positive x-axis.
    """
    caustic = np.asarray(caustic)
    delta_y = caustic[:, 0] - centre_yx[0]
    delta_x = caustic[:, 1] - centre_yx[1]

    angle = np.arctan2(delta_y, delta_x)[np.argmax(np.hypot(delta_y, delta_x))]

    long_axis = delta_x * np.cos(angle) + delta_y * np.sin(angle)
    short_axis = -delta_x * np.sin(angle) + delta_y * np.cos(angle)

    return (
        float(np.abs(long_axis).max()),
        float(np.abs(short_axis).max()),
        float(np.degrees(angle) % 180.0),
    )


def _mass_classes():
    """
    The mass profile classes the SIEP projection accepts, by class rather than
    by elimination.

    ``lp_basis.Basis`` subclasses ``MassProfile``, so "any ``MassProfile`` that
    is not an ``ExternalShear``" picks the **MGE lens light** as the lens mass
    on this pipeline's own model shape. The projection is only defined for an
    isothermal-like potential anyway, so the family is named explicitly:
    anything else returns a sentinel rather than a plausible wrong answer.
    """
    import autogalaxy as ag

    return (
        ag.mp.Isothermal,
        ag.mp.IsothermalSph,
        ag.mp.PowerLaw,
        ag.mp.PowerLawSph,
    )


def _is_admissible_mass(profile) -> bool:
    import autogalaxy as ag

    if isinstance(profile, (ag.lp_basis.Basis, ag.LightProfile)):
        return False

    if isinstance(profile, ag.mp.ExternalShear):
        return False

    return isinstance(profile, _mass_classes())


def _lens_plane_mass_profile_list_from(tracer) -> list:
    """
    Every mass profile of the **lens plane** (``tracer.planes[0]``), in order.

    Selection is restricted to the first plane because a mass profile or an
    ``ExternalShear`` attached to the *source* galaxy is not part of the lens
    this projection describes: taken tracer-wide it would be eligible for the
    vector sum's ``ell_comps`` and would name itself in the ``mass_profile``
    column.
    """
    import autogalaxy as ag

    if not tracer.planes:
        return []

    return tracer.planes[0].cls_list_from(cls=ag.mp.MassProfile)


def _admissible_mass_list_from(tracer) -> list:
    """Every Isothermal/PowerLaw-family lens-plane mass profile, in order."""
    return [
        profile
        for profile in _lens_plane_mass_profile_list_from(tracer)
        if _is_admissible_mass(profile)
    ]


def _shear_from(tracer):
    import autogalaxy as ag

    return next(
        (
            profile
            for profile in _lens_plane_mass_profile_list_from(tracer)
            if isinstance(profile, ag.mp.ExternalShear)
        ),
        None,
    )


def _mass_and_shear_from(tracer):
    """
    The lens's mass profile and its external shear, both **selected by class**.

    Returns ``(mass, shear)``, either of which may be ``None``: ``mass`` when
    the tracer holds no Isothermal/PowerLaw-family profile (a shear-only galaxy,
    or a mass model this projection does not cover), ``shear`` when there is no
    ``ExternalShear``. With more than one admissible mass profile the first is
    returned and the projections record that in ``WittWynne.reason``.
    """
    mass_list = _admissible_mass_list_from(tracer)
    return (mass_list[0] if mass_list else None), _shear_from(tracer)


def _distances_from(tracer) -> Tuple[float, float, float, float, float]:
    """``(d_ol, d_ls, z_lens, z_source, h)``, the distances in ``h^-1 Mpc``."""
    z_lens, z_source = tracer.plane_redshifts[0], tracer.plane_redshifts[-1]
    h = float(tracer.cosmology.H0) / 100.0
    d_ol = tracer.cosmology.angular_diameter_distance_to_earth_in_kpc_from(
        redshift=z_lens
    )
    d_ls = tracer.cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=z_lens, redshift_1=z_source
    )
    # kpc -> Mpc -> h^-1 Mpc.
    return float(d_ol) / 1e3 * h, float(d_ls) / 1e3 * h, z_lens, z_source, h


def _mass_note_from(mass_list: list) -> str:
    if len(mass_list) < 2:
        return ""

    names = ", ".join(type(mass).__name__ for mass in mass_list)
    return (
        f"{len(mass_list)} admissible mass profiles ({names}); projected the "
        f"first, {type(mass_list[0]).__name__}"
    )


def _sentinel_from(
    reason: str,
    source_centre: Tuple[float, float],
    distances: Optional[Tuple[float, float, float, float, float]] = None,
) -> WittWynne:
    """
    An invalid ``WittWynne``: NaN parameters, ``valid=False`` and ``reason``.

    The source and the distances are kept when they are known, so a catalogue
    row for a lens that could not be projected still says which lens it was.
    """
    d_ol, d_ls, z_lens, z_source, h = distances or (np.nan,) * 5

    return WittWynne(
        centre=(np.nan, np.nan),
        source=(source_centre[1], source_centre[0]),
        e=np.nan,
        b=np.nan,
        pa_deg=np.nan,
        d_ol=d_ol,
        d_ls=d_ls,
        z_lens=z_lens,
        z_source=z_source,
        h=h,
        valid=False,
        reason=reason,
    )


def witt_wynne_from_tracer(
    tracer,
    grid,
    source_centre: Tuple[float, float],
    centre: Tuple[float, float] = None,
    caustic_pixel_scale: float = 0.05,
) -> WittWynne:
    """
    Project a tracer onto SIEP parameters by matching its tangential caustic.

    ``b`` is the effective Einstein radius (the radius of the circle enclosing
    the same area as the tangential critical curve), the position angle is the
    orientation of the caustic's long axis, and ``e`` is chosen so the SIEP
    astroid matches both caustic semi-axes as closely as it can. The caustic is
    a property of the whole tracer, so secondary perturbers and external shear
    are folded in automatically rather than being modelled term by term. This is
    the projection to ship: on a 136-case grid its verdict agrees with
    ``al.PointSolver`` 68/68 inside the caustic and 51/68 outside it, against
    66/68 and 22/68 for the vector sum.

    ``source_centre`` is the source-plane coordinate in **PyAutoLens**
    ``(y, x)`` order; ``centre``, if given, overrides the mass profile's centre
    and is in the solver's ``(x, y)`` order.

    Returns a **sentinel** ``WittWynne`` (``valid=False``) rather than raising
    when the tracer holds no admissible mass profile (a shear-only or
    light-only galaxy) or no tangential caustic (a sub-critical lens).
    """
    import autolens as al

    distances = _distances_from(tracer)

    mass_list = _admissible_mass_list_from(tracer)

    if not mass_list:
        return _sentinel_from(
            reason="no Isothermal/PowerLaw-family mass profile in the tracer",
            source_centre=source_centre,
            distances=distances,
        )

    mass = mass_list[0]
    centre_yx = mass.centre if centre is None else (centre[1], centre[0])

    lens_calc = al.LensCalc.from_tracer(tracer=tracer)

    b = float(
        lens_calc.einstein_radius_from(grid=grid, pixel_scale=caustic_pixel_scale)
    )

    caustic_list = lens_calc.tangential_caustic_list_from(
        grid=grid, pixel_scale=caustic_pixel_scale
    )

    if len(caustic_list) == 0 or np.asarray(caustic_list[0]).size == 0:
        return _sentinel_from(
            reason="no tangential caustic: the lens is sub-critical on this grid",
            source_centre=source_centre,
            distances=distances,
        )

    a_long, a_short, angle_deg = caustic_semi_axes_from(
        caustic=np.asarray(caustic_list[0]), centre_yx=centre_yx
    )

    if not np.isfinite(b) or b <= 0.0 or min(a_long, a_short) <= 0.0:
        return _sentinel_from(
            reason=(
                f"degenerate caustic fit: b={b:.4g}, semi-axes "
                f"({a_long:.4g}, {a_short:.4g})"
            ),
            source_centre=source_centre,
            distances=distances,
        )

    d_ol, d_ls, z_lens, z_source, h = distances

    return WittWynne(
        centre=(centre_yx[1], centre_yx[0]),
        source=(source_centre[1], source_centre[0]),
        e=ellipticity_from_caustic(b=b, a_long=a_long, a_short=a_short),
        b=b,
        pa_deg=(angle_deg + 90.0) % 180.0,
        d_ol=d_ol,
        d_ls=d_ls,
        z_lens=z_lens,
        z_source=z_source,
        h=h,
        reason=_mass_note_from(mass_list),
    )


"""
__The Vector-Sum Projection__

Schechter's literal prescription is to add the ellipticity and the shear as
vectors and throw away the components perpendicular to their sum. Both
quantities live in the 2-theta plane, so the sum is well defined -- but three
conversions are needed first.

**Density to potential.** ``ell_comps`` describe the *density*; the SIEP's ``e``
describes the *potential*. Expanding both to first order in ellipticity, an
isothermal density with axis ratio ``q`` has a relative quadrupole
``(1 - q) / 2`` in convergence, while a potential ellipticity ``e`` gives
``3 e / 2``. Matching them gives the familiar factor of three, ``e ~ (1 - q)/3``.

**Shear to potential ellipticity.** At the Einstein radius the SIEP quadrupole
``b e r cos 2 theta`` and the shear quadrupole ``gamma r^2 cos 2 theta`` are
equal in amplitude when ``e = gamma``, so shear enters the sum with unit weight.

**Sign.** The shear enters the 2-theta sum with a minus sign, as
``-(gamma_1, gamma_2)`` -- see ``__Conventions__`` in the module docstring.

Both projections use the same ``b``, so any difference between them is due to
``e`` and the position angle alone.
"""

# Below this the ellipticity and the shear have cancelled and the SIEP is
# effectively circular: `images_from_source` divides by `e`, and the unfixed
# guide returned positions of ~1e7 arcsec here rather than saying so.
E_MINIMUM = 1e-3


def witt_wynne_vector_sum(
    tracer,
    grid,
    source_centre: Tuple[float, float],
    centre: Tuple[float, float] = None,
    caustic_pixel_scale: float = 0.05,
) -> WittWynne:
    """
    Project a tracer onto SIEP parameters by summing the ellipticity and shear
    vectors.

    ``source_centre`` is in **PyAutoLens** ``(y, x)`` order; ``centre``, if
    given, is in the solver's ``(x, y)`` order. Unlike the caustic-matched
    projection this one reads the mass profile's own ``ell_comps``, so picking
    the wrong profile silently mis-projects it -- the profile is selected by
    class, and a tracer with none returns a sentinel.

    Returns a **sentinel** ``WittWynne`` (``valid=False``) rather than raising
    when there is no admissible mass profile, or when the ellipticity and the
    shear cancel to ``e < E_MINIMUM`` (1e-3), which happens exactly when
    ``e_potential = gamma`` and the two are aligned.
    """
    import autolens as al
    import autogalaxy as ag

    distances = _distances_from(tracer)

    mass_list = _admissible_mass_list_from(tracer)
    shear = _shear_from(tracer)

    if not mass_list:
        return _sentinel_from(
            reason="no Isothermal/PowerLaw-family mass profile in the tracer",
            source_centre=source_centre,
            distances=distances,
        )

    mass = mass_list[0]
    centre_yx = mass.centre if centre is None else (centre[1], centre[0])

    axis_ratio, angle_deg = ag.convert.axis_ratio_and_angle_from(
        ell_comps=mass.ell_comps
    )
    e_potential = (1.0 - float(axis_ratio)) / 3.0

    gamma_1 = 0.0 if shear is None else float(shear.gamma_1)
    gamma_2 = 0.0 if shear is None else float(shear.gamma_2)

    component_1 = e_potential * np.cos(2.0 * np.radians(angle_deg)) - gamma_1
    component_2 = e_potential * np.sin(2.0 * np.radians(angle_deg)) - gamma_2

    e = float(np.hypot(component_1, component_2))
    angle_sum = np.degrees(np.arctan2(component_2, component_1)) / 2.0

    if not np.isfinite(e) or e < E_MINIMUM:
        return _sentinel_from(
            reason=(
                f"ellipticity and shear cancel: vector-sum e = {e:.3g}, below "
                f"{E_MINIMUM:g}"
            ),
            source_centre=source_centre,
            distances=distances,
        )

    b = float(
        al.LensCalc.from_tracer(tracer=tracer).einstein_radius_from(
            grid=grid, pixel_scale=caustic_pixel_scale
        )
    )

    if not np.isfinite(b) or b <= 0.0:
        return _sentinel_from(
            reason=f"degenerate Einstein radius: b={b:.4g}",
            source_centre=source_centre,
            distances=distances,
        )

    d_ol, d_ls, z_lens, z_source, h = distances

    return WittWynne(
        centre=(centre_yx[1], centre_yx[0]),
        source=(source_centre[1], source_centre[0]),
        e=e,
        b=b,
        pa_deg=(angle_sum + 90.0) % 180.0,
        d_ol=d_ol,
        d_ls=d_ls,
        z_lens=z_lens,
        z_source=z_source,
        h=h,
        reason=_mass_note_from(mass_list),
    )


"""
__Writing the isit4or2or1 Input__

The ``.in`` file is seven whitespace-separated lines, read by the C++ in this
order:

    x_lens   y_lens
    x_source y_source
    ellipticity
    einstein_radius
    position_angle
    D_ol D_ls
    z_lens z_source

``zero_centre=True`` (the default) translates the lens to the origin and the
source with it. The solver is translation invariant, so every predicted position
and lag is unchanged and only the absolute sky coordinates are withheld -- which
is what makes it safe to share a projected model from a proprietary survey.

The distances written are on the cosmology's own ``h``. Feeding this file to the
compiled `isit4or2or1` reproduces the positions and magnifications exactly, but
its lags come out a factor ``h / 0.7`` smaller, because the C++ fixes
``h = 0.7`` in its time constant. Scale by ``0.7 / h`` to compare the two.

Neither writer checks ``model.valid`` -- a sentinel model writes ``nan`` fields.
Check it before calling them.
"""

ISIT_CSV_HEADER = (
    "name,x_lens,y_lens,x_source,y_source,ellipticity,einstein_radius,"
    "position_angle,d_ol,d_ls,z_lens,z_source"
)


def write_isit_input(path, model: WittWynne, zero_centre: bool = True) -> Path:
    """Write the seven-line `isit4or2or1` input file and return its path."""
    if zero_centre:
        model = model.zeroed()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{model.centre[0]:.6e} {model.centre[1]:.6e}\n"
        f"{model.source[0]:.6e} {model.source[1]:.6e}\n"
        f"{model.e:.6e}\n"
        f"{model.b:.6e}\n"
        f"{model.pa_deg:.6e}\n"
        f"{model.d_ol:.6g} {model.d_ls:.6g}\n"
        f"{model.z_lens:.6g} {model.z_source:.6g}\n"
    )
    return path


def isit_csv_row(name: str, model: WittWynne, zero_centre: bool = True) -> str:
    """The same twelve fields as one comma-separated row (no header line)."""
    if zero_centre:
        model = model.zeroed()

    return ",".join(
        [
            name,
            f"{model.centre[0]:.6f}",
            f"{model.centre[1]:.6f}",
            f"{model.source[0]:.6f}",
            f"{model.source[1]:.6f}",
            f"{model.e:.6f}",
            f"{model.b:.6f}",
            f"{model.pa_deg:.6f}",
            f"{model.d_ol:.4f}",
            f"{model.d_ls:.4f}",
            f"{model.z_lens:.4f}",
            f"{model.z_source:.4f}",
        ]
    )
