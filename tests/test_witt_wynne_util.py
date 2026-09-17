"""
The numerics and the failure contract of ``catalogue/scripts/witt_wynne_util.py``.

That module projects a finished lens model onto Witt-Wynne SIEP parameters and
solves the resulting quartic for the 4/2/1 verdict a transient broker reads. It
is a port of someone else's C++, fixed after an independent numerical review,
and none of it needs a fit, a dataset or an aggregator to check. Five things are
pinned here, each one a finding of that review:

1. **The port still reproduces `isit4or2or1`.** The SN 2025wny rows shipped with
   the Zenodo record are hard-coded below, both the source-in and the
   image-A-in examples, to the six significant figures the C++ prints.
2. **The solver agrees with an independent oracle.** A brute-force Newton
   solver of the SIEP lens equation -- no quartic, no shared code -- is run on
   the same off-axis sources. Positions agree to 1e-8 and, crucially, so does
   the *verdict*: before the wrong-branch root filter the quartic returned only
   ever 4 or 2 images, so the "1" of `isit4or2or1` never fired.
3. **The degeneracies are sentinels, not numbers.** A source on a potential
   axis, at the lens centre, or an ``e`` outside ``(0, 1)`` returns an all-NaN
   row with the verdict ``-1``. The unfixed port returned finite, plausible and
   wrong positions instead, which is the more dangerous failure.
4. **The MGE lens light is not the lens mass.** ``lp_basis.Basis`` subclasses
   ``MassProfile``, so "the first mass profile that is not an ExternalShear"
   picks the lens light on exactly this pipeline's model shape -- an 84 degree
   position-angle error, silently. The projection of a galaxy with an MGE light
   must equal the projection of the same galaxy without it.
5. **Nothing raises on a lens it cannot project.** A sub-critical lens, a
   shear-only galaxy and a vector sum whose ellipticity and shear cancel each
   return an invalid ``WittWynne`` with a reason, so a catalogue producer can
   skip that lens instead of aborting the bundle.

Plus the writers, whose field order is a contract with a program in another
language that cannot tell it apart from a wrong one.

JAX-free and fit-free: ``autolens`` / ``autogalaxy`` are imported inside the
tests that build a tracer, and no non-linear search runs.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "_witt_wynne_util_under_test",
    PROJECT_ROOT / "catalogue" / "scripts" / "witt_wynne_util.py",
)
witt_wynne_util = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(witt_wynne_util)


# ---------------------------------------------------------------------------
# SN 2025wny: the worked examples shipped with isit4or2or1 v1.0
# (Zenodo DOI 10.5281/zenodo.20086659, CC-BY-4.0), as the guide pins them.
# ---------------------------------------------------------------------------

WNY = dict(
    centre=(6.593, 6.295),
    source=(7.019908, 6.481430),
    image_a=(5.4450, 7.6350),
    e=2.848743e-01,
    b=1.685944,
    pa_deg=2.271276e01,
    d_ol=745.89,
    d_ls=867.04,
    z_lens=0.375,
    z_source=2.008,
)

# Columns: xpos ypos mag angle lags, from 2025wny_source.out.
WNY_SOURCE_OUT = np.array(
    [
        [4.84040, 5.58275, 2.75755, -67.8834, 125.383],
        [5.54144, 7.53257, -1.26685, 40.3545, 152.259],
        [6.74068, 4.69140, -1.23602, 5.26186, 153.516],
        [9.19257, 7.39661, 1.74532, -67.0344, 0.0],
    ]
)

# Columns: xpos ypos mag angle lags, from 2025wny_imageA.out
# (source at 6.93244 6.59150).
WNY_IMAGE_OUT = np.array(
    [
        [4.68748, 5.91044, 2.76318, -78.5903, 114.580],
        [5.44500, 7.63500, -1.55821, 40.5872, 133.573],
        [6.80686, 4.81970, -0.973223, 8.24825, 159.014],
        [9.06068, 7.60014, 1.76825, -62.1260, 0.0],
    ]
)

# The .out files are written with the C++ default of six significant figures.
WNY_TOLERANCE = 5e-4


def _wny_columns(x_s, y_s):
    """The five `.out` columns of the SN 2025wny model for a source position."""
    x_image, y_image, magnification, angle = witt_wynne_util.images_from_source(
        x_s=x_s,
        y_s=y_s,
        e=WNY["e"],
        b=WNY["b"],
        pa_deg=WNY["pa_deg"],
        centre=WNY["centre"],
    )
    lags = witt_wynne_util.time_lags_days(
        x_image=x_image,
        y_image=y_image,
        x_s=x_s,
        y_s=y_s,
        e=WNY["e"],
        b=WNY["b"],
        pa_deg=WNY["pa_deg"],
        d_ol=WNY["d_ol"],
        d_ls=WNY["d_ls"],
        z_lens=WNY["z_lens"],
        z_source=WNY["z_source"],
        centre=WNY["centre"],
        h=0.7,
    )
    return np.stack([x_image, y_image, magnification, angle, lags], axis=1)


def test_regression_against_the_2025wny_source_example():
    columns = _wny_columns(x_s=WNY["source"][0], y_s=WNY["source"][1])

    assert witt_wynne_util.n_images_from(columns[:, 0], columns[:, 1]) == 4
    assert np.abs(columns - WNY_SOURCE_OUT).max() < WNY_TOLERANCE


def test_regression_against_the_2025wny_image_a_example():
    x_s, y_s = witt_wynne_util.source_from_image(
        x_i=WNY["image_a"][0],
        y_i=WNY["image_a"][1],
        e=WNY["e"],
        b=WNY["b"],
        pa_deg=WNY["pa_deg"],
        centre=WNY["centre"],
    )

    assert np.abs(np.array([x_s, y_s]) - np.array([6.93244, 6.59150])).max() < 1e-4

    columns = _wny_columns(x_s=x_s, y_s=y_s)

    assert witt_wynne_util.n_images_from(columns[:, 0], columns[:, 1]) == 4
    assert np.abs(columns - WNY_IMAGE_OUT).max() < WNY_TOLERANCE


# ---------------------------------------------------------------------------
# An independent brute-force oracle for the SIEP lens equation
#
# Newton's method from a dense polar seed grid on
# ``psi = b sqrt(x^2 + y^2 / (1 - e)^2)`` in the registered frame. It shares no
# code with the module: no quartic, no ACLE, no scaled (p, q).
# ---------------------------------------------------------------------------


def _oracle_deflection(x, y, e, b):
    g = np.sqrt(x * x + y * y / (1.0 - e) ** 2)
    return b * x / g, b * y / ((1.0 - e) ** 2 * g)


def _oracle_hessian(x, y, e, b):
    g = np.sqrt(x * x + y * y / (1.0 - e) ** 2)
    f = (1.0 - e) ** 2
    return (
        b * (y * y / f) / g**3,
        b * x * x / (f * g**3),
        -b * x * y / (f * g**3),
    )


def _oracle_images(x_s, y_s, e, b, n_r=30, n_t=120, iterations=60):
    """Every image of a source at ``(x_s, y_s)``, registered frame, (x, y)."""
    radius = np.linspace(0.02 * b, 4.0 * b, n_r)
    theta = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)
    radius, theta = np.meshgrid(radius, theta, indexing="ij")

    x = (radius * np.cos(theta)).ravel().astype("float64")
    y = (radius * np.sin(theta)).ravel().astype("float64")

    with np.errstate(all="ignore"):
        for _ in range(iterations):
            deflection_x, deflection_y = _oracle_deflection(x, y, e, b)
            f_x, f_y = x - deflection_x - x_s, y - deflection_y - y_s

            psi_xx, psi_yy, psi_xy = _oracle_hessian(x, y, e, b)
            a_11, a_22, a_12 = 1.0 - psi_xx, 1.0 - psi_yy, -psi_xy

            determinant = a_11 * a_22 - a_12 * a_12
            determinant = np.where(np.abs(determinant) < 1e-30, np.nan, determinant)

            delta_x = (a_22 * f_x - a_12 * f_y) / determinant
            delta_y = (-a_12 * f_x + a_11 * f_y) / determinant

            step = np.hypot(delta_x, delta_y)
            cap = np.minimum(1.0, 0.25 * b / np.maximum(step, 1e-300))

            x, y = x - delta_x * cap, y - delta_y * cap

            bad = (
                ~np.isfinite(x)
                | ~np.isfinite(y)
                | (np.hypot(x, y) > 1e3 * b)
                | (np.hypot(x, y) < 1e-12)
            )
            x, y = np.where(bad, np.nan, x), np.where(bad, np.nan, y)

        converged = np.isfinite(x) & np.isfinite(y)
        x, y = x[converged], y[converged]

        deflection_x, deflection_y = _oracle_deflection(x, y, e, b)
        solved = np.hypot(x - deflection_x - x_s, y - deflection_y - y_s) < 1e-11

    images = []

    for x_i, y_i in zip(x[solved], y[solved]):
        if all(
            np.hypot(x_i - x_o, y_i - y_o) >= 1e-7 * max(b, 1.0) for x_o, y_o in images
        ):
            images.append((float(x_i), float(y_i)))

    return np.array(images).reshape(-1, 2)


def _oracle_cases():
    """
    Off-axis sources spanning the astroid, its neighbourhood and far outside it.

    ``t`` parametrises the astroid ``(cos^3 t, sin^3 t)`` and ``scale`` is the
    multiple of it, so 0.5 is deep inside the caustic (4 images), 1.5 outside
    (2, or 1 where the second image has been lost) and 4.0 far outside.
    """
    cases = []
    position_angles = [0.0, 37.0, 90.0]

    for scale in [0.5, 0.95, 1.5, 4.0]:
        for e in [0.1, 0.2, 0.3]:
            for t in [0.9, 2.3, -0.4]:
                cases.append(
                    (scale, e, t, position_angles[len(cases) % len(position_angles)])
                )

    return cases


def test_solver_matches_an_independent_brute_force_oracle():
    b = 1.2
    centre = (0.3, -0.2)

    verdicts = []

    for scale, e, t, pa_deg in _oracle_cases():
        a_long, a_short = witt_wynne_util.siep_astroid_semi_axes(e=e, b=b)

        x_reg_s = scale * np.cos(t) ** 3 * a_long
        y_reg_s = -scale * np.sin(t) ** 3 * a_short

        x_s, y_s = witt_wynne_util.to_detector(
            x_reg_s, y_reg_s, pa_deg=pa_deg, centre=centre
        )

        x_image, y_image, _, _ = witt_wynne_util.images_from_source(
            x_s=x_s, y_s=y_s, e=e, b=b, pa_deg=pa_deg, centre=centre
        )
        verdict = witt_wynne_util.n_images_from(x_image, y_image)

        truth = _oracle_images(x_s=x_reg_s, y_s=y_reg_s, e=e, b=b)

        assert verdict == len(truth), (scale, e, t, pa_deg, verdict, len(truth))

        x_reg, y_reg = witt_wynne_util.to_registered(
            x_image, y_image, pa_deg=pa_deg, centre=centre
        )

        separation = np.hypot(
            truth[:, 0][:, None] - x_reg[None, :],
            truth[:, 1][:, None] - y_reg[None, :],
        ).min(axis=1)

        assert separation.max() < 1e-8, (scale, e, t, pa_deg, separation.max())

        verdicts.append(verdict)

    assert len(verdicts) >= 30
    # The 1-image verdict is the one the unfixed solver could never return, and
    # a case that is genuinely 2 images is what tells it apart from a filter
    # that simply drops roots.
    assert verdicts.count(1) >= 5
    assert verdicts.count(2) >= 5
    assert verdicts.count(4) >= 5


# ---------------------------------------------------------------------------
# The degeneracy table
# ---------------------------------------------------------------------------


def _degenerate_sources():
    """
    ``(label, x_s, y_s, e)`` at ``b = 1.2``, ``pa_deg = 90`` (where the
    registered and detector frames coincide), covering every row of the review's
    degeneracy table.
    """
    b, e = 1.2, 0.2
    a_long, a_short = witt_wynne_util.siep_astroid_semi_axes(e=e, b=b)

    return [
        ("major axis, inside the caustic", 0.5 * a_long, 0.0, e),
        ("major axis, outside the caustic", 1.5 * a_long, 0.0, e),
        ("minor axis, inside the caustic", 0.0, -0.5 * a_short, e),
        ("minor axis, outside the caustic", 0.0, -1.5 * a_short, e),
        ("the lens centre", 0.0, 0.0, e),
        ("the major cusp", a_long, 0.0, e),
        ("the minor cusp", 0.0, -a_short, e),
        ("a nanoarcsecond off the major axis", 0.5 * a_long, 1e-9, e),
        ("e = 0 exactly", 0.05, 0.03, 0.0),
        ("e = 1 exactly", 0.05, 0.03, 1.0),
        ("e below zero", 0.05, 0.03, -0.05),
        ("e above one", 0.05, 0.03, 1.5),
        ("e not a number", 0.05, 0.03, np.nan),
    ]


@pytest.mark.parametrize("case", _degenerate_sources(), ids=lambda case: case[0])
def test_degenerate_sources_return_a_nan_row_and_the_sentinel_verdict(case):
    label, x_s, y_s, e = case

    x_image, y_image, magnification, angle = witt_wynne_util.images_from_source(
        x_s=x_s, y_s=y_s, e=e, b=1.2, pa_deg=90.0
    )

    assert witt_wynne_util.n_images_from(x_image, y_image) == -1
    assert witt_wynne_util.N_IMAGES_SENTINEL == -1

    for values in (x_image, y_image, magnification, angle):
        assert values.size == 1
        assert np.all(np.isnan(values))

    # Nothing downstream turns the sentinel back into a number.
    lags = witt_wynne_util.time_lags_days(
        x_image=x_image,
        y_image=y_image,
        x_s=x_s,
        y_s=y_s,
        e=e if np.isfinite(e) else 0.2,
        b=1.2,
        pa_deg=90.0,
        d_ol=745.89,
        d_ls=867.04,
        z_lens=0.5,
        z_source=1.5,
    )
    assert np.all(np.isnan(lags))


def test_a_source_far_enough_off_axis_is_still_solved():
    """The on-axis screen is a screen, not a blanket: 0.01" off it solves."""
    b, e = 1.2, 0.2
    a_long, _ = witt_wynne_util.siep_astroid_semi_axes(e=e, b=b)

    x_image, y_image, _, _ = witt_wynne_util.images_from_source(
        x_s=0.5 * a_long, y_s=0.01, e=e, b=b, pa_deg=90.0
    )

    assert witt_wynne_util.n_images_from(x_image, y_image) == 4


# ---------------------------------------------------------------------------
# The projection: the mass profile is picked by class
# ---------------------------------------------------------------------------


def _grid():
    import autolens as al

    return al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)


def _lens_mass():
    import autolens as al
    import autogalaxy as ag

    return al.mp.Isothermal(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.75, angle=40.0),
        einstein_radius=1.2,
    )


def _external_shear(magnitude=0.05, angle=70.0):
    import autolens as al
    import autogalaxy as ag

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=magnitude, angle=angle)
    return al.mp.ExternalShear(gamma_1=gamma_1, gamma_2=gamma_2)


def _mge_basis():
    """An MGE lens light, its centre deliberately offset from the mass."""
    import autolens as al

    return al.lp_basis.Basis(
        profile_list=[
            al.lp.Gaussian(
                centre=(0.15, -0.1),
                ell_comps=(0.05, 0.05),
                intensity=1.0,
                sigma=0.1 * (index + 1),
            )
            for index in range(5)
        ]
    )


def _tracer_from(lens):
    import autolens as al

    return al.Tracer(
        galaxies=[lens, al.Galaxy(redshift=1.5)], cosmology=al.cosmo.Planck15()
    )


def test_an_mge_light_basis_is_never_taken_for_the_lens_mass():
    import autolens as al

    mass, shear = _lens_mass(), _external_shear()

    with_light = _tracer_from(
        al.Galaxy(redshift=0.5, bulge=_mge_basis(), mass=mass, shear=shear)
    )

    picked_mass, picked_shear = witt_wynne_util._mass_and_shear_from(tracer=with_light)

    assert isinstance(picked_mass, al.mp.Isothermal)
    assert isinstance(picked_shear, al.mp.ExternalShear)
    assert picked_mass.centre == mass.centre

    without_light = _tracer_from(al.Galaxy(redshift=0.5, mass=mass, shear=shear))

    grid, source_centre = _grid(), (0.03, 0.05)

    lit = witt_wynne_util.witt_wynne_from_tracer(
        tracer=with_light, grid=grid, source_centre=source_centre
    )
    dark = witt_wynne_util.witt_wynne_from_tracer(
        tracer=without_light, grid=grid, source_centre=source_centre
    )

    assert lit.valid and dark.valid
    assert abs(lit.e - dark.e) < 1e-3
    assert abs(lit.pa_deg - dark.pa_deg) < 0.1
    assert np.abs(np.array(lit.centre) - np.array(dark.centre)).max() < 1e-3

    # The vector sum reads the mass profile's own ell_comps, so picking the
    # light there rotates the potential outright.
    lit_sum = witt_wynne_util.witt_wynne_vector_sum(
        tracer=with_light, grid=grid, source_centre=source_centre
    )
    dark_sum = witt_wynne_util.witt_wynne_vector_sum(
        tracer=without_light, grid=grid, source_centre=source_centre
    )

    assert lit_sum.valid and dark_sum.valid
    assert abs(lit_sum.e - dark_sum.e) < 1e-3
    assert abs(lit_sum.pa_deg - dark_sum.pa_deg) < 0.1


def test_a_mass_profile_on_the_source_galaxy_is_not_the_lens():
    """
    Selection is per plane, not tracer-wide.

    A mass profile or an ``ExternalShear`` on the *source* galaxy belongs to a
    second deflector behind the lens, not to the lens this projection
    describes. Taken tracer-wide it would be eligible: the vector sum would read
    its ``ell_comps`` and the producer's ``mass_profile`` column would name it,
    and with the source's own shear picked the potential is rotated outright.
    """
    import autolens as al
    import autogalaxy as ag

    mass, shear = _lens_mass(), _external_shear()

    lens = al.Galaxy(redshift=0.5, mass=mass, shear=shear)
    source_with_mass = al.Galaxy(
        redshift=1.5,
        mass=al.mp.Isothermal(
            centre=(0.9, -0.8),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.4, angle=115.0),
            einstein_radius=0.6,
        ),
        shear=_external_shear(magnitude=0.2, angle=5.0),
    )

    tracer = al.Tracer(galaxies=[lens, source_with_mass], cosmology=al.cosmo.Planck15())

    picked_mass, picked_shear = witt_wynne_util._mass_and_shear_from(tracer=tracer)

    assert picked_mass.centre == mass.centre
    assert picked_mass.einstein_radius == mass.einstein_radius
    assert picked_shear.gamma_1 == shear.gamma_1
    assert picked_shear.gamma_2 == shear.gamma_2

    # One admissible profile, so no "2 admissible mass profiles" caveat either.
    assert len(witt_wynne_util._admissible_mass_list_from(tracer)) == 1

    grid, source_centre = _grid(), (0.03, 0.05)

    with_source_mass = witt_wynne_util.witt_wynne_vector_sum(
        tracer=tracer, grid=grid, source_centre=source_centre
    )
    lens_only = witt_wynne_util.witt_wynne_vector_sum(
        tracer=_tracer_from(lens), grid=grid, source_centre=source_centre
    )

    assert with_source_mass.valid and lens_only.valid
    assert with_source_mass.e == lens_only.e
    assert with_source_mass.pa_deg == lens_only.pa_deg


def test_a_second_mass_profile_is_recorded_rather_than_ignored():
    import autolens as al

    lens = al.Galaxy(
        redshift=0.5,
        mass=_lens_mass(),
        perturber=al.mp.IsothermalSph(centre=(1.0, 1.0), einstein_radius=0.15),
        shear=_external_shear(),
    )

    model = witt_wynne_util.witt_wynne_from_tracer(
        tracer=_tracer_from(lens), grid=_grid(), source_centre=(0.03, 0.05)
    )

    assert model.valid
    assert "2 admissible mass profiles" in model.reason
    assert "Isothermal" in model.reason


# ---------------------------------------------------------------------------
# The projection: everything it cannot project returns a sentinel
# ---------------------------------------------------------------------------


def test_a_sub_critical_lens_returns_a_sentinel_rather_than_raising():
    import autolens as al

    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.001),
    )

    model = witt_wynne_util.witt_wynne_from_tracer(
        tracer=_tracer_from(lens), grid=_grid(), source_centre=(0.03, 0.05)
    )

    assert not model.valid
    assert "caustic" in model.reason
    assert np.isnan(model.e) and np.isnan(model.b) and np.isnan(model.pa_deg)
    assert model.n_images() == -1


def test_a_shear_only_galaxy_returns_a_sentinel_from_both_projections():
    import autolens as al

    tracer = _tracer_from(al.Galaxy(redshift=0.5, shear=_external_shear()))

    for projection in (
        witt_wynne_util.witt_wynne_from_tracer,
        witt_wynne_util.witt_wynne_vector_sum,
    ):
        model = projection(tracer=tracer, grid=_grid(), source_centre=(0.03, 0.05))

        assert not model.valid
        assert "mass profile" in model.reason
        assert np.isnan(model.e)
        # The lens is still identified: its redshifts survive for the CSV row.
        assert model.z_lens == 0.5 and model.z_source == 1.5


def test_the_vector_sum_reports_ellipticity_and_shear_cancelling():
    """
    At ``q = 0.7`` the potential ellipticity is ``(1 - q) / 3 = 0.1``, which an
    aligned ``gamma = 0.10`` cancels exactly. The unfixed guide divided by that
    zero and returned image positions of ~1e7 arcsec.
    """
    import autolens as al
    import autogalaxy as ag

    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=40.0),
            einstein_radius=1.2,
        ),
        shear=_external_shear(magnitude=0.10, angle=40.0),
    )

    model = witt_wynne_util.witt_wynne_vector_sum(
        tracer=_tracer_from(lens), grid=_grid(), source_centre=(0.03, 0.05)
    )

    assert not model.valid
    assert "cancel" in model.reason
    assert model.n_images() == -1

    # The caustic-matched projection has no such singularity: it reads the
    # tracer's own caustic, which is merely rounder.
    caustic = witt_wynne_util.witt_wynne_from_tracer(
        tracer=_tracer_from(lens), grid=_grid(), source_centre=(0.03, 0.05)
    )
    assert caustic.valid and caustic.e > 0.0


# ---------------------------------------------------------------------------
# The writers: a field order shared with a program in another language
# ---------------------------------------------------------------------------


def _model():
    return witt_wynne_util.WittWynne(
        centre=(-0.2, 0.3),
        source=(-0.16, 0.33),
        e=7.965893e-02,
        b=1.187194,
        pa_deg=112.790997,
        d_ol=878.6177660230832,
        d_ls=686.4784027811058,
        z_lens=0.5,
        z_source=1.5,
        h=0.6774,
    )


def test_write_isit_input_writes_the_seven_documented_lines(tmp_path):
    model = _model()

    path = witt_wynne_util.write_isit_input(
        path=tmp_path / "witt_wynne.in", model=model, zero_centre=False
    )

    lines = path.read_text().splitlines()

    assert len(lines) == 7
    assert [len(line.split()) for line in lines] == [2, 2, 1, 1, 1, 2, 2]

    assert [float(value) for value in lines[0].split()] == list(model.centre)
    assert [float(value) for value in lines[1].split()] == list(model.source)
    assert float(lines[2]) == pytest.approx(model.e, rel=1e-6)
    assert float(lines[3]) == pytest.approx(model.b, rel=1e-6)
    assert float(lines[4]) == pytest.approx(model.pa_deg, rel=1e-6)
    assert [float(value) for value in lines[5].split()] == pytest.approx(
        [model.d_ol, model.d_ls], rel=1e-5
    )
    assert [float(value) for value in lines[6].split()] == [
        model.z_lens,
        model.z_source,
    ]


def test_zero_centre_hides_the_sky_position_and_changes_nothing_else(tmp_path):
    model = _model()

    zeroed = (
        witt_wynne_util.write_isit_input(
            path=tmp_path / "zeroed.in", model=model, zero_centre=True
        )
        .read_text()
        .splitlines()
    )

    assert [float(value) for value in zeroed[0].split()] == [0.0, 0.0]
    assert [float(value) for value in zeroed[1].split()] == pytest.approx(
        [model.source[0] - model.centre[0], model.source[1] - model.centre[1]],
        abs=1e-9,
    )
    # Lines 3-7 are untouched by the translation.
    absolute = (
        witt_wynne_util.write_isit_input(
            path=tmp_path / "absolute.in", model=model, zero_centre=False
        )
        .read_text()
        .splitlines()
    )
    assert zeroed[2:] == absolute[2:]

    x_image, y_image, magnification, _ = model.images()
    lags = model.lags(x_image=x_image, y_image=y_image)

    x_zeroed, y_zeroed, magnification_zeroed, _ = model.zeroed().images()
    lags_zeroed = model.zeroed().lags(x_image=x_zeroed, y_image=y_zeroed)

    assert np.abs((x_zeroed + model.centre[0]) - x_image).max() == 0.0
    assert np.abs((y_zeroed + model.centre[1]) - y_image).max() == 0.0
    assert np.abs(magnification_zeroed - magnification).max() == 0.0
    assert np.abs(lags_zeroed - lags).max() == 0.0


def test_isit_csv_row_matches_the_header_field_for_field():
    model = _model()

    header = witt_wynne_util.ISIT_CSV_HEADER.split(",")
    row = witt_wynne_util.isit_csv_row(name="a_lens", model=model, zero_centre=False)
    fields = row.split(",")

    assert header == [
        "name",
        "x_lens",
        "y_lens",
        "x_source",
        "y_source",
        "ellipticity",
        "einstein_radius",
        "position_angle",
        "d_ol",
        "d_ls",
        "z_lens",
        "z_source",
    ]
    assert len(fields) == len(header) == 12
    assert fields[0] == "a_lens"

    assert [float(value) for value in fields[1:]] == pytest.approx(
        [
            model.centre[0],
            model.centre[1],
            model.source[0],
            model.source[1],
            model.e,
            model.b,
            model.pa_deg,
            model.d_ol,
            model.d_ls,
            model.z_lens,
            model.z_source,
        ],
        rel=1e-6,
        abs=1e-6,
    )
