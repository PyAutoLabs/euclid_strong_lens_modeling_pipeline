"""
``render_north_up.mirrored_fit_from`` is what every north-up release figure is drawn from,
so it is tested by the physics the figures rely on: on a simulated lens, the mirrored fit's
model image is the native model image flipped top to bottom, with the same chi-squared and
figure of merit — for a light-profile source and for a Delaunay pixelized source whose
adapt image and image-plane mesh grid must be mirrored with the data.
"""
import numpy as np
import pytest

al = pytest.importorskip("autolens")

from scripts.tools import render_north_up  # noqa: E402

PIXEL_SCALE = 0.1
SHAPE = (60, 60)
MASK_RADIUS = 2.5


def galaxies_from():
    lens = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, -0.12), ell_comps=(0.15, -0.08), intensity=1.0,
                           effective_radius=0.6, sersic_index=3.0),
        mass=al.mp.Isothermal(centre=(0.04, -0.1), ell_comps=(0.2, 0.1), einstein_radius=1.1),
        shear=al.mp.ExternalShear(gamma_1=0.04, gamma_2=-0.06),
    )
    source = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(centre=(0.2, 0.15), ell_comps=(-0.1, 0.25), intensity=0.5,
                           effective_radius=0.2, sersic_index=1.0),
    )
    return lens, source


def dataset_from(lens, source):
    """The lens image plus fixed-seed Gaussian noise: synthetic data in memory, not a simulated dataset."""
    grid = al.Grid2D.uniform(shape_native=SHAPE, pixel_scales=PIXEL_SCALE)
    image = np.asarray(al.Tracer(galaxies=[lens, source]).image_2d_from(grid=grid).native)
    noise = 0.05 * np.ones(SHAPE)
    dataset = al.Imaging(
        data=al.Array2D.no_mask(values=image + noise * np.random.default_rng(1).standard_normal(SHAPE),
                                pixel_scales=PIXEL_SCALE),
        noise_map=al.Array2D.no_mask(values=noise, pixel_scales=PIXEL_SCALE),
        psf=al.Convolver.from_gaussian(shape_native=(11, 11), sigma=0.1, pixel_scales=PIXEL_SCALE),
    )
    # An off-centre mask, so a wrong mask flip cannot hide behind symmetry.
    mask = al.Mask2D.circular(shape_native=SHAPE, pixel_scales=PIXEL_SCALE, radius=MASK_RADIUS,
                              centre=(0.3, -0.2))
    return dataset.apply_mask(mask=mask)


def assert_mirrored(fit, mirrored):
    np.testing.assert_allclose(
        np.asarray(mirrored.model_data.native),
        np.flipud(np.asarray(fit.model_data.native)),
        atol=1e-8 * np.max(np.abs(fit.model_data)),
    )
    np.testing.assert_allclose(mirrored.chi_squared, fit.chi_squared, rtol=1e-9)
    np.testing.assert_allclose(mirrored.figure_of_merit, fit.figure_of_merit,
                               rtol=1e-6)


def test__light_profile_fit__mirrors_exactly():
    lens, source = galaxies_from()
    fit = al.FitImaging(dataset=dataset_from(lens, source), tracer=al.Tracer(galaxies=[lens, source]))
    mirrored = render_north_up.mirrored_fit_from(fit)

    assert_mirrored(fit, mirrored)
    assert mirrored.dataset.mask.mask_centre[0] == pytest.approx(-fit.dataset.mask.mask_centre[0])
    row = render_north_up.witness_from("tile", "vis_lp", fit, mirrored, fit.figure_of_merit)
    assert row["status"] == "pass"


def test__pixelized_fit__mirrors_with_adapt_images_and_mesh_grid():
    lens, source = galaxies_from()
    dataset = dataset_from(lens, source)

    adapt_data = al.Tracer(galaxies=[lens, source]).galaxy_image_2d_dict_from(
        grid=dataset.grids.lp
    )[source]
    adapt_data = adapt_data + 0.01 * np.max(adapt_data)
    mesh_grid = al.image_mesh.Hilbert(pixels=80, weight_power=3.5, weight_floor=0.01) \
        .image_plane_mesh_grid_from(mask=dataset.mask, adapt_data=adapt_data)

    pixelized = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.Delaunay(pixels=mesh_grid.shape[0]),
            regularization=al.reg.Constant(coefficient=1.0),
        ),
    )
    adapt_images = al.AdaptImages(
        galaxy_image_dict={pixelized: adapt_data},
        galaxy_image_plane_mesh_grid_dict={pixelized: mesh_grid},
    )
    fit = al.FitImaging(dataset=dataset, tracer=al.Tracer(galaxies=[lens, pixelized]),
                        adapt_images=adapt_images)
    mirrored = render_north_up.mirrored_fit_from(fit)

    assert_mirrored(fit, mirrored)
    np.testing.assert_allclose(
        np.asarray(mirrored.inversion.reconstruction),
        np.asarray(fit.inversion.reconstruction),
        rtol=1e-6, atol=1e-6 * np.max(np.abs(fit.inversion.reconstruction)),
    )


def test__witness__a_singular_mirrored_log_determinant_is_judged_on_the_images():
    lens, source = galaxies_from()
    fit = al.FitImaging(dataset=dataset_from(lens, source), tracer=al.Tracer(galaxies=[lens, source]))
    mirrored = render_north_up.mirrored_fit_from(fit)

    class SingularFigureOfMerit:
        def __getattr__(self, name):
            if name == "figure_of_merit":
                raise np.linalg.LinAlgError("Matrix is not positive definite")
            return getattr(mirrored, name)

    row = render_north_up.witness_from("tile", "vis_pix", fit, SingularFigureOfMerit(), fit.figure_of_merit)
    assert row["status"] == "pass_no_fom"
    assert np.isnan(row["mirrored_figure_of_merit"])
