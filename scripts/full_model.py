"""
Full SLaM Pipeline
==================

Runs the complete Source, Light and Mass (SLaM) pipeline on a Euclid VIS dataset:

  SOURCE LP      — parametric MGE source + isothermal mass (fast initialisation)
  SOURCE PIX 1   — pixelised source initialisation with rectangular mesh
  SOURCE PIX 2   — refined pixelisation with adapt image
  LIGHT LP       — lens light refinement with source/mass fixed
  MASS TOTAL     — final PowerLaw mass model

For a detailed walkthrough of every step see ``start_here.py``, which fits the
SOURCE LP stage in isolation with full inline documentation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import util


"""
__SOURCE LP PIPELINE__

The SOURCE LP PIPELINE uses one search to initialize a robust model for the
source galaxy's light:

 - Models the lens galaxy's light as an MGE with 2 x 20 Gaussians.
 - Uses an ``Isothermal`` model for the lens's total mass distribution with an
   ``ExternalShear``.
 - Models the source galaxy's light as an MGE with 1 x 20 Gaussians.

The mass and source models from this search initialize the SOURCE PIX PIPELINE
searches that follow.
"""


def source_lp(
    settings_search,
    analysis,
    lens_bulge,
    mass,
    shear,
    source_bulge,
    redshift_lens: float = 0.5,
    redshift_source: float = 1.0,
    n_batch: int = 50,
):
    import autofit as af
    import autolens as al

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=None,
                mass=mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
            ),
        ),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=200,
        n_batch=n_batch,
        n_like_max=200000,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Rectangular adaptive mesh + Adapt regularization.  Creates the adapt image
for the refined pixelisation in SOURCE PIX 2.

``mesh_pixels_yx`` fixes the mesh resolution; it cannot be a free parameter
because JAX requires statically shaped arrays.

Image positions are computed automatically from the SOURCE LP result to prevent
unphysical source reconstructions.  An adapt image is computed from the SOURCE
LP result and passed to the analysis.
"""


def source_pix_1(
    settings_search,
    analysis,
    source_lp_result,
    mesh_init,
    regularization_init,
    n_batch: int = 20,
):
    import autofit as af
    import autolens as al

    mass = al.util.chaining.mass_from(
        mass=source_lp_result.model.galaxies.lens.mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    shear = source_lp_result.model.galaxies.lens.shear

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Refined pixelisation using the adapt image from SOURCE PIX 1.  The
``RectangularAdaptImage`` mesh and ``Adapt`` regularization adapt the source
pixels and regularization weights to the source's morphology.
"""


def source_pix_2(
    settings_search,
    analysis,
    source_lp_result,
    source_pix_result_1,
    mesh,
    regularization,
    n_batch: int = 20,
):
    import autofit as af
    import autolens as al

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh,
                    regularization=regularization,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=75,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__LIGHT LP PIPELINE__

Refines the lens light (2×20 Gaussians) with mass and source fixed.

The lens light model is fit from scratch (not seeded from SOURCE LP) because the
earlier mass and source models may not have been precise enough for an accurate
lens-light subtraction.
"""


def light_lp(
    settings_search,
    analysis,
    source_result_for_lens,
    source_result_for_source,
    lens_bulge,
    lens_disk=None,
    n_batch: int = 30,
):
    import autofit as af
    import autolens as al

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=lens_disk,
                mass=source_result_for_lens.instance.galaxies.lens.mass,
                shear=source_result_for_lens.instance.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=300,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

PowerLaw + ExternalShear mass model, priors seeded from SOURCE PIX result.

Positions are computed from the SOURCE PIX 2 result for more precise multiple
image positions.  The shear prior is reset to broad uniform priors because the
``PowerLaw`` model can absorb azimuthal structure previously captured by shear.
"""


def mass_total(
    settings_search,
    analysis,
    source_result_for_lens,
    source_result_for_source,
    light_result,
    mass,
    reset_shear_prior: bool = False,
    n_batch: int = 30,
):
    import autofit as af
    import autolens as al

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_result_for_lens.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    bulge = light_result.instance.galaxies.lens.bulge
    disk = light_result.instance.galaxies.lens.disk

    if not reset_shear_prior:
        shear = source_result_for_lens.model.galaxies.lens.shear
    else:
        shear = al.mp.ExternalShear

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=bulge,
                disk=disk,
                mass=mass,
                shear=shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=150,
        n_batch=n_batch,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def fit(
    dataset_name: str,
    sample_name: str = None,
    iterations_per_quick_update: int = 5000,
):
    from autolens import conf

    project_root = Path(__file__).parent.parent
    conf.instance.push(
        new_path=project_root / "config", output_path=project_root / "output"
    )

    import autofit as af
    import autolens as al

    d = util.load_vis_dataset(dataset_name, sample_name=sample_name)

    # full_model uses a sparse operator for faster pixelisation fits
    dataset = d.dataset.apply_sparse_operator()

    settings_search = af.SettingsSearch(
        path_prefix=(
            Path(sample_name) / dataset_name
            if sample_name is not None
            else Path(dataset_name)
        ),
        unique_tag="slam",
        info={"magzero": d.magzero},
        session=None,
    )

    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __SOURCE LP PIPELINE__

    Isothermal mass + MGE lens + MGE source.  Provides the initial model and
    adapt image for the pixelised source stages that follow.
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=d.positions_likelihood_list,
        use_jax=True,
        title_prefix="VIS",
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        dataset_main_path=d.dataset_main_path,
        **settings_search.info,
    )

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=d.dataset_centre,
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.UniformPrior(
        lower_limit=d.dataset_centre[0] - 0.05, upper_limit=d.dataset_centre[0] + 0.05
    )
    mass.centre.centre_1 = af.UniformPrior(
        lower_limit=d.dataset_centre[1] - 0.05, upper_limit=d.dataset_centre[1] + 0.05
    )

    source_lp_result = source_lp(
        settings_search=settings_search,
        analysis=analysis,
        lens_bulge=lens_bulge,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear),
        source_bulge=al.model_util.mge_model_from(
            mask_radius=d.mask_radius, total_gaussians=20, centre_prior_is_uniform=False
        ),
        redshift_lens=redshift_lens,
        redshift_source=redshift_source,
    )

    """
    __SOURCE PIX PIPELINE 1__

    Rectangular adaptive mesh + Adapt regularization.  Creates the adapt image
    for the refined pixelisation in SOURCE PIX 2.

    ``mesh_pixels_yx`` fixes the mesh resolution; it cannot be a free parameter
    because JAX requires statically shaped arrays.
    """
    mesh_pixels_yx = 28
    mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = util.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        use_jax=True,
        title_prefix="VIS",
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        dataset_main_path=d.dataset_main_path,
        **settings_search.info,
    )

    source_pix_result_1 = source_pix_1(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        mesh_init=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
        regularization_init=af.Model(al.reg.Adapt),
    )

    """
    __SOURCE PIX PIPELINE 2__

    Refined pixelisation using the adapt image from SOURCE PIX 1.
    """
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = util.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
        title_prefix="VIS",
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        dataset_main_path=d.dataset_main_path,
        **settings_search.info,
    )

    source_pix_result_2 = source_pix_2(
        settings_search=settings_search,
        analysis=analysis,
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_result_1,
        mesh=af.Model(al.mesh.RectangularAdaptImage, shape=mesh_shape),
        regularization=af.Model(al.reg.Adapt),
    )

    """
    __LIGHT LP PIPELINE__

    Refines the lens light (2×30 Gaussians) with mass and source fixed.
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        use_jax=True,
        title_prefix="VIS",
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        dataset_main_path=d.dataset_main_path,
        **settings_search.info,
    )

    light_result = light_lp(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        lens_bulge=al.model_util.mge_model_from(
            mask_radius=d.mask_radius,
            total_gaussians=20,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
            centre=d.dataset_centre,
        ),
        lens_disk=None,
    )

    """
    __MASS TOTAL PIPELINE__

    PowerLaw + ExternalShear mass model, priors seeded from SOURCE LP result.
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_pix_result_2.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        use_jax=True,
        title_prefix="VIS",
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        dataset_main_path=d.dataset_main_path,
        **settings_search.info,
    )

    mass_result = mass_total(
        settings_search=settings_search,
        analysis=analysis,
        source_result_for_lens=source_pix_result_1,
        source_result_for_source=source_pix_result_2,
        light_result=light_result,
        mass=af.Model(al.mp.PowerLaw),
        reset_shear_prior=True,
    )

    return source_lp_result, mass_result


if __name__ == "__main__":
    sample_name, dataset_name, iterations_per_quick_update = util.parse_fit_args()
    fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
    )
