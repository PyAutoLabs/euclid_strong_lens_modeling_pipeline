"""
Euclid Pipeline: Start Here
===========================

This script fits an initial lens model to a Euclid strong lens dataset. It is the
recommended entry point — run this first before any of the pipelines in ``scripts/``.

**Installation:** see https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline

**Running as a black box:** pass it the dataset name and it fits automatically. Results
and visualisations are written to ``output/``. For small samples, browsing ``output/``
directly is sufficient. For large samples use the ``workflow/`` scripts to export
.csv, .fits, and .png summaries via the database aggregator.

**Questions:** contact James Nightingale on the Euclid Consortium Slack.

__Initial Lens Model__

Fits a Multi Gaussian Expansion (MGE) for the lens and source light together with a
Singular Isothermal Ellipsoid (SIE) + shear mass model. The MGE decomposes each galaxy
into ~15–100 Gaussians whose intensities are solved via linear algebra ("inversion"),
keeping the non-linear parameter space small (~15 parameters) and well-conditioned. This
makes the fit fast and highly likely to reach the global maximum, especially on a GPU.

__SLaM Overview__

``scripts/full_model.py`` extends this fit using the Source, Light and Mass (SLaM)
pipeline: a sequence of chained searches that progressively build up a complex lens model.
Each stage is seeded from the previous result, ensuring robust convergence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import util


def fit(
    dataset_name: str,
    sample_name: str = None,
    iterations_per_quick_update: int = 5000,
):
    from autolens import conf

    project_root = Path(__file__).parent
    conf.instance.push(
        new_path=project_root / "config", output_path=project_root / "output"
    )

    import autofit as af
    import autolens as al

    """
    __Dataset__

    ``util.load_vis_dataset`` performs all standard dataset preparation in one call:

    1. Resolves paths — ``dataset/<dataset_name>/<dataset_name>.fits``
    2. Reads the multi-HDU FITS file and maps wavebands to HDU indices
    3. Loads the VIS image, noise-map, and PSF (pixel scale 0.1"/px)
    4. Finds the brightest central pixel to anchor model priors
    5. Loads ``info.json`` metadata (mask radius/centre, redshifts, …)
    6. Reads the FITS header for the photometric zero-point (MAGZERO) and WCS
    7. Applies ``mask_extra_galaxies.fits`` noise scaling if present
    8. Applies a circular mask (radius from argument, info.json, or 3.0" default)
    9. Applies standard adaptive over-sampling (4/2/1× in radial bins)
    10. Loads the lowest-resolution band PSF + FWHM for aperture photometry

    See ``util.load_vis_dataset`` for full parameter documentation.
    """
    d = util.load_vis_dataset(dataset_name, sample_name=sample_name)

    """
    __Settings AutoFit__

    Controls output paths and search behaviour. ``unique_tag`` sets the subfolder
    name inside ``output/<sample>/<dataset_name>/`` for this particular fit.
    """
    settings_search = af.SettingsSearch(
        path_prefix=(
            Path(sample_name) / dataset_name
            if sample_name is not None
            else Path(dataset_name)
        ),
        unique_tag="initial_lens_model",
        info={"magzero": d.magzero},
        session=None,
    )

    """
    __Redshifts__

    For a single-plane lens, PyAutoLens units are dimensionless so redshifts do not
    affect the lens model. These are placeholders; photometric redshifts are estimated
    after modeling via SED fitting of the latent-variable fluxes.
    """
    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __Model: MGE Lens + SIE Mass + MGE Source__

    - Lens light:  40 Gaussians (2 sets of 20), 4 non-linear + ~40 linear parameters.
    - Lens mass:   Isothermal ellipsoid + ExternalShear, 5 non-linear parameters.
      Centre fixed to the brightest pixel for this initial fit.
    - Source light: 20 Gaussians, 4 non-linear + ~20 linear parameters.

    Total: ~15 non-linear parameters.  Linear parameters are solved at every
    likelihood evaluation and add negligible sampling cost.
    """
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=d.dataset_centre,
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = d.dataset_centre[0]
    mass.centre.centre_1 = d.dataset_centre[1]

    source_bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                mass=mass,
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge),
        )
    )

    """
    __Analysis__

    ``util.AnalysisImaging`` extends the built-in PyAutoLens analysis to compute
    latent variables (aperture fluxes, magnification) and output RGB visualisations.

    JAX accelerates likelihood evaluation on GPUs; without a GPU it falls back to
    multithreaded CPU execution (slower but still functional).
    """
    analysis = util.AnalysisImaging(
        dataset=d.dataset,
        positions_likelihood_list=d.positions_likelihood_list,
        use_jax=True,
        dataset_main_path=d.dataset_main_path,
        title_prefix="VIS",
        plot_rgb=True,
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        **settings_search.info,
    )

    """
    __Search__

    Nautilus nested sampling.  ``n_live=100`` balances accuracy and speed for this
    15-parameter model.  ``batch_size=50`` controls GPU parallelism.
    ``n_like_max`` stops runaway fits (most complete well under 100 000 evaluations).
    """
    search = af.Nautilus(
        name="vis",
        **settings_search.search_dict,
        n_live=100,
        batch_size=50,
        iterations_per_quick_update=iterations_per_quick_update,
        n_like_max=100000,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


if __name__ == "__main__":
    sample_name, dataset_name, iterations_per_quick_update = util.parse_fit_args()
    fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
    )
