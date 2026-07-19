"""
MGE Lens-Only Pipeline
======================

Fits a Multi Gaussian Expansion (MGE) to the lens light only — no mass model,
no source.  Provides a clean lens light subtraction for visual inspection of the
lensed arcs and as an input to downstream pipelines.

``fit()``          — VIS band only.
``fit_waveband()`` — all non-VIS bands, with the lens model fixed to the VIS result.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import util


def fit(
    dataset_name: str,
    sample_name: str = None,
    iterations_per_quick_update: int = 50000,
):
    from autolens import conf

    project_root = Path(__file__).parent.parent
    conf.instance.push(
        new_path=project_root / "config", output_path=project_root / "output"
    )

    import autofit as af
    import autolens as al

    d = util.load_vis_dataset(dataset_name, sample_name=sample_name)

    settings_search = af.SettingsSearch(
        path_prefix=(
            Path(sample_name) / dataset_name
            if sample_name is not None
            else Path(dataset_name)
        ),
        unique_tag="mge_lens_only",
        info={"magzero": d.magzero},
        session=None,
    )

    analysis = util.AnalysisImaging(
        dataset=d.dataset,
        use_jax=True,
        title_prefix="VIS",
        dataset_main_path=d.dataset_main_path,
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        **settings_search.info,
    )

    bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=d.dataset_centre,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, bulge=bulge),
        ),
    )

    search = af.Nautilus(
        name="vis",
        **settings_search.search_dict,
        n_live=75,
        batch_size=50,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def fit_waveband(
    dataset_name: str,
    vis_result,
    sample_name: str = None,
    iterations_per_quick_update: int = 50000,
):
    """
    Fit all non-VIS wavebands in the dataset, with the lens model fixed to the
    VIS result.  A sub-pixel astrometric offset (DatasetModel) is fitted for
    each secondary band.
    """
    import autofit as af
    import autolens as al
    from autolens import conf
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    conf.instance.push(
        new_path=project_root / "config", output_path=project_root / "output"
    )

    conf.instance["visualize"]["general"]["units"][
        "cb_unit"
    ] = r"$\,\,\mathrm{e^{-}}\,\mathrm{s^{-1}}$"

    # Re-use the dataset index dict from any waveband load; we only need the
    # path and HDU mapping here so load without full over-sampling overhead.
    project_root = Path(__file__).parent.parent
    if sample_name is not None:
        dataset_main_path = project_root / "dataset" / sample_name / dataset_name
    else:
        dataset_main_path = project_root / "dataset" / dataset_name
    dataset_fits_name = f"{dataset_name}.fits"

    dataset_index_dict = util.dataset_instrument_hdu_dict_via_fits_from(
        dataset_path=dataset_main_path,
        dataset_fits_name=dataset_fits_name,
        image_tag="_BGSUB",
    )

    import json

    try:
        with open(dataset_main_path / "info.json") as f:
            info = json.load(f)
    except FileNotFoundError:
        info = {}

    mask_radius = info["mask_radius"]

    # Load lowest-resolution PSF once — it is the same for all bands.
    header_primary = al.header_obj_from(
        file_path=dataset_main_path / dataset_fits_name, hdu=0
    )
    lowest_resolution_waveband = header_primary.get("WORST_BAND", None).lower()
    lowest_resolution_waveband_index = dataset_index_dict.get(
        lowest_resolution_waveband, None
    )
    psf_lowest_resolution = al.Convolver.from_fits(
        file_path=dataset_main_path / dataset_fits_name,
        hdu=lowest_resolution_waveband_index * 3 + 2,
        pixel_scales=0.1,
        normalize=True,
    )
    psf_lowest_resolution_fwhm = float(header_primary.get("WORST_PSF_MER", None))
    if psf_lowest_resolution_fwhm is None or psf_lowest_resolution_fwhm < -98:
        psf_lowest_resolution_fwhm = float(header_primary.get("WORST_PSF_HDR", None))

    for dataset_waveband, dataset_index in dataset_index_dict.items():
        if dataset_waveband == "vis":
            continue

        dataset = al.Imaging.from_fits(
            data_path=dataset_main_path / dataset_fits_name,
            data_hdu=dataset_index * 3 + 1,
            noise_map_path=dataset_main_path / dataset_fits_name,
            noise_map_hdu=dataset_index * 3 + 3,
            psf_path=dataset_main_path / dataset_fits_name,
            psf_hdu=dataset_index * 3 + 2,
            pixel_scales=0.1,
            check_noise_map=False,
        )

        dataset_centre = dataset.data.brightest_sub_pixel_coordinate_in_region_from(
            region=(-0.3, 0.3, -0.3, 0.3), box_size=2
        )

        try:
            header = al.header_obj_from(
                file_path=dataset_main_path / dataset_fits_name,
                hdu=dataset_index * 3 + 1,
            )
            magzero = header["MAGZERO"]
        except FileNotFoundError:
            header = None
            magzero = None

        from astropy.wcs import WCS

        pixel_wcs = WCS(header).celestial if header is not None else None

        try:
            mask_extra_galaxies = al.Mask2D.from_fits(
                file_path=dataset_main_path / "mask_extra_galaxies.fits",
                pixel_scales=0.1,
                invert=True,
            )
            dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
        except FileNotFoundError:
            pass

        mask_centre = info.get("mask_centre") or (0.0, 0.0)
        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=mask_radius,
            centre=mask_centre,
        )
        dataset = dataset.apply_mask(mask=mask)

        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=dataset.grid,
            sub_size_list=[4, 2, 1],
            radial_list=[0.1, 0.3],
            centre_list=[dataset_centre],
        )
        dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        settings_search = af.SettingsSearch(
            path_prefix=(
                Path(sample_name) / dataset_name
                if sample_name is not None
                else Path(dataset_name)
            ),
            unique_tag="mge_lens_only",
            info={"magzero": magzero},
            session=None,
        )

        analysis = util.AnalysisImaging(
            dataset=dataset,
            use_jax=True,
            title_prefix=dataset_waveband.upper(),
            dataset_main_path=dataset_main_path,
            skip_rgb_plot=True,
            psf_lowest_resolution=psf_lowest_resolution,
            psf_lowest_resolution_fwhm=psf_lowest_resolution_fwhm,
            pixel_wcs=pixel_wcs,
            **settings_search.info,
        )

        dataset_model = af.Model(al.DatasetModel)
        dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
            lower_limit=-0.2, upper_limit=0.2
        )
        dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
            lower_limit=-0.2, upper_limit=0.2
        )

        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=vis_result.instance.galaxies.lens.redshift,
                    bulge=vis_result.instance.galaxies.lens.bulge,
                ),
            ),
            dataset_model=dataset_model,
        )

        search = af.Nautilus(
            name=dataset_waveband,
            **settings_search.search_dict,
            n_live=75,
            batch_size=50,
            iterations_per_quick_update=iterations_per_quick_update,
        )

        search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


if __name__ == "__main__":
    sample_name, dataset_name, iterations_per_quick_update = util.parse_fit_args()

    vis_result = fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    fit_waveband(
        dataset_name=dataset_name,
        vis_result=vis_result,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
    )
