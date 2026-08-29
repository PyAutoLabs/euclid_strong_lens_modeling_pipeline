"""
Sersic Lens Model Pipeline
===========================

Fits a linear Sersic bulge to both the lens and source galaxy, with the mass
model fixed to the result of ``scripts/initial_lens_model.py`` (the initial MGE fit).

``fit_sersic()``   — VIS band, Sersic light profiles, fixed mass from ``initial_lens_model``.
``fit_waveband()`` — all non-VIS bands with the Sersic result fixed.

The Sersic profile diverges at the galaxy centre, so this pipeline uses a
higher-order over-sampling scheme derived from the tracer of the prior fit.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import util
from scripts.initial_lens_model import fit
from scripts.lens_model_waveband import fit_waveband


def fit_sersic(
    dataset_name: str,
    vis_result,
    sample_name: str = None,
    iterations_per_quick_update: int = 5000,
):
    from autolens import conf

    project_root = Path(__file__).parent.parent
    conf.instance.push(
        new_path=project_root / "config",
        output_path=project_root / os.environ.get("PYAUTO_OUTPUT_DIR", "output"),
    )

    import numpy as np
    import autofit as af
    import autolens as al

    d = util.load_vis_dataset(dataset_name, sample_name=sample_name)

    """
    __Over Sampling (Sersic)__

    The Sersic profile diverges at its centre; we replace the standard over-
    sampling with a tracer-derived scheme that concentrates sub-pixels near both
    the lens and source centres.
    """
    tracer = vis_result.max_log_likelihood_tracer

    traced_grid = tracer.traced_grid_2d_list_from(grid=d.dataset.grid)[-1]
    source_centre = tracer.galaxies[1].bulge.profile_list[0].centre

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=traced_grid,
        sub_size_list=[16, 4, 2],
        radial_list=[0.1, 0.3],
        centre_list=[source_centre],
    )
    over_sample_size_lens = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=d.dataset.grid,
        sub_size_list=[16, 4, 1],
        radial_list=[0.1, 0.3],
        centre_list=[d.dataset_centre],
    )
    over_sample_size = np.where(
        over_sample_size > over_sample_size_lens,
        over_sample_size,
        over_sample_size_lens,
    )
    over_sample_size = al.Array2D(values=over_sample_size, mask=d.dataset.mask)
    dataset = d.dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    settings_search = af.SettingsSearch(
        path_prefix=(
            Path(sample_name) / dataset_name
            if sample_name is not None
            else Path(dataset_name)
        ),
        unique_tag="sersic_lens_model",
        info={"magzero": d.magzero},
        session=None,
    )

    """
    __Model__

    Linear Sersic for lens and source; mass and shear fixed from ``initial_lens_model``.
    Centre priors seeded from the MGE result.
    """
    lens_bulge = af.Model(al.lp_linear.Sersic)
    lens_bulge.centre.centre_0 = (
        vis_result.model_centred.galaxies.lens.bulge.profile_list[0].centre.centre_0
    )
    lens_bulge.centre.centre_1 = (
        vis_result.model_centred.galaxies.lens.bulge.profile_list[0].centre.centre_1
    )

    source_bulge = af.Model(al.lp_linear.Sersic)
    source_bulge.centre.centre_0 = (
        vis_result.model_centred.galaxies.source.bulge.profile_list[0].centre.centre_0
    )
    source_bulge.centre.centre_1 = (
        vis_result.model_centred.galaxies.source.bulge.profile_list[0].centre.centre_1
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                bulge=lens_bulge,
                mass=vis_result.instance.galaxies.lens.mass,
                shear=vis_result.instance.galaxies.lens.shear,
            ),
            source=af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge),
        )
    )

    analysis = util.AnalysisImaging(
        dataset=dataset,
        use_jax=True,
        title_prefix="VIS",
        dataset_main_path=d.dataset_main_path,
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        **settings_search.info,
    )

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
    (
        sample_name,
        dataset_name,
        iterations_per_quick_update,
        number_of_cores,
        use_cpu,
        skip_pix,
    ) = util.parse_fit_args()

    # Bypass vis_pix — the Sersic fit only needs vis_lp (which has the MGE
    # source.bulge, SIE mass and shear). vis_pix replaces source.bulge with a
    # pixelization, so its instance cannot seed the Sersic source prior.
    vis_lp_result = fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
        number_of_cores=number_of_cores,
        use_cpu=use_cpu,
        skip_pix=True,
    )

    sersic_result = fit_sersic(
        dataset_name=dataset_name,
        vis_result=vis_lp_result,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    # Multi-waveband follow-on intentionally disabled — this run extends vis_lp
    # with a Sersic fit only. Run ``scripts/sersic_lens_model_waveband.py`` to
    # chain the non-VIS bands on afterwards, or re-enable the call below.
    # fit_waveband(
    #     dataset_name=dataset_name,
    #     unique_tag="sersic_lens_model",
    #     vis_result=sersic_result,
    #     use_sersic_over_sampling=True,
    #     sample_name=sample_name,
    #     iterations_per_quick_update=iterations_per_quick_update,
    # )
