"""
Sersic Lens Model Pipeline
===========================

__What This Script Is For__

``fit_sersic`` refits a VIS dataset that has already been modeled by
``scripts/initial_lens_model.py``, replacing both galaxies' light with a single
linear ``Sersic`` profile each and holding the mass model fixed at the values that
fit found.

The initial fit describes each galaxy with a Multi Gaussian Expansion: 40 Gaussians
for the lens, 20 for the source. That is the right tool for *lens modeling* — it is
flexible, its intensities solve linearly, and it keeps the non-linear parameter
space small. It is not what a photometry catalogue wants. A catalogue row needs
standard structural parameters for each galaxy — a centre, an ellipticity, an
effective radius and a Sersic index — and a basis of Gaussians has none of those
as parameters. So this script fits the profile that does.

Those six parameters per galaxy are exactly what
``catalogue/scripts/lens_sersic.py`` and ``catalogue/scripts/source_sersic.py``
scrape out of these results into ``lens_sersic.csv`` and ``source_sersic.csv``.
Intensity is not among them: an ``lp_linear.Sersic`` has its intensity solved by
linear algebra at every likelihood evaluation, so it never enters the non-linear
samples. That is also what makes the profile portable across bands —
``scripts/lens_model_waveband.py`` fixes this Sersic's *shape* and re-solves its
intensity against each band's own data, which is where the matched multi-band
photometry for SED fitting and photometric redshifts comes from.

__Why The Mass Model Is Fixed__

The lens mass and external shear enter the model as instances of the ``vis_lp``
result, not as free components. The geometry was already solved, on the same VIS
data, by a fit built for the job; freeing it again would spend this search's
budget re-deriving a known answer, and would leave the light parameters this
script exists to measure marginalised over a mass uncertainty the initial fit had
already resolved. With the mass fixed, the whole twelve-parameter space is light.

__Why It Chains Off vis_lp, Not vis_pix__

``initial_lens_model.fit`` is called with ``skip_pix=True``, so this script chains
off the light-profile search ``vis_lp`` rather than the pixelized search
``vis_pix``. This is not an optimisation. The source Sersic's centre priors are
read from ``galaxies.source.bulge``, and the pixelized stage replaces the source
bulge with a ``Pixelization`` — there is no bulge left in that result to seed a
prior from. A pixelized source reconstruction is also not a Sersic and cannot be
converted into one.

__Running It__

Run as a script, it does the ``vis_lp`` fit and then the Sersic fit::

    python scripts/sersic_lens_model.py --sample=<sample> --dataset=<name>

Results are written to ``sersic_lens_model/vis`` inside the dataset's output
folder. The multi-waveband follow-on is deliberately not run from here; use
``scripts/sersic_lens_model_waveband.py``, the SED chain driver, which runs the
same two stages and then ``fit_waveband`` over every remaining band.

Of the shared pipeline arguments, ``--number_of_cores`` and ``--use_cpu`` are
forwarded to the upstream ``vis_lp`` fit; the Sersic search itself always runs on
JAX. ``--skip_pix`` is ignored: it is forced to ``True`` regardless, for the reason
above.

New to the pipeline? Read ``start_here.py`` in the repository root first: it covers
installation, the dataset contract, masking and over-sampling, the MGE, Nautilus
and JAX, and how to read ``output/``. This script assumes all of it and explains
only what is specific to the Sersic fit.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import util
from scripts.initial_lens_model import fit


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

    """
    __Dataset__

    The same VIS dataset the ``vis_lp`` fit was run on, loaded the same way:
    ``util.load_vis_dataset`` handles the FITS layout, noise scaling, mask, PSF, WCS
    and zero-point in one call. Its steps are documented individually in
    ``scripts/initial_lens_model.py``, and the dataset contract it reads is described
    in ``start_here.py``.
    """
    d = util.load_vis_dataset(dataset_name, sample_name=sample_name)

    """
    __Over Sampling (Sersic)__

    The standard over-sampling applied by ``load_vis_dataset`` (4x4 sub-pixels within
    0.1" of the lens centre, 2x2 out to 0.3", 1x1 beyond) is replaced here for two
    reasons. A Sersic profile diverges at its centre, so 4x4 is not fine enough there:
    the flux in the central pixels comes out wrong, and that error propagates straight
    into the effective radius and Sersic index this script exists to measure. And the
    standard scheme is built around the lens centre alone — it knows nothing about
    where the lensed source's light falls.

    Two maps are built and the larger sub-grid is kept at every pixel:

    - The *source* map, computed on the grid traced back to the source plane by the
      ``vis_lp`` tracer, and centred on the source. The source centre is taken from
      the first Gaussian of the MGE source basis, which stands in for the centre of
      the whole basis. Sub-sizes 16 / 4 / 2.
    - The *lens* map, computed on the image-plane grid and centred on the dataset
      centre (the brightest central pixel). Sub-sizes 16 / 4 / 1.

    The source map has to be built on the traced grid because the source's centre is
    a source-plane position: it is the *lensed* image of that centre, wherever the
    arcs land, that needs the fine sub-grid in the image plane.

    ``scripts/lens_model_waveband.py`` applies the identical scheme to the other
    bands when the SED chain passes it ``use_sersic_over_sampling=True``.
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

    """
    __Settings AutoFit__

    ``unique_tag="sersic_lens_model"`` keeps this fit beside, rather than inside, the
    ``initial_lens_model`` results of the same dataset. With the search named ``vis``
    below, results land in
    ``output/<sample>/<dataset>/sersic_lens_model/vis/`` — the path the catalogue
    Sersic scrapers look for.
    """
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

    A linear ``Sersic`` for the lens and a linear ``Sersic`` for the source, with the
    lens mass and external shear fixed:

     - Lens light: ``lp_linear.Sersic`` [6 free parameters — centre, elliptical
       components, effective radius, Sersic index].
     - Source light: ``lp_linear.Sersic`` [6 free parameters].
     - Lens mass: ``Isothermal`` + ``ExternalShear``, both passed as instances of the
       ``vis_lp`` result [0 free parameters].

    Twelve non-linear parameters in total. Each profile's ``intensity`` is absent from
    that count by design: ``lp_linear`` profiles solve intensity by linear algebra at
    every likelihood evaluation, so brightness costs the sampler nothing and is never
    a prior that can be got wrong.

    The two centres are not fixed, but neither do they start from the broad config
    default. ``vis_result.model_centred`` is the ``vis_lp`` model with its priors
    re-centred on that fit's maximum likelihood values, so taking ``centre_0`` and
    ``centre_1`` from it anchors each Sersic on the position the MGE already found
    for that galaxy — the lens's on the image-plane centre, the source's on its
    source-plane centre — while leaving them free to move.

    This is also the reason a pixelized result cannot be used here: the source's two
    centre priors are read from ``galaxies.source.bulge``, which the ``vis_pix``
    search replaces with a ``Pixelization``.

    The redshifts below are the same dimensionless placeholders the initial fit uses:
    for a single-plane lens they do not affect the model.
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

    """
    __Analysis & Search__

    No positions likelihood is passed, unlike the fits in
    ``scripts/initial_lens_model.py``. Positions exist to reject mass models that
    demagnify the source into an unphysical reconstruction, and there is no mass
    model to reject here — it is fixed.

    Nautilus settles this twelve-parameter space with a much smaller live-point set
    than the ``vis_lp`` fit needs (``n_live=100`` against 750), and ``n_like_max``
    caps a runaway fit. ``batch_size`` controls how many models are evaluated
    simultaneously on the GPU; JAX is always on for this search.
    """
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
