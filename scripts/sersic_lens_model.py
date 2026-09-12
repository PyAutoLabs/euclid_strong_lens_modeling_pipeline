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

``initial_lens_model.fit`` is called with ``stage="vis_lp"``, so this script chains
off the light-profile search ``vis_lp`` rather than the pixelized search
``vis_pix``. This is not an optimisation. The source Sersic's centre priors are
read from ``galaxies.source.bulge``, and the pixelized stage replaces the source
bulge with a ``Pixelization`` — there is no bulge left in that result to seed a
prior from. A pixelized source reconstruction is also not a Sersic and cannot be
converted into one.

__Variants__

``--variant`` runs one of four alternative fits, added to find out why the
lens-light Sersic index piles up at the ``n = 5`` prior edge in the DR1 catalogue.
They are meant to be run over the same lenses and compared:

- ``baseline`` — the unmodified model, so the others have a like-for-like
  comparison on the same data.
- ``wide_n`` — the lens Sersic index prior widened to ``Uniform(0.5, 10.0)``, past
  the config edge. The source Sersic is untouched. Separates "the prior stopped it"
  from "the data want a high ``n``".
- ``central_noise`` — the noise map multiplied by a Gaussian bowl at the lens light
  centre (``1 + 9 exp(-r^2 / 2 * 0.17"^2)``). The data and the model are identical
  to ``baseline``; only the weight of the central pixels changes. Answers whether a
  handful of pixels at the centre are driving the index.
- ``sersic_point`` — the lens galaxy gains a compact MGE ``point`` component (five
  linear Gaussians, shared free centre), so an unresolved nucleus has somewhere to
  go other than the Sersic's cusp. Sixteen parameters rather than twelve.

Each writes to ``sersic_lens_model_<variant>/vis``, beside the others under the
same dataset. **Without ``--variant`` nothing changes**: the fit, the model and the
output path are exactly what they were before variants existed.

__Running It__

Run as a script, it does the ``vis_lp`` fit and then the Sersic fit::

    python scripts/sersic_lens_model.py --sample=<sample> --dataset=<name>
    python scripts/sersic_lens_model.py --dataset=<name> --variant=wide_n

Results are written to ``sersic_lens_model/vis`` (or
``sersic_lens_model_<variant>/vis``) inside the dataset's output folder. The
multi-waveband follow-on is deliberately not run from here; use
``scripts/sersic_lens_model_waveband.py``, the SED chain driver, which runs the
same two stages and then ``fit_waveband`` over every remaining band.

Of the shared pipeline arguments, ``--number_of_cores`` and ``--use_cpu`` are
forwarded to the upstream ``vis_lp`` fit; the Sersic search itself always runs on
JAX. ``--stage`` is ignored: it is forced to ``"vis_lp"`` regardless, for the reason
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


# The `central_noise` variant's single setting: the noise map is multiplied by
# `1 + 9 exp(-r^2 / 2 * 0.17"^2)` at the lens light centre, so the central pixels
# enter the likelihood at a tenth of their weight and the Sersic cusp is no longer
# pinned by them. One setting, not a ladder — a ladder was considered and dropped.
CENTRAL_NOISE_AMPLITUDE = 9.0
CENTRAL_NOISE_SIGMA_ARCSEC = 0.17

# The `wide_n` variant's lens Sersic index prior. The config prior is
# `Uniform(0.8, 5.0)` (`config/priors/light/linear/sersic.yaml`), and 5.0 is the
# edge the June catalogue piles up against; 10.0 is past any physical value, so a
# fit that still lands at 5 is telling us about the data rather than the prior.
WIDE_N_LOWER_LIMIT = 0.5
WIDE_N_UPPER_LIMIT = 10.0

# The `sersic_point` variant's nucleus: five linear Gaussians with a shared free
# centre and shared ell_comps, sigma log-spaced from 0.01" to twice the pixel
# scale. Four free parameters, so the model goes from twelve to sixteen.
POINT_TOTAL_GAUSSIANS = 5
POINT_SIGMA_MIN = 0.01


def unique_tag_from(variant: str = None) -> str:
    """
    The ``unique_tag`` a Sersic fit writes under.

    Without a variant this is ``"sersic_lens_model"``, unchanged from before
    variants existed — that is the whole contract: an existing run, an existing
    result directory and every other project that calls `fit_sersic` are untouched.
    A variant appends its name, so the variants sit beside each other under one
    dataset and ``catalogue/scripts/lens_sersic.py --unique_tag`` scrapes each.

    Parameters
    ----------
    variant
        One of ``util.VARIANTS``, or ``None``.
    """
    if variant is None:
        return "sersic_lens_model"

    if variant not in util.VARIANTS:
        raise ValueError(
            f"unique_tag_from: unknown variant {variant!r}; "
            f"expected one of {util.VARIANTS} or None."
        )

    return f"sersic_lens_model_{variant}"


def sersic_model_from(
    lens_centre,
    source_centre,
    mass,
    shear,
    variant: str = None,
    pixel_scales: float = None,
    point_centre=None,
) -> "af.Collection":
    """
    Compose the Sersic model: a linear ``Sersic`` for the lens and one for the
    source, with the lens mass and external shear fixed.

    This is the model `fit_sersic` hands to its search, split out of it so that the
    four variants can be built and their priors inspected without loading a dataset
    or running a search — the same split `scripts/initial_lens_model.py` makes with
    ``vis_lp_model_from``. ``tests/test_sersic_variants.py`` is the caller that does
    that.

    - Lens light: ``lp_linear.Sersic`` [6 free parameters — centre, elliptical
      components, effective radius, Sersic index].
    - Source light: ``lp_linear.Sersic`` [6 free parameters].
    - Lens mass: ``Isothermal`` + ``ExternalShear``, both passed as instances of the
      ``vis_lp`` result [0 free parameters].

    Twelve non-linear parameters in total (sixteen for ``sersic_point``). Each
    profile's ``intensity`` is absent from that count by design: ``lp_linear``
    profiles solve intensity by linear algebra at every likelihood evaluation, so
    brightness costs the sampler nothing and is never a prior that can be got wrong.

    The redshifts below are the same dimensionless placeholders the initial fit
    uses: for a single-plane lens they do not affect the model.

    Parameters
    ----------
    lens_centre
        The lens light's ``(centre_0, centre_1)`` priors, taken from
        ``vis_result.model_centred``.
    source_centre
        The source light's ``(centre_0, centre_1)`` priors, from the same place.
    mass
        The lens mass, an instance of the ``vis_lp`` result — not a model.
    shear
        The external shear, likewise an instance.
    variant
        One of ``util.VARIANTS``, or ``None`` for the unmodified model. ``baseline``
        and ``central_noise`` also return the unmodified model: ``baseline`` is the
        re-run of it on the new data, and ``central_noise`` changes the *noise map*
        rather than the model.
    pixel_scales
        The dataset's pixel scale in arcseconds. Required by ``sersic_point`` only,
        which sizes its Gaussians against it.
    point_centre
        The ``(y, x)`` centre the point component's shared centre prior is placed
        on, ±0.1". Required by ``sersic_point`` only.

    Returns
    -------
    af.Collection
        The lens and source galaxies, ready to fit.
    """
    import autofit as af
    import autolens as al

    if variant is not None and variant not in util.VARIANTS:
        raise ValueError(
            f"sersic_model_from: unknown variant {variant!r}; "
            f"expected one of {util.VARIANTS} or None."
        )

    lens_bulge = af.Model(al.lp_linear.Sersic)
    lens_bulge.centre.centre_0 = lens_centre[0]
    lens_bulge.centre.centre_1 = lens_centre[1]

    if variant == "wide_n":
        lens_bulge.sersic_index = af.UniformPrior(
            lower_limit=WIDE_N_LOWER_LIMIT, upper_limit=WIDE_N_UPPER_LIMIT
        )

    source_bulge = af.Model(al.lp_linear.Sersic)
    source_bulge.centre.centre_0 = source_centre[0]
    source_bulge.centre.centre_1 = source_centre[1]

    lens_kwargs = {}

    if variant == "sersic_point":
        if pixel_scales is None or point_centre is None:
            raise ValueError(
                "sersic_model_from: variant 'sersic_point' needs both "
                "`pixel_scales` and `point_centre`."
            )
        lens_kwargs["point"] = al.model_util.mge_point_model_from(
            pixel_scales=pixel_scales,
            total_gaussians=POINT_TOTAL_GAUSSIANS,
            centre=point_centre,
            sigma_min=POINT_SIGMA_MIN,
        )

    return af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                bulge=lens_bulge,
                mass=mass,
                shear=shear,
                **lens_kwargs,
            ),
            source=af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge),
        )
    )


def fit_sersic(
    dataset_name: str,
    vis_result,
    sample_name: str = None,
    iterations_per_quick_update: int = 5000,
    variant: str = None,
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
    if variant == "central_noise":
        # All the MGE Gaussians of the `vis_lp` lens bulge share one centre, so the
        # first profile's carries it. This is the lens light centre the fit found,
        # not the brightest pixel: the bowl has to sit where the Sersic cusp is.
        noise_inflation = {
            "centre": vis_result.instance.galaxies.lens.bulge.profile_list[0].centre,
            "amplitude": CENTRAL_NOISE_AMPLITUDE,
            "sigma_arcsec": CENTRAL_NOISE_SIGMA_ARCSEC,
        }
    else:
        noise_inflation = None

    d = util.load_vis_dataset(
        dataset_name, sample_name=sample_name, noise_inflation=noise_inflation
    )

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
        sub_size_list=[16, 4, 2],
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

    A variant appends its name, so the four sit beside each other under one dataset
    in ``sersic_lens_model_<variant>/vis/`` and
    ``catalogue/scripts/lens_sersic.py --unique_tag`` scrapes each of them with no
    change. Without ``--variant`` the tag is exactly what it has always been, so
    every existing result and every other project is untouched.
    """
    unique_tag = unique_tag_from(variant)

    settings_search = af.SettingsSearch(
        path_prefix=(
            Path(sample_name) / dataset_name
            if sample_name is not None
            else Path(dataset_name)
        ),
        unique_tag=unique_tag,
        info={"magzero": d.magzero},
        session=None,
    )

    """
    __Model__

    Composed by `sersic_model_from`, which is this block split out so the four
    variants' models can be built and inspected without a dataset or a search.
    The centre priors come from ``vis_result.model_centred`` — the ``vis_lp`` model
    with its priors re-centred on that fit's maximum likelihood values — so each
    Sersic starts on the position the MGE already found for that galaxy while
    staying free to move.
    """
    lens_model_centred = vis_result.model_centred.galaxies.lens.bulge.profile_list[0]
    source_model_centred = vis_result.model_centred.galaxies.source.bulge.profile_list[
        0
    ]

    model = sersic_model_from(
        lens_centre=(
            lens_model_centred.centre.centre_0,
            lens_model_centred.centre.centre_1,
        ),
        source_centre=(
            source_model_centred.centre.centre_0,
            source_model_centred.centre.centre_1,
        ),
        mass=vis_result.instance.galaxies.lens.mass,
        shear=vis_result.instance.galaxies.lens.shear,
        variant=variant,
        pixel_scales=d.dataset.pixel_scales[0],
        point_centre=d.dataset_centre,
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
        stage,
        variant,
    ) = util.parse_fit_args(with_variant=True)

    # Bypass vis_pix — the Sersic fit only needs vis_lp (which has the MGE
    # source.bulge, SIE mass and shear). vis_pix replaces source.bulge with a
    # pixelization, so its instance cannot seed the Sersic source prior.
    vis_lp_result = fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
        number_of_cores=number_of_cores,
        use_cpu=use_cpu,
        stage="vis_lp",
    )

    sersic_result = fit_sersic(
        dataset_name=dataset_name,
        vis_result=vis_lp_result,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
        variant=variant,
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
