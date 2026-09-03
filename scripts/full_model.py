"""
Full SLaM Pipeline
==================

Runs the complete Source, Light and Mass (SLaM) pipeline on a Euclid VIS dataset::

    python scripts/full_model.py --dataset=<name> --sample=<sample>

__Before You Start__

If you are new to this pipeline, read ``start_here.py`` first. Installation and the
command line, the dataset contract, masking and over-sampling, what a Multi Gaussian
Expansion (MGE) is, the SIE + shear initial mass model, the Nautilus / JAX / GPU
basics, the idea of a pixelized source and the layout of ``output/`` are all covered
there, and are deliberately *not* repeated here.

For a line-by-line walkthrough of every step of a fit, read
``scripts/initial_lens_model.py``: it fits one light-profile search (``vis_lp``)
followed by one pixelized search (``vis_pix``) with full inline documentation. This
script is that same idea generalised into five chained searches.

This script accepts the shared pipeline arguments (``util.parse_fit_args``), but its
``fit`` function uses only ``--dataset``, ``--sample`` and
``--iterations_per_quick_update``. Every stage here runs on JAX and every stage is
run, so ``--use_cpu``, ``--number_of_cores`` and ``--stage`` have no effect.

__SLaM (Source, Light and Mass)__

This script is an introduction to the Source, (lens) Light and Mass (SLaM) pipelines.
These are advanced modeling pipelines which use many aspects of core PyAutoLens
functionality to automate the modeling of strong lenses.

The idea is simple: rather than throwing a complex model at the data in one go and
hoping the sampler finds the global maximum, each search solves one part of the
problem with everything else simplified or held fixed, and hands its result to the
next search as priors or as a fixed instance.

__Prerequisites__

Before reading this script, it helps to be familiar with the following key concepts.
The references are scripts in the ``autolens_workspace`` repository
(https://github.com/PyAutoLabs/autolens_workspace):

- **Non-linear search chaining** (``scripts/guides/modeling/chaining.py``): linking
  models together in a sequence, such as transitioning from a light profile source to
  a pixelized source. This is the mechanism the whole script is built on.

- **Pixelizations** (``scripts/imaging/features/pixelization/modeling.py``, and
  ``.../pixelization/delaunay.py`` for the mesh family used here): structures which
  reconstruct the source galaxy on a pixel grid rather than with an analytic profile.

- **Adaptive pixelizations** (``scripts/imaging/features/pixelization/adaptive.py``):
  pixelizations whose source pixels and regularization weights adapt to the unlensed
  morphology of the source.

- **Multi Gaussian Expansions (MGE)**
  (``scripts/imaging/features/multi_gaussian_expansion/modeling.py``): galaxy light
  modeled as a sum of Gaussians. Used here for the lens light throughout, and for the
  source in the first search before the pixelization takes over. ``start_here.py``
  gives the short version.

- The generic PyAutoLens introduction to these pipelines is
  ``scripts/guides/modeling/slam_start_here.py``. Read it for the concepts, not the
  settings: its worked example uses a different mesh family to the Delaunay meshes
  used here.

If any of these concepts are unfamiliar you can still run the script, but reviewing
the referenced examples later will deepen your understanding of how and why SLaM
pipelines are structured as they are.

__Overview__

The SLaM pipelines strategically chain together sequential searches, each designed to
exploit the results of the last. This provides a fully automated framework for fitting
large samples of strong lenses with complex models.

Each stage targets a specific aspect of the strong lens model:

- **Source**: establish a robust source model. For a pixelized source this means
  finding accurate mesh and regularization parameters, and building the "adapt image"
  the mesh adapts to.
- **Light**: model the lens light, with the source and mass models fixed from the
  source stages.
- **Mass**: fit a detailed, higher-complexity mass model, using the source and lens
  light models established earlier.

Models set up in earlier stages guide those used in later ones. The Delaunay mesh and
``AdaptSplit`` regularization chosen for the source stages are what the final mass
measurement is made against.

__Pipeline Structure__

This script implements five searches, each as a function below with its own
documentation block, called in order by ``fit`` at the bottom of the file. All five
are written to the ``slam`` unique tag of the dataset's output folder:

1. **SOURCE LP** (``source_lp[1]``) — MGE lens light (2 x 20 Gaussians),
   ``Isothermal`` + ``ExternalShear`` mass and an MGE source (1 x 20 Gaussians), all
   free. Fast, well-conditioned, and the origin of everything that follows: the mass
   priors, the multiple-image positions and the first adapt image.

2. **SOURCE PIX 1** (``source_pix[1]``) — the source becomes a ``Delaunay``
   pixelization on a uniform ``Overlay`` image-plane grid, with the lens light fixed
   to SOURCE LP and the mass and shear chained forward as models. Its purpose is not
   the mass model but the reconstruction itself, which becomes the adapt image for the
   next stage.

3. **SOURCE PIX 2** (``source_pix[2]``) — the same Delaunay mesh, but the image-plane
   grid is now drawn by a ``Hilbert`` mesh weighted by that adapt image. The lens
   light, mass and shear are all fixed instances, so the only free parameters are
   those of the ``AdaptSplit`` regularization. This is the source model used by every
   later stage.

4. **LIGHT LP** (``light[1]``) — the lens light is refit from scratch (2 x 20
   Gaussians) against a source and mass that are now accurate enough for a clean
   lens-light subtraction.

5. **MASS TOTAL** (``mass_total[1]``) — the ``Isothermal`` mass is promoted to a
   ``PowerLaw``, with priors initialized from SOURCE PIX 1, the lens light fixed from
   LIGHT LP and the source fixed from SOURCE PIX 2.

__Design Choices__

There are many design choices that go into the SLaM pipelines, which we discuss now.

The SLaM pipelines are designed around pixelized source modeling. Pixelized sources
are necessary for fitting complex mass models, which the SLaM pipelines automate the
fitting of. The design considerations below all follow from that:

- **Source First**: the pipeline starts with the source, because complex mass models
  (e.g. a ``PowerLaw``, or composite models with stars and dark matter) require
  pixelized source modeling rather than simple light profiles. This step establishes a
  robust pixelized source using a simpler mass model (``Isothermal`` with
  ``ExternalShear``).

- **Image Positions**: for pixelized source modeling, specifying the positions of the
  multiple images of the lensed source is crucial to prevent unphysical
  reconstructions, where the mass model demagnifies the source into a blob that fits
  the data with no lensing at all. The positions are not input by hand: they are
  estimated automatically from the previous stage's mass and source result.

- **Adapt Images**: an adaptive pixelization uses an "adapt image" — a lens-light
  subtracted image in which only the lensed source emission remains — to place source
  pixels and set regularization weights according to the source's morphology. The
  adapt image is therefore only set once a good model of the source is available.

- **Lens Light Before Mass**: modeling the lens light accurately requires deblending
  the lens and source emission, which a robust pixelized source model makes possible.
  The lens light is refined while the mass model is still the simple ``Isothermal``,
  so the harder mass fit begins from a clean lens-light subtraction.

- **Mass Model Last**: the most complex mass model fitting is saved for last, and
  benefits from the prior refinement of both the source and the lens light.

These design choices enable the SLaM pipelines to deliver precise and automated lens
modeling while optimizing each stage for robustness and efficiency.

__This Script__

Using a SOURCE LP, two SOURCE PIX, a LIGHT LP and a MASS TOTAL search, this script
fits a Euclid VIS ``Imaging`` dataset of a strong lens where in the final model:

 - The lens galaxy's light is a bulge with a Multi Gaussian Expansion (MGE) light
   profile [fixed from LIGHT LP].
 - The lens galaxy's total mass distribution is a ``PowerLaw`` plus an
   ``ExternalShear``.
 - The source galaxy's light is a ``Pixelization``: a ``Delaunay`` mesh with
   ``AdaptSplit`` regularization [fixed from SOURCE PIX 2].
"""

import os
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

Everything is fit with light profiles here, and nothing is held fixed. That is
the point of the stage: the MGE's intensities are solved linearly, so the
non-linear parameter space stays small and well-conditioned and the search is
very likely to find the global maximum. A pixelized source started from nothing
is not — it needs a mass model that is already roughly right, and an adapt image
of the source, neither of which exist until this search has run.

__Settings__:

 - Mass Centre: the ``Isothermal`` centre is given a uniform prior spanning
   +/- 0.05" of the dataset centre (the brightest central pixel), rather than
   being left fully free. The mass centre is expected to lie very close to the
   lens light centre, and restricting it this tightly keeps the initial search
   stable. Later stages chain the mass forward from this result rather than
   setting it up again from scratch.
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
    iterations_per_quick_update: int = 5000,
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
        iterations_per_quick_update=iterations_per_quick_update,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 1__

Delaunay mesh + AdaptSplit regularization.  Creates the adapt image for the
refined pixelisation in SOURCE PIX 2.

This is the first of two pixelized searches, and it exists because of a
chicken-and-egg problem. An adaptive pixelization needs an "adapt image" — a
lens-light subtracted image containing only the lensed source emission — to
decide where to put source pixels and how strongly to regularize them. The
SOURCE LP result can supply one, but not a good one: the true source is often
more complex than the MGE fitted to it, so the emission it predicts is smoother
than the real thing. This search therefore fits a pixelization whose real
product is its own reconstruction, which SOURCE PIX 2 then adapts to.

The lens light is fixed to the SOURCE LP instance; the mass and shear are
chained forward as models, so they are re-fit against the pixelized source.

The Delaunay source pixels are the traced positions of an image-plane grid,
which is built here (uniform ``Overlay`` grid over the mask, plus a ring of
edge points) and handed to the analysis via ``AdaptImages``.  The number of
source pixels is therefore fixed by that grid, not a free parameter: JAX
requires statically shaped arrays.

``reg.AdaptSplit`` — not ``reg.Adapt`` — is mandatory for the Delaunay family:
Delaunay neighbours come from a ``scipy.spatial.Delaunay`` call on the traced
source-plane grid, which cannot be traced under ``jit``/``grad``.

__Positions__

Image positions are computed automatically from the SOURCE LP result to prevent
unphysical source reconstructions — those where the mass model demagnifies the
source so heavily that a featureless blob reproduces the data. They are not
input by hand; deriving them from the previous result is a key part of what
makes these pipelines automatic.

__Adapt Images__

An adapt image is computed from the SOURCE LP result and passed to the analysis,
together with the image-plane mesh grid the Delaunay source pixels are traced
from (both are built in ``fit`` below).

__Mass Chaining__

``unfix_mass_centre=True`` is passed to ``mass_from`` for consistency with the
standard SLaM API. It has nothing to unfix here: the SOURCE LP mass centre was
already free within its +/- 0.05" prior rather than fixed to a value.
"""


def source_pix_1(
    settings_search,
    analysis,
    source_lp_result,
    mesh_init,
    regularization_init,
    n_batch: int = 20,
    iterations_per_quick_update: int = 5000,
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
        iterations_per_quick_update=iterations_per_quick_update,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__SOURCE PIX PIPELINE 2__

Refined pixelisation using the adapt image from SOURCE PIX 1.  The image-plane
grid is now drawn by a ``Hilbert`` image mesh weighted by that adapt image, so
Delaunay source pixels concentrate where the source is bright, and
``reg.AdaptSplit`` adapts the regularization weights to the source's
morphology.

The lens light, mass and shear are all passed as instances — the lens light from
SOURCE LP, the mass and shear from SOURCE PIX 1 — so the only free parameters in
this search are those of the regularization scheme. It is a small, cheap search
(hence its low ``n_live``) whose job is to settle the source reconstruction that
LIGHT LP and MASS TOTAL are then fit against.

Because the mass is fixed here, this result carries no mass posterior worth
chaining. The MASS TOTAL priors are therefore taken from SOURCE PIX 1, the last
search in which the mass was free.
"""


def source_pix_2(
    settings_search,
    analysis,
    source_lp_result,
    source_pix_result_1,
    mesh,
    regularization,
    n_batch: int = 20,
    iterations_per_quick_update: int = 5000,
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
        iterations_per_quick_update=iterations_per_quick_update,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__LIGHT LP PIPELINE__

Refines the lens light (2 x 20 Gaussians) with mass and source fixed:

 - The lens light is a new MGE, fit from scratch [free].
 - The ``Isothermal`` mass and ``ExternalShear`` are fixed from SOURCE PIX 1.
 - The source is fixed from SOURCE PIX 2, passed through by
   ``source_custom_model_from`` with ``source_is_model=False``.

The lens light model is fit from scratch (not seeded from SOURCE LP) because the
earlier mass and source models may not have been precise enough for an accurate
lens-light subtraction. The lens light and the lensed source overlap on the sky,
so a source that is even slightly wrong pushes its residuals into the lens light
model; now that the source is a converged pixelized reconstruction, the
deblending is trustworthy and the light model is worth refitting properly.

This ordering is what makes the final mass fit possible: MASS TOTAL takes this
lens light as a fixed instance, so it never has to fit light and a ``PowerLaw``
at the same time.
"""


def light_lp(
    settings_search,
    analysis,
    source_result_for_lens,
    source_result_for_source,
    lens_bulge,
    lens_disk=None,
    n_batch: int = 30,
    iterations_per_quick_update: int = 5000,
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
        iterations_per_quick_update=iterations_per_quick_update,
    )

    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


"""
__MASS TOTAL PIPELINE__

The final search, fitting the mass model the whole pipeline exists to measure:

 - The lens light is the MGE fixed from LIGHT LP.
 - The mass is a ``PowerLaw`` with an ``ExternalShear``, with priors initialized
   from the ``Isothermal`` fit of SOURCE PIX 1 via ``al.util.chaining.mass_from``
   (the ``PowerLaw`` adds a ``slope``, which the ``Isothermal`` fixed at 2.0).
 - The source is carried through from SOURCE PIX 2 by
   ``al.util.chaining.source_from``, so the mass is measured against the source
   reconstruction those stages settled on.

__Positions__

Positions are computed from the SOURCE PIX 2 result rather than reused from
SOURCE LP. The pixelized source reconstructs the source plane far better than
the MGE did, so the multiple image positions it predicts are more precise, and
the position threshold they impose on this search is correspondingly tighter.

__Shear__

``fit`` calls this stage with ``reset_shear_prior=True``, which replaces the
chained shear with a fresh ``ExternalShear`` model: the search starts from the
broad uniform priors of ``config/priors``, not from the SOURCE PIX posterior.
Chaining a narrow shear prior into this search would be a mistake: the
``PowerLaw`` slope changes how much azimuthal structure the mass profile itself
can produce, so shear values that were the best fit alongside an ``Isothermal``
are not the ones that best fit alongside a ``PowerLaw``. The shear has to be free
to move.
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
    iterations_per_quick_update: int = 5000,
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
        iterations_per_quick_update=iterations_per_quick_update,
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
        new_path=project_root / "config",
        output_path=project_root / os.environ.get("PYAUTO_OUTPUT_DIR", "output"),
    )

    import numpy as np
    import autofit as af
    import autolens as al

    """
    __Dataset__

    ``util.load_vis_dataset`` performs all standard dataset preparation in one
    call: FITS layout, noise scaling, mask, over-sampling, PSF, WCS and zero-point.
    Every step is documented individually in ``scripts/initial_lens_model.py``, and
    the dataset contract it reads — including ``info.json``, which is where the mask
    radius comes from — is described in ``start_here.py``.

    Every search in this script runs under JAX, so the dataset is fitted exactly as
    it is loaded. The Numba sparse operator that ``scripts/initial_lens_model.py``
    applies to its pixelized stage is the CPU route's tool, applied only under
    ``--use_cpu``; under JAX the pixelized inversion uses JAX's own linear algebra
    on the plain dataset.
    """
    d = util.load_vis_dataset(dataset_name, sample_name=sample_name)

    dataset = d.dataset

    """
    __Settings AutoFit__

    Controls output paths and search behaviour. ``unique_tag="slam"`` places all five
    searches of this pipeline together in ``output/<sample>/<dataset>/slam/``, one
    folder per search name (``source_lp[1]``, ``source_pix[1]``, and so on).
    """
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

    """
    __Redshifts__

    For a single-plane lens, PyAutoLens units are dimensionless so the redshifts do
    not affect the lens model. These are placeholders; photometric redshifts are
    estimated after modeling via SED fitting of the latent-variable fluxes.
    """
    redshift_lens = 0.5
    redshift_source = 1.0

    """
    __SOURCE LP PIPELINE__

    Isothermal mass + MGE lens + MGE source, all free.  Provides the initial model
    and adapt image for the pixelised source stages that follow.

    The analysis uses the multiple-image positions that came with the dataset:
    ``util.load_vis_dataset`` reads ``positions.json`` if it is present, and
    otherwise derives positions from the segmentation source flux map, leaving them
    unused if neither is available. Later stages replace them with positions derived
    from the previous search's result.
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
        iterations_per_quick_update=iterations_per_quick_update,
    )

    """
    __SOURCE PIX PIPELINE 1__

    Delaunay mesh + AdaptSplit regularization.  Creates the adapt image for the
    refined pixelisation in SOURCE PIX 2.

    Delaunay source pixels are the traced positions of an image-plane grid, so
    that grid is built here and passed to the analysis inside ``AdaptImages``.
    A uniform ``Overlay(26, 26)`` grid clipped to the circular Euclid mask gives
    the interior pixels; ``append_with_circle_edge_points`` adds a ring of 30
    points just outside the mask edge, whose source pixels are zeroed
    (``zeroed_pixels``) so the reconstruction is not distorted by the mask
    boundary.  ``pixels`` is therefore fixed by the grid rather than a free
    parameter: JAX requires statically shaped arrays.

    The grid is uniform at this stage because the only adapt image available is the
    one predicted by the SOURCE LP MGE source, which is too smooth to be worth
    weighting a mesh by.  SOURCE PIX 2 does that, using this search's own
    reconstruction.

    The positions handed to the analysis are recomputed from the SOURCE LP result
    with ``positions_likelihood_from``. Its threshold is not a hand-chosen number: it
    is set by how closely the positions trace to one another in the source plane
    under the best-fit mass model, multiplied by ``factor=3.0`` so plausible mass
    models are not rejected, and floored at ``minimum_threshold=0.2``.
    """
    mask = dataset.mask
    edge_pixels_total = 30

    image_mesh = al.image_mesh.Overlay(shape=(26, 26))

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=mask)

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=d.mask_radius + mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )
    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    # `load_vis_dataset` only sets `over_sample_size_lp`; the pixelised stages
    # need their own over-sampling, concentrated where the source is bright.
    signal_to_noise_threshold = 3.0
    over_sample_size_pixelization = np.where(
        galaxy_image_name_dict["('galaxies', 'source')"] > signal_to_noise_threshold,
        4,
        2,
    )
    over_sample_size_pixelization = al.Array2D(
        values=over_sample_size_pixelization, mask=mask
    )
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=dataset.grids.lp.over_sample_size,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

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
        mesh_init=al.mesh.Delaunay(
            pixels=image_plane_mesh_grid.shape[0],
            zeroed_pixels=edge_pixels_total,
        ),
        regularization_init=al.reg.AdaptSplit,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    """
    __SOURCE PIX PIPELINE 2__

    Refined pixelisation using the adapt image from SOURCE PIX 1.  The
    image-plane grid is redrawn by a ``Hilbert`` image mesh weighted by that
    adapt image, so Delaunay source pixels concentrate on the bright parts of
    the source and its faintest regions are reconstructed with far fewer pixels.
    ``weight_power=3.5`` sets how sharply the point density follows the adapt
    image, and ``weight_floor=0.01`` keeps the faint outskirts from being left
    with no pixels at all.

    Like the ``Overlay`` grid before it, this grid is computed here, before the
    search, and stays fixed for its duration — which is why the lens light, mass
    and shear are all fixed instances in this search: with the mesh frozen, the
    only thing left to fit is the regularization, and freeing the mass alongside
    it would reintroduce degeneracies that are slow and difficult to sample.

    ``pixels=500`` follows the Euclid sibling ``scripts/initial_lens_model.py``
    rather than the ``autolens_workspace`` ``delaunay.py`` example, which uses
    1000: Euclid VIS cut-outs are small, and 1000 source pixels inflates VRAM
    for no gain here.
    """
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )

    hilbert_pixels = 500

    image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01
    )

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
    )

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=d.mask_radius + mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )

    # LIGHT LP and MASS TOTAL reuse this `adapt_images`, so they inherit the
    # same mesh grid as the source instance they chain from.
    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

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
        mesh=al.mesh.Delaunay(
            pixels=image_plane_mesh_grid.shape[0],
            zeroed_pixels=edge_pixels_total,
        ),
        regularization=al.reg.AdaptSplit,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    """
    __LIGHT LP PIPELINE__

    Refines the lens light (a fresh MGE of 2 x 20 Gaussians) with the mass fixed from
    SOURCE PIX 1 and the source fixed from SOURCE PIX 2.

    The analysis reuses the ``adapt_images`` built for SOURCE PIX 2, so the fixed
    source instance is evaluated on the mesh grid it was fitted with. No positions
    likelihood is needed: the mass model is not being varied here, so there are no
    unphysical mass solutions to guard against.
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
        iterations_per_quick_update=iterations_per_quick_update,
    )

    """
    __MASS TOTAL PIPELINE__

    ``PowerLaw`` + ``ExternalShear`` mass model, with the mass priors seeded from
    SOURCE PIX 1 — the last search in which the mass was free — the lens light fixed
    from LIGHT LP and the source fixed from SOURCE PIX 2.

    The positions likelihood is rebuilt from SOURCE PIX 2, whose pixelized source
    gives a better source-plane reconstruction, and therefore more precise multiple
    image positions, than the SOURCE LP fit used earlier.
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
        iterations_per_quick_update=iterations_per_quick_update,
    )

    return source_lp_result, mass_result


if __name__ == "__main__":
    (
        sample_name,
        dataset_name,
        iterations_per_quick_update,
        number_of_cores,
        use_cpu,
        stage,
    ) = util.parse_fit_args()
    fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
    )
