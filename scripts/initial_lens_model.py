"""
Euclid Pipeline: Initial Lens Model
=====================================

This script fits an initial lens model to a Euclid strong lens dataset. It is the
recommended entry point — run this first before any of the pipelines in ``scripts/``.

It is also the implementation behind ``start_here.py``, which imports ``fit`` from
here. Read that file for everything this one deliberately does not repeat:
installation and the GPU setup ("__Installation__", "__JAX And GPUs__"), running the
pipeline as a black box ("__Running The Pipeline__"), what lands in ``output/`` and how
to summarise a large sample with ``workflow/`` ("__Reading The Output__", "__Where To Go
Next: Workflow__"), and the science concepts the code below implements — the Multi
Gaussian Expansion, the mass model, and pixelized sources.

__Initial Lens Model__

Two searches, described in ``start_here.py``, "__The Two Searches__": ``vis_lp`` fits an
MGE lens light, an SIE + shear mass model and an MGE source together; ``vis_pix`` then
fixes the lens light, frees the mass centre, and replaces the source with a pixelized
Delaunay reconstruction. The prose below documents how each is built, line by line.

``--stage`` selects which of the two runs: ``all`` (the default) runs both in this
process, ``vis_lp`` stops after the first, and ``vis_pix`` runs only the second,
loading the ``vis_lp`` result from disk. See "__Search: vis_pix__" below for why the
CPU route splits the two into separate processes.

``scripts/full_model.py`` extends this fit with the full Source, Light and Mass (SLaM)
chain — see ``start_here.py``, "__SLaM: Source, Light And Mass__".

**Questions:** contact James Nightingale on the Euclid Consortium Slack.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import util


def fit(
    dataset_name: str,
    sample_name: str = None,
    iterations_per_quick_update: int = 5000,
    number_of_cores: int = 1,
    use_cpu: bool = False,
    stage: str = "all",
    skip_pix: bool = None,
):
    """
    Fit the initial Euclid lens model: an MGE light + SIE mass ``vis_lp`` search
    followed by a pixelized-source ``vis_pix`` search.

    Parameters
    ----------
    dataset_name
        Name of the dataset subdirectory inside ``dataset/<sample_name>/``.
    sample_name
        Optional sample subdirectory inside ``dataset/``.
    iterations_per_quick_update
        Sampler iterations between on-the-fly visualisation updates.
    number_of_cores
        Number of CPU cores used by the ``vis_pix`` Nautilus search.
    use_cpu
        Disable JAX and use the CPU sparse operator for the pixelization.
    stage
        Which of the two searches to run.

        - ``"all"`` (default) — run ``vis_lp``, then ``vis_pix``, in this
          process, and return the ``vis_pix`` result.
        - ``"vis_lp"`` — run ``vis_lp`` and return its result without running
          ``vis_pix``. Used by ``sersic_lens_model.py``, whose Sersic source
          prior is seeded from ``galaxies.source.bulge`` — which the pixelized
          fit replaces.
        - ``"vis_pix"`` — run ``vis_pix`` only. The ``vis_lp`` search is not
          re-run: its completed output must already be on disk, and is loaded
          from there. If it is not complete this raises ``RuntimeError``
          immediately, before any fitting, rather than silently re-running
          ``vis_lp`` in a process configured for the pixelized stage.
    skip_pix
        Deprecated alias for ``stage``: ``skip_pix=True`` means
        ``stage="vis_lp"``. Kept so existing callers keep working; new code
        should pass ``stage``.
    """
    if skip_pix is not None and skip_pix:
        stage = "vis_lp"

    if stage not in ("all", "vis_lp", "vis_pix"):
        raise ValueError(
            f"stage must be one of 'all', 'vis_lp' or 'vis_pix', but got {stage!r}."
        )

    from autolens import conf

    project_root = Path(__file__).parent.parent
    conf.instance.push(
        new_path=project_root / "config",
        output_path=project_root / os.environ.get("PYAUTO_OUTPUT_DIR", "output"),
    )

    import autofit as af
    import autolens as al

    """
    __Dataset__

    ``util.load_vis_dataset`` performs all standard dataset preparation in one
    call — HDU discovery, header metadata, noise scaling, the circular mask at the
    ``info.json`` radius, over-sampling, and the multiple-image positions. The ten
    steps are listed in ``start_here.py``, "__How A Dataset Is Loaded__", and
    documented per-parameter on the function itself in ``util.py``.

    What matters for the code below is the single ``EuclidDataset`` it returns,
    whose attributes this script reads rather than re-deriving: ``d.dataset`` is
    the masked, over-sampled imaging; ``d.dataset_centre`` (the brightest central
    pixel) and ``d.mask_radius`` anchor the model priors; and
    ``d.positions_likelihood_list``, ``d.magzero``, ``d.pixel_wcs`` and the
    lowest-resolution PSF are handed to the analysis.
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

    # Tighten the lens ell_comps TruncatedGaussianPrior bounds from the
    # library default of [-1, 1] to [-0.5, 0.5]. Beyond that, the MGE forms
    # a multi-blob shape that absorbs lensed-source flux into the lens light
    # model — a known systematic. ell_comps in [-0.5, 0.5] corresponds to
    # axis ratios q >= ~0.17 (at the diagonal corner), which is plenty wide.
    #
    # mge_model_from gives each of the 2 bases its own independent ell_comps
    # prior shared across all 20 gaussians within the basis. Preserve that by
    # creating exactly 2 fresh priors and reassigning them by basis-slice.
    for j in range(2):
        ell_0 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-0.5, upper_limit=0.5
        )
        ell_1 = af.TruncatedGaussianPrior(
            mean=0.0, sigma=0.3, lower_limit=-0.5, upper_limit=0.5
        )
        for i in range(20):
            g = lens_bulge.profile_list[j * 20 + i]
            g.ell_comps.ell_comps_0 = ell_0
            g.ell_comps.ell_comps_1 = ell_1

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = d.dataset_centre[0]
    mass.centre.centre_1 = d.dataset_centre[1]

    source_bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=False,
        centre=d.dataset_centre,
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

    ``use_jax`` follows ``--use_cpu``; see ``start_here.py``, "__JAX And GPUs__".
    """
    analysis = util.AnalysisImaging(
        dataset=d.dataset,
        positions_likelihood_list=d.positions_likelihood_list,
        use_jax=not use_cpu,
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

    Nautilus nested sampling.  ``n_live=750`` balances accuracy and speed for this
    15-parameter model.
    ``n_like_max`` stops runaway fits (most complete well under 200 000 evaluations).
    """
    search = af.Nautilus(
        name="vis_lp",
        **settings_search.search_dict,
        n_live=750,
        batch_size=50,
        iterations_per_quick_update=iterations_per_quick_update,
        n_like_max=200000,
    )

    """
    __Stage: vis_pix Requires A Completed vis_lp__

    Under ``--stage vis_pix`` this process must not run ``vis_lp``; it must load
    the one that a previous ``--stage vis_lp`` run left on disk. ``search.fit``
    would do exactly that on its own — it short-circuits to
    ``result_via_completed_fit`` when ``paths.is_complete`` — but if the result
    is *not* there it would instead run the whole light-profile fit silently,
    inside a process configured for the pixelized stage. So the condition is
    checked here and the run is failed instead.

    The check is the library's own, reached the same way ``search.fit`` reaches
    it: attach the (analysis-modified) model and the unique tag to the paths so
    the output directory and its identifier resolve, restore any zipped result,
    and read ``paths.is_complete`` — the ``.completed`` marker file that
    ``paths.completed()`` writes at the end of a successful fit.
    """
    if stage == "vis_pix":
        search.paths.model = analysis.modify_model(model)
        search.paths.unique_tag = search.unique_tag
        search.paths.restore()

        if not search.paths.is_complete:
            sample_arg = f"--sample={sample_name} " if sample_name is not None else ""
            raise RuntimeError(
                f"--stage vis_pix requires a completed vis_lp result, but none "
                f"was found at:\n\n    {search.paths.output_path}\n\n"
                f"Run the light-profile stage first:\n\n"
                f"    python scripts/initial_lens_model.py "
                f"{sample_arg}--dataset={dataset_name} --stage vis_lp\n"
            )

    source_lp_result = search.fit(
        model=model, analysis=analysis, **settings_search.fit_dict
    )

    if stage == "vis_lp":
        return source_lp_result

    """
    __Source Pix__

    The second search, ``vis_pix``, replaces the MGE source with a pixelized
    reconstruction on a Delaunay mesh and re-fits the mass against it.
    ``start_here.py``, "__Pixelized Sources__", gives the reason: high-redshift
    sources are typically clumpy, asymmetric and multi-component, a sum of
    concentric Gaussians cannot represent that, and a source model too rigid to
    fit the source is compensated for by distorting the *mass* model — the thing
    the pipeline exists to measure. Reconstructing the source pixel by pixel
    removes that bias, so the mass model improves even though nothing about the
    mass model itself has changed.

    Everything from here to the end of ``fit`` builds that one search, and it is
    the most intricate part of the pipeline, so it is documented step by step
    below. The ``vis_lp`` result is used four times over: it supplies the lens
    light to hold fixed, the adapt image that decides where source pixels go, the
    shear priors, and the multiple-image positions that veto unphysical
    solutions.

    The standalone tutorial for this mesh, outside the Euclid context, is
    ``scripts/imaging/features/pixelization/delaunay.py`` in the
    `autolens_workspace <https://github.com/PyAutoLabs/autolens_workspace>`_. The
    code below follows its adaptive-Delaunay section, with the Euclid-specific
    settings noted as they appear.
    """
    dataset = d.dataset
    mask = dataset.mask
    mask_radius = d.mask_radius

    """
    __Sparse Operator__

    A pixelized fit solves a large linear system at every likelihood evaluation,
    and the expensive ingredient in it is the PSF: the mapping between image
    pixels and source pixels has to be convolved before the source can be solved
    for. The sparse operator precomputes the PSF-and-noise-map products that
    convolution needs, once, when the dataset is built. Each likelihood evaluation
    then reuses them instead of recomputing them. The operator is attached to the
    returned dataset and picked up automatically by the fit, and it is applied
    only here, because only a pixelized fit performs that inversion — ``vis_lp``
    above runs on the plain ``d.dataset``.

    The two calls build the same operator by different routes.
    ``apply_sparse_operator`` uses JAX, which is the GPU path;
    ``apply_sparse_operator_cpu`` uses a Numba CPU implementation. ``--use_cpu``
    selects the latter, and is the same switch that sets ``use_jax=False`` on the
    analysis below — so it does not merely move the same arithmetic to another
    device, it changes which implementation of the linear algebra is used.
    """
    if use_cpu:
        dataset = dataset.apply_sparse_operator_cpu()
    else:
        dataset = dataset.apply_sparse_operator()

    """
    __Image Mesh__

    A Delaunay source mesh is not laid down in the source plane. Its vertices are
    (y, x) points defined in the *image* plane, which are ray-traced to the source
    plane by whichever mass model the sampler is currently trying, and
    triangulated there. That image-plane grid therefore has to be built before the
    search starts, and it stays fixed for the whole search.

    The ``Hilbert`` image mesh draws those points from the source's own light. It
    turns an adapt image (next section) into a weight map and runs a Hilbert curve
    over it, so points cluster where the weight is high: the source's bright
    clumps get many source pixels and blank sky gets few. ``weight_power=3.5`` is
    the power the weights are raised to, and controls how aggressively points
    migrate into the brightest regions; ``weight_floor=0.01`` sets a minimum
    weight, so the faint outskirts still receive some pixels rather than none.

    ``pixels=500`` is the total drawn. The autolens_workspace ``delaunay.py``
    example uses 1000 on its simulated data; 500 is the right number here because
    Euclid VIS cut-outs are small — a postage stamp a few arcseconds across at
    0.1"/pixel, so the analysis mask contains only a few thousand image pixels.
    Drawing 1000 source pixels into that would over-resolve the data, making every
    likelihood evaluation more expensive while inviting the reconstruction to fit
    noise.
    """
    hilbert_pixels = 500

    image_mesh = al.image_mesh.Hilbert(
        pixels=hilbert_pixels, weight_power=3.5, weight_floor=0.01
    )

    """
    __Adapt Image__

    An adapt image is the previous fit's view of one galaxy on its own.
    ``galaxy_name_image_dict_via_result_from`` takes the maximum likelihood
    ``vis_lp`` model and, for each galaxy, subtracts the model images of every
    other galaxy from the data and divides by the noise map. The entry keyed
    ``"('galaxies', 'source')"`` is therefore a signal-to-noise map of the
    lens-light-subtracted lensed source: an image of where the source's light
    actually falls, with the lens removed. A floor at a small fraction of its peak
    is applied so that no pixel is zero or negative, which would break the
    weighting downstream.

    That image steers three things. Here it is the ``adapt_data`` the Hilbert mesh
    weights its points by; below it sets the per-pixel over-sampling; and in the
    model it is what ``reg.AdaptSplit`` uses to vary the regularization strength
    across the source.

    This is why the two searches must run in this order. The ``vis_lp`` MGE fit is
    not the final answer, but it is a good enough deblending of lens and source to
    say where the source is bright, which is all the adaptation needs.
    ``image_plane_mesh_grid_from`` combines it with the mask and returns the 500
    image-plane points.
    """
    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )

    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
    )

    """
    __Edge Zeroing__

    Source pixels at the edge of the mesh are the ones most likely to go wrong.
    Nothing constrains them from outside, so the solver is free to give them
    bright values that mop up residuals from the lens-light subtraction at the
    mask boundary, and that corruption then spreads inwards through the
    regularization.

    The fix is to include such pixels deliberately and force them to zero.
    ``append_with_circle_edge_points`` appends a ring of 30 points to the
    image-plane grid, placed just outside the mask edge — at ``mask_radius`` plus
    half a pixel. They are ray-traced and triangulated along with all the others,
    so the mesh has a proper boundary, and ``zeroed_pixels=edge_pixels_total`` on
    the ``Delaunay`` mesh below tells the inversion to hold their reconstructed
    values at zero. The reconstruction is then bounded by pixels that are known to
    be blank rather than by pixels free to absorb whatever is left over.
    """
    edge_pixels_total = 30

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )

    """
    __Handing The Mesh To The Fit__

    ``AdaptImages`` is the object that carries all of the above into the
    model-fit. It pairs the source galaxy's model path with two things: the adapt
    image, which the regularization adapts to, and the image-plane mesh grid built
    above, which the ``Delaunay`` mesh uses as its vertices. Both travel to the
    analysis as one argument, so nothing has to be recomputed per sample.

    With the ring appended the grid is final, and its length is the number of
    source pixels: 500 Hilbert points plus 30 edge points. That length is what is
    passed to the mesh below as ``pixels``, rather than the pixel count being a
    free parameter of the model — and it has to be fixed, because JAX compiles the
    likelihood function against statically shaped arrays. A source-pixel count
    that varied from sample to sample would change those shapes and force a
    recompile on every evaluation.
    """
    adapt_images = al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    """
    __Pixelization Over Sampling__

    ``load_vis_dataset`` sets over-sampling for the light profiles only, in radial
    bins around the lens centre. The pixelization needs its own, and it wants it
    somewhere else: not where the lens light is steep, but where the *lensed
    source* is bright, because that is where a coarse image grid would smear the
    mapping between image pixels and source pixels.

    The source adapt image is already a signal-to-noise map, so the rule is a
    single threshold. Image pixels where the source's signal-to-noise exceeds 3
    are over-sampled 4x; everywhere else 2x. ``apply_over_sampling`` writes that
    map onto the dataset as ``over_sample_size_pixelization``, passing the
    existing light-profile over-sampling straight through unchanged.
    """
    import numpy as np

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

    """
    __Analysis And Positions__

    The analysis is rebuilt on the over-sampled dataset, with ``adapt_images``
    attached and with a tightened positional constraint. Everything else — the
    latent variables, the RGB visualisation, the WCS — is as it was for
    ``vis_lp``.

    A pixelized source has a failure mode a parameterised one does not. A mass
    model that demagnifies the source heavily can reproduce the data with a
    bright, featureless blob and score a high likelihood while being physically
    wrong. The defence is the multiple images.
    ``source_lp_result.positions_likelihood_from`` takes the maximum likelihood
    ``vis_lp`` mass model, solves for where it places the source's multiple
    images, and returns a likelihood penalty that rejects any model which cannot
    ray-trace them back to a common point in the source plane.

    ``factor=3.0`` multiplies the threshold that model implies, widening it so a
    mass model merely somewhat different from ``vis_lp``'s is not vetoed;
    ``minimum_threshold=0.2`` rounds the threshold up if it comes out below 0.2",
    so a ``vis_lp`` fit that happened to trace its positions very precisely cannot
    hand the next search an impossibly tight constraint. Note that this replaces
    ``d.positions_likelihood_list``, which ``vis_lp`` used: the constraint is now
    derived from a converged mass model rather than from the dataset's own
    positions. Deriving it rather than requiring it by hand is a large part of
    what makes the pipeline automatic.
    """
    analysis = util.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        use_jax=not use_cpu,
        dataset_main_path=d.dataset_main_path,
        title_prefix="VIS",
        plot_rgb=True,
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        **settings_search.info,
    )

    """
    __Model: Fixed Lens Light + Free Mass Centre + Pixelized Source__

    The lens light is no longer fitted. ``bulge`` is the ``vis_lp`` *instance* —
    the 40-Gaussian MGE frozen at its maximum likelihood parameters — because
    deblending the lens is a solved problem by this point, and freezing it keeps
    the search's whole budget on the mass and the source.

    The mass is a fresh ``Isothermal`` whose centre is now free, given a uniform
    prior spanning ±0.1" of ``d.dataset_centre``. In ``vis_lp`` that centre was
    pinned exactly to ``d.dataset_centre``, the brightest pixel; with a reliable
    source reconstruction in hand, the small offset between the light centroid and
    the mass centroid can be measured rather than assumed, and the box keeps that
    freedom from turning back into a degeneracy. The shear is chained forward as a
    *model*, so it is re-fitted, but from the priors ``vis_lp`` inferred.

    The source becomes a ``Pixelization``: a ``Delaunay`` mesh whose ``pixels`` is
    the fixed length of the image-plane grid and whose ``zeroed_pixels`` is the
    edge ring, regularized by ``reg.AdaptSplit``.

    ``reg.AdaptSplit`` — not ``reg.Adapt`` — is mandatory for the Delaunay family:
    ``reg.Adapt`` cannot JIT on it, because Delaunay neighbours come from a
    ``scipy.spatial.Delaunay`` call on the traced source-plane grid, which cannot
    be traced under ``jit``/``grad``. Both pixelized stages of
    ``scripts/full_model.py`` use the same pairing for the same reason.

    The "adapt" half is doing scientific work as well as satisfying the compiler:
    it uses the adapt image to vary the regularization strength across the source,
    so the bright, well-constrained regions are smoothed less than the faint ones
    instead of the whole reconstruction being smoothed uniformly.
    """
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.UniformPrior(
        lower_limit=d.dataset_centre[0] - 0.1, upper_limit=d.dataset_centre[0] + 0.1
    )
    mass.centre.centre_1 = af.UniformPrior(
        lower_limit=d.dataset_centre[1] - 0.1, upper_limit=d.dataset_centre[1] + 0.1
    )

    shear = source_lp_result.model.galaxies.lens.shear

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                mass=mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=al.mesh.Delaunay(
                        pixels=image_plane_mesh_grid.shape[0],
                        zeroed_pixels=edge_pixels_total,
                    ),
                    regularization=al.reg.AdaptSplit,
                ),
            ),
        ),
    )

    """
    __Search: vis_pix__

    Nautilus again, with settings that reflect how much more each likelihood
    evaluation now costs. ``n_live=300``, against ``vis_lp``'s 750: the space is
    both smaller (the lens light is fixed, and the pixelization contributes only
    its regularization parameters) and far more expensive per evaluation, since
    every one solves an inversion. ``n_batch=15`` is the number of likelihood
    evaluations Nautilus requests per step, kept low so the memory those
    inversions need in parallel stays manageable. ``n_like_max=100000`` caps a
    pathological fit at half the ``vis_lp`` ceiling, for the same reason.

    ``number_of_cores`` is added to the search dict here and only here. ``vis_lp``
    takes its parallelism from the batch of models it evaluates on the device,
    whereas this search is the one that benefits from a multiprocessing pool —
    which is why ``--use_cpu`` and ``--number_of_cores`` are usually passed
    together, and why ``--stage`` exists.

    On CPU the two searches are run as two separate processes: ``--stage vis_lp``
    first, with JAX on CPU, and then ``--stage vis_pix``, with the Numba sparse
    operator and a multiprocessing pool of ``--number_of_cores``. The two stages run as separate Python processes. This is a conservative default: PyAutoFit documents an XLA deadlock for a forked worker whose likelihood touches JAX, and the DR1 science runs were submitted this way. A local control test (hpc/diagnostics/jax_fork_control.py) did not reproduce a hang for the CPU route, whose vis_pix likelihood is Numba; production sampler sizes and large pools are untested, so the boundary is kept until a cluster run passes. hpc/README.md has the measured table. The second process does not
    re-fit ``vis_lp``: it loads the completed result the first one wrote, and
    fails immediately if that result is missing (see "__Stage: vis_pix Requires A
    Completed vis_lp__" above). ``hpc/README.md`` documents the submission
    scripts that run the pair in order. On GPU there is no such constraint and
    the default ``--stage all`` runs both searches in one process.
    """
    vis_pix_search_dict = {
        **settings_search.search_dict,
        "number_of_cores": number_of_cores,
    }

    search = af.Nautilus(
        name="vis_pix",
        **vis_pix_search_dict,
        n_live=300,
        n_batch=15,
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
    ) = util.parse_fit_args()
    fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
        number_of_cores=number_of_cores,
        use_cpu=use_cpu,
        stage=stage,
    )
