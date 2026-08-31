"""
MGE Lens-Only Pipeline
======================

__What This Is For__

Sometimes the question is not "what is the lens model?" but "is there a lensed
source here at all?". In a Euclid cut-out the lens galaxy is the bright thing in
the middle and the arcs sit on top of it, so a candidate is hard to grade by eye
until the lens light has been taken away.

This script takes it away. It fits a Multi Gaussian Expansion (MGE) to the lens
light and to nothing else — there is no mass model and no source in the model at
all — and the fit's lens-light-subtracted image is the thing you look at. Use it
to eyeball a candidate or triage a sample; use ``scripts/initial_lens_model.py``
when you want an Einstein radius, a source reconstruction, or photometry to
publish.

__Why It Is Fast__

Two reasons, both consequences of what has been left out.

Nothing is lensed. With no mass profile the model has a single plane, so a
likelihood evaluation is a light profile evaluated on a grid and convolved with
the PSF — no ray tracing, no source plane, no pixelization.

And the MGE is cheap to sample. The lens light is 40 Gaussians (two sets of 20),
but every Gaussian's ``intensity`` is solved by linear algebra at each
evaluation rather than sampled, leaving only a handful of non-linear parameters
for the search to explore. That combination is what makes a subtraction you can
run over a whole candidate list.

__What To Watch Out For__

The MGE is deliberately flexible, and here it is the *only* thing in the model.
Nothing represents the lensed source, so any source flux the Gaussians can
absorb, they will — most of all where the arcs sit closest to the lens light.
``scripts/initial_lens_model.py`` guards against a related failure by narrowing
its MGE ellipticity priors, so that the Gaussians cannot form the multi-blob
shape that eats lensed-source flux; this script does not, because it is not
trying to measure anything.

So read the subtracted image as an indication of where the source emission is,
not as a measurement of how much of it there is. If it looks promising, fit it
properly.

__What It Fits__

- ``fit()`` — the VIS band. Returns the result the second stage is conditioned
  on.
- ``fit_waveband()`` — every other band in the cut-out, with the VIS lens light
  held fixed and only a small astrometric offset free, giving the same
  subtraction band by band.

Run it as::

    python scripts/mge_lens_only.py --sample=<sample> --dataset=<name>

New to the pipeline? Read ``start_here.py`` in the repository root first: it
covers installation, the dataset contract, masking and over-sampling, what an
MGE is, Nautilus and JAX, and how to read ``output/``. This script assumes all
of it.
"""

import os
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

    """
    __Configs__

    The pipeline's ``config/`` directory is pushed onto the configuration stack
    — priors, visualisation settings and the latent toggles all come from there
    — and results are written under the directory named by ``PYAUTO_OUTPUT_DIR``
    (default ``output/``).
    """
    project_root = Path(__file__).parent.parent
    conf.instance.push(
        new_path=project_root / "config",
        output_path=project_root / os.environ.get("PYAUTO_OUTPUT_DIR", "output"),
    )

    import autofit as af
    import autolens as al

    """
    __Dataset__

    One call does the whole VIS preparation: finds the FITS file, locates the
    VIS HDU triplet, loads image, noise map and PSF, reads ``info.json`` and the
    header, applies the extra-galaxy noise scaling and the circular mask, sets
    the standard over-sampling and loads the worst-seeing band's PSF for the
    aperture-flux latents. ``start_here.py`` walks through what each of those
    steps is for; ``util.load_vis_dataset`` documents the returned fields.

    The mask radius comes from the dataset's ``info.json`` and is never a
    command-line argument. It matters more than it looks: it sets the outer
    extent of the Gaussians the MGE is built from, so it is a modeling choice
    and not just a crop.
    """
    d = util.load_vis_dataset(dataset_name, sample_name=sample_name)

    """
    __Settings AutoFit__

    Results land at::

        <PYAUTO_OUTPUT_DIR>/<sample>/<dataset>/mge_lens_only/vis/<hash>/

    The ``mge_lens_only`` tag keeps this subtraction beside, rather than on top
    of, any full lens model of the same lens, and is the string the ``workflow/``
    aggregator examples query on to collect these fits across a sample.

    ``magzero``, the photometric zero-point from the VIS header, rides along in
    ``info`` so the analysis can express fluxes in physical units.
    """
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

    """
    __Analysis__

    The pipeline's ``AnalysisImaging``, which adds the RGB visualisation and the
    Euclid latent catalogue on top of the library analysis, and runs the
    likelihood through JAX.

    No ``positions_likelihood_list`` is passed, unlike the full pipelines. That
    likelihood penalty exists to reject mass models that fail to trace the
    multiple images back to a common source position — and there is no mass
    model here for it to act on.
    """
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

    """
    __Model: Lens Light Only__

    One galaxy, one component: an MGE bulge of 40 Gaussians, built as two sets
    of 20 by ``mge_model_from``, sharing a centre initialised on the brightest
    pixel found near the lens centre. Each Gaussian's intensity is solved
    linearly, so the search sees only the handful of non-linear parameters that
    set the shape of the two sets.

    There is no ``mass`` and no ``source`` in this collection, and that absence
    is the entire design of this script: it is what makes the fit quick, and
    what means the residual image is a look at the source rather than a model of
    it.

    The redshift is a label. With no mass profile nothing is lensed, so no
    distance-dependent quantity is computed and the value never enters the fit.
    """
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

    """
    __Search__

    Nautilus nested sampling, named ``vis`` after the band, with ``n_live=75``
    — a small live-point count is enough for a parameter space this shallow.
    ``iterations_per_quick_update`` sets how often the figures on disk are
    refreshed, so the subtraction can be watched as it improves.
    """
    search = af.Nautilus(
        name="vis",
        **settings_search.search_dict,
        n_live=75,
        batch_size=50,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    """
    __Result: What To Look At__

    The result folder's ``image/`` directory is refreshed throughout the fit.
    Because this model has one plane, the fit subplot written there is the
    single-plane one — data, model data, lens-light-subtracted image, normalized
    residual map — and the third of those panels is why you ran the script: the
    data with the fitted lens light removed, in which the arcs should stand out
    if they are there.

    ``files/`` alongside it holds the machine-readable result, and the returned
    object is what ``fit_waveband`` below conditions every other band on.
    """
    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def fit_waveband(
    dataset_name: str,
    vis_result,
    sample_name: str = None,
    iterations_per_quick_update: int = 50000,
):
    """
    Subtract the lens light from every non-VIS band of the cut-out.

    The cut-out's other bands — NISP near-infrared, and the external
    ground-based imaging where it exists — each get their own fit, with the MGE
    from the VIS result frozen as an instance and only a small astrometric
    offset between the two pixel grids left free. The Gaussian intensities are
    still solved linearly against each band's own data, so the subtraction
    adapts to how bright the lens is in that band even though its shape does
    not change.

    As in ``fit`` above there is no mass model and no source: this produces a
    lens-light-subtracted image per band for inspection, not a multi-band lens
    model. For fitted multi-band photometry, run
    ``scripts/sersic_lens_model_waveband.py``, the SED chain.
    """
    import autofit as af
    import autolens as al
    from autolens import conf
    from pathlib import Path

    """
    __Configs__

    The configuration push is repeated because this function is also usable on
    its own, given a VIS result. ``cb_unit`` relabels the figure colour bars in
    electrons per second, the units of the NISP and external imaging, where VIS
    is in ADU per second (see ``util.ab_mag_via_flux_from``).
    """
    project_root = Path(__file__).parent.parent
    conf.instance.push(
        new_path=project_root / "config",
        output_path=project_root / os.environ.get("PYAUTO_OUTPUT_DIR", "output"),
    )

    conf.instance["visualize"]["general"]["units"][
        "cb_unit"
    ] = r"$\,\,\mathrm{e^{-}}\,\mathrm{s^{-1}}$"

    """
    __Dataset Wavebands__

    ``util.load_vis_dataset`` is the VIS loader — it prepares one band and
    returns it ready to fit. Here every *other* band has to be prepared in turn,
    so the FITS file is opened directly and only the path and the band-to-HDU
    mapping are taken from it.

    That mapping is discovered from the HDU names: anything ending in ``_BGSUB``
    is an image HDU, and the band name is what is left when the tag is stripped.
    A band's position in that list gives its three HDUs, because the file is a
    repeating ``(<BAND>_BGSUB, <BAND>_PSF, <BAND>_RMS)`` triplet after the
    PRIMARY header — image at ``i * 3 + 1``, PSF at ``i * 3 + 2``, noise map at
    ``i * 3 + 3``.
    """
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

    """
    __Mask Metadata__

    ``mask_radius`` is read from the dataset's ``info.json`` — it is a required
    key, not a command-line argument, and a dataset without it raises here
    rather than being masked at some invented default. Every band is then masked
    exactly as VIS was, which is what makes their subtractions comparable.
    """
    try:
        with open(dataset_main_path / "info.json") as f:
            info = json.load(f)
    except FileNotFoundError:
        info = {}

    mask_radius = info["mask_radius"]

    """
    __Lowest Resolution PSF__

    The aperture-flux latent variables are matched-aperture photometry, measured
    on the lens image convolved to the resolution of the worst-seeing band in
    the cut-out. That band is named by the PRIMARY header's ``WORST_BAND`` key
    and its FWHM by the ``WORST_PSF_*`` keys, both stamped by the upstream
    Euclid cut-out generator rather than by this pipeline.

    It is the same reference PSF for every band, so it is loaded once here
    instead of inside the loop. If ``WORST_BAND`` is missing or names a band the
    cut-out does not contain, the aperture latents are skipped with a warning
    and the fits themselves proceed unaffected.
    """
    header_primary = al.header_obj_from(
        file_path=dataset_main_path / dataset_fits_name, hdu=0
    )
    # `WORST_BAND` / `WORST_PSF_*` are stamped by the upstream Euclid cut-out
    # generator. When they are absent the aperture-flux latents degrade to NaN
    # rather than crashing the whole multi-band run.
    worst_band_attr = header_primary.get("WORST_BAND", None)
    if worst_band_attr is None:
        print(
            f"[WARN] {dataset_name}: WORST_BAND missing in primary header — "
            "skipping aperture-flux latent variables.",
            flush=True,
        )
        psf_lowest_resolution = None
        psf_lowest_resolution_fwhm = None
    else:
        lowest_resolution_waveband = worst_band_attr.lower()
        lowest_resolution_waveband_index = dataset_index_dict.get(
            lowest_resolution_waveband, None
        )
        if lowest_resolution_waveband_index is None:
            print(
                f"[WARN] {dataset_name}: WORST_BAND={worst_band_attr} not present in "
                "dataset HDU list — skipping aperture-flux latent variables.",
                flush=True,
            )
            psf_lowest_resolution = None
            psf_lowest_resolution_fwhm = None
        else:
            psf_lowest_resolution = al.Convolver.from_fits(
                file_path=dataset_main_path / dataset_fits_name,
                hdu=lowest_resolution_waveband_index * 3 + 2,
                pixel_scales=0.1,
                normalize=True,
            )
            psf_lowest_resolution_fwhm = util.psf_fwhm_arcsec_from_primary_header(
                header=header_primary,
                dataset_name=dataset_name,
            )

    """
    __Waveband Loop__

    One independent fit per band, VIS skipped because it has already been
    fitted and is what the rest are conditioned on. Each pass repeats the whole
    preparation for its band — load, centre, header, noise scaling, mask,
    over-sampling — then runs its own short search.
    """
    for dataset_waveband, dataset_index in dataset_index_dict.items():
        if dataset_waveband == "vis":
            continue

        """
        __Dataset (This Band)__

        Image, PSF and noise map for this band, from the triplet arithmetic
        above.

        The pixel scale is 0.1"/pixel for every band and so is hard-coded: a MER
        cut-out puts the NISP and external imaging on the same grid as VIS. The
        NIR and ground-based data are therefore blurrier rather than coarser,
        and that blurring is carried by each band's own PSF, loaded from its own
        HDU. It is also the reason the lens light shape is taken from VIS and
        not refitted here — a broader PSF smooths out exactly the structure an
        MGE would need in order to be constrained.

        The brightest sub-pixel is searched for in a box around the centre of
        the frame, and is used to concentrate the over-sampling below.
        """
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

        """
        __Zero Point And WCS__

        ``MAGZERO`` is this band's photometric zero-point, passed into the
        analysis so fluxes can be expressed as magnitudes and microJansky; the
        WCS is carried through so the fitted lens centre can be written out in
        sky coordinates.
        """
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

        """
        __Noise Scaling__

        Neighbouring galaxies and artefacts inside the mask are not modelled.
        Where the dataset ships ``mask_extra_galaxies.fits``, those pixels are
        given a large noise value instead, so the likelihood stops trying to fit
        them and the MGE is not pulled off the lens by a bright neighbour. A
        dataset without the file simply skips this.
        """
        try:
            mask_extra_galaxies = al.Mask2D.from_fits(
                file_path=dataset_main_path / "mask_extra_galaxies.fits",
                pixel_scales=0.1,
                invert=True,
            )
            dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
        except FileNotFoundError:
            pass

        """
        __Mask__

        The same circular aperture as the VIS fit — same radius, same centre,
        both from ``info.json`` — so the lens light being subtracted here is
        constrained over the same region it was in VIS.
        """
        mask_centre = info.get("mask_centre") or (0.0, 0.0)
        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=mask_radius,
            centre=mask_centre,
        )
        dataset = dataset.apply_mask(mask=mask)

        """
        __Over Sampling__

        The standard radial scheme: 4x4 sub-pixels within 0.1" of the lens
        centre, 2x2 out to 0.3", 1x1 beyond. Light profiles change fastest in
        the galaxy's core, so that is where evaluating one point per pixel would
        be inaccurate — and it is where all of this model's light is.
        """
        over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
            grid=dataset.grid,
            sub_size_list=[4, 2, 1],
            radial_list=[0.1, 0.3],
            centre_list=[dataset_centre],
        )
        dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        """
        __Settings AutoFit__

        Each band is a search named after itself, under the same tag as the VIS
        fit::

            <PYAUTO_OUTPUT_DIR>/<sample>/<dataset>/mge_lens_only/<band>/<hash>/

        so a lens's whole set of subtractions sits in one folder, VIS included.
        """
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

        """
        __Analysis__

        ``title_prefix`` stamps the band onto every figure, so subtractions from
        different bands cannot be confused once they are out of their folders.
        ``skip_rgb_plot=True`` drops the RGB subplot, which is built from the
        dataset's colour thumbnails and is the same picture for every band — it
        is already in the VIS output.
        """
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

        """
        __Dataset Model__

        The only free part of the model: a (y, x) offset between this band's
        pixel grid and the grid the VIS fit was performed on, with a uniform
        prior of ±0.2" — two pixels at 0.1"/pixel. Registration between
        instruments is not perfect at that scale, and an unmodelled shift would
        show up as a ring of residuals around the lens centre in the subtracted
        image, which is exactly the region being inspected.
        """
        dataset_model = af.Model(al.DatasetModel)
        dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
            lower_limit=-0.2, upper_limit=0.2
        )
        dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
            lower_limit=-0.2, upper_limit=0.2
        )

        """
        __Model: VIS Lens Light, Fixed__

        The lens bulge is taken straight from ``vis_result.instance``, so every
        Gaussian's shape, size and position is fixed to what VIS measured, and
        the search is left with the two offset parameters. Still no mass and no
        source.

        The lens's brightness in this band is not inherited: an MGE is built
        from linear light profiles, whose intensities are solved by linear
        algebra at each likelihood evaluation, so they are re-solved against
        this band's data. The shape comes from VIS, the amount of light comes
        from the band being fitted.
        """
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

        """
        __Search__

        A two-parameter search, so it is short. Its output folder carries the
        same single-plane fit subplot as the VIS fit — data, model data,
        lens-light-subtracted image, normalized residual map — which is the
        subtraction to look at for this band.
        """
        search = af.Nautilus(
            name=dataset_waveband,
            **settings_search.search_dict,
            n_live=75,
            batch_size=50,
            iterations_per_quick_update=iterations_per_quick_update,
        )

        search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


if __name__ == "__main__":
    (
        sample_name,
        dataset_name,
        iterations_per_quick_update,
        number_of_cores,
        use_cpu,
        skip_pix,
    ) = util.parse_fit_args()

    """
    __Running Standalone__

    VIS first, then every other band against it — the two stages of this script
    in the order they have to run.

    The shared argument parser is used so that one command line works across all
    the pipelines, but this script only acts on ``--sample``, ``--dataset`` and
    ``--iterations_per_quick_update``. ``--number_of_cores``, ``--use_cpu`` and
    ``--skip_pix`` are accepted and ignored: there is no pixelized stage to skip
    and no CPU-parallel search to size, because every search here runs under
    JAX on a model with a handful of parameters. ``mask_radius`` is not an
    argument at all — it comes from the dataset's ``info.json``.
    """
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
