"""
Multi-Waveband Lens Model Pipeline
====================================

__Why Fit The Other Bands At All__

A Euclid cut-out is not one image. Alongside VIS it carries the NISP
near-infrared bands and, over most of the sky, external ground-based imaging
(DES, CFIS, HSC, Pan-STARRS and others), each stored as its own
``(<BAND>_BGSUB, <BAND>_PSF, <BAND>_RMS)`` HDU triplet in the same FITS file.

The lens model is derived from VIS, because VIS is the sharpest data and
therefore the data that constrains geometry. But a lens model on its own does
not give you a spectral energy distribution. To place the lens and the lensed
source on an SED — and from there estimate photometric redshifts and stellar
masses — you need the flux of *each galaxy separately* in *each band*, and in a
strong lens the two galaxies overlap on the sky. Aperture photometry cannot
separate them; a lens model can. So this script goes back to every non-VIS band
and measures its flux with the lens model already known.

__What Is Fixed And What Is Free__

Everything geometric is fixed to the VIS result handed in as ``vis_result``:
the lens light, the mass, the external shear and the source light all enter the
model as ``instance`` objects, not free model components. The only parameters
the sampler varies are the two components of a ``DatasetModel`` grid offset,
which absorbs any residual astrometric misalignment between this band's pixel
grid and VIS.

That leaves the question of where the per-band photometry comes from, given
that nothing in the model is free to change brightness. The answer is that
brightness was never a sampled parameter in the first place: the lens and source
light profiles are *linear* light profiles (the MGE Gaussians of
``initial_lens_model.py``, or the ``lp_linear.Sersic`` of
``sersic_lens_model.py``), whose intensities are solved by linear algebra at
every likelihood evaluation. Fixing the instance fixes the shapes; the
intensities are re-solved against this band's own data. The shape comes from
VIS, the flux comes from the band.

__Running It__

Run as a script it does the whole chain: an ``initial_lens_model`` ``vis_lp``
fit on VIS, then one search per remaining band under the same tag::

    python scripts/lens_model_waveband.py --sample=<sample> --dataset=<name>

``fit_waveband`` is also imported by ``scripts/sersic_lens_model_waveband.py``,
the SED chain driver, which passes the Sersic VIS result and
``unique_tag="sersic_lens_model"`` instead.

__Downstream__

The per-band results are the input to two catalogue producers:
``catalogue/scripts/magnitudes.py`` scrapes their latent fluxes into
``magnitudes.csv``, the photometry table the SED fitting works from, and
``catalogue/scripts/multi_wavelength.py`` stacks one row per band into
``fit_multi_wavelength.png`` for visual inspection. Both default to reading
``output_sed``, the output tree the SED chain is run under.

New to the pipeline? Read ``start_here.py`` in the repository root first: it
covers installation, the dataset and FITS header contracts, masking and
over-sampling, the MGE, Nautilus and JAX, and how to read ``output/``. This
script assumes all of it and explains only what is specific to fitting a second
waveband.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import util
from scripts.initial_lens_model import fit


def fit_waveband(
    dataset_name: str,
    unique_tag: str,
    vis_result,
    use_sersic_over_sampling: bool = False,
    sample_name: str = None,
    iterations_per_quick_update: int = 5000,
):
    import numpy as np
    import json
    import autofit as af
    import autolens as al
    from astropy.wcs import WCS
    from autolens import conf

    """
    __Configs__

    The pipeline's own ``config/`` directory is pushed onto the configuration
    stack, and results are written to the directory named by
    ``PYAUTO_OUTPUT_DIR`` (default ``output/``). The SED chain sets that
    variable to ``output_sed``, because this script writes one search per band
    per lens and would otherwise bury the main output tree.

    ``cb_unit`` is the colour bar label used by every figure the run produces.
    VIS imaging is in ADU per second while the NISP and external bands are in
    electrons per second (see ``util.ab_mag_via_flux_from``), so the label is
    set to match the data being fitted here.
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
    __Dataset__

    The same dataset directory the VIS fit read:
    ``dataset/<sample>/<name>/<name>.fits``, with ``<sample>`` omitted for a
    flat layout. Nothing new is downloaded or prepared here — the other bands
    were always in that one file.

    ``util.load_vis_dataset`` is not used below, because it is the *VIS* loader:
    it picks the ``vis`` HDU, applies that band's mask and over-sampling and
    returns a single prepared dataset. This script needs the same preparation
    repeated for every other band, so it opens the FITS file directly.
    """
    if sample_name is not None:
        dataset_main_path = project_root / "dataset" / sample_name / dataset_name
    else:
        dataset_main_path = project_root / "dataset" / dataset_name
    dataset_fits_name = f"{dataset_name}.fits"

    """
    __Dataset Wavebands__

    Which bands a cut-out contains is discovered, not configured. Every HDU name
    ending in ``_BGSUB`` is an image HDU, and the band name is what remains once
    the tag is stripped: ``des_g``, ``vis``, ``nir_y`` and so on. The dictionary
    maps each band to its *position* in that list of image HDUs, counting from
    zero.

    Because the FITS contract is a repeating
    ``(<BAND>_BGSUB, <BAND>_PSF, <BAND>_RMS)`` triplet after the PRIMARY header,
    a band at position ``i`` has its image at HDU ``i * 3 + 1``, its PSF at
    ``i * 3 + 2`` and its RMS noise map at ``i * 3 + 3``. That is the arithmetic
    used throughout the loop below — you never write an HDU index by hand.
    """
    dataset_index_dict = util.dataset_instrument_hdu_dict_via_fits_from(
        dataset_path=dataset_main_path,
        dataset_fits_name=dataset_fits_name,
        image_tag="_BGSUB",
    )

    """
    __Mask Metadata__

    The mask radius and centre come from the dataset's ``info.json`` — they are
    never command-line arguments, so every band of a lens is masked exactly as
    its VIS fit was, and the fluxes measured here are comparable to the VIS
    fluxes. If the file or a key is missing the fallbacks are a 3.0" radius
    centred on the frame.
    """
    try:
        with open(dataset_main_path / "info.json") as f:
            info = json.load(f)
    except FileNotFoundError:
        info = {}

    mask_radius = info.get("mask_radius") or 3.0
    mask_centre = info.get("mask_centre") or (0.0, 0.0)

    """
    __Lowest Resolution PSF__

    The four aperture-flux latent variables are matched-aperture photometry: the
    model lens image is convolved to the resolution of the *worst-seeing* band
    in the cut-out and summed within 1, 2, 3 and 4 times that band's PSF FWHM,
    so that a VIS flux and a ground-based flux can be compared like for like in
    an SED fit.

    That reference band is named by the ``WORST_BAND`` key of the PRIMARY
    header, and its FWHM by the ``WORST_PSF_*`` keys — both stamped by the
    upstream Euclid cut-out generator, not by this pipeline. The PSF is the same
    for every band being fitted, so it is loaded once here rather than inside
    the loop.

    If ``WORST_BAND`` is missing, or names a band this cut-out does not contain,
    the aperture latents are skipped with a warning and the fits themselves are
    unaffected. A missing FWHM is treated far more harshly inside
    ``util.psf_fwhm_arcsec_from_primary_header``, which raises: the aperture
    radii are multiples of it, so a guessed value would silently corrupt the
    photometry this whole script exists to produce.
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
    fitted — its result is the ``vis_result`` every one of these fits is
    conditioned on. Each iteration repeats the whole dataset preparation for its
    band: load, centre, header, noise scaling, mask, over-sampling, then the
    search.
    """
    for dataset_waveband, dataset_index in dataset_index_dict.items():
        if dataset_waveband == "vis":
            continue

        """
        __Dataset (This Band)__

        Image, PSF and noise map for this band, from the triplet arithmetic
        described above.

        The pixel scale is 0.1"/pixel for *every* band, which is why it is
        hard-coded here rather than read per band: a MER cut-out resamples the
        NISP and external imaging onto the same grid as VIS. "Lower resolution"
        therefore does not mean bigger pixels — the NIR and ground-based data
        are blurrier, not coarser, and that blurring is carried entirely by each
        band's own PSF, loaded from its own HDU and convolved with the model at
        every likelihood evaluation.

        This is also the deeper reason the lens model is held fixed. A band
        whose PSF is two or three times wider than VIS simply does not resolve
        the arcs well enough to constrain an Einstein radius or a light profile
        shape; asked to fit them it would return a poorly constrained model and
        contaminate the photometry. Given the geometry, the same data constrain
        flux perfectly well.
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

        """
        __Lens Centre__

        The brightest sub-pixel is searched for in a small box around the *lens*
        centre from ``info.json``, not around the centre of the frame, so a lens
        that sits off-centre in its cut-out still anchors the over-sampling on
        the right pixel. In this band the brightest pixel can land a fraction of
        a pixel away from where it falls in VIS, which is exactly the
        misalignment the grid offset below is there to absorb.
        """
        cy, cx = mask_centre
        dataset_centre = dataset.data.brightest_sub_pixel_coordinate_in_region_from(
            region=(cy - 0.3, cy + 0.3, cx - 0.3, cx + 0.3), box_size=2
        )

        """
        __Zero Point And WCS__

        ``MAGZERO`` is the photometric zero-point of *this* band's image HDU. It
        is passed into the analysis below and is what turns a fitted flux in
        image units into an AB magnitude and then microJansky — without it the
        µJy latents cannot be computed, and a band's photometry is not
        comparable with any other band's. The WCS is carried through so the
        fitted lens centre can be written back out in sky coordinates.
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

        pixel_wcs = WCS(header).celestial if header is not None else None

        """
        __Noise Scaling__

        Neighbouring galaxies and artefacts inside the mask are not modelled;
        their pixels are instead given a large noise value so the likelihood
        stops caring about them. DR1 preprocessing writes that map as
        ``segmentation/artefact_binary.fits`` and older datasets as
        ``mask_extra_galaxies.fits``, so both names are tried in that order, and
        neither being present is fine.

        The map is applied only if it was cut out at the same size as this
        band's image; one made for a differently sized frame is skipped rather
        than forced on.
        """
        for noise_mask_path in (
            dataset_main_path / "segmentation" / "artefact_binary.fits",
            dataset_main_path / "mask_extra_galaxies.fits",
        ):
            try:
                mask_extra_galaxies = al.Mask2D.from_fits(
                    file_path=noise_mask_path,
                    pixel_scales=0.1,
                    invert=True,
                )
            except FileNotFoundError:
                continue
            if mask_extra_galaxies.shape_native == dataset.shape_native:
                dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
            break

        """
        __Mask__

        The same circular mask as the VIS fit: same radius, same centre, read
        from the same ``info.json``. Using one aperture across all bands is what
        makes the fluxes that come out of them a colour rather than a collection
        of unrelated numbers.
        """
        mask = al.Mask2D.circular(
            shape_native=dataset.shape_native,
            pixel_scales=dataset.pixel_scales,
            radius=mask_radius,
            centre=mask_centre,
        )
        dataset = dataset.apply_mask(mask=mask)

        """
        __Over Sampling__

        Two schemes, chosen by ``use_sersic_over_sampling``.

        The default (``False``, how the ``initial_lens_model`` chain runs) is
        the standard radial scheme: 4x4 sub-pixels within 0.1" of the lens
        centre, 2x2 out to 0.3", 1x1 beyond.

        The Sersic chain passes ``True``. A Sersic profile diverges at its
        centre, so evaluating it on one point per pixel is not accurate enough
        near either galaxy's core. This branch therefore builds two maps — one
        concentrated on the *source* centre, computed on the grid traced back to
        the source plane by the VIS tracer, and one concentrated on the lens
        centre in the image plane — and keeps the larger sub-grid of the two at
        every pixel. It reads ``tracer.galaxies[1].bulge.centre``, which is why
        it is only ever passed by the Sersic chain: it needs a source bulge that
        is a single profile with a centre.
        """
        if not use_sersic_over_sampling:
            over_sample_size = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=dataset.grid,
                    sub_size_list=[4, 2, 1],
                    radial_list=[0.1, 0.3],
                    centre_list=[dataset_centre],
                )
            )
            dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)
        else:
            tracer = vis_result.max_log_likelihood_tracer
            traced_grid = tracer.traced_grid_2d_list_from(grid=dataset.grid)[-1]
            source_centre = tracer.galaxies[1].bulge.centre

            over_sample_size = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=traced_grid,
                    sub_size_list=[16, 4, 2],
                    radial_list=[0.1, 0.3],
                    centre_list=[source_centre],
                )
            )
            over_sample_size_lens = (
                al.util.over_sample.over_sample_size_via_radial_bins_from(
                    grid=dataset.grid,
                    sub_size_list=[16, 4, 1],
                    radial_list=[0.1, 0.3],
                    centre_list=[dataset_centre],
                )
            )
            over_sample_size = np.where(
                over_sample_size > over_sample_size_lens,
                over_sample_size,
                over_sample_size_lens,
            )
            over_sample_size = al.Array2D(values=over_sample_size, mask=mask)
            dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        """
        __Settings AutoFit__

        Results land at::

            <PYAUTO_OUTPUT_DIR>/<sample>/<dataset>/<unique_tag>/<band>/<hash>/

        The search is named after the band, so all of a lens's bands sit side by
        side under one tag — ``initial_lens_model`` when this script is run
        directly, ``sersic_lens_model`` when the SED chain calls it. That layout
        is what the catalogue producers walk, and their ``--unique_tag``
        argument selects which stage they read.

        ``magzero`` rides along in ``info``, from where the analysis picks it up
        to convert fitted fluxes into microJansky.
        """
        settings_search = af.SettingsSearch(
            path_prefix=(
                Path(sample_name) / dataset_name
                if sample_name is not None
                else Path(dataset_name)
            ),
            unique_tag=unique_tag,
            info={"magzero": magzero},
            session=None,
        )

        """
        __Analysis__

        The pipeline's ``AnalysisImaging``, so each band's fit computes the
        Euclid latent catalogue — the lens, lensed-source and source fluxes in
        µJy and the four matched aperture fluxes — which is the actual product
        of this script.

        ``title_prefix`` labels every figure with the band so the outputs are
        not interchangeable at a glance, and ``skip_rgb_plot=True`` suppresses
        the RGB subplot: it is built from the dataset's colour thumbnails and is
        identical for every band, so it is worth plotting once with the VIS fit
        rather than again in every band's output folder.
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

        The one thing that is genuinely free. A ``DatasetModel`` extends the
        model with a (y, x) offset between this band's grid and the grid the VIS
        model was fitted on, given a uniform prior of ±0.2" — two pixels at
        0.1"/pixel.

        It is needed because astrometric registration between instruments is
        rarely perfect at the precision a lens model works to, and an
        unmodelled shift of even a fraction of a pixel would be absorbed as
        residuals around the lens centre and arcs, biasing the very fluxes this
        fit is measuring. Two parameters is a cheap price for removing that.
        """
        dataset_model = af.Model(al.DatasetModel)
        dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
            lower_limit=-0.2, upper_limit=0.2
        )
        dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
            lower_limit=-0.2, upper_limit=0.2
        )

        """
        __Model: VIS Lens Model, Fixed__

        Lens light, mass, shear and source light are all taken from
        ``vis_result.instance`` — fitted values, not priors, so the sampler
        cannot move any of them. Combined with the two offset parameters above,
        the whole model has two non-linear dimensions, which is why a band
        finishes in a fraction of the time the VIS fit took.

        What is *not* frozen is brightness. Both bulges are linear light
        profiles, so their intensities never were sampled parameters; they are
        solved by linear algebra against whichever data the analysis is given.
        Handing the same shapes to this band's data therefore produces this
        band's fluxes, and doing that across every band produces an SED.
        """
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(
                    al.Galaxy,
                    redshift=vis_result.instance.galaxies.lens.redshift,
                    bulge=vis_result.instance.galaxies.lens.bulge,
                    mass=vis_result.instance.galaxies.lens.mass,
                    shear=vis_result.instance.galaxies.lens.shear,
                ),
                source=af.Model(
                    al.Galaxy,
                    redshift=vis_result.instance.galaxies.source.redshift,
                    bulge=vis_result.instance.galaxies.source.bulge,
                ),
            ),
            dataset_model=dataset_model,
        )

        """
        __Search__

        Nautilus again, but a much smaller one than the VIS fit: ``n_live=75``
        is ample for a two-parameter space, and ``n_like_max=50000`` stops a
        band that refuses to converge from stalling the rest of the sample's SED
        chain. ``iterations_per_quick_update`` sets how often the on-the-fly
        figures are refreshed while it runs.
        """
        search = af.Nautilus(
            name=dataset_waveband,
            **settings_search.search_dict,
            n_live=75,
            batch_size=50,
            iterations_per_quick_update=iterations_per_quick_update,
            n_like_max=50000,
        )

        search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


if __name__ == "__main__":
    (
        sample_name,
        dataset_name,
        iterations_per_quick_update,
        number_of_cores,
        use_cpu,
        stage,
    ) = util.parse_fit_args()

    """
    __Running Standalone__

    Run directly, this script is self-contained: it produces the VIS result it
    needs, then fits every other band against it. The arguments are the ones
    every pipeline here takes — ``--dataset``, ``--sample``,
    ``--iterations_per_quick_update`` and the rest. ``mask_radius`` is not among
    them; it always comes from the dataset's ``info.json``.

    Both stages write under the same ``initial_lens_model`` tag, VIS as
    ``vis_lp`` and each other band under its own name. For the SED chain proper
    — where the bands are fitted against a Sersic VIS model instead — run
    ``scripts/sersic_lens_model_waveband.py``.
    """
    # `stage="vis_lp"` is forced (the `--stage` flag is not consulted here):
    # the multi-band model takes `vis_result.instance.galaxies.source.bulge`,
    # which only exists on the `vis_lp` result — `vis_pix` replaces the source
    # bulge with a pixelization.
    vis_lp_result = fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
        number_of_cores=number_of_cores,
        use_cpu=use_cpu,
        stage="vis_lp",
    )

    fit_waveband(
        dataset_name=dataset_name,
        unique_tag="initial_lens_model",
        vis_result=vis_lp_result,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
    )
