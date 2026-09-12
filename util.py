from astropy.io import fits
from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple
import logging

import matplotlib.pyplot as plt
from autolens import conf as _conf
from autolens import output_to_json

import autofit as af
import autolens as al
import autolens.plot as aplt


def _find_local_maxima(flux: np.ndarray) -> List[tuple]:
    """
    Return ``(value, row, col)`` for every interior local maximum, sorted descending.

    A pixel is a local maximum if it is strictly brighter than its four
    orthogonal neighbours. Border pixels are skipped.
    """
    ny, nx = flux.shape
    maxima = []
    for r in range(1, ny - 1):
        for c in range(1, nx - 1):
            v = flux[r, c]
            if (
                v > flux[r - 1, c]
                and v > flux[r + 1, c]
                and v > flux[r, c - 1]
                and v > flux[r, c + 1]
            ):
                maxima.append((float(v), r, c))
    maxima.sort(reverse=True)
    return maxima


def _pixel_to_arcsec(
    row: int, col: int, ny: int, nx: int, pixel_scale: float
) -> List[float]:
    """
    Convert ``(row, col)`` to **PyAutoLens** ``[y, x]`` arcsec using the half-pixel
    offset convention.
    """
    y = (ny / 2 - 0.5 - row) * pixel_scale
    x = (col - nx / 2 + 0.5) * pixel_scale
    return [y, x]


def _compute_positions_from_source_flux(
    source_flux: np.ndarray,
    noise_map: Optional[np.ndarray],
    pixel_scale: float,
    n_positions: int = 4,
) -> List[List[float]]:
    """
    Compute up to *n_positions* multiple-image positions from a source flux map.

    Mirrors the logic in ``preprocess/segmentation.py``, which is the canonical
    writer of ``positions.json``. This function is the fallback used by
    `load_vis_dataset` when no ``positions.json`` is present but the dataset
    ships a ``segmentation/source_flux.fits`` map.

    Local maxima above a signal-to-noise threshold of 3.0 are taken as candidate
    multiple images. If none of the selected positions lies on the opposite side
    of the lens from the brightest one (i.e. there is no counter-image), the
    threshold is walked down in steps of 0.1 until a counter-image is found,
    which then replaces the weakest position.

    Parameters
    ----------
    source_flux
        2D source flux map (native shape).
    noise_map
        2D noise map of the same shape, or ``None`` to skip signal-to-noise
        filtering.
    pixel_scale
        Pixel scale in arcsec used to convert pixel indices to arcsec.
    n_positions
        Maximum number of positions returned.

    Returns
    -------
    list[list[float]]
        List of ``[y, x]`` arcsec positions, brightest first.
    """
    SNR_THRESHOLD = 3.0
    SNR_STEP = 0.1
    ny, nx = source_flux.shape

    if noise_map is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_map = np.where(noise_map > 0, source_flux / noise_map, 0.0)
    else:
        snr_map = None

    all_maxima = _find_local_maxima(source_flux)
    maxima = [
        (v, r, c)
        for v, r, c in all_maxima
        if snr_map is None or snr_map[r, c] > SNR_THRESHOLD
    ]

    if not maxima:
        return []

    selected = maxima[:n_positions]
    positions = [_pixel_to_arcsec(r, c, ny, nx, pixel_scale) for _, r, c in selected]

    has_counter = any(
        p[0] * positions[0][0] < 0 or p[1] * positions[0][1] < 0 for p in positions[1:]
    )
    if not has_counter and snr_map is not None:
        threshold = SNR_THRESHOLD - SNR_STEP
        while threshold >= 0:
            lower_maxima = sorted(
                [(v, r, c) for v, r, c in all_maxima if snr_map[r, c] > threshold],
                reverse=True,
            )
            for v, r, c in lower_maxima:
                candidate = _pixel_to_arcsec(r, c, ny, nx, pixel_scale)
                if candidate not in positions and (
                    candidate[0] * positions[0][0] < 0
                    or candidate[1] * positions[0][1] < 0
                ):
                    positions[-1] = candidate
                    break
            else:
                threshold -= SNR_STEP
                continue
            break

    return positions


def subplot_rgb(
    arrays: List[al.Array2DRGB],
    titles: Optional[List[str]] = None,
    output_path=None,
    output_filename: str = "rgb",
    output_format: str = "png",
) -> None:
    """
    __RGB Subplot__

    Plot a list of `Array2DRGB` objects as a grid of subplots and save to disk.

    This is the Euclid-specific RGB subplot function. It uses the new `aplt.plot_array`
    function which detects RGB arrays and skips colormap / colorbar handling automatically.

    Parameters
    ----------
    arrays
        List of `Array2DRGB` objects to plot as individual subplot panels.
    titles
        Optional list of panel title strings, one per array. Defaults to empty strings.
    output_path
        Directory path to save the figure. ``None`` calls ``plt.show()`` instead.
    output_filename
        Base filename (without extension) for the output file.
    output_format
        Output file format, e.g. ``"png"``.
    """
    from autoarray.plot.utils import (
        subplot_save,
        conf_subplot_figsize,
        hide_unused_axes,
    )

    n = len(arrays)
    if n == 0:
        return

    try:
        shape_map = _conf.instance["visualize"]["general"]["subplot_shape"]
        for key in sorted(shape_map.keys(), key=lambda k: int(k)):
            if n <= int(key):
                shape_str = shape_map[key]
                nrows, ncols = eval(shape_str)
                break
        else:
            import math

            ncols = math.ceil(math.sqrt(n))
            nrows = math.ceil(n / ncols)
    except Exception:
        import math

        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

    figsize = conf_subplot_figsize(nrows, ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, array in enumerate(arrays):
        title = titles[i] if titles is not None and i < len(titles) else ""
        aplt.plot_array(array, ax=axes[i], title=title)

    hide_unused_axes(axes)
    subplot_save(fig, output_path or "", output_filename, output_format)


def ab_mag_via_flux_from(flux, magzero, xp=np):
    """
    Convert image flux values (ADU) into calibrated astronomical AB magnitudes.

    This uses the standard relation:
        m_AB = -2.5 log10(flux) + magzero

    `flux` and `magzero` must be in consistent units. Euclid VIS, NISP and EXT data are
    typically in different units (E.g. VIS is ADU / second, NISP and EXT are
    electrons / second). However, because `magzero` is also in these units, this
    function does not need to know the specific units of `flux`.

    Parameters
    ----------
    flux : float or xp.ndarray
        Measured flux value(s) in image units (ADU). Must be strictly positive.
    magzero : float
        Photometric zero-point defining the AB magnitude system for the image.

    Returns
    -------
    ab_mag : float or xp.ndarray
        The corresponding AB magnitude(s).
    """
    ab_mag = -2.5 * xp.log10(flux) + magzero
    return ab_mag


def flux_mujy_via_ab_mag_from(ab_mag, xp=np):
    """
    Convert AB magnitudes into flux density expressed in microJansky (µJy).

    This uses the AB definition where a source with 0 mag has a flux of 3631 Jy.

    Parameters
    ----------
    ab_mag : float or array-like
        AB magnitude value(s).

    Returns
    -------
    flux_mujy : float or xp.ndarray
        Flux densities in microJansky (µJy).
    """
    flux_mujy = 3631e6 * 10 ** (-0.4 * ab_mag)
    return flux_mujy


def aperture_flux_from(image_2d, centre, radius_pixels, xp=np):
    """
    Measure enclosed flux inside a single circular aperture on an image.

    Given an image and a central coordinate (typically the lens centre),
    compute the total pixel flux within a circular aperture of specified radius.

    Parameters
    ----------
    image_2d : 2D xp.ndarray
        Image data (NumPy or JAX array).
    centre : (float, float)
        (y, x) coordinate defining the centre of the aperture in pixel units.
    radius_pixels : float
        Aperture radius in pixel units.
    xp : array module, optional
        Array namespace (default: numpy). Can also be jax.numpy.

    Returns
    -------
    float
        Total flux inside the circular aperture.
    """
    y0, x0 = centre
    yy, xx = xp.indices(image_2d.shape)

    rr = xp.sqrt((yy - y0) ** 2 + (xx - x0) ** 2)

    # mask = 1 inside aperture, 0 outside (ensures JAX-safety)
    mask = (rr <= radius_pixels).astype(image_2d.dtype)

    # Equivalent to summing only those inside aperture, but no boolean indexing
    return xp.sum(image_2d * mask)


def psf_fwhm_arcsec_from_primary_header(header, dataset_name: str) -> float:
    """
    Read the worst-band PSF FWHM, in arcsec, from a primary FITS header.

    The Euclid cut-out generator stamps the worst-seeing band's PSF FWHM under
    one of three keys, in order of preference: ``WORST_PSF_MER`` (the OU-MER
    measured value), ``WORST_PSF_HDR`` (the cut-out pipeline's value) and
    ``WORST_PSF``. A value of ``-99`` is the Euclid "not measured" sentinel and
    is skipped.

    Raises
    ------
    ValueError
        If all three keys are missing or hold the sentinel. This deliberately
        refuses to silently fall back: the aperture-flux latent variables are
        computed at multiples of this FWHM, so a wrong value silently corrupts
        the matched-aperture photometry used for SED fitting.
    """
    for key in ("WORST_PSF_MER", "WORST_PSF_HDR", "WORST_PSF"):
        value = header.get(key, None)
        if value is None:
            continue
        value = float(value)
        if value >= -98:
            return value

    raise ValueError(
        f"{dataset_name}: WORST_PSF_MER / WORST_PSF_HDR / WORST_PSF all missing "
        "in primary header — refusing to silently fall back; aperture-flux "
        "latent variables would be wrong. (Test-run setting.)"
    )


def dataset_instrument_hdu_dict_via_fits_from(
    dataset_path, dataset_fits_name, image_tag: str = "_FLUX"
):
    """
    Load a dictionary mapping dataset instruments (e.g. DES_g, NIR_Y) to their index in a multi-extension
    fits file.

    Parameters
    ----------
    dataset_path
        The path where the multi-extension fits file is stored.
    dataset_fits_name
        The name of the multi-extension fits file.
    image_tag
        The tag appended to the instrument name of the image HDU, e.g. _FLUX, _IMAGE, which is used to pick
        out the image HDUs from the fits file and ignore other HDUs like noise maps or PSFs.

    Returns
    -------
    A dictionary mapping dataset names to their index in the fits file.
    """
    hdu_list = fits.open(dataset_path / dataset_fits_name)

    # Build dictionary: {name: index}
    hdu_dict = {}
    for i, hdu in enumerate(hdu_list):
        name = hdu.name if hdu.name else ("PRIMARY" if i == 0 else f"UNNAMED_{i}")
        hdu_dict[name] = i

    instrument_dict = {}
    counter = 0

    for hdu in hdu_list:
        name = hdu.name
        if name.endswith(image_tag):
            band = name.replace(image_tag, "").lower()
            instrument_dict[band] = counter
            counter += 1

    return instrument_dict


class VisualizerImaging(al.VisualizerImaging):
    """
    __RGB Visualizer__

    In built into **PyAutoLens** are `Visualizer` objects that output images of the dataset, fit, tracer and other
    quantities to hard-disk.

    These images for in the `image` folder of th modeling results. They are used for quick inspection of the fit and
    by the workflow functionality to produce new images of the results quickly.

    However, the source code visualizers cannot access quantities that are outside the inputs of the source-code,
    such as the RGB images of the dataset.

    The API below shows how a custom visualizer can be created that can access these quantities, output them to
    hard-disk in the modeling folder results and in the workflow examples are used to produce new images of the results.
    """

    @staticmethod
    def visualize_before_fit(
        analysis,
        paths: af.AbstractPaths,
        model: af.AbstractPriorModel,
    ):
        """
        PyAutoFit calls this function immediately before the non-linear search begins.

        It visualizes objects which do not change throughout the model fit like the dataset.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """
        skip_rgb_plot = analysis.kwargs.get("skip_rgb_plot", False)
        if skip_rgb_plot:
            return

        dataset = analysis.dataset
        dataset_main_path = analysis.kwargs["dataset_main_path"]

        visualizer = al.VisualizerImaging()

        visualizer.visualize_before_fit(
            analysis=analysis,
            paths=paths,
            model=model,
        )

        # Load the images. DR1 tile dumps ship `.jpg` thumbnails, earlier
        # datasets `.png`, so try each extension in turn.
        def _open_rgb(stem):
            for ext in (".png", ".jpg", ".jpeg"):
                path = dataset_main_path / f"{stem}{ext}"
                if path.exists():
                    return np.array(Image.open(path))
            return None

        img0 = _open_rgb("rgb_0")
        img1 = _open_rgb("rgb_1")
        if img0 is None or img1 is None:
            return

        mask = al.Mask2D.all_false(
            shape_native=(img0.shape[0], img0.shape[1]),
            pixel_scales=dataset.pixel_scales,
            origin=dataset.mask.origin,
        )

        img0 = al.Array2DRGB(values=img0, mask=mask)
        img1 = al.Array2DRGB(values=img1, mask=mask)

        mask_rgb = al.Mask2D.circular(
            shape_native=(img0.shape[0], img0.shape[1]),
            pixel_scales=dataset.pixel_scales,
            radius=dataset.mask.circular_radius,
            origin=dataset.mask.origin,
        )

        img0_masked = al.Array2DRGB(values=img0, mask=mask_rgb)
        img1_masked = al.Array2DRGB(values=img1, mask=mask_rgb)

        subplot_rgb(
            arrays=[img0, img1, img0_masked, img1_masked],
            titles=["RGB 0", "RGB 1", "RGB 0 Masked", "RGB 1 Masked"],
            output_path=paths.image_path,
            output_filename="rgb",
            output_format="png",
        )


class LatentEuclid(al.LatentLens):
    """
    Euclid latent catalogue: the PyAutoLens library latents (config-enabled via
    this pipeline's ``config/latent.yaml``, dispatched through
    ``autolens.analysis.latent.LATENT_FUNCTIONS``) plus four Euclid-only FWHM
    aperture-flux µJy latents. Declared on the pipeline ``AnalysisImaging`` as
    ``Latent = LatentEuclid`` (mirrors ``Visualizer``).

    The aperture latents stay here because they need pipeline-specific kwargs
    (``psf_lowest_resolution`` / ``psf_lowest_resolution_fwhm``) that don't
    belong in PyAutoLens. With the static ``Latent`` API the composition is
    explicit (no MRO gymnastics), and the ``FitImaging`` is still built once
    and reused for both the library and aperture blocks.
    """

    APERTURE_LATENT_KEYS = [
        "total_lens_flux_1_fwhm_mujy",
        "total_lens_flux_2_fwhm_mujy",
        "total_lens_flux_3_fwhm_mujy",
        "total_lens_flux_4_fwhm_mujy",
    ]

    @staticmethod
    def keys(analysis):
        from autolens.analysis.latent import latent_keys_enabled

        return list(latent_keys_enabled()) + LatentEuclid.APERTURE_LATENT_KEYS

    @staticmethod
    def variables(analysis, parameters, model):
        """
        Library latent values (config-enabled subset of ``LATENT_FUNCTIONS``,
        in ``latent_keys_enabled()`` order) followed by the four FWHM
        aperture-flux µJy values, computed on the lens-galaxy image convolved
        with ``psf_lowest_resolution`` and centred at its brightest pixel. The
        ``FitImaging`` is built once and shared. Returns a tuple positionally
        aligned with :meth:`keys`.

        The four aperture values are **matched-aperture** photometry: the lens
        image is convolved to the resolution of the worst-seeing band across
        all MER bands (``psf_lowest_resolution``, selected by the dataset's
        ``WORST_BAND`` header key) and the flux is measured inside circular
        apertures of 1, 2, 3 and 4 times that band's PSF FWHM — hence the
        ``total_lens_flux_{1,2,3,4}_fwhm_mujy`` keys. Matching the aperture to
        the worst band is what makes the per-band fluxes comparable, and so
        usable for SED fitting.

        If the analysis carries no worst-band PSF (``WORST_BAND`` absent from
        the dataset, or naming a band not in the cut-out) the four values are
        NaN and are dropped from the written latent summary; the fit itself is
        unaffected.

        The instance is built with ``latent_instance_from`` rather than
        ``model.instance_from_vector``, because this method overrides
        ``LatentLens.variables`` and so must build its own: the latent engine
        evaluates it inside a per-sample ``jax.jit``, where checking the
        ``vis_lp`` ordered-MGE-bases assertion applies a Python ``not`` to a
        traced boolean and raises ``TracerBoolConversionError``, which the
        engine turns into a NaN row for every sample — no latent output at all
        (PyAutoLens#732).
        """
        from autolens.analysis.latent import (
            LATENT_FUNCTIONS,
            latent_instance_from,
            latent_keys_enabled,
        )

        xp = analysis._xp
        magzero = analysis.kwargs.get("magzero", None)

        instance = latent_instance_from(model=model, parameters=parameters, xp=xp)
        fit = analysis.fit_from(instance=instance)
        context = {"fit": fit, "magzero": magzero, "xp": xp}

        library_keys = latent_keys_enabled()
        library_value_dict = {k: LATENT_FUNCTIONS[k](**context) for k in library_keys}
        library_value_dict.update(
            LatentEuclid._source_flux_latents_on_uniform_grid(
                fit=fit, magzero=magzero, keys=library_keys, xp=xp
            )
        )
        library_values = tuple(library_value_dict[k] for k in library_keys)

        try:
            image = fit.galaxy_image_dict[fit.tracer.galaxies[0]]
            image_native = analysis.to_ndarray_2d(image=image, xp=xp)

            flat_index = xp.argmax(image_native)
            y, x = xp.unravel_index(flat_index, image.shape_native)

            psf_lowest_resolution = analysis.kwargs["psf_lowest_resolution"]
            psf_lowest_resolution_fwhm = analysis.kwargs["psf_lowest_resolution_fwhm"]

            image_convolved_to_lowest = psf_lowest_resolution.convolved_image_from(
                image=image, blurring_image=None, xp=xp
            )
            image_convolved_to_lowest_native = analysis.to_ndarray_2d(
                image=image_convolved_to_lowest, xp=xp
            )

            radius = psf_lowest_resolution_fwhm / (0.1 * 2.0)
            aperture_values = tuple(
                flux_mujy_via_ab_mag_from(
                    ab_mag=ab_mag_via_flux_from(
                        flux=aperture_flux_from(
                            image_2d=image_convolved_to_lowest_native,
                            centre=(y, x),
                            radius_pixels=radius * multiplier,
                            xp=xp,
                        ),
                        magzero=magzero,
                        xp=xp,
                    ),
                    xp=xp,
                )
                for multiplier in (1.0, 2.0, 3.0, 4.0)
            )
        except (AttributeError, KeyError):
            aperture_values = (xp.nan, xp.nan, xp.nan, xp.nan)

        return library_values + aperture_values

    # Keys re-evaluated by `_source_flux_latents_on_uniform_grid`: the two
    # unlensed source-flux latents and the magnification built from them.
    SOURCE_FLUX_LATENT_KEYS = [
        "total_source_flux",
        "total_source_flux_mujy",
        "magnification",
    ]

    @staticmethod
    def _source_flux_latents_on_uniform_grid(fit, magzero, keys, xp):
        """
        Re-evaluate the *unlensed* source-flux latents on a uniform
        over-sample-4 version of the fit's masked grid, and rebuild
        ``magnification`` from them. Returns a ``{key: value}`` dict covering
        whichever of :attr:`SOURCE_FLUX_LATENT_KEYS` are enabled; every other
        latent is left on the production light-profile grid.

        Why. ``load_vis_dataset`` gives the fit a *lens-centred radial*
        over-sampling map (4x4 within 0.3" of the lens centre, 2x2 beyond).
        That map is built for the lens light and for the arcs, and both are
        measured correctly on it. The unlensed source, however, is a compact
        profile whose wings run straight out into the sub-size-2 outer bin,
        where they are under-integrated — so ``total_source_flux`` depends on
        where the radial bins happen to fall relative to the source, and
        ``magnification`` (a lensed / unlensed ratio whose numerator is an
        image-plane sum over the arcs, and so unaffected) inherits the whole
        error. With the production ``[4, 4, 2]`` bins the arcs agree with the
        simulator to +0.03 % while the magnification was still 0.21 % out
        (autolens_profiling#235).

        The reference is therefore the grid the simulator integrates its own
        truth fluxes on: the same mask, uniform ``over_sample_size=4``
        (``scripts/simulator.py`` builds
        ``al.Grid2D.uniform(..., over_sample_size=4)`` and sums
        ``source_galaxy.image_2d_from`` on it). This touches no modelling
        stage and no other latent — only the reference grid of the unlensed
        source-flux integral.

        The pixelized (inversion) term is unchanged: it is integrated on the
        source mesh, not on any image-plane over-sampling map, so it is read
        back from the library helper as-is.
        """
        from autolens.analysis.latent import (
            LATENT_FUNCTIONS,
            _pixelized_source_flux,
            ab_mag_via_flux_from as library_ab_mag_via_flux_from,
            flux_mujy_via_ab_mag_from as library_flux_mujy_via_ab_mag_from,
        )

        wanted = [key for key in LatentEuclid.SOURCE_FLUX_LATENT_KEYS if key in keys]
        if not wanted:
            return {}

        try:
            tracer = fit.tracer_linear_light_profiles_to_light_profiles
            # The reference grid is built on NumPy, without ``xp``: the mask is
            # static, and under JAX ``Grid2D.from_mask`` reaches ``jnp.nonzero``,
            # which needs a size known at trace time -> ``ConcretizationTypeError``
            # inside the latent engine's per-sample ``jax.jit`` (PyAutoLens#732).
            # The traced quantity is the source image evaluated on it, so ``xp``
            # stays on ``image_2d_from`` below.
            grid_uniform = al.Grid2D.from_mask(
                mask=fit.dataset.grids.lp.mask, over_sample_size=4
            )
            source_image = tracer.galaxies[-1].image_2d_from(grid=grid_uniform, xp=xp)
        except (AttributeError, IndexError):
            return {key: xp.nan for key in wanted}

        source_flux = xp.sum(source_image.array) + _pixelized_source_flux(
            fit=fit, xp=xp
        )

        values = {}
        if "total_source_flux" in wanted:
            values["total_source_flux"] = source_flux

        # The uJy latents keep the library's magzero contract: without a
        # zero-point they are NaN, which is what `LATENT_FUNCTIONS` already
        # returned (with its one-time warning), so leave those values alone.
        if magzero is None:
            return values

        source_flux_mujy = library_flux_mujy_via_ab_mag_from(
            ab_mag=library_ab_mag_via_flux_from(
                flux=source_flux, magzero=magzero, xp=xp
            ),
            xp=xp,
        )
        if "total_source_flux_mujy" in wanted:
            values["total_source_flux_mujy"] = source_flux_mujy
        if "magnification" in wanted:
            values["magnification"] = (
                LATENT_FUNCTIONS["total_lensed_source_flux_mujy"](
                    fit=fit, magzero=magzero, xp=xp
                )
                / source_flux_mujy
            )

        return values


class AnalysisImaging(al.AnalysisImaging):
    """
    Sets the custom RGB visualizer ensuring the RGB subplot is output, and
    declares the Euclid latent catalogue (``LatentEuclid`` — library latents
    plus four FWHM aperture-flux latents).
    """

    Visualizer = VisualizerImaging
    Latent = LatentEuclid

    def to_ndarray_2d(self, image, xp):

        array_2d = xp.zeros(image.mask.shape, dtype=image.dtype)

        if xp is np:

            array_2d[image.mask.slim_to_native_tuple] = image.array

        else:

            array_2d = array_2d.at[image.mask.slim_to_native_tuple].set(image.array)

        return array_2d

    def save_results(self, paths: af.DirectoryPaths, result):
        """
        At the end of a model-fit, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis it outputs the following:

        - The maximum log likelihood tracer of the fit.
        - ``wcs.json``: where the fitted lens sits on the sky and where its lensed source's multiple
          images fall in the image plane — the record ``wcs_dict_from`` builds.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        result
            The result of a model fit, including the non-linear search, samples and maximum likelihood tracer.
        """
        super().save_results(paths=paths, result=result)

        try:
            fit = result.max_log_likelihood_fit
        except Exception as e:  # noqa: BLE001 — a finished search outlives its record
            logging.getLogger(__name__).warning(
                "wcs.json: the maximum log likelihood fit could not be built, so a "
                f"pixelized source's clumps are not recorded ({type(e).__name__}: {e})"
            )
            fit = None

        wcs_dict = wcs_dict_from(
            tracer=result.max_log_likelihood_tracer,
            data=self.dataset.data,
            pixel_wcs=self.kwargs["pixel_wcs"],
            fit=fit,
        )

        output_to_json(
            obj=wcs_dict,
            file_path=paths._files_path / "wcs.json",
        )


# ---------------------------------------------------------------------------
# The WCS record (files/wcs.json)
# ---------------------------------------------------------------------------

# The `al.PointSolver` settings the lensed source's image-plane positions are
# solved with. They are `scripts/simulator.py`'s (`positions_from_tracer`), so
# the images a fit records for a tracer and the `positions.json` the simulator
# solves for the same tracer agree to `pixel_scale_precision`.
LENSED_SOURCE_PIXEL_SCALE_PRECISION = 0.005
LENSED_SOURCE_MAGNIFICATION_THRESHOLD = 0.1

# The `Inversion.source_clumps_from` settings a pixelized source's clumps are
# found with — the library's own `subplot_mappings` defaults
# (`autoarray/config/visualize/general.yaml`, `inversion:`), restated here so
# the record does not move when a visualization config does. A mesh pixel is in
# a clump if its reconstructed value exceeds `threshold` times the maximum;
# ~0.5 isolates one smooth source, ~0.2 merges two nearby galaxies into one
# clump, ~0.8 splits a source into its star-forming knots.
SOURCE_CLUMP_THRESHOLD = 0.5
SOURCE_CLUMP_MIN_PIXELS = 3
SOURCE_CLUMP_TOTAL = 5

# The pixelized source is always the last plane, mapped by the inversion's one
# mapper (every pipeline model has a single pixelized plane).
SOURCE_CLUMP_MAPPER_INDEX = 0


def source_centre_from(tracer) -> Optional[Tuple[float, float]]:
    """
    The source-plane (y, x) centre of the source galaxy's light, in arcsec, or
    ``None`` when the source carries no light profile with a centre.

    The source galaxy is the tracer's last — the order every pipeline model and
    ``truth.json`` use, and the one ``LatentLens`` indexes by. Its light is the
    first light profile the galaxy holds: a single MGE ``Basis`` in ``vis_lp``,
    whose ``centre`` is the centre its Gaussians share, or a ``Sersic`` in the
    SED chain. A pixelized source (``vis_pix`` and the Delaunay stages) has no
    light profile and so no centre — there is nothing to solve for.
    """
    source_galaxy = tracer.galaxies[-1]

    for profile in source_galaxy.cls_list_from(cls=al.LightProfile):
        centre = getattr(profile, "centre", None)
        if centre is not None:
            return float(centre[0]), float(centre[1])

    return None


def lensed_source_image_positions_from(
    tracer, data, source_centre: Tuple[float, float]
) -> al.Grid2DIrregular:
    """
    The image-plane (y, x) positions, in arcsec, that the tracer maps onto
    ``source_centre``: the lens equation solved with ``al.PointSolver`` on a
    uniform grid spanning the cut-out, exactly as ``scripts/simulator.py``
    solves a mock's ``positions.json``.

    The solver tiles the image plane with triangles, keeps those which trace
    onto the source-plane point and subdivides them down to
    ``LENSED_SOURCE_PIXEL_SCALE_PRECISION`` arcsec, then drops images whose
    magnification is below ``LENSED_SOURCE_MAGNIFICATION_THRESHOLD``. A source
    that is not multiply imaged returns fewer than two positions.
    """
    grid = al.Grid2D.uniform(
        shape_native=data.shape_native,
        pixel_scales=data.pixel_scales,
        origin=data.origin,
    )

    solver = al.PointSolver.for_grid(
        grid=grid,
        pixel_scale_precision=LENSED_SOURCE_PIXEL_SCALE_PRECISION,
        magnification_threshold=LENSED_SOURCE_MAGNIFICATION_THRESHOLD,
    )

    return solver.solve(tracer=tracer, source_plane_coordinate=source_centre)


def pixelized_source_clumps_from(fit) -> List[dict]:
    """
    The bright clumps of a pixelized source's reconstruction, each with its
    source-plane peak and the image-plane pixels it maps to — read off the
    fit's mapper, not solved.

    ``Inversion.source_clumps_from`` thresholds the reconstruction at
    ``SOURCE_CLUMP_THRESHOLD`` times its maximum, splits what is left into
    connected groups over the mesh neighbour graph, drops groups smaller than
    ``SOURCE_CLUMP_MIN_PIXELS`` and keeps the ``SOURCE_CLUMP_TOTAL`` brightest.
    ``Inversion.mappings_from`` then reads each clump's multiple images off the
    mapper's mapping matrix as connected image-plane regions
    (``autoarray.inversion.mappings``, PyAutoArray#517). Per clump this returns:

    - ``peak_y_arcsec`` / ``peak_x_arcsec`` — the source-plane centre of the
      clump's brightest mesh pixel (the peak, not the mean of the clump);
    - ``peak_value`` — the reconstructed value there;
    - ``mesh_pixels`` — how many mesh pixels the clump spans;
    - ``image_y_arcsec`` / ``image_x_arcsec`` — one entry per image region: the
      brightest pixel of the source's *model* image inside that region (what a
      fibre is pointed at; ``ImageRegion.brightest_coordinate_from``), largest
      region first.

    Clumps are ordered brightest first, so the first is the source's main
    structure and the rest are companions or star-forming knots.
    """
    inversion = fit.inversion
    mapper = inversion.cls_list_from(cls=al.Mapper)[SOURCE_CLUMP_MAPPER_INDEX]

    reconstruction = np.asarray(inversion.reconstruction_dict[mapper])
    mesh_grid = np.asarray(mapper.source_plane_mesh_grid)
    model_image = fit.model_images_of_planes_list[-1]

    mappings = inversion.mappings_from(
        mapper_index=SOURCE_CLUMP_MAPPER_INDEX,
        threshold=SOURCE_CLUMP_THRESHOLD,
        min_pixels=SOURCE_CLUMP_MIN_PIXELS,
        total_clumps=SOURCE_CLUMP_TOTAL,
    )

    clumps = []

    for mapping in mappings:
        pix_indexes = np.asarray(mapping.pix_indexes)
        peak_index = int(pix_indexes[int(np.argmax(reconstruction[pix_indexes]))])

        images = [
            region.brightest_coordinate_from(array=model_image)
            for region in mapping.image_regions
            if len(region.slim_indexes) > 0
        ]

        clumps.append(
            {
                "peak_y_arcsec": float(mesh_grid[peak_index, 0]),
                "peak_x_arcsec": float(mesh_grid[peak_index, 1]),
                "peak_value": float(reconstruction[peak_index]),
                "mesh_pixels": int(pix_indexes.shape[0]),
                "image_y_arcsec": [float(y) for y, _ in images],
                "image_x_arcsec": [float(x) for _, x in images],
            }
        )

    return clumps


def wcs_dict_from(tracer, data, pixel_wcs, fit=None) -> dict:
    """
    The record ``AnalysisImaging.save_results`` writes to ``files/wcs.json``:
    where the fitted lens sits on the sky, and where its lensed source's
    multiple images fall.

    Always present:

    - ``crval_ra_deg`` / ``crval_dec_deg`` — the maximum-likelihood lens light
      centre converted to RA / Dec through ``pixel_wcs``. Despite the FITS-style
      name this is not the cut-out's reference pixel;
      ``catalogue/scripts/magnitudes.py`` reads the RA back as its
      ``crval_ra_deg`` label column.
    - ``crpix_x`` / ``crpix_y`` — the 1-based WCS pixel of the image-plane
      origin ``(0, 0)``.
    - ``source_model`` — how the source was located: ``"light_profile"`` (its
      light has a centre — the MGE ``Basis`` of ``vis_lp``, the ``Sersic`` of
      the SED chain), ``"pixelized"`` (a ``Pixelization`` source and a ``fit``
      to read its reconstruction from — ``vis_pix``, the Delaunay stages) or
      ``"none"`` (neither, or no ``fit`` was given for a pixelized source).

    Present when the source was located (``source_model != "none"``):

    - ``source_centre_y_arcsec`` / ``source_centre_x_arcsec`` — the source
      position the lens equation is solved for, in the source plane: the light
      centre (``source_centre_from``), or the peak of the brightest clump of the
      reconstruction.
    - ``lensed_source_image_y_arcsec`` / ``lensed_source_image_x_arcsec`` and
      ``lensed_source_image_ra_deg`` / ``lensed_source_image_dec_deg`` — that
      position's multiple images in the image plane, in arcsec and on the sky,
      one entry per image (``lensed_source_image_positions_from``, the
      ``al.PointSolver`` route). Empty lists mean the solver found no image;
      the four are *absent* if the solver raised, which is logged rather than
      raised so a search that has finished is never lost to its own record.

    Present for a pixelized source whose clumps could be read
    (``source_model == "pixelized"``):

    - ``source_clumps`` — one entry per bright clump of the reconstruction,
      brightest first (``pixelized_source_clumps_from``): its ``peak_*`` in the
      source plane and its ``image_*`` in the image plane read off the fit's
      mapper — the brightest model pixel of each image region — plus the same
      images on the sky as ``image_ra_deg`` / ``image_dec_deg``. Absent if the
      clump finder raised (logged), in which case the solver keys are absent
      too, since there is no peak to solve for. The two routes differ on
      purpose: the solver keys are the sub-pixel lens-equation solution for one
      point, the mapper keys are data pixels of every clump and can merge two
      images into one arc or split one across a critical curve.

    Nothing is ever written as ``null``: PyAutoFit's ``output_to_json`` drops
    ``None``-valued keys on write, so an unavailable value is an absent key,
    and ``source_model`` says why.

    Parameters
    ----------
    tracer
        The maximum log likelihood tracer of the fit; its first galaxy is the
        lens, its last the source.
    data
        The fitted image, whose geometry converts image-plane arcsec to WCS
        pixels and whose extent bounds the solver's grid.
    pixel_wcs
        The dataset's celestial ``astropy.wcs.WCS``, converting those pixels to
        RA / Dec.
    fit
        The maximum log likelihood ``FitImaging``, needed only for a pixelized
        source (its inversion holds the reconstruction and the mapper). ``None``
        records such a source as ``"none"``.

    Notes
    -----
    Every sky value goes through the FITS pixel the light sits in, so it is
    right whichever way the cut-out is oriented. The array is loaded from the
    FITS without a row flip (``autonerves.fitsable``), so its native row 0 is
    FITS row 1 — the bottom row of a standard north-up image — and PyAutoLens's
    positive ``y`` (native row 0 upward on its own plots) is therefore the FITS
    row-1 direction: south, for a north-up ``CD`` matrix. Do not read a
    ``lensed_source_image_y_arcsec`` as "north of the lens"; read the
    ``_dec_deg`` beside it.
    """
    logger = logging.getLogger(__name__)

    def sky_from(scaled_coordinates_2d):
        pixel_y, pixel_x = data.geometry.pixel_coordinates_wcs_2d_from(
            scaled_coordinates_2d=scaled_coordinates_2d
        )
        ra_deg, dec_deg = pixel_wcs.wcs_pix2world(pixel_x, pixel_y, 1)
        return float(ra_deg), float(dec_deg)

    lens_light_centre = tracer.galaxies[0].bulge.centre
    lens_ra_deg, lens_dec_deg = sky_from(lens_light_centre)

    data_centre_wcs_pix_y, data_centre_wcs_pix_x = (
        data.geometry.pixel_coordinates_wcs_2d_from(scaled_coordinates_2d=(0.0, 0.0))
    )

    wcs_dict = {
        "crpix_x": data_centre_wcs_pix_x,
        "crpix_y": data_centre_wcs_pix_y,
        "crval_ra_deg": lens_ra_deg,
        "crval_dec_deg": lens_dec_deg,
        "source_model": "none",
    }

    source_centre = source_centre_from(tracer=tracer)

    if source_centre is not None:
        wcs_dict["source_model"] = "light_profile"

    elif fit is not None and fit.tracer.planes[-1].has(cls=al.Pixelization):
        wcs_dict["source_model"] = "pixelized"

        try:
            clumps = pixelized_source_clumps_from(fit=fit)
        except Exception as e:  # noqa: BLE001 — a finished search outlives its record
            logger.warning(
                "wcs.json: the pixelized source's clumps could not be read off the "
                f"fit's mapper; `source_clumps` is not written ({type(e).__name__}: {e})"
            )
            return wcs_dict

        for clump in clumps:
            sky = [
                sky_from((y, x))
                for y, x in zip(clump["image_y_arcsec"], clump["image_x_arcsec"])
            ]
            clump["image_ra_deg"] = [ra for ra, _ in sky]
            clump["image_dec_deg"] = [dec for _, dec in sky]

        wcs_dict["source_clumps"] = clumps

        if clumps:
            source_centre = (clumps[0]["peak_y_arcsec"], clumps[0]["peak_x_arcsec"])

    if source_centre is None:
        return wcs_dict

    wcs_dict["source_centre_y_arcsec"] = source_centre[0]
    wcs_dict["source_centre_x_arcsec"] = source_centre[1]

    try:
        positions = lensed_source_image_positions_from(
            tracer=tracer, data=data, source_centre=source_centre
        )
    except Exception as e:  # noqa: BLE001 — a finished search outlives its record
        logger.warning(
            "wcs.json: the point solver raised for the source centre "
            f"{source_centre}; the lensed-source image positions are not written "
            f"({type(e).__name__}: {e})"
        )
        return wcs_dict

    positions = [(float(y), float(x)) for y, x in np.asarray(positions).reshape(-1, 2)]
    sky = [sky_from(position) for position in positions]

    wcs_dict["lensed_source_image_y_arcsec"] = [y for y, _ in positions]
    wcs_dict["lensed_source_image_x_arcsec"] = [x for _, x in positions]
    wcs_dict["lensed_source_image_ra_deg"] = [ra for ra, _ in sky]
    wcs_dict["lensed_source_image_dec_deg"] = [dec for _, dec in sky]

    return wcs_dict


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class EuclidDataset:
    """
    Container for all objects produced by loading a Euclid VIS dataset.

    Returned by `load_vis_dataset`; pass attribute access (`d.dataset`,
    `d.magzero`, etc.) into pipelines instead of re-deriving each value.
    """

    dataset: object  # al.Imaging — masked and over-sampled
    dataset_main_path: Path
    dataset_fits_name: str
    dataset_index_dict: dict  # waveband name -> HDU index
    dataset_centre: tuple  # (y, x) of brightest central pixel
    info: dict  # contents of info.json (empty dict if absent)
    header: object  # FITS header of VIS image HDU
    magzero: Optional[float]  # photometric zero-point from header
    pixel_wcs: object  # astropy WCS for sky coordinate conversion
    psf_lowest_resolution: object  # al.Convolver at the worst-seeing band
    psf_lowest_resolution_fwhm: Optional[float]  # FWHM of that PSF in arcsec
    mask_radius: float  # circular mask radius used (arcsec)
    positions_likelihood_list: object  # list[al.PositionsLH] or None


def load_vis_dataset(
    dataset_name: str,
    image_tag: str = "_BGSUB",
    sample_name: str = None,
) -> EuclidDataset:
    """
    Load and prepare a Euclid VIS imaging dataset for lens modeling.

    This function centralises the dataset setup that is common to every
    pipeline: loading the FITS file, reading the header, applying the extra-
    galaxy noise mask, applying the circular analysis mask, setting standard
    over-sampling, and loading the lowest-resolution PSF for aperture
    photometry.

    Parameters
    ----------
    dataset_name
        Name of the dataset subdirectory inside ``dataset/``.  The main FITS
        file is assumed to be ``dataset/<dataset_name>/<dataset_name>.fits``.
    image_tag
        Tag appended to instrument names in the FITS HDU headers to identify
        image HDUs (default ``"_BGSUB"``).

    Returns
    -------
    EuclidDataset
        Dataclass containing the prepared dataset and all associated metadata.

    Notes
    -----
    ``pixel_scale`` and ``mask_radius`` are required fields in the dataset's
    ``info.json`` file.  See the project README for the expected format.
    """
    from astropy.wcs import WCS

    # util.py lives in the project root; dataset/ is a sibling directory.
    project_root = Path(__file__).parent
    if sample_name is not None:
        dataset_main_path = project_root / "dataset" / sample_name / dataset_name
    else:
        dataset_main_path = project_root / "dataset" / dataset_name
    dataset_fits_name = f"{dataset_name}.fits"

    dataset_index_dict = dataset_instrument_hdu_dict_via_fits_from(
        dataset_path=dataset_main_path,
        dataset_fits_name=dataset_fits_name,
        image_tag=image_tag,
    )

    # Some datasets stamp the image HDUs with the other tag; retry before giving up.
    if "vis" not in dataset_index_dict:
        for fallback_tag in ("_FLUX", "_BGSUB"):
            if fallback_tag == image_tag:
                continue
            dataset_index_dict = dataset_instrument_hdu_dict_via_fits_from(
                dataset_path=dataset_main_path,
                dataset_fits_name=dataset_fits_name,
                image_tag=fallback_tag,
            )
            if "vis" in dataset_index_dict:
                break

    vis_index = dataset_index_dict["vis"]

    with open(dataset_main_path / "info.json") as f:
        info = json.load(f)

    pixel_scale = info["pixel_scale"]

    dataset = al.Imaging.from_fits(
        data_path=dataset_main_path / dataset_fits_name,
        data_hdu=vis_index * 3 + 1,
        noise_map_path=dataset_main_path / dataset_fits_name,
        noise_map_hdu=vis_index * 3 + 3,
        psf_path=dataset_main_path / dataset_fits_name,
        psf_hdu=vis_index * 3 + 2,
        pixel_scales=pixel_scale,
        check_noise_map=False,
    )

    # The mask centre is the lens centre. If the dataset ships a segmentation
    # lens flux map its peak is the most reliable estimate; otherwise fall back
    # to `info.json` and finally to the frame centre.
    lens_flux_path = dataset_main_path / "segmentation" / "lens_flux.fits"
    if lens_flux_path.exists():
        lens_flux = fits.getdata(lens_flux_path).astype(np.float32)
        if lens_flux.shape == dataset.shape_native:
            peak_row, peak_col = np.unravel_index(
                int(np.nanargmax(lens_flux)), lens_flux.shape
            )
            ny, nx = lens_flux.shape
            mask_centre = (
                (ny / 2 - 0.5 - peak_row) * pixel_scale,
                (peak_col - nx / 2 + 0.5) * pixel_scale,
            )
        else:
            mask_centre = info.get("mask_centre") or (0.0, 0.0)
    else:
        mask_centre = info.get("mask_centre") or (0.0, 0.0)

    cy, cx = mask_centre

    # Search for the brightest pixel around the lens centre, not the frame
    # centre, so offset lenses are handled correctly.
    dataset_centre = dataset.data.brightest_sub_pixel_coordinate_in_region_from(
        region=(cy - 0.3, cy + 0.3, cx - 0.3, cx + 0.3), box_size=2
    )

    try:
        header = al.header_obj_from(
            file_path=dataset_main_path / dataset_fits_name,
            hdu=vis_index * 3 + 1,
        )
        magzero = header.get("MAGZERO", None)
    except FileNotFoundError:
        header = None
        magzero = None

    pixel_wcs = WCS(header).celestial if header is not None else None

    # Noise-scaling mask. The DR1 preprocessing writes
    # `segmentation/artefact_binary.fits`; older datasets (including the shipped
    # example) ship `mask_extra_galaxies.fits`. Try both, in that order.
    for noise_mask_path in (
        dataset_main_path / "segmentation" / "artefact_binary.fits",
        dataset_main_path / "mask_extra_galaxies.fits",
    ):
        try:
            mask_extra_galaxies = al.Mask2D.from_fits(
                file_path=noise_mask_path,
                pixel_scales=pixel_scale,
                invert=True,
            )
        except FileNotFoundError:
            continue
        # A mask cut out at a different size cannot be applied to this dataset.
        if mask_extra_galaxies.shape_native == dataset.shape_native:
            dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
        break

    mask_radius = info["mask_radius"]

    # Clamp the mask radius to the frame: for an offset lens an `info.json`
    # radius can run the circular mask off the edge of the cut-out.
    ny, nx = dataset.shape_native
    half_y = ny / 2 * pixel_scale
    half_x = nx / 2 * pixel_scale
    max_radius = min(half_x - abs(cx), half_y - abs(cy))
    if mask_radius > max_radius:
        mask_radius = round(max_radius, 6)

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
        centre=mask_centre,
    )
    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        # [4,4,2] since 2026-09-08 (autolens_profiling#235): sub-size 1 causes gradient issues,
        # and sub-size 2 in the 0.1-0.3" annulus under-integrates a compact source by ~0.6 % (magnification cross-check).
        sub_size_list=[4, 4, 2],
        radial_list=[0.1, 0.3],
        centre_list=[dataset_centre],
    )
    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

    # Lowest-resolution PSF across all MER bands — used for aperture photometry.
    header_primary = al.header_obj_from(
        file_path=dataset_main_path / dataset_fits_name,
        hdu=0,
    )

    # `WORST_BAND` / `WORST_PSF_*` are an *input contract* on the dataset: they
    # are stamped by the upstream Euclid cut-out generator, and neither this
    # pipeline nor PyAutoReduce writes them.
    #
    # `WORST_BAND` names the worst-seeing band across all MER bands (e.g.
    # `DES_G`); lower-cased it indexes the HDU list to find that band's PSF,
    # which the aperture-flux latents are convolved to. The FWHM itself comes
    # from `WORST_PSF_MER` / `WORST_PSF_HDR` / `WORST_PSF` (see
    # `psf_fwhm_arcsec_from_primary_header`).
    #
    # Two degradation paths, deliberately different:
    #
    #  - `WORST_BAND` missing, or naming a band absent from the cut-out: warn
    #    and skip the four aperture latents (they come out NaN). The fit itself
    #    is unaffected.
    #  - `WORST_BAND` present but every FWHM key missing or `-99`: raise. The
    #    aperture radii are multiples of that FWHM, so a guessed value would
    #    silently corrupt the photometry rather than fail.
    worst_band = header_primary.get("WORST_BAND", None)
    if worst_band is not None:
        lowest_resolution_waveband = worst_band.lower()
        lowest_resolution_waveband_index = dataset_index_dict.get(
            lowest_resolution_waveband, None
        )
        if lowest_resolution_waveband_index is None:
            print(
                f"[WARN] {dataset_name}: WORST_BAND={worst_band} not present in dataset "
                "HDU list — skipping aperture-flux latent variables.",
                flush=True,
            )
            psf_lowest_resolution = None
            psf_lowest_resolution_fwhm = None
        else:
            psf_lowest_resolution = al.Convolver.from_fits(
                file_path=dataset_main_path / dataset_fits_name,
                hdu=lowest_resolution_waveband_index * 3 + 2,
                pixel_scales=pixel_scale,
                normalize=True,
            )
            psf_lowest_resolution_fwhm = psf_fwhm_arcsec_from_primary_header(
                header=header_primary,
                dataset_name=dataset_name,
            )
    else:
        print(
            f"[WARN] {dataset_name}: WORST_BAND missing in primary header — "
            "skipping aperture-flux latent variables.",
            flush=True,
        )
        psf_lowest_resolution = None
        psf_lowest_resolution_fwhm = None

    try:
        positions = al.Grid2DIrregular(
            values=al.from_json(file_path=dataset_main_path / "positions.json")
        )
        # A single position is not a multiple-image constraint; treat as absent.
        if len(positions) == 1:
            raise FileNotFoundError
        positions_likelihood_list = [al.PositionsLH(threshold=0.2, positions=positions)]
    except FileNotFoundError:
        # No `positions.json`: derive positions from the segmentation source
        # flux map and the VIS noise map, mirroring `preprocess/segmentation.py`.
        source_flux_path = dataset_main_path / "segmentation" / "source_flux.fits"
        if (
            source_flux_path.exists()
            and fits.getdata(source_flux_path).shape == dataset.shape_native
        ):
            source_flux = fits.getdata(source_flux_path).astype(np.float32)
            try:
                noise_map = fits.getdata(
                    dataset_main_path / dataset_fits_name,
                    ext=vis_index * 3 + 3,
                ).astype(np.float32)
            except Exception:
                noise_map = None
            pos_list = _compute_positions_from_source_flux(
                source_flux=source_flux,
                noise_map=noise_map,
                pixel_scale=pixel_scale,
            )
            if len(pos_list) >= 2:
                positions = al.Grid2DIrregular(values=pos_list)
                positions_likelihood_list = [
                    al.PositionsLH(threshold=0.2, positions=positions)
                ]
            else:
                positions_likelihood_list = None
        else:
            positions_likelihood_list = None

    return EuclidDataset(
        dataset=dataset,
        dataset_main_path=dataset_main_path,
        dataset_fits_name=dataset_fits_name,
        dataset_index_dict=dataset_index_dict,
        dataset_centre=dataset_centre,
        info=info,
        header=header,
        magzero=magzero,
        pixel_wcs=pixel_wcs,
        psf_lowest_resolution=psf_lowest_resolution,
        psf_lowest_resolution_fwhm=psf_lowest_resolution_fwhm,
        mask_radius=mask_radius,
        positions_likelihood_list=positions_likelihood_list,
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_fit_args(with_seed: bool = False):
    """
    Parse the standard command-line arguments shared by all pipeline scripts.

    Parameters
    ----------
    with_seed
        If True, also parse ``--seed`` and return it as a seventh tuple element.
        Off by default so that every script which does not take a seed keeps the
        six-tuple it already unpacks; ``scripts/initial_lens_model.py`` is the one
        caller that passes True.

    Returns
    -------
    (sample_name, dataset_name, iterations_per_quick_update, number_of_cores,
     use_cpu, stage) or, when ``with_seed=True``, that tuple with ``seed``
    appended
        ``stage`` is one of ``"all"``, ``"vis_lp"`` or ``"vis_pix"``, and
        selects which of the two searches in ``scripts/initial_lens_model.py``
        the run performs. ``mask_radius`` is always read from the dataset's
        ``info.json``.

        ``seed`` is the Nautilus random seed, an ``int`` or ``None`` when the
        flag is not given (unseeded, the default).

        The six-tuple's last element used to be the boolean ``skip_pix``.
        ``--skip_pix`` is still accepted as a deprecated alias for
        ``--stage vis_lp`` — it emits a deprecation line on stderr and resolves
        to the string ``"vis_lp"``, so the tuple length is unchanged.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="PyAutoLens Euclid Pipeline")
    parser.add_argument(
        "--sample",
        metavar="name",
        required=False,
        default=None,
        help="Sample subdirectory inside dataset/ containing the dataset.",
    )
    parser.add_argument(
        "--dataset",
        metavar="name",
        required=True,
        help="Name of the dataset subdirectory inside dataset/<sample>/.",
    )
    parser.add_argument(
        "--iterations_per_quick_update",
        metavar="int",
        required=False,
        default=5000,
        help="Number of sampler iterations between on-the-fly visualisation updates.",
    )
    parser.add_argument(
        "--number_of_cores",
        metavar="int",
        required=False,
        default=1,
        help="Number of CPU cores for non-JAX Nautilus searches (e.g. vis_pix).",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        default=False,
        help="CPU mode: disables JAX and applies the sparse operator for vis_pix.",
    )
    parser.add_argument(
        "--stage",
        metavar="name",
        required=False,
        choices=["all", "vis_lp", "vis_pix"],
        default=None,
        help=(
            "Which of the two searches to run: 'all' (default) runs vis_lp and "
            "then vis_pix in one process; 'vis_lp' runs the MGE light-profile "
            "fit only and returns its result; 'vis_pix' runs the pixelized "
            "source fit only, and requires a completed vis_lp result on disk "
            "(it fails immediately if there is none)."
        ),
    )
    parser.add_argument(
        "--skip_pix",
        action="store_true",
        default=False,
        help="Deprecated alias for --stage vis_lp.",
    )
    if with_seed:
        parser.add_argument(
            "--seed",
            metavar="int",
            type=int,
            required=False,
            default=None,
            help=(
                "Random seed for the vis_lp Nautilus search. Omitted (the "
                "default) the search is unseeded. The seed is part of the run "
                "identifier, so a different seed writes to a different output "
                "directory."
            ),
        )
    args = parser.parse_args()

    stage = args.stage

    if args.skip_pix:
        if stage is not None and stage != "vis_lp":
            parser.error(
                f"--skip_pix is the deprecated spelling of --stage vis_lp and "
                f"cannot be combined with --stage {stage}."
            )
        print("--skip_pix is deprecated; use --stage vis_lp", file=sys.stderr)
        stage = "vis_lp"

    if stage is None:
        stage = "all"

    parsed = (
        args.sample,
        args.dataset,
        int(args.iterations_per_quick_update),
        int(args.number_of_cores),
        args.use_cpu,
        stage,
    )

    if with_seed:
        return parsed + (args.seed,)

    return parsed
