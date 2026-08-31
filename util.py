from astropy.io import fits
from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional

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
        """
        from autolens.analysis.latent import LATENT_FUNCTIONS, latent_keys_enabled

        xp = analysis._xp
        magzero = analysis.kwargs.get("magzero", None)

        instance = model.instance_from_vector(vector=parameters)
        fit = analysis.fit_from(instance=instance)
        context = {"fit": fit, "magzero": magzero, "xp": xp}

        library_keys = latent_keys_enabled()
        library_values = tuple(LATENT_FUNCTIONS[k](**context) for k in library_keys)

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
        - The World Coordinate System (WCS) information of the dataset, which is used to convert between pixel and
          world coordinates.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        result
            The result of a model fit, including the non-linear search, samples and maximum likelihood tracer.
        """
        super().save_results(paths=paths, result=result)

        lens_light_centre = result.max_log_likelihood_tracer.galaxies[0].bulge.centre

        lens_light_centre_wcs_pix = (
            self.dataset.data.geometry.pixel_coordinates_wcs_2d_from(
                scaled_coordinates_2d=lens_light_centre
            )
        )
        lens_light_centre_wcs_pix_y = lens_light_centre_wcs_pix[0]
        lens_light_centre_wcs_pix_x = lens_light_centre_wcs_pix[1]

        pixel_wcs = self.kwargs["pixel_wcs"]

        ra_c_deg, dec_c_deg = pixel_wcs.wcs_pix2world(
            lens_light_centre_wcs_pix_x, lens_light_centre_wcs_pix_y, 1
        )

        data_centre_wcs_pix = self.dataset.data.geometry.pixel_coordinates_wcs_2d_from(
            scaled_coordinates_2d=(0.0, 0.0)
        )
        data_centre_wcs_pix_y = data_centre_wcs_pix[0]
        data_centre_wcs_pix_x = data_centre_wcs_pix[1]

        wcs_dict = {
            "crpix_x": data_centre_wcs_pix_x,
            "crpix_y": data_centre_wcs_pix_y,
            "crval_ra_deg": float(ra_c_deg),
            "crval_dec_deg": float(dec_c_deg),
        }

        output_to_json(
            obj=wcs_dict,
            file_path=paths._files_path / "wcs.json",
        )


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
        sub_size_list=[4, 2, 1],
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


def parse_fit_args():
    """
    Parse the standard command-line arguments shared by all pipeline scripts.

    Returns
    -------
    (sample_name, dataset_name, iterations_per_quick_update, number_of_cores,
     use_cpu, skip_pix)
        ``mask_radius`` is always read from the dataset's ``info.json``.
    """
    import argparse

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
        "--skip_pix",
        action="store_true",
        default=False,
        help="Skip the pixelization stage and return after the MGE light-profile fit.",
    )
    args = parser.parse_args()
    return (
        args.sample,
        args.dataset,
        int(args.iterations_per_quick_update),
        int(args.number_of_cores),
        args.use_cpu,
        args.skip_pix,
    )
