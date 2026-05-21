from astropy.io import fits
from dataclasses import dataclass
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional

import matplotlib.pyplot as plt
from autoconf import conf as _conf
from autoconf.dictable import output_to_json

import autofit as af
import autolens as al
import autolens.plot as aplt
from autogalaxy.operate.lens_calc import LensCalc


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

        # Load the images
        try:
            img0 = np.array(Image.open(dataset_main_path / "rgb_0.png"))
            img1 = np.array(Image.open(dataset_main_path / "rgb_1.png"))
        except FileNotFoundError:
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


class AnalysisImaging(al.AnalysisImaging):
    """
    Sets the custom RGB visualizer above ensuring the RGB subplot is output.
    """

    Visualizer = VisualizerImaging

    LATENT_KEYS = [
        "latent.total_lens_flux",
        "latent.total_lens_flux_1_fwhm",
        "latent.total_lens_flux_2_fwhm",
        "latent.total_lens_flux_3_fwhm",
        "latent.total_lens_flux_4_fwhm",
        "latent.total_lensed_source_flux",
        "latent.total_source_flux",
        "latent.magnification",
        "latent.effective_einstein_radius",
    ]

    def to_ndarray_2d(self, image, xp):

        array_2d = xp.zeros(image.mask.shape, dtype=image.dtype)

        if xp is np:

            array_2d[image.mask.slim_to_native_tuple] = image.array

        else:

            array_2d = array_2d.at[image.mask.slim_to_native_tuple].set(image.array)

        return array_2d

    def compute_latent_variables(self, parameters, model):
        """
        A latent variable is not a model parameter but can be derived from the model. Its value and errors may be
        of interest and aid in the interpretation of a model-fit.

        This code implements a simple example of a latent variable, the magn

        By overwriting this method we can manually specify latent variables that are calculated and output to
        a `latent.csv` file, which mirrors the `samples.csv` file.

        In the example below, the `latent.csv` file will contain at least two columns with the shear magnitude and
        angle sampled by the non-linear search.

        This function is called at the end of search, following one of two schemes depending on the settings in
        `output.yaml`:

        1) Call for every search sample, which produces a complete `latent/samples.csv` which mirrors the normal
        `samples.csv` file but takes a long time to compute.

        2) Call only for N random draws from the posterior inferred at the end of the search, which only produces a
        `latent/latent_summary.json` file with the median and 1 and 3 sigma errors of the latent variables but is
        fast to compute.

        You can add your own custom latent variables here, if you have particular quantities that you
        would like to output to the `latent.csv` file.

        Parameters
        ----------
        parameters : array-like
            The parameter vector of the model sample. This will typically come from the non-linear search.
            Inside this method it is mapped back to a model instance via `model.instance_from_vector`.
        model : Model
            The model object defining how the parameter vector is mapped to an instance. Passed explicitly
            so that this function can be used inside JAX transforms (`vmap`, `jit`) with `functools.partial`.

        Returns
        -------
        A dictionary mapping every latent variable name to its value.

        """

        xp = self._xp

        instance = model.instance_from_vector(vector=parameters)

        fit = self.fit_from(instance=instance)
        tracer = fit.tracer_linear_light_profiles_to_light_profiles

        magzero = self.kwargs.get("magzero", None)

        if magzero is None:
            raise ValueError(
                "MAGZERO must be provided in analysis kwargs to compute latent variables."
            )

        # LENS LIGHT FLUX IN MICROJANSKY, INCLUDING APERTURE FLUXES

        try:

            image = fit.galaxy_image_dict[fit.tracer.galaxies[0]]
            image_native = self.to_ndarray_2d(image=image, xp=xp)

            total_lens_flux = xp.sum(image_native)
            lens_ab_mag = ab_mag_via_flux_from(
                flux=total_lens_flux, magzero=magzero, xp=xp
            )
            total_lens_flux_muJy = flux_mujy_via_ab_mag_from(ab_mag=lens_ab_mag, xp=xp)

            flat_index = xp.argmax(image_native)
            y, x = xp.unravel_index(flat_index, image.shape_native)

            psf_lowest_resolution = self.kwargs["psf_lowest_resolution"]
            psf_lowest_resolution_fwhm = self.kwargs["psf_lowest_resolution_fwhm"]

            image_convolved_to_lowest = psf_lowest_resolution.convolved_image_from(
                image=image, blurring_image=None, xp=xp
            )

            aperture_multipliers = np.array([1.0, 2.0, 3.0, 4.0])

            radius = psf_lowest_resolution_fwhm / (0.1 * 2.0)
            aperture_radii = [
                radius * multiplier for multiplier in aperture_multipliers
            ]

            image_convolved_to_lowest_native = self.to_ndarray_2d(
                image=image_convolved_to_lowest, xp=xp
            )

            total_lens_flux_aperture_list = [
                aperture_flux_from(
                    image_2d=image_convolved_to_lowest_native,
                    centre=(y, x),
                    radius_pixels=radius_pixels,
                    xp=xp,
                )
                for radius_pixels in aperture_radii
            ]

            # convert each aperture flux to magnitude and µJy
            total_lens_mag_aperture_list = [
                ab_mag_via_flux_from(
                    flux=total_lens_flux_aperture, magzero=magzero, xp=xp
                )
                for total_lens_flux_aperture in total_lens_flux_aperture_list
            ]
            total_lens_flux_muJy_aperture_list = [
                flux_mujy_via_ab_mag_from(ab_mag=m, xp=xp)
                for m in total_lens_mag_aperture_list
            ]

        except AttributeError:

            total_lens_flux_muJy = xp.nan
            total_lens_flux_muJy_aperture_list = [xp.nan, xp.nan, xp.nan, xp.nan]

        # LENSED SOURCE FLUX IN MICROJANSKY

        try:

            lensed_source_image = fit.galaxy_image_dict[fit.tracer.galaxies[-1]]

            total_lensed_source_flux = xp.sum(lensed_source_image.array)
            lensed_source_ab_mag = ab_mag_via_flux_from(
                flux=total_lensed_source_flux, magzero=magzero, xp=xp
            )
            total_lensed_source_flux_muJy = flux_mujy_via_ab_mag_from(
                ab_mag=lensed_source_ab_mag, xp=xp
            )

        except AttributeError:
            total_lensed_source_flux_muJy = xp.nan

        # SOURCE FLUX IN MICROJANSKY

        try:
            source_image = tracer.galaxies[-1].image_2d_from(
                grid=self.dataset.grids.lp, xp=xp
            )
            total_source_flux = xp.sum(source_image.array)
            source_ab_mag = ab_mag_via_flux_from(
                flux=total_source_flux, magzero=magzero, xp=xp
            )
            total_source_flux_muJy = flux_mujy_via_ab_mag_from(
                ab_mag=source_ab_mag, xp=xp
            )

        except AttributeError:
            total_source_flux_muJy = xp.nan

        # MAGNIFICATION

        try:
            magnification = total_lensed_source_flux / total_source_flux
        except AttributeError:
            magnification = xp.nan

        # EFFECTIVE EINSTEIN RADIUS

        try:
            lens_calc = LensCalc.from_mass_obj(tracer)
            if self._use_jax:
                # JIT-friendly path. `AnalysisDataset.LATENT_BATCH_MODE = "jit"`
                # in PyAutoGalaxy means `Analysis.compute_latent_samples` wraps
                # this with per-sample `jax.jit`, not `jax.vmap` — the inner
                # ZeroSolver is vmap-incompatible upstream. A fixed fan of
                # seeds at ~1 arcsec from the (preprocessed) lens centre
                # covers the typical Euclid range of Einstein radii.
                import jax.numpy as jnp

                init_guess = jnp.array(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [-1.0, 0.0],
                        [0.0, -1.0],
                    ]
                )
                effective_einstein_radius = lens_calc.einstein_radius_jit_from(
                    init_guess=init_guess,
                )
            else:
                effective_einstein_radius = lens_calc.einstein_radius_from(
                    grid=self.dataset.grids.lp,
                )
        except ValueError:
            effective_einstein_radius = xp.nan

        return (
            total_lens_flux_muJy,
            total_lens_flux_muJy_aperture_list[0],
            total_lens_flux_muJy_aperture_list[1],
            total_lens_flux_muJy_aperture_list[2],
            total_lens_flux_muJy_aperture_list[3],
            total_lensed_source_flux_muJy,
            total_source_flux_muJy,
            magnification,
            effective_einstein_radius,
        )

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
    psf_lowest_resolution_fwhm: float  # FWHM of that PSF in pixels
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

    dataset_centre = dataset.data.brightest_sub_pixel_coordinate_in_region_from(
        region=(-0.3, 0.3, -0.3, 0.3), box_size=2
    )

    try:
        header = al.header_obj_from(
            file_path=dataset_main_path / dataset_fits_name,
            hdu=vis_index * 3 + 1,
        )
        magzero = header["MAGZERO"]
    except FileNotFoundError:
        header = None
        magzero = None

    pixel_wcs = WCS(header).celestial if header is not None else None

    try:
        mask_extra_galaxies = al.Mask2D.from_fits(
            file_path=dataset_main_path / "mask_extra_galaxies.fits",
            pixel_scales=pixel_scale,
            invert=True,
        )
        dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
    except FileNotFoundError:
        pass

    mask_radius = info["mask_radius"]
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

    # Lowest-resolution PSF across all MER bands — used for aperture photometry.
    header_primary = al.header_obj_from(
        file_path=dataset_main_path / dataset_fits_name,
        hdu=0,
    )

    lowest_resolution_waveband = header_primary.get("WORST_BAND", None).lower()
    lowest_resolution_waveband_index = dataset_index_dict.get(
        lowest_resolution_waveband, None
    )

    psf_lowest_resolution = al.Convolver.from_fits(
        file_path=dataset_main_path / dataset_fits_name,
        hdu=lowest_resolution_waveband_index * 3 + 2,
        pixel_scales=pixel_scale,
        normalize=True,
    )

    # Use OU-MER FWHM if valid; fall back to cutout-pipeline value if -99.
    psf_lowest_resolution_fwhm = float(header_primary.get("WORST_PSF_MER", None))
    if psf_lowest_resolution_fwhm is None or psf_lowest_resolution_fwhm < -98:
        psf_lowest_resolution_fwhm = float(header_primary.get("WORST_PSF_HDR", None))

    try:
        positions = al.Grid2DIrregular(
            values=al.from_json(file_path=dataset_main_path / "positions.json")
        )
        positions_likelihood_list = [al.PositionsLH(threshold=0.3, positions=positions)]
    except FileNotFoundError:
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
    (sample_name, dataset_name, iterations_per_quick_update)
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
    args = parser.parse_args()
    return (
        args.sample,
        args.dataset,
        int(args.iterations_per_quick_update),
    )
