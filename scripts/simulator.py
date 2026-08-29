"""
Euclid Pipeline: Simulator
==========================

Simulate a Euclid strong lens in the *exact* format this pipeline reads, so that you can
fit simulated data and check whether PyAutoLens recovers the values you put in.

Two modes, one script:

- ``--from-params`` (default) builds an analytic lens from the truth values written at the
  top of this file: an ``Isothermal`` + ``ExternalShear`` mass, a ``Sersic`` lens light and a
  ``Sersic`` source. Use it to make a clean, well-understood test lens.
- ``--from-result`` takes a **fit you have already run** and resimulates it: the tracer is
  rebuilt from that result's ``model.json`` + maximum-log-likelihood sample, and the bands,
  PSFs, zero-points, WCS and noise levels are read off the dataset the fit was made on. This
  is the *"I have fitted a lens, now resimulate it"* workflow.

Every simulation writes a **truth file** (``truth.json``) beside the data holding every model
parameter, the true per-band fluxes (counts and µJy), the four aperture lens fluxes, the true
magnification and the true Einstein radius. Nothing about the simulation has to be
reconstructed by re-running the script.

__Output__

A simulated dataset is a normal dataset of this pipeline — every fitting script reads it with
no changes::

    dataset/<output_sample>/<output_dataset>/
        <output_dataset>.fits        # PRIMARY + (<BAND>_BGSUB, <BAND>_PSF, <BAND>_RMS) x N
        info.json                    # pixel_scale, mask_radius, mask_centre
        truth.json                   # everything that went in, and the fluxes that came out
        positions.json               # true multiple-image positions
        mask_extra_galaxies.fits     # all-zero (the simulation has no extra galaxies)
        rgb_0.png / rgb_1.png        # colour thumbnails, used by the RGB visualiser
        segmentation/
            lens_flux.fits           # mask centre is taken from this map's peak
            source_flux.fits         # positions fall back to this map's local maxima
            artefact_flux.fits       # all-zero
            artefact_binary.fits     # all-zero: the preferred noise-scaling mask
            lens_binary.fits         # lens_flux thresholded

Fit it exactly like the real example dataset::

    python scripts/simulator.py --output-dataset=euclid_dr1_like
    python scripts/initial_lens_model.py --dataset=euclid_dr1_like --sample=simulated

__Resimulating a fit__

``--from-result`` resolves a result directory the same way ``scripts/diagnose_latent.py``
does (and reuses its ``resolve_files_path`` helper), so the arguments are the same::

    python scripts/simulator.py --from-result \
        --sample=q1_walsmley --dataset=102018665_NEG570040238507752998 \
        --unique_tag=sersic_lens_model --search=vis \
        --output-dataset=102018665_resimulated

The tracer is whatever the fit inferred — an MGE ``initial_lens_model/vis_lp`` result
resimulates as an MGE, a ``sersic_lens_model/vis`` result resimulates as the Sersic-on-Sersic
lens the DR1 resimulation programme wants.

**Prior-edge rule.** Real Euclid fits frequently pin the lens-light ``sersic_index`` against
the upper prior edge at 5. Simulating a lens at the prior edge bakes that artefact into the
mock, so when the inferred index is at (or above) ``--sersic-index-prior-edge`` it is replaced
by ``--sersic-index-replacement`` (default 3.0, the middle of the [2, 4] range the DR1
programme specified). **Both** values are recorded in ``truth.json``
(``sersic_index_inferred`` / ``sersic_index_simulated``), so a later analysis can ask whether
the real lens was genuinely at the edge.

**Single-band fit, multi-band mock.** A VIS fit constrains VIS intensities only. When
``--from-result`` writes more than one band it applies the same fitted intensities to every
band — a *flat* SED. ``truth.json`` records this as ``sed: "flat"``. For a colour-correct
resimulation, fit each band first (``scripts/sersic_lens_model_waveband.py``) and simulate
each band from its own result.

__What is idealised, and what is faithful__

Faithful: the FITS layout, the band set, the pixel scale, the zero-points, the noise level and
the PSF widths are all taken from real Euclid Q1 imaging.

Idealised: the PSFs are circular Gaussians of the right FWHM rather than the real, structured
Euclid PSF stamps (in ``--from-result`` mode the *real* PSF stamps of the source dataset are
reused instead); the noise is Gaussian at a constant per-band sigma rather than
pixel-dependent Poisson; the background is exactly zero; and the NIR bands sit on the VIS
pixel grid, which is what the Euclid MER cut-outs this pipeline consumes already do.

__Test mode__

Under ``PYAUTO_TEST_MODE`` the dataset is written to
``$PYAUTO_OUTPUT_DIR/simulator/<output_sample>/<output_dataset>/`` (default
``output/simulator/...``, which is gitignored) instead of ``dataset/``, so a smoke run can
never overwrite the committed dataset. Pass ``--force-dataset-dir`` to write to ``dataset/``
anyway.

``--dataset`` / ``--sample`` name the *input* dataset and are used only by ``--from-result``;
in ``--from-params`` mode they are accepted and ignored, so this script can carry the smoke
runner's global ``args_default``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import util
from diagnose_latent import resolve_files_path


SIMULATOR_VERSION = "1.0"

DEFAULT_OUTPUT_SAMPLE = "simulated"
DEFAULT_OUTPUT_DATASET = "euclid_dr1_like"

"""
__Bands__

The four DR1 bands this pipeline models, with the conventions read off the shipped Euclid Q1
cut-out ``dataset/q1_walsmley/102018665_NEG570040238507752998/``:

- ``magzero``     — the AB zero-point stamped on that band's ``_BGSUB`` header.
- ``psf_fwhm``    — the ``FWHM`` card of that band's ``_PSF`` header, in arcsec.
- ``psf_shape``   — the PSF stamp size that band uses (21x21 for VIS, 33x33 for NIR).
- ``noise_sigma`` — the median of that band's ``_RMS`` map.
- ``lens_ab_mag`` / ``source_ab_mag`` — the *total* AB magnitude the simulated lens light and
  the *lensed* (image-plane) source are scaled to in this band. A red lens and a bluer source,
  which is what a Euclid lens sample looks like.

Every band sits on the VIS pixel grid, because the MER cut-outs this pipeline reads already
resample the NISP bands onto it.
"""
BANDS = {
    "VIS": {
        "magzero": 24.6,
        "psf_fwhm": 0.2143,
        "psf_shape": (21, 21),
        "noise_sigma": 0.0040,
        "lens_ab_mag": 19.2,
        "source_ab_mag": 21.8,
    },
    "NIR_Y": {
        "magzero": 29.8,
        "psf_fwhm": 0.5037,
        "psf_shape": (33, 33),
        "noise_sigma": 2.356,
        "lens_ab_mag": 18.9,
        "source_ab_mag": 21.9,
    },
    "NIR_J": {
        "magzero": 30.0,
        "psf_fwhm": 0.4614,
        "psf_shape": (33, 33),
        "noise_sigma": 2.389,
        "lens_ab_mag": 18.7,
        "source_ab_mag": 22.0,
    },
    "NIR_H": {
        "magzero": 29.9,
        "psf_fwhm": 0.5875,
        "psf_shape": (33, 33),
        "noise_sigma": 2.228,
        "lens_ab_mag": 18.6,
        "source_ab_mag": 22.1,
    },
}

DEFAULT_BANDS = ["VIS", "NIR_Y", "NIR_J", "NIR_H"]

"""
__Truth Values (--from-params)__

The analytic lens. Chosen so the lens is unambiguously resolved at the Euclid VIS pixel scale
of 0.1"/pixel:

- ``einstein_radius=1.2"`` is 12 VIS pixels, so the arcs are far outside the lens light.
- ``sersic_index=3.0`` for the lens sits in the middle of the [0.8, 5.0] prior the Sersic
  pipelines use — deliberately *not* at the prior edge (see the prior-edge rule above).
- the source is offset from the mass centre so the ring breaks into an asymmetric set of
  multiple images rather than a degenerate Einstein ring.
"""
TRUTH = {
    "redshift_lens": 0.5,
    "redshift_source": 1.0,
    "lens_light": {
        "centre": (0.0, 0.0),
        "axis_ratio": 0.85,
        "angle": 45.0,
        "effective_radius": 0.8,
        "sersic_index": 3.0,
    },
    "lens_mass": {
        "centre": (0.0, 0.0),
        "axis_ratio": 0.75,
        "angle": 40.0,
        "einstein_radius": 1.2,
    },
    "shear": {"gamma_1": 0.03, "gamma_2": -0.02},
    "source_light": {
        "centre": (0.08, 0.12),
        "axis_ratio": 0.7,
        "angle": 100.0,
        "effective_radius": 0.15,
        "sersic_index": 1.0,
    },
}

DEFAULT_SHAPE_NATIVE = (100, 100)
DEFAULT_PIXEL_SCALE = 0.1
DEFAULT_MASK_RADIUS = 3.0

# Euclid VIS nominal exposure time. The simulator runs with Poisson noise off, so this only
# sets the (unused) Poisson branch's scale; the noise actually written is Gaussian at each
# band's `noise_sigma`.
EXPOSURE_TIME = 565.0

# A TAN gnomonic WCS anchored on the frame centre. The real cut-outs inherit the parent MER
# tile's reference pixel (a large negative CRPIX); an equivalent frame-centred WCS is used here
# because nothing downstream depends on the tile origin, only on a valid celestial WCS
# (`util.load_vis_dataset` builds `WCS(header).celestial`, and `AnalysisImaging.save_results`
# converts the lens centre to sky coordinates for the catalogue's `crval_ra_deg` column).
WCS_CRVAL = (57.38288, -51.0)


def json_number(value):
    """
    ``float(value)``, or ``None`` when it is not finite.

    Resimulating a degenerate result (a ``PYAUTO_TEST_MODE`` fit's fake samples, say) can give a
    galaxy zero flux, which sends the magnitude conversions to infinity. ``None`` records that
    honestly rather than writing a non-standard ``Infinity`` token into ``truth.json``.
    """
    value = float(value)
    return value if np.isfinite(value) else None


def gaussian_psf_from(fwhm_arcsec, shape_native, pixel_scale):
    """
    A circular Gaussian PSF ``Convolver`` of the given FWHM, normalised to unit sum.
    """
    import autolens as al

    sigma = fwhm_arcsec / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    return al.Convolver.from_gaussian(
        shape_native=shape_native,
        pixel_scales=pixel_scale,
        sigma=sigma,
        normalize=True,
    )


def image_wcs_header_from(shape_native, pixel_scale, magzero, band):
    """
    The header cards of a ``<BAND>_BGSUB`` image HDU: the AB zero-point plus a TAN WCS.

    ``MAGZERO`` lives on the image HDU (not the primary) because that is where
    ``util.load_vis_dataset`` reads it from.
    """
    header = fits.Header()
    header["EQUINOX"] = (2000.0, "Mean equinox")
    header["RADESYS"] = ("ICRS", "Equatorial coordinate system")
    header["CTYPE1"] = ("RA---TAN", "Right ascension, gnomonic projection")
    header["CUNIT1"] = ("deg", "Units of coordinate increment and value")
    header["CRVAL1"] = (WCS_CRVAL[0], "[deg] Coordinate value at reference point")
    header["CRPIX1"] = (
        shape_native[1] / 2.0 + 0.5,
        "Pixel coordinate of reference point",
    )
    header["CD1_1"] = (-pixel_scale / 3600.0, "Linear projection matrix")
    header["CD1_2"] = (0.0, "Linear projection matrix")
    header["CTYPE2"] = ("DEC--TAN", "Declination, gnomonic projection")
    header["CUNIT2"] = ("deg", "Units of coordinate increment and value")
    header["CRVAL2"] = (WCS_CRVAL[1], "[deg] Coordinate value at reference point")
    header["CRPIX2"] = (
        shape_native[0] / 2.0 + 0.5,
        "Pixel coordinate of reference point",
    )
    header["CD2_1"] = (0.0, "Linear projection matrix")
    header["CD2_2"] = (pixel_scale / 3600.0, "Linear projection matrix")
    header["MAGZERO"] = (magzero, "AB zeropoint")
    header["FILTER"] = (band, "Euclid filter for flux image")

    return header


def write_dataset_fits(file_path, band_data, worst_band, worst_psf_fwhm, pixel_scale):
    """
    Write the multi-extension FITS this pipeline reads.

    The layout is not negotiable: ``util.dataset_instrument_hdu_dict_via_fits_from`` assigns
    each ``<BAND>_BGSUB`` extension an ordinal in file order and
    ``util.load_vis_dataset`` then indexes the image, PSF and noise map as
    ``3i + 1``, ``3i + 2`` and ``3i + 3``. So the HDUs must be exactly

        PRIMARY, (<BAND>_BGSUB, <BAND>_PSF, <BAND>_RMS) x N

    and the primary header must name every extension in ``EXT_1 .. EXT_3N``, matching the
    Euclid cut-out generator.

    ``WORST_BAND`` and ``WORST_PSF_MER`` / ``WORST_PSF_HDR`` are what
    ``util.psf_fwhm_arcsec_from_primary_header`` reads to size the four aperture-flux latent
    variables. They are stamped upstream for real data; here the simulator knows them exactly.

    Parameters
    ----------
    band_data
        Ordered mapping ``band -> {"image", "psf", "rms", "magzero"}`` of 2D arrays.
    """
    primary = fits.PrimaryHDU()

    hdus = [primary]
    for band, entry in band_data.items():
        for suffix, data, header in (
            (
                "BGSUB",
                entry["image"].astype(np.float32),
                image_wcs_header_from(
                    shape_native=entry["image"].shape,
                    pixel_scale=pixel_scale,
                    magzero=entry["magzero"],
                    band=band,
                ),
            ),
            ("PSF", entry["psf"].astype(np.float32), fits.Header()),
            ("RMS", entry["rms"].astype(np.float32), fits.Header()),
        ):
            extname = f"{band}_{suffix}"
            if suffix == "PSF":
                header["FILTER"] = (band, "Euclid filter for PSF image")
                header["FWHM"] = (
                    float(entry["psf_fwhm"]),
                    "The central PSF stamp FWHM value in arcseconds",
                )
            hdus.append(fits.ImageHDU(data=data, header=header, name=extname))
            primary.header[f"EXT_{len(hdus) - 1}"] = (
                extname,
                f"Extension name for {band} {suffix}",
            )

    # The Euclid cut-out generator writes these three as HIERARCH cards (their names exceed the
    # 8-character FITS limit). Spelling the prefix explicitly writes the identical card without
    # astropy's VerifyWarning; `header.get("WORST_BAND")` reads it back unchanged.
    primary.header["HIERARCH WORST_BAND"] = (worst_band, "Band with worst PSF")
    primary.header["HIERARCH WORST_PSF_HDR"] = (
        float(worst_psf_fwhm),
        "PSF FWHM of worst band [arcsec]",
    )
    primary.header["HIERARCH WORST_PSF_MER"] = (
        float(worst_psf_fwhm),
        "MER FWHM of worst band [arcsec]",
    )

    file_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(file_path, overwrite=True)


def write_segmentation(dataset_path, lens_image, lensed_source_image, pixel_scale):
    """
    Write the ``segmentation/`` maps the pipeline's optional inputs chain reads.

    Only the five maps with a consumer are written:

    - ``lens_flux.fits``      — ``util.load_vis_dataset`` takes the analysis mask centre from
      this map's brightest pixel (preferred over ``info.json``'s ``mask_centre``).
    - ``source_flux.fits``    — the multiple-image positions fall back to this map's local
      maxima when ``positions.json`` is absent.
    - ``artefact_binary.fits``— the preferred noise-scaling mask (all-zero: the simulation has
      no artefacts, so nothing is scaled).
    - ``artefact_flux.fits``  — read by ``preprocess/segmentation.py`` for its diagnostic PNG.
    - ``lens_binary.fits``    — likewise.

    ``source_binary.fits`` exists in the DR1 preprocessing outputs but has no consumer in this
    repository, so it is not written.
    """
    seg_path = dataset_path / "segmentation"
    seg_path.mkdir(parents=True, exist_ok=True)

    zeros = np.zeros(lens_image.shape, dtype=np.uint8)

    lens_binary = (lens_image > 0.05 * float(np.max(lens_image))).astype(np.uint8)

    for name, data in (
        ("lens_flux", lens_image.astype(np.float32)),
        ("source_flux", lensed_source_image.astype(np.float32)),
        ("artefact_flux", zeros),
        ("artefact_binary", zeros),
        ("lens_binary", lens_binary),
    ):
        header = fits.Header()
        header["PIXSCAY"] = pixel_scale
        header["PIXSCAX"] = pixel_scale
        fits.PrimaryHDU(data=data, header=header).writeto(
            seg_path / f"{name}.fits", overwrite=True
        )


def write_rgb_thumbnails(dataset_path, band_images):
    """
    Write the ``rgb_0.png`` / ``rgb_1.png`` colour thumbnails.

    Two consumers want them, and both degrade silently without them:
    ``util.VisualizerImaging.visualize_before_fit`` needs **both** to write the fit's
    ``image/rgb.png`` (which ``scripts/build_inspect.py`` collects into the inspection bundle as
    ``rgb.png``), and ``preprocess/segmentation.py`` draws ``rgb_0.png`` in the top-left panel of
    its diagnostic figure.

    The composite is red = reddest band, green = middle band, blue = VIS, arcsinh-stretched and
    clipped at the 99.5th percentile — the standard astronomical thumbnail recipe.
    ``rgb_1.png`` is the same composite at a harder stretch, mirroring the two-thumbnail
    convention of the real Euclid cut-outs. They are written at the data's own resolution, so
    they cost a few tens of KB rather than the few hundred a rendered figure would.
    """
    bands = list(band_images)
    blue = "VIS" if "VIS" in band_images else bands[0]
    red = bands[-1]
    green = bands[len(bands) // 2]

    def composite(stretch):
        channels = []
        for band in (red, green, blue):
            image = np.asarray(band_images[band], dtype=np.float64)
            scale = np.percentile(image, 99.5)
            if scale <= 0.0:
                scale = np.max(np.abs(image)) or 1.0
            stretched = np.arcsinh(stretch * image / scale) / np.arcsinh(stretch)
            channels.append(np.clip(stretched, 0.0, 1.0))
        return (np.stack(channels, axis=-1) * 255.0).astype(np.uint8)

    for name, stretch in (("rgb_0.png", 10.0), ("rgb_1.png", 100.0)):
        Image.fromarray(composite(stretch), mode="RGB").save(
            dataset_path / name, optimize=True
        )


def positions_from_tracer(tracer, grid, source_centre, source_flux_2d, pixel_scale):
    """
    The true multiple-image positions of the source centre.

    ``al.PointSolver`` traces the image plane back to the source plane and returns the
    positions that map to ``source_centre`` — the exact answer for a simulated lens, where a
    real dataset has to mark them by eye. If the solver returns fewer than two images (a source
    that is not multiply imaged), fall back to the same local-maxima search
    ``preprocess/segmentation.py`` and ``util.load_vis_dataset`` use on the source flux map, so
    a ``positions.json`` is always written.
    """
    import autolens as al

    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.005, magnification_threshold=0.1
    )
    positions = solver.solve(tracer=tracer, source_plane_coordinate=source_centre)

    if len(positions) >= 2:
        return al.Grid2DIrregular(values=[[float(y), float(x)] for y, x in positions])

    fallback = util._compute_positions_from_source_flux(
        source_flux=source_flux_2d.astype(np.float32),
        noise_map=None,
        pixel_scale=pixel_scale,
    )
    return al.Grid2DIrregular(values=fallback)


def tracer_from_params():
    """
    The analytic ``--from-params`` tracer: ``Isothermal`` + ``ExternalShear`` mass, ``Sersic``
    lens light, ``Sersic`` source, all at unit intensity.

    The intensities are rescaled per band by :func:`simulate` so each band hits its target AB
    magnitude; because a Sersic image is linear in ``intensity`` this is exact.
    """
    import autolens as al

    light = TRUTH["lens_light"]
    mass = TRUTH["lens_mass"]
    shear = TRUTH["shear"]
    source = TRUTH["source_light"]

    lens_galaxy = al.Galaxy(
        redshift=TRUTH["redshift_lens"],
        bulge=al.lp.Sersic(
            centre=light["centre"],
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=light["axis_ratio"], angle=light["angle"]
            ),
            intensity=1.0,
            effective_radius=light["effective_radius"],
            sersic_index=light["sersic_index"],
        ),
        mass=al.mp.Isothermal(
            centre=mass["centre"],
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=mass["axis_ratio"], angle=mass["angle"]
            ),
            einstein_radius=mass["einstein_radius"],
        ),
        shear=al.mp.ExternalShear(gamma_1=shear["gamma_1"], gamma_2=shear["gamma_2"]),
    )

    source_galaxy = al.Galaxy(
        redshift=TRUTH["redshift_source"],
        bulge=al.lp.Sersic(
            centre=source["centre"],
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=source["axis_ratio"], angle=source["angle"]
            ),
            intensity=1.0,
            effective_radius=source["effective_radius"],
            sersic_index=source["sersic_index"],
        ),
    )

    return al.Tracer(galaxies=[lens_galaxy, source_galaxy])


def scaled_tracer_from(tracer, lens_scale, source_scale):
    """
    A copy of ``tracer`` whose lens-light and source-light intensities are multiplied by the
    given factors.

    The per-band images are produced by scaling the unit-intensity images (a light profile is
    linear in ``intensity``, so this is exact), but ``truth.json``'s model block and its latent
    values must describe the *actual* profiles that were written. This rebuilds the tracer at
    the VIS-band scaling so both agree with the pixels on disk.

    Handles a ``Basis`` (an MGE lens light, as ``--from-result`` produces) by recursing into
    ``profile_list``.
    """
    import copy

    def scale(profile, factor):
        profile_list = getattr(profile, "profile_list", None)
        if profile_list is not None:
            for entry in profile_list:
                scale(entry, factor)
        elif hasattr(profile, "intensity"):
            profile.intensity = profile.intensity * factor

    scaled = copy.deepcopy(tracer)

    for galaxy, factor in (
        (scaled.galaxies[0], lens_scale),
        (scaled.galaxies[-1], source_scale),
    ):
        for name in ("bulge", "disk", "light"):
            profile = getattr(galaxy, name, None)
            if profile is not None:
                scale(profile, factor)

    return scaled


def apply_sersic_index_prior_edge_rule(galaxy, prior_edge, replacement):
    """
    Lower a lens-light ``sersic_index`` that has piled up against its prior edge.

    Real Euclid fits routinely return ``sersic_index = 5``, the upper prior limit — an artefact
    of the fit, not a measurement. Simulating at that value would bake the artefact into the
    mock, so it is replaced. Returns ``(inferred, simulated)``; both are recorded in
    ``truth.json`` so a later analysis can ask whether the lens was genuinely at the edge.

    Profiles without a ``sersic_index`` (an MGE lens light, say) are left untouched and report
    ``(None, None)``.
    """
    bulge = getattr(galaxy, "bulge", None)
    if bulge is None or not hasattr(bulge, "sersic_index"):
        return None, None

    inferred = float(bulge.sersic_index)
    if inferred < prior_edge:
        return inferred, inferred

    bulge.sersic_index = replacement
    print(
        f"[simulator] lens sersic_index {inferred} is at the prior edge "
        f"({prior_edge}); simulating with {replacement} instead.",
        flush=True,
    )
    return inferred, float(replacement)


def tracer_from_result(args, output_path):
    """
    Rebuild the tracer of a finished fit from its ``model.json`` and maximum-log-likelihood
    sample.

    This is the same resolution ``scripts/diagnose_latent.py`` performs — its
    ``resolve_files_path`` is imported rather than reimplemented, so "the newest converged
    result" means the same thing in both scripts.
    """
    from autofit import from_dict
    import autolens as al  # noqa: F401  needed so `from_dict` resolves al.* classes

    sample_name = args.sample if args.sample else None

    search_dir = output_path
    if sample_name is not None:
        search_dir = search_dir / sample_name
    search_dir = search_dir / args.dataset / args.unique_tag / args.search

    files_path = resolve_files_path(search_dir, result_hash=args.result_hash)
    print(f"[simulator] resimulating result: {files_path}", flush=True)

    with open(files_path / "samples_summary.json") as f:
        summary = from_dict(json.load(f))
    with open(files_path / "model.json") as f:
        model = from_dict(json.load(f))
    summary.model = model

    parameters = summary.max_log_likelihood_sample.parameter_lists_for_model(model)
    instance = model.instance_from_vector(vector=parameters)

    galaxies = list(instance.galaxies)

    return al.Tracer(galaxies=galaxies), files_path


def band_setup_from_dataset(dataset_path, dataset_name, band_names):
    """
    Read the band conventions of an existing dataset: PSF stamp, PSF FWHM, zero-point and
    noise level per band, plus the worst-seeing band.

    Used by ``--from-result`` so a resimulated lens carries the *real* PSF stamps and noise
    levels of the data it was fitted to, rather than idealised Gaussians.
    """
    fits_name = f"{dataset_name}.fits"
    index_dict = util.dataset_instrument_hdu_dict_via_fits_from(
        dataset_path=dataset_path, dataset_fits_name=fits_name, image_tag="_BGSUB"
    )

    with fits.open(dataset_path / fits_name) as hdu_list:
        primary_header = hdu_list[0].header

        setup = {}
        for band_lower, index in index_dict.items():
            band = band_lower.upper()
            if band_names is not None and band not in band_names:
                continue
            psf = np.asarray(hdu_list[index * 3 + 2].data, dtype=np.float64)
            setup[band] = {
                "magzero": float(hdu_list[index * 3 + 1].header["MAGZERO"]),
                "psf_kernel": psf / psf.sum(),
                "psf_fwhm": float(hdu_list[index * 3 + 2].header.get("FWHM", np.nan)),
                "noise_sigma": float(
                    np.median(
                        np.asarray(hdu_list[index * 3 + 3].data, dtype=np.float64)
                    )
                ),
            }

        worst_band = primary_header.get("WORST_BAND", None)
        worst_psf_fwhm = (
            util.psf_fwhm_arcsec_from_primary_header(
                header=primary_header, dataset_name=dataset_name
            )
            if worst_band is not None
            else None
        )

    if worst_band is not None:
        worst_band = str(worst_band).strip().upper()

    return setup, worst_band, worst_psf_fwhm


def latents_via_truth_from(dataset_name, output_sample, tracer, magzero, loadable):
    """
    Evaluate the pipeline's own latent catalogue (``util.LatentEuclid``) on the truth model.

    The dataset is loaded back off disk exactly as a fitting script would, so these are the
    values a *perfect* fit would recover — the known answers the latent unit tests assert
    against. Every parameter is fixed, so the model has zero free parameters and the parameter
    vector is empty.

    ``util.load_vis_dataset`` only resolves paths under ``dataset/``, so when the simulation was
    redirected to the test-mode scratch directory there is nothing to load back and
    ``(None, None)`` is returned — ``truth.json`` then records why the block is absent.
    """
    import autofit as af

    if not loadable:
        return None, None

    d = util.load_vis_dataset(dataset_name, sample_name=output_sample)

    names = (
        ["lens", "source"]
        if len(tracer.galaxies) == 2
        else [f"galaxy_{index}" for index in range(len(tracer.galaxies))]
    )
    model = af.Collection(
        galaxies=af.Collection(
            **{
                name: af.Model.from_instance(galaxy)
                for name, galaxy in zip(names, tracer.galaxies)
            }
        )
    )

    analysis = util.AnalysisImaging(
        dataset=d.dataset,
        positions_likelihood_list=None,
        use_jax=False,
        dataset_main_path=d.dataset_main_path,
        title_prefix="VIS",
        plot_rgb=False,
        skip_rgb_plot=True,
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        magzero=magzero,
    )

    keys = util.LatentEuclid.keys(analysis)
    values = util.LatentEuclid.variables(analysis=analysis, parameters=[], model=model)

    return {key: float(value) for key, value in zip(keys, values)}, d


def aperture_lens_fluxes_from(lens_image_2d, worst_psf_fwhm, pixel_scale, magzero):
    """
    The four aperture lens fluxes at 1, 2, 3 and 4 x the worst band's PSF FWHM, in µJy.

    Computed with the pipeline's own helpers (``util.aperture_flux_from``,
    ``util.ab_mag_via_flux_from``, ``util.flux_mujy_via_ab_mag_from``) and the same radius
    convention as ``util.LatentEuclid.variables`` (``fwhm / (0.1 * 2) * k`` pixels), so the
    numbers are directly comparable with the ``total_lens_flux_{1,2,3,4}_fwhm_mujy`` latents.

    ``lens_image_2d`` must therefore be the VIS-band lens light convolved with the **worst
    band's** PSF — matched-aperture photometry degrades every band to the worst seeing — and
    ``magzero`` the VIS zero-point, which is exactly the pair the latent uses.
    """
    peak = np.unravel_index(int(np.argmax(lens_image_2d)), lens_image_2d.shape)
    radius = worst_psf_fwhm / (0.1 * 2.0)

    radii = [radius * multiplier for multiplier in (1.0, 2.0, 3.0, 4.0)]
    fluxes = [
        json_number(
            util.flux_mujy_via_ab_mag_from(
                ab_mag=util.ab_mag_via_flux_from(
                    flux=util.aperture_flux_from(
                        image_2d=lens_image_2d,
                        centre=peak,
                        radius_pixels=aperture_radius,
                    ),
                    magzero=magzero,
                )
            )
        )
        for aperture_radius in radii
    ]

    return fluxes, radii, [int(peak[0]), int(peak[1])]


def simulate(args):
    """
    Build the tracer, simulate every band, write the dataset and write ``truth.json``.
    """
    from autolens import conf

    project_root = Path(__file__).parent.parent

    output_root = project_root / os.environ.get("PYAUTO_OUTPUT_DIR", "output")
    conf.instance.push(new_path=project_root / "config", output_path=output_root)

    import autolens as al
    from autogalaxy.operate.lens_calc import LensCalc

    test_mode = int(os.environ.get("PYAUTO_TEST_MODE", "0")) > 0
    if test_mode and not args.force_dataset_dir:
        dataset_root = output_root / "simulator"
        print(
            f"[simulator] PYAUTO_TEST_MODE is set — writing to {dataset_root} rather than "
            "dataset/ so the committed dataset is never clobbered "
            "(--force-dataset-dir overrides).",
            flush=True,
        )
    else:
        dataset_root = project_root / "dataset"

    dataset_path = dataset_root / args.output_sample / args.output_dataset
    dataset_path.mkdir(parents=True, exist_ok=True)

    band_names = (
        [band.strip().upper() for band in args.bands.split(",")] if args.bands else None
    )

    """
    __Tracer__
    """
    sersic_index_inferred = None
    sersic_index_simulated = None
    source_from = None

    if args.from_result:
        tracer, files_path = tracer_from_result(args, output_path=output_root)
        source_from = str(files_path)
        (
            sersic_index_inferred,
            sersic_index_simulated,
        ) = apply_sersic_index_prior_edge_rule(
            galaxy=tracer.galaxies[0],
            prior_edge=args.sersic_index_prior_edge,
            replacement=args.sersic_index_replacement,
        )
        source_dataset_path = (
            project_root / "dataset" / args.sample / args.dataset
            if args.sample
            else project_root / "dataset" / args.dataset
        )
        band_setup, worst_band, worst_psf_fwhm = band_setup_from_dataset(
            dataset_path=source_dataset_path,
            dataset_name=args.dataset,
            band_names=band_names,
        )
        if worst_band is None or worst_band not in band_setup:
            # The source dataset's worst-seeing band was not one of the bands being written
            # (or its primary header carried no WORST_BAND). Fall back to the widest PSF among
            # the bands actually simulated, so the four aperture latents still have a scale.
            worst_band = max(band_setup, key=lambda b: band_setup[b]["psf_fwhm"] or 0.0)
            worst_psf_fwhm = band_setup[worst_band]["psf_fwhm"]
        sed = "flat"
    else:
        tracer = tracer_from_params()
        sersic_index_inferred = None
        sersic_index_simulated = TRUTH["lens_light"]["sersic_index"]
        band_setup = {
            band: {
                "magzero": BANDS[band]["magzero"],
                "psf_kernel": None,
                "psf_fwhm": BANDS[band]["psf_fwhm"],
                "noise_sigma": BANDS[band]["noise_sigma"],
            }
            for band in (band_names or DEFAULT_BANDS)
        }
        worst_band = max(band_setup, key=lambda b: band_setup[b]["psf_fwhm"])
        worst_psf_fwhm = band_setup[worst_band]["psf_fwhm"]
        sed = "per-band AB magnitudes (see truth.json bands block)"

    """
    __Grid__

    The simulation grid. ``over_sample_size=4`` evaluates every pixel on a 4x4 sub-grid, which
    is what keeps the steep Sersic centres and the highly magnified arcs accurate. All truth
    fluxes below are integrated on this same grid, so the truth file is internally consistent
    with the pixels that were written.
    """
    shape_native = tuple(int(v) for v in args.shape.split(","))
    pixel_scale = args.pixel_scale

    grid = al.Grid2D.uniform(
        shape_native=shape_native, pixel_scales=pixel_scale, over_sample_size=4
    )

    lens_galaxy = tracer.galaxies[0]
    source_galaxy = tracer.galaxies[-1]

    galaxy_image_dict = tracer.galaxy_image_2d_dict_from(grid=grid)
    lens_image_unit = np.asarray(
        galaxy_image_dict[lens_galaxy].native, dtype=np.float64
    )
    lensed_source_image_unit = np.asarray(
        galaxy_image_dict[source_galaxy].native, dtype=np.float64
    )
    source_image_unit = np.asarray(
        source_galaxy.image_2d_from(grid=grid).native, dtype=np.float64
    )

    lens_sum_unit = float(np.sum(lens_image_unit))
    lensed_source_sum_unit = float(np.sum(lensed_source_image_unit))
    source_sum_unit = float(np.sum(source_image_unit))

    """
    __Simulate Each Band__

    Every band is simulated on the same grid with its own PSF, zero-point and noise level. In
    ``--from-params`` mode the lens and source images are rescaled so each band hits its target
    AB magnitude, which is what gives the mock a colour; in ``--from-result`` mode the fitted
    intensities are used unchanged in every band (a flat SED — see the module docstring).
    """
    rng = np.random.default_rng(args.seed)

    # The reference band for the truth model, the aperture fluxes and the latent block. VIS is
    # the band every fitting script models, so it is used whenever it is present; a VIS-less
    # band selection falls back to the first band written.
    reference_band = "VIS" if "VIS" in band_setup else next(iter(band_setup))

    band_data = {}
    band_psf = {}
    truth_fluxes = {}
    reference_lens_image = None

    for band, setup in band_setup.items():
        if setup["psf_kernel"] is None:
            psf = gaussian_psf_from(
                fwhm_arcsec=setup["psf_fwhm"],
                shape_native=BANDS.get(band, {}).get("psf_shape", (21, 21)),
                pixel_scale=pixel_scale,
            )
            psf_kernel = np.asarray(psf.kernel.native, dtype=np.float64)
        else:
            psf_kernel = setup["psf_kernel"]
            psf = al.Convolver(
                kernel=al.Array2D.no_mask(values=psf_kernel, pixel_scales=pixel_scale),
                normalize=True,
            )

        magzero = setup["magzero"]

        if args.from_result:
            lens_scale = 1.0
            source_scale = 1.0
        else:
            lens_target_counts = 10.0 ** (-0.4 * (BANDS[band]["lens_ab_mag"] - magzero))
            source_target_counts = 10.0 ** (
                -0.4 * (BANDS[band]["source_ab_mag"] - magzero)
            )
            lens_scale = lens_target_counts / lens_sum_unit
            source_scale = source_target_counts / lensed_source_sum_unit

        lens_image = lens_image_unit * lens_scale
        lensed_source_image = lensed_source_image_unit * source_scale
        source_image = source_image_unit * source_scale

        noise_sigma = setup["noise_sigma"]

        """
        `al.SimulatorImaging` performs the PSF convolution and builds the noise-map. Poisson
        noise is switched off and a constant noise-map of `noise_sigma` requested instead
        (`noise_if_add_noise_false`), then Gaussian noise at exactly that sigma is added below.
        Euclid MER cut-outs are background-subtracted mosaics whose RMS maps are near-constant,
        so constant Gaussian noise is both the closer match and the cleanly-known truth.
        """
        simulator = al.SimulatorImaging(
            exposure_time=EXPOSURE_TIME,
            psf=psf,
            background_sky_level=0.0,
            add_poisson_noise_to_data=False,
            include_poisson_noise_in_noise_map=False,
            noise_if_add_noise_false=noise_sigma,
        )

        image_convolved = np.asarray(
            simulator.via_image_from(
                image=al.Array2D.no_mask(
                    values=lens_image + lensed_source_image, pixel_scales=pixel_scale
                )
            ).data.native,
            dtype=np.float64,
        )
        lens_image_convolved = np.asarray(
            simulator.via_image_from(
                image=al.Array2D.no_mask(values=lens_image, pixel_scales=pixel_scale)
            ).data.native,
            dtype=np.float64,
        )

        data = image_convolved + rng.normal(
            loc=0.0, scale=noise_sigma, size=image_convolved.shape
        )
        rms = np.full(image_convolved.shape, noise_sigma, dtype=np.float64)

        band_psf[band] = psf
        band_data[band] = {
            "image": data,
            "psf": psf_kernel,
            "rms": rms,
            "magzero": magzero,
            "psf_fwhm": setup["psf_fwhm"],
        }

        lens_counts = float(np.sum(lens_image))
        lensed_source_counts = float(np.sum(lensed_source_image))
        source_counts = float(np.sum(source_image))

        truth_fluxes[band] = {
            "magzero": magzero,
            "psf_fwhm": setup["psf_fwhm"],
            "psf_shape": list(psf_kernel.shape),
            "noise_sigma": noise_sigma,
            "lens_intensity_scale": lens_scale,
            "source_intensity_scale": source_scale,
            "lens_flux_counts": lens_counts,
            "lensed_source_flux_counts": lensed_source_counts,
            "source_flux_counts": source_counts,
            "lens_flux_mujy": json_number(
                util.flux_mujy_via_ab_mag_from(
                    ab_mag=util.ab_mag_via_flux_from(flux=lens_counts, magzero=magzero)
                )
            ),
            "lensed_source_flux_mujy": json_number(
                util.flux_mujy_via_ab_mag_from(
                    ab_mag=util.ab_mag_via_flux_from(
                        flux=lensed_source_counts, magzero=magzero
                    )
                )
            ),
            "source_flux_mujy": json_number(
                util.flux_mujy_via_ab_mag_from(
                    ab_mag=util.ab_mag_via_flux_from(
                        flux=source_counts, magzero=magzero
                    )
                )
            ),
        }

        if band == reference_band:
            reference_lens_image = lens_image
            reference_lens_image_convolved = lens_image_convolved
            reference_lensed_source_image = lensed_source_image
            reference_lens_scale = lens_scale
            reference_source_scale = source_scale

    """
    __Write The Dataset__
    """
    write_dataset_fits(
        file_path=dataset_path / f"{args.output_dataset}.fits",
        band_data=band_data,
        worst_band=worst_band,
        worst_psf_fwhm=worst_psf_fwhm,
        pixel_scale=pixel_scale,
    )

    info = {
        "pixel_scale": pixel_scale,
        "mask_radius": args.mask_radius,
        "mask_centre": [
            float(v) for v in getattr(lens_galaxy.mass, "centre", (0.0, 0.0))
        ],
    }
    with open(dataset_path / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    fits.PrimaryHDU(data=np.zeros(shape_native, dtype=np.uint8)).writeto(
        dataset_path / "mask_extra_galaxies.fits", overwrite=True
    )

    write_rgb_thumbnails(
        dataset_path=dataset_path,
        band_images={band: entry["image"] for band, entry in band_data.items()},
    )

    write_segmentation(
        dataset_path=dataset_path,
        lens_image=reference_lens_image_convolved,
        lensed_source_image=reference_lensed_source_image,
        pixel_scale=pixel_scale,
    )

    """
    __Positions__
    """
    positions = positions_from_tracer(
        tracer=tracer,
        grid=grid,
        source_centre=tuple(float(v) for v in source_galaxy.bulge.centre),
        source_flux_2d=reference_lensed_source_image,
        pixel_scale=pixel_scale,
    )
    al.output_to_json(obj=positions, file_path=dataset_path / "positions.json")

    """
    __Truth__

    Everything that went into the simulation, plus the quantities a fit is meant to recover.
    """
    lens_calc = LensCalc.from_mass_obj(tracer)
    try:
        einstein_radius_effective = float(lens_calc.einstein_radius_from(grid=grid))
    except (ValueError, AttributeError):
        einstein_radius_effective = float("nan")

    try:
        magnification_point = float(
            np.sum(
                np.abs(
                    np.asarray(
                        lens_calc.magnification_2d_via_hessian_from(grid=positions)
                    )
                )
            )
        )
    except (ValueError, AttributeError):
        magnification_point = float("nan")

    reference_magzero = band_data[reference_band]["magzero"]

    """
    The truth model is the VIS-scaled tracer: the per-band images were made by scaling
    unit-intensity images, so the tracer that actually describes the VIS pixels on disk is the
    one whose intensities carry the VIS scale factors. Every other band's scaling is recorded
    in the `bands` block.
    """
    truth_tracer = scaled_tracer_from(
        tracer=tracer,
        lens_scale=reference_lens_scale,
        source_scale=reference_source_scale,
    )

    """
    Matched-aperture photometry degrades every band to the worst seeing, so the aperture
    latents convolve the lens light with the *worst band's* PSF, not VIS's. Reproduce that here
    so `aperture_lens_flux_mujy` is directly comparable with the four aperture latents.
    """
    lens_image_worst_psf = np.asarray(
        band_psf[worst_band]
        .convolved_image_via_real_space_from(
            image=al.Array2D.no_mask(
                values=reference_lens_image, pixel_scales=pixel_scale
            ),
            blurring_image=None,
        )
        .native,
        dtype=np.float64,
    )

    aperture_fluxes, aperture_radii, aperture_centre = aperture_lens_fluxes_from(
        lens_image_2d=lens_image_worst_psf,
        worst_psf_fwhm=worst_psf_fwhm,
        pixel_scale=pixel_scale,
        magzero=reference_magzero,
    )

    latents, d = latents_via_truth_from(
        dataset_name=args.output_dataset,
        output_sample=args.output_sample,
        tracer=truth_tracer,
        magzero=reference_magzero,
        loadable=dataset_root == project_root / "dataset",
    )

    truth = {
        "simulator_version": SIMULATOR_VERSION,
        "mode": "from-result" if args.from_result else "from-params",
        "source_result": source_from,
        "seed": args.seed,
        "sed": sed,
        "dataset": {
            "sample": args.output_sample,
            "name": args.output_dataset,
            "shape_native": list(shape_native),
            "pixel_scale": pixel_scale,
            "mask_radius": args.mask_radius,
            "mask_centre_info_json": info["mask_centre"],
            "mask_centre_used": (
                [float(v) for v in d.dataset.mask.mask_centre]
                if d is not None
                else None
            ),
            "over_sample_size": 4,
        },
        "conventions": {
            "bands.*_flux_counts": (
                "Integrated over the whole simulation frame, before PSF convolution, in the "
                "band's own image units."
            ),
            "bands.*_flux_mujy": (
                "The counts above converted with util.ab_mag_via_flux_from + "
                "util.flux_mujy_via_ab_mag_from at that band's MAGZERO."
            ),
            "latents": (
                "util.LatentEuclid evaluated on the truth model after loading the written "
                "dataset back through util.load_vis_dataset. These are integrated over the "
                "*masked* grid (mask_radius), so they are a few per cent below the full-frame "
                "band fluxes above; they are the values a perfect fit recovers."
            ),
            "aperture_lens_flux_mujy.values": (
                "Independent full-frame counterpart of the four total_lens_flux_k_fwhm_mujy "
                "latents: the VIS lens light convolved with the worst band's PSF, summed in "
                "circular apertures of fwhm/(0.1*2)*k pixels, at the VIS zero-point."
            ),
            "magnification.area": (
                "Ratio of the image-plane to source-plane integrated source flux over the "
                "whole frame; the latent 'magnification' is the same ratio on the masked grid."
            ),
            "magnification.point": (
                "Sum of |mu| at the multiple images in 'positions', via "
                "LensCalc.magnification_2d_via_hessian_from. A different quantity from "
                "'area' — they are expected to disagree."
            ),
            "einstein_radius.effective": (
                "LensCalc.einstein_radius_from on the simulation grid — the tangential "
                "critical curve's area-equivalent radius, which differs from the Isothermal "
                "'model_parameter' by the mass model's ellipticity convention."
            ),
        },
        "bands": truth_fluxes,
        "worst_band": worst_band,
        "worst_psf_fwhm": float(worst_psf_fwhm),
        "model": model_truth_from(truth_tracer),
        "sersic_index_inferred": sersic_index_inferred,
        "sersic_index_simulated": sersic_index_simulated,
        "aperture_lens_flux_mujy": {
            "band": reference_band,
            "psf_band": worst_band,
            "radii_pixels": aperture_radii,
            "radii_fwhm_multiples": [1.0, 2.0, 3.0, 4.0],
            "centre_pixels": aperture_centre,
            "values": aperture_fluxes,
        },
        "magnification": {
            "area": (
                lensed_source_sum_unit / source_sum_unit
                if source_sum_unit != 0.0
                else None
            ),
            "point": json_number(magnification_point),
        },
        "einstein_radius": {
            "model_parameter": json_number(
                getattr(lens_galaxy.mass, "einstein_radius", float("nan"))
            ),
            "effective": json_number(einstein_radius_effective),
        },
        "positions": [[float(y), float(x)] for y, x in positions],
        "latents": latents,
        "latents_skipped_reason": (
            None
            if latents is not None
            else (
                "the dataset was written outside dataset/ (PYAUTO_TEST_MODE scratch "
                "directory), which util.load_vis_dataset cannot resolve"
            )
        ),
    }

    with open(dataset_path / "truth.json", "w") as f:
        json.dump(truth, f, indent=4)

    print(f"[simulator] wrote {dataset_path}", flush=True)
    print(f"[simulator] bands: {list(band_data)}", flush=True)
    print(
        f'[simulator] worst band {worst_band} (FWHM {worst_psf_fwhm:.4f}"), '
        f"{len(positions)} multiple images, "
        f"magnification {truth['magnification']['area']}",
        flush=True,
    )


def model_truth_from(tracer):
    """
    A JSON-serialisable dump of every galaxy, profile and parameter in the tracer.

    ``al.to_dict`` is not used because it round-trips through PyAutoFit's class registry; this
    is a flat, human-readable record meant to be read by eye and asserted against in tests.
    """
    galaxies = {}

    for index, galaxy in enumerate(tracer.galaxies):
        profiles = {}
        for name in ("bulge", "disk", "mass", "shear", "light"):
            profile = getattr(galaxy, name, None)
            if profile is None:
                continue
            parameters = {}
            for parameter in (
                "centre",
                "ell_comps",
                "intensity",
                "effective_radius",
                "sersic_index",
                "einstein_radius",
                "slope",
                "gamma_1",
                "gamma_2",
            ):
                value = getattr(profile, parameter, None)
                if value is None:
                    continue
                if isinstance(value, (tuple, list, np.ndarray)):
                    parameters[parameter] = [float(v) for v in np.asarray(value)]
                else:
                    parameters[parameter] = float(value)
            entry = {"type": type(profile).__name__, "parameters": parameters}
            profile_list = getattr(profile, "profile_list", None)
            if profile_list is not None:
                entry["profile_list"] = [
                    {
                        "type": type(sub).__name__,
                        "parameters": {
                            parameter: (
                                [float(v) for v in np.asarray(getattr(sub, parameter))]
                                if isinstance(
                                    getattr(sub, parameter), (tuple, list, np.ndarray)
                                )
                                else float(getattr(sub, parameter))
                            )
                            for parameter in (
                                "centre",
                                "ell_comps",
                                "intensity",
                                "sigma",
                                "effective_radius",
                                "sersic_index",
                            )
                            if getattr(sub, parameter, None) is not None
                        },
                    }
                    for sub in profile_list
                ]
            profiles[name] = entry
        galaxies[f"galaxy_{index}"] = {
            "redshift": float(galaxy.redshift),
            "profiles": profiles,
        }

    return galaxies


def parse_args():
    """
    Command-line arguments.

    ``--dataset`` and ``--sample`` name the *input* dataset (the one the resimulated fit was
    made on). They are only used by ``--from-result``, but are always accepted, because the
    smoke runner appends one global ``args_default`` (``--dataset=... --sample=...``) to every
    entry in ``smoke_tests.txt``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a Euclid strong lens in this pipeline's dataset format, either from "
            "analytic truth values or by resimulating a fit you have already run."
        )
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--from-params",
        action="store_true",
        help=(
            "Simulate the analytic lens defined by the truth values at the top of this file. "
            "This is the default when --from-result is not given."
        ),
    )
    mode.add_argument(
        "--from-result",
        action="store_true",
        help=(
            "Rebuild the tracer from a finished fit's model.json + max-log-likelihood sample "
            "instead of the analytic truth values in this file."
        ),
    )
    parser.add_argument(
        "--dataset",
        metavar="name",
        default=None,
        help=(
            "Input dataset subdirectory (--from-result only; ignored otherwise). This is the "
            "dataset the resimulated fit was made on, and the source of the band, PSF, "
            "zero-point and noise conventions."
        ),
    )
    parser.add_argument(
        "--sample",
        metavar="name",
        default=None,
        help="Input sample subdirectory inside dataset/ (--from-result only).",
    )
    parser.add_argument(
        "--unique_tag",
        metavar="name",
        default="initial_lens_model",
        help="Pipeline stage folder holding the result. Default: initial_lens_model.",
    )
    parser.add_argument(
        "--search",
        metavar="name",
        default="vis_lp",
        help="Search name within the stage folder (e.g. vis_lp, vis_pix, vis).",
    )
    parser.add_argument(
        "--result_hash",
        metavar="hash",
        default=None,
        help=(
            "Result hash subdirectory. Default: the most recently modified hash directory "
            "that contains samples_summary.json + model.json."
        ),
    )

    parser.add_argument(
        "--output-sample",
        metavar="name",
        default=DEFAULT_OUTPUT_SAMPLE,
        help=f"Sample directory written to. Default: '{DEFAULT_OUTPUT_SAMPLE}'.",
    )
    parser.add_argument(
        "--output-dataset",
        metavar="name",
        default=DEFAULT_OUTPUT_DATASET,
        help=f"Dataset directory written to. Default: '{DEFAULT_OUTPUT_DATASET}'.",
    )
    parser.add_argument(
        "--force-dataset-dir",
        action="store_true",
        help=(
            "Write to dataset/ even under PYAUTO_TEST_MODE, which otherwise redirects the "
            "output to $PYAUTO_OUTPUT_DIR/simulator/ so a smoke run cannot overwrite the "
            "committed dataset."
        ),
    )

    parser.add_argument(
        "--bands",
        metavar="list",
        default=None,
        help=(
            "Comma-separated bands to simulate. Default: VIS,NIR_Y,NIR_J,NIR_H "
            "(--from-params) or every band of the input dataset (--from-result)."
        ),
    )
    parser.add_argument(
        "--shape",
        metavar="ny,nx",
        default=",".join(str(v) for v in DEFAULT_SHAPE_NATIVE),
        help="Image shape in pixels. Default: 100,100.",
    )
    parser.add_argument(
        "--pixel-scale",
        metavar="float",
        type=float,
        default=DEFAULT_PIXEL_SCALE,
        help="Arcsec per pixel. Default: 0.1 (Euclid VIS).",
    )
    parser.add_argument(
        "--mask-radius",
        metavar="float",
        type=float,
        default=DEFAULT_MASK_RADIUS,
        help="Circular analysis mask radius written to info.json, in arcsec. Default: 3.0.",
    )
    parser.add_argument(
        "--seed",
        metavar="int",
        type=int,
        default=1,
        help="Random seed for the Gaussian noise. Default: 1.",
    )

    parser.add_argument(
        "--sersic-index-prior-edge",
        metavar="float",
        type=float,
        default=5.0,
        help=(
            "A resimulated lens-light sersic_index at or above this value is treated as pinned "
            "against the prior edge and replaced. Default: 5.0."
        ),
    )
    parser.add_argument(
        "--sersic-index-replacement",
        metavar="float",
        type=float,
        default=3.0,
        help=(
            "The sersic_index simulated in place of a prior-edge value; the middle of the "
            "[2, 4] range the DR1 resimulation programme specified. Default: 3.0."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    simulate(parse_args())
