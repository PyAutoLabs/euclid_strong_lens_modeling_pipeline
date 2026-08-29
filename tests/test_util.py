"""
Unit tests for the shared pipeline helpers in ``util.py``.

Scope: the pure helpers and the dataset-loading contracts that the DR1 port
introduced — the six-tuple CLI, the noise-mask fallback chain, the worst-band
PSF header contract, the ``positions.json`` fallback maths, and the latent
key set.

These tests are deliberately **JAX-free** (no ``use_jax=True`` anywhere) and
run no non-linear search, so the whole module is a few seconds. Fits, latent
*values* and visualisation belong to the smoke suite, not here.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import util  # noqa: E402


EXAMPLE_SAMPLE = "q1_walsmley"
EXAMPLE_DATASET = "102018665_NEG570040238507752998"
EXAMPLE_PATH = PROJECT_ROOT / "dataset" / EXAMPLE_SAMPLE / EXAMPLE_DATASET

# `load_vis_dataset` resolves `dataset/` relative to `util.py`, so a dataset
# built for a test has to live inside the repository's own dataset tree. This
# sample name is created and destroyed by the `tmp_sample` fixture below.
TMP_SAMPLE = "_pytest_util"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def push_config():
    """
    Point the library at this repository's ``config/`` so ``config/latent.yaml``
    governs which latents are enabled.
    """
    from autolens import conf

    conf.instance.push(
        new_path=PROJECT_ROOT / "config", output_path=PROJECT_ROOT / "output"
    )


@pytest.fixture(scope="session")
def tmp_sample():
    """
    A throwaway sample directory inside ``dataset/`` that tests can populate
    with dataset variants. Removed whole at the end of the session.
    """
    sample_path = PROJECT_ROOT / "dataset" / TMP_SAMPLE
    shutil.rmtree(sample_path, ignore_errors=True)
    sample_path.mkdir(parents=True)
    try:
        yield sample_path
    finally:
        shutil.rmtree(sample_path, ignore_errors=True)


def _make_dataset_copy(sample_path: Path, name: str) -> Path:
    """
    Copy the shipped example dataset's FITS + ``info.json`` under a new dataset
    name, with **no** noise-scaling mask of any kind.
    """
    dataset_path = sample_path / name
    dataset_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXAMPLE_PATH / f"{EXAMPLE_DATASET}.fits", dataset_path / f"{name}.fits")
    shutil.copy(EXAMPLE_PATH / "info.json", dataset_path / "info.json")
    return dataset_path


def _write_band_mask(path: Path, row_slice: slice, shape=(100, 100)):
    """
    Write a noise-scaling mask FITS marking one horizontal band of rows.

    ``load_vis_dataset`` loads these with ``invert=True``, so a marked (``1``)
    pixel becomes ``False`` in the ``Mask2D`` and is exactly what
    ``apply_noise_scaling`` raises to ``1e8``.
    """
    array = np.zeros(shape, dtype=np.float64)
    array[row_slice, :] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, array, overwrite=True)


# Two disjoint bands, both comfortably inside the 3.5" circular analysis mask
# (rows 15-85) and clear of the +/-0.3" brightest-pixel search box (rows 47-53).
ARTEFACT_ROWS = slice(40, 43)
EXTRA_GALAXIES_ROWS = slice(58, 61)
SCALED_NOISE = 1e7  # anything at or above this was raised to `noise_value=1e8`


def _noise_native(euclid_dataset):
    return np.asarray(euclid_dataset.dataset.noise_map.native)


# ---------------------------------------------------------------------------
# parse_fit_args — the six-tuple CLI
# ---------------------------------------------------------------------------


def test_parse_fit_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--dataset=abc"])

    result = util.parse_fit_args()

    assert len(result) == 6
    sample, dataset, iterations, cores, use_cpu, skip_pix = result
    assert sample is None
    assert dataset == "abc"
    assert iterations == 5000
    assert cores == 1
    assert use_cpu is False
    assert skip_pix is False


def test_parse_fit_args_all_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dataset=abc",
            "--sample=xyz",
            "--iterations_per_quick_update=250",
            "--number_of_cores=8",
            "--use_cpu",
            "--skip_pix",
        ],
    )

    sample, dataset, iterations, cores, use_cpu, skip_pix = util.parse_fit_args()

    assert (sample, dataset) == ("xyz", "abc")
    assert (iterations, cores) == (250, 8)
    assert isinstance(iterations, int) and isinstance(cores, int)
    assert use_cpu is True
    assert skip_pix is True


def test_parse_fit_args_requires_dataset(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--sample=xyz"])

    with pytest.raises(SystemExit):
        util.parse_fit_args()


# ---------------------------------------------------------------------------
# psf_fwhm_arcsec_from_primary_header — the WORST_PSF_* contract
# ---------------------------------------------------------------------------


def test_psf_fwhm_prefers_mer_over_hdr():
    header = fits.Header({"WORST_PSF_MER": 1.25, "WORST_PSF_HDR": 1.40})

    assert util.psf_fwhm_arcsec_from_primary_header(header, "d") == pytest.approx(1.25)


def test_psf_fwhm_skips_the_minus_99_sentinel():
    """``-99`` is the Euclid "OU-MER did not measure this" sentinel."""
    header = fits.Header({"WORST_PSF_MER": -99.0, "WORST_PSF_HDR": 1.40})

    assert util.psf_fwhm_arcsec_from_primary_header(header, "d") == pytest.approx(1.40)


def test_psf_fwhm_raises_when_every_key_is_absent():
    """
    The function refuses to guess: the four aperture-flux latents are evaluated
    at multiples of this FWHM, so a wrong value silently corrupts the
    matched-aperture photometry rather than failing loudly.
    """
    with pytest.raises(ValueError, match="WORST_PSF_MER"):
        util.psf_fwhm_arcsec_from_primary_header(fits.Header(), "some_tile")


def test_psf_fwhm_on_the_shipped_example_dataset():
    header = fits.getheader(EXAMPLE_PATH / f"{EXAMPLE_DATASET}.fits", 0)

    assert util.psf_fwhm_arcsec_from_primary_header(
        header, EXAMPLE_DATASET
    ) == pytest.approx(float(header["WORST_PSF_MER"]))


def test_load_vis_dataset_degrades_when_worst_band_is_absent(tmp_sample, capsys):
    """
    ``WORST_BAND`` is stamped by the upstream Euclid cut-out generator, not by
    this pipeline. When it is missing the aperture-flux latents are skipped
    (``None``) with a warning rather than the load failing.
    """
    name = "no_worst_band"
    dataset_path = _make_dataset_copy(tmp_sample, name)
    with fits.open(dataset_path / f"{name}.fits", mode="update") as hdul:
        del hdul[0].header["WORST_BAND"]

    d = util.load_vis_dataset(name, sample_name=TMP_SAMPLE)

    assert d.psf_lowest_resolution is None
    assert d.psf_lowest_resolution_fwhm is None
    assert "WORST_BAND missing" in capsys.readouterr().out


def test_load_vis_dataset_reads_worst_band_on_the_example(tmp_sample):
    d = util.load_vis_dataset(EXAMPLE_DATASET, sample_name=EXAMPLE_SAMPLE)

    assert d.psf_lowest_resolution is not None
    assert d.psf_lowest_resolution_fwhm == pytest.approx(1.2929935, abs=1e-6)


# ---------------------------------------------------------------------------
# load_vis_dataset — the noise-scaling mask fallback chain
# ---------------------------------------------------------------------------


def test_noise_mask_prefers_segmentation_artefact_binary(tmp_sample):
    """
    ``segmentation/artefact_binary.fits`` is what DR1 preprocessing writes; when
    both it and the older ``mask_extra_galaxies.fits`` are present it wins, and
    the loop stops — the second mask is never applied.
    """
    name = "both_masks"
    dataset_path = _make_dataset_copy(tmp_sample, name)
    _write_band_mask(
        dataset_path / "segmentation" / "artefact_binary.fits", ARTEFACT_ROWS
    )
    _write_band_mask(dataset_path / "mask_extra_galaxies.fits", EXTRA_GALAXIES_ROWS)

    noise = _noise_native(util.load_vis_dataset(name, sample_name=TMP_SAMPLE))

    assert (noise[ARTEFACT_ROWS, 50] >= SCALED_NOISE).all()
    assert (noise[EXTRA_GALAXIES_ROWS, 50] < SCALED_NOISE).all()


def test_noise_mask_falls_back_to_mask_extra_galaxies(tmp_sample):
    """The shipped example dataset's layout: no ``segmentation/`` directory."""
    name = "legacy_mask_only"
    dataset_path = _make_dataset_copy(tmp_sample, name)
    _write_band_mask(dataset_path / "mask_extra_galaxies.fits", EXTRA_GALAXIES_ROWS)

    noise = _noise_native(util.load_vis_dataset(name, sample_name=TMP_SAMPLE))

    assert (noise[EXTRA_GALAXIES_ROWS, 50] >= SCALED_NOISE).all()
    assert (noise[ARTEFACT_ROWS, 50] < SCALED_NOISE).all()


def test_noise_mask_absent_applies_no_scaling(tmp_sample):
    name = "no_masks"
    _make_dataset_copy(tmp_sample, name)

    noise = _noise_native(util.load_vis_dataset(name, sample_name=TMP_SAMPLE))

    assert (noise < SCALED_NOISE).all()


def test_noise_mask_of_the_wrong_shape_is_refused(tmp_sample):
    """
    A mask cut out at a different size cannot be applied to this dataset; the
    shape guard skips it instead of raising.
    """
    name = "wrong_shape_mask"
    dataset_path = _make_dataset_copy(tmp_sample, name)
    _write_band_mask(
        dataset_path / "segmentation" / "artefact_binary.fits",
        ARTEFACT_ROWS,
        shape=(60, 60),
    )

    noise = _noise_native(util.load_vis_dataset(name, sample_name=TMP_SAMPLE))

    assert (noise < SCALED_NOISE).all()


# ---------------------------------------------------------------------------
# _find_local_maxima / _compute_positions_from_source_flux
# ---------------------------------------------------------------------------


def test_find_local_maxima_orders_by_brightness_and_skips_the_border():
    flux = np.zeros((7, 7))
    flux[2, 2] = 5.0
    flux[4, 4] = 9.0
    flux[0, 3] = 100.0  # on the border — must be ignored

    maxima = util._find_local_maxima(flux)

    assert maxima == [(9.0, 4, 4), (5.0, 2, 2)]


def test_pixel_to_arcsec_uses_the_half_pixel_convention():
    # Centre of a 10 x 10 frame at 0.1"/pixel: rows 4/5 and cols 4/5 straddle
    # the origin at +/- half a pixel.
    assert util._pixel_to_arcsec(4, 5, 10, 10, 0.1) == pytest.approx([0.05, 0.05])
    assert util._pixel_to_arcsec(5, 4, 10, 10, 0.1) == pytest.approx([-0.05, -0.05])


def _synthetic_source_flux(shape=(41, 41)):
    """Two peaks placed symmetrically either side of the frame centre."""
    flux = np.zeros(shape)
    flux[10, 20] = 40.0
    flux[30, 20] = 30.0
    return flux


def test_compute_positions_returns_a_counter_image_pair():
    flux = _synthetic_source_flux()
    noise = np.ones_like(flux)

    positions = util._compute_positions_from_source_flux(
        source_flux=flux, noise_map=noise, pixel_scale=0.1
    )

    assert len(positions) == 2
    # Brightest first, and the pair straddles the lens centre in y.
    assert positions[0] == pytest.approx(util._pixel_to_arcsec(10, 20, 41, 41, 0.1))
    assert positions[0][0] * positions[1][0] < 0


def test_compute_positions_filters_below_the_signal_to_noise_threshold():
    """Peaks below S/N 3 are not multiple images."""
    flux = _synthetic_source_flux()
    noise = np.full_like(flux, 100.0)

    assert (
        util._compute_positions_from_source_flux(
            source_flux=flux, noise_map=noise, pixel_scale=0.1
        )
        == []
    )


def test_compute_positions_caps_at_n_positions():
    flux = np.zeros((41, 41))
    for row, value in ((6, 40.0), (14, 35.0), (26, 30.0), (34, 25.0), (38, 20.0)):
        flux[row, 20] = value

    positions = util._compute_positions_from_source_flux(
        source_flux=flux, noise_map=np.ones_like(flux), pixel_scale=0.1, n_positions=3
    )

    assert len(positions) == 3


def test_compute_positions_without_a_noise_map_skips_filtering():
    positions = util._compute_positions_from_source_flux(
        source_flux=_synthetic_source_flux(), noise_map=None, pixel_scale=0.1
    )

    assert len(positions) == 2


# ---------------------------------------------------------------------------
# LatentEuclid — the key set this pipeline's config/latent.yaml enables
# ---------------------------------------------------------------------------


EXPECTED_LATENT_KEYS = [
    # config-enabled library latents, in `latent_keys_enabled()` order
    "total_lens_flux",
    "total_lensed_source_flux",
    "total_source_flux",
    "total_lens_flux_mujy",
    "total_lensed_source_flux_mujy",
    "total_source_flux_mujy",
    "magnification",
    "effective_einstein_radius",
    # Euclid-only FWHM aperture-flux latents
    "total_lens_flux_1_fwhm_mujy",
    "total_lens_flux_2_fwhm_mujy",
    "total_lens_flux_3_fwhm_mujy",
    "total_lens_flux_4_fwhm_mujy",
]


def test_latent_euclid_keys_match_the_pipeline_config():
    """
    12 keys: the eight the library enables through this repository's
    ``config/latent.yaml`` followed by the four aperture-flux latents that
    ``LatentEuclid`` adds. ``keys()`` is static and does not read the analysis.
    """
    assert util.LatentEuclid.keys(None) == EXPECTED_LATENT_KEYS


def test_latent_euclid_aperture_keys_are_the_tail_of_keys():
    keys = util.LatentEuclid.keys(None)

    assert keys[-4:] == util.LatentEuclid.APERTURE_LATENT_KEYS


def test_analysis_imaging_declares_latent_euclid():
    assert util.AnalysisImaging.Latent is util.LatentEuclid


def test_latent_yaml_enables_every_library_key_the_pipeline_documents():
    with open(PROJECT_ROOT / "config" / "latent.yaml") as f:
        text = f.read()

    for key in (
        "total_lens_flux_mujy",
        "total_lensed_source_flux_mujy",
        "total_source_flux_mujy",
        "magnification",
        "effective_einstein_radius",
    ):
        assert f"{key}: true" in text
