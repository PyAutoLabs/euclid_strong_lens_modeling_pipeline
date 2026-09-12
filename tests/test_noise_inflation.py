"""
The Gaussian noise inflation behind the ``central_noise`` Sersic variant.

``util.inflate_noise_map_gaussian`` multiplies the noise map by
``1 + A exp(-r^2 / 2 sigma^2)`` and changes nothing else. These tests load the
shipped example dataset with and without the ``noise_inflation`` argument and
compare the two, which pins the three things that would make the variant
meaningless without failing anything:

- that the profile really is the Gaussian bowl, not a top hat (the shape is the
  whole point — ``Imaging.apply_noise_scaling``, the method that already exists,
  is a top-hat *replacement* and is why this helper had to be written);
- that the bowl reaches its intended depth at the centre and has decayed away far
  from it, so the variant neither under- nor over-reaches;
- that the data and the PSF come back untouched, so the fit sees the same photons.

JAX-free, no search, one dataset load per case.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import util  # noqa: E402


EXAMPLE_SAMPLE = "q1_walsmley"
EXAMPLE_DATASET = "102018665_NEG570040238507752998"

AMPLITUDE = 9.0
SIGMA_ARCSEC = 0.17
CENTRE = (0.0, 0.0)


@pytest.fixture(scope="module", autouse=True)
def push_config():
    from autolens import conf

    conf.instance.push(
        new_path=PROJECT_ROOT / "config", output_path=PROJECT_ROOT / "output"
    )


@pytest.fixture(scope="module")
def plain():
    return util.load_vis_dataset(EXAMPLE_DATASET, sample_name=EXAMPLE_SAMPLE)


@pytest.fixture(scope="module")
def inflated():
    return util.load_vis_dataset(
        EXAMPLE_DATASET,
        sample_name=EXAMPLE_SAMPLE,
        noise_inflation={
            "centre": CENTRE,
            "amplitude": AMPLITUDE,
            "sigma_arcsec": SIGMA_ARCSEC,
        },
    )


def radii_squared(dataset):
    """
    The squared distance in arcseconds from ``CENTRE`` to each unmasked pixel.
    ``grid`` and ``noise_map`` share the same slim ordering.
    """
    grid = np.asarray(dataset.grid)

    return (grid[:, 0] - CENTRE[0]) ** 2 + (grid[:, 1] - CENTRE[1]) ** 2


def test_noise_map_is_multiplied_by_the_gaussian_bowl(plain, inflated):
    ratio = np.asarray(inflated.dataset.noise_map) / np.asarray(plain.dataset.noise_map)

    expected = 1.0 + AMPLITUDE * np.exp(
        -radii_squared(plain.dataset) / (2.0 * SIGMA_ARCSEC**2)
    )

    assert ratio == pytest.approx(expected, rel=1e-12)


def test_the_bowl_reaches_its_depth_at_the_centre(plain, inflated):
    """
    No pixel centre sits exactly on the lens centre, so the peak ratio is a little
    under ``1 + A``; at this pixel scale it should still be within a tenth of it.
    """
    ratio = np.asarray(inflated.dataset.noise_map) / np.asarray(plain.dataset.noise_map)

    assert ratio.max() > 1.0 + 0.9 * AMPLITUDE
    assert ratio.max() <= 1.0 + AMPLITUDE


def test_the_bowl_has_decayed_away_far_from_the_centre(plain, inflated):
    """
    Beyond five sigma the noise map must be the one the pipeline always had: the
    variant is a statement about the central pixels, not a global reweighting.
    """
    ratio = np.asarray(inflated.dataset.noise_map) / np.asarray(plain.dataset.noise_map)

    far = radii_squared(plain.dataset) > (5.0 * SIGMA_ARCSEC) ** 2

    assert far.any()
    # `A exp(-25/2)` is 3.4e-5, so 1e-4 is the bound the profile itself sets.
    assert ratio[far] == pytest.approx(1.0, abs=1e-4)


def test_the_data_and_the_psf_are_untouched(plain, inflated):
    assert np.array_equal(
        np.asarray(plain.dataset.data), np.asarray(inflated.dataset.data)
    )
    assert np.array_equal(
        np.asarray(plain.dataset.psf.kernel), np.asarray(inflated.dataset.psf.kernel)
    )


def test_no_noise_inflation_is_the_dataset_the_pipeline_always_loaded(plain):
    """
    The default path. Every existing caller passes nothing, and must get back the
    noise map it got before this argument existed.
    """
    again = util.load_vis_dataset(
        EXAMPLE_DATASET, sample_name=EXAMPLE_SAMPLE, noise_inflation=None
    )

    assert np.array_equal(
        np.asarray(plain.dataset.noise_map), np.asarray(again.dataset.noise_map)
    )
