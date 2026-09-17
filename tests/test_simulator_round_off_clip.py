"""
``scripts/simulator.py::simulated_image_from`` — the convolution round-off clip.

Every model image the simulator builds is non-negative (a Sersic is), but PSF-convolving one
returns pixels of order ``-4e-18`` where the true value is zero: floating-point round-off, about
``1e-17`` of the image peak. PyAutoArray's imaging simulator draws its Poisson noise **before**
it checks ``add_poisson_noise_to_data`` (``autoarray/dataset/imaging/simulator.py:246``), so
``np.random.poisson`` is handed a negative lambda and raises ``ValueError: lam < 0`` — even
though this script switches Poisson noise off entirely. Four of the 100 ``dr1_sep1_sersics``
tiles died that way on 2026-09-13 (indices 10, 62, 63 and 69 of the sorted sample); 95 of the
others went through untouched, which is what makes it a trap rather than a visible break.

The clip cannot be applied to the simulator's *input*: the negatives are made by the
convolution, not carried into it. ``simulated_image_from`` therefore runs the convolution
itself, clips, and hands the already-convolved image back.

What these tests pin: the round-off no longer raises, the library genuinely does raise without
the clip (so the guard is load-bearing rather than decorative), a genuinely negative image is
*not* quietly zeroed, and a clean image comes out bit-identical to the unguarded path.

The PSF here is a delta kernel, so "convolution" is the identity and the negative pixel under
test is exactly the one that reaches the Poisson draw.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import autolens as al  # noqa: E402

from scripts import simulator  # noqa: E402


PIXEL_SCALE = 0.1
ROUND_OFF = -1.0e-19


def delta_simulator():
    """
    The simulator this pipeline actually builds (``band_simulator_from`` — Poisson noise off, a
    constant noise-map, no background sky), with a delta-function PSF so the convolution is the
    identity and the negative pixel under test is exactly the one that reaches the Poisson draw.
    """
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1.0

    return simulator.band_simulator_from(
        psf=al.Convolver(
            kernel=al.Array2D.no_mask(values=kernel, pixel_scales=PIXEL_SCALE),
            normalize=True,
        ),
        noise_sigma=0.01,
    )


def image_with(corner_value):
    image = np.zeros((11, 11))
    image[5, 5] = 1.0
    image[0, 0] = corner_value
    return image


def test_a_round_off_negative_no_longer_raises():
    """
    The defect, directly: a pixel at ``-1e-19`` used to kill the run inside the Poisson draw.
    """
    result = simulator.simulated_image_from(
        simulator=delta_simulator(),
        image=image_with(ROUND_OFF),
        pixel_scale=PIXEL_SCALE,
    )

    assert result.min() >= 0.0
    assert result[5, 5] == pytest.approx(1.0)


def test_the_library_really_does_raise_without_the_clip():
    """
    The control. Without this the test above would pass just as well against a library that had
    never had the bug, and the guard would be unfalsifiable decoration.
    """
    with pytest.raises(ValueError, match="lam"):
        delta_simulator().via_image_from(
            image=al.Array2D.no_mask(
                values=image_with(ROUND_OFF), pixel_scales=PIXEL_SCALE
            )
        )


def test_a_genuinely_negative_image_is_not_clipped_away():
    """
    The clip is for round-off and nothing else: a real negative — here half the image peak, some
    seventeen orders of magnitude above round-off — must stop the run rather than be zeroed.
    """
    with pytest.raises(SystemExit, match="genuinely negative"):
        simulator.simulated_image_from(
            simulator=delta_simulator(), image=image_with(-0.5), pixel_scale=PIXEL_SCALE
        )


def test_a_clean_image_is_unchanged():
    """
    Nothing else moved: an image with no negatives comes out of the guarded path exactly as it
    comes out of the library's own.
    """
    image = image_with(0.0)

    guarded = simulator.simulated_image_from(
        simulator=delta_simulator(), image=image, pixel_scale=PIXEL_SCALE
    )
    direct = np.asarray(
        delta_simulator()
        .via_image_from(
            image=al.Array2D.no_mask(values=image, pixel_scales=PIXEL_SCALE)
        )
        .data.native,
        dtype=np.float64,
    )

    assert guarded == pytest.approx(direct, abs=0.0, rel=0.0)
