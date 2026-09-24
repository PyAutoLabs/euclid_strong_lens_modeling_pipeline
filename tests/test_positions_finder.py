"""
Tests for the model-guided multiple-image finder (``positions_finder.py``).

The finder is numpy/scipy only. Synthetic image sets are generated with its own
tracer (``tests/test_positions_gate.py`` pins that tracer to ``al.Tracer``), and
the solver is checked against an independent brute-force least-squares root
search and the analytic SIS double.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import least_squares

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import positions_finder as pf  # noqa: E402

QUAD_PARAMS = (1.2, 0.15, 0.05, 0.04, -0.03)
QUAD_SOURCE = (0.03, 0.05)
DOUBLE_PARAMS = (1.0, 0.1, -0.05, 0.02, 0.03)
DOUBLE_SOURCE = (0.35, 0.2)


def brute_force_images(params, source, extent=3.0, n=25):
    """Independent oracle: least-squares roots of the lens equation from a start grid."""
    sols = []
    for y0 in np.linspace(-extent, extent, n):
        for x0 in np.linspace(-extent, extent, n):
            res = least_squares(
                lambda t: pf.source_positions(t[None, :], params)[0] - np.asarray(source),
                [y0, x0],
                xtol=1e-14,
                ftol=1e-14,
            )
            if np.hypot(*res.fun) < 1e-9 and np.hypot(*res.x) > 0.05:
                if all(np.hypot(*(res.x - s)) > 1e-4 for s in sols):
                    sols.append(res.x)
    return np.array(sols)


# ---------------------------------------------------------------------------
# solve_images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, source, n_images",
    [(QUAD_PARAMS, QUAD_SOURCE, 4), (DOUBLE_PARAMS, DOUBLE_SOURCE, 2)],
    ids=["quad", "double"],
)
def test_solve_images_round_trips_through_source_positions(params, source, n_images):
    images, mu = pf.solve_images(params, source, half_width=3.0)
    assert len(images) == n_images and len(mu) == n_images
    # Every solved image traces back to the source within the tolerance...
    traced = pf.source_positions(images, params)
    assert np.hypot(*(traced - np.asarray(source)).T).max() < pf.SOLVE_TOL
    # ...and the known images are all recovered.
    truth = brute_force_images(params, source)
    assert len(truth) == n_images
    for t in truth:
        assert np.hypot(*(images - t).T).min() < 1e-3
    # Ordered by |mu|, none inside the central-image exclusion.
    assert np.all(np.diff(np.abs(mu)) <= 0)
    assert np.hypot(images[:, 0], images[:, 1]).min() >= pf.CENTRAL_RADIUS


def test_solve_images_quad_parities():
    _, mu = pf.solve_images(QUAD_PARAMS, QUAD_SOURCE, half_width=3.0)
    # Two minima (mu > 0) and two saddles (mu < 0).
    assert (mu > 0).sum() == 2 and (mu < 0).sum() == 2


def test_solve_images_sis_double_positions_and_magnifications():
    # SIS: images at r = thetaE +/- beta, mu = r / (r - thetaE). (The tracer clips
    # q at 0.99999, so the "SIS" is exact only to ~1e-5.)
    theta_e, beta = 1.0, 0.3
    images, mu = pf.solve_images((theta_e, 0.0, 0.0, 0.0, 0.0), (0.0, beta), half_width=2.5)
    assert len(images) == 2
    np.testing.assert_allclose(images[0], [0.0, theta_e + beta], atol=1e-4)
    np.testing.assert_allclose(images[1], [0.0, -(theta_e - beta)], atol=1e-4)
    np.testing.assert_allclose(mu, [(theta_e + beta) / beta, -(theta_e - beta) / beta], rtol=1e-3)


def test_solve_images_near_ring_model_returns_roots_not_a_chain():
    # A nearly round lens with the source almost on axis: a long chain of grid
    # minima sits below SOLVE_TOL along the ring, but only true roots are images.
    params = (1.0, 0.02, 0.0, 0.0, 0.0)
    images, _ = pf.solve_images(params, (0.001, 0.002), half_width=2.0)
    assert 2 <= len(images) <= 4
    traced = pf.source_positions(images, params)
    assert np.hypot(*(traced - np.array([0.001, 0.002])).T).max() < pf.SOLVE_ROOT_TOL


def test_solve_images_outside_the_grid_finds_nothing():
    images, mu = pf.solve_images(QUAD_PARAMS, QUAD_SOURCE, half_width=0.5)
    assert images.shape == (0, 2) and mu.shape == (0,)
