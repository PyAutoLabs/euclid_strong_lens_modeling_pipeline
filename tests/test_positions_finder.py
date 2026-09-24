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


# ---------------------------------------------------------------------------
# Synthetic flux maps
# ---------------------------------------------------------------------------

N_PIX = 60
PS = 0.1
# A pixel centre of the 60 x 60 frame, off the origin so the centre is honoured.
CENTRE = (0.05, -0.05)


def render(points, amps, n=N_PIX, sigma=1.0):
    """Gaussian blobs (sigma in pixels) of peak ``amps`` at (y, x) arcsec ``points``."""
    yy, xx = np.mgrid[0:n, 0:n]
    flux = np.zeros((n, n))
    for (y, x), a in zip(points, amps):
        r, c = pf.arcsec_to_pixel(y, x, n, n, PS)
        flux += a * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma**2))
    return flux


@pytest.fixture(scope="module")
def quad():
    images, mu = pf.solve_images(QUAD_PARAMS, QUAD_SOURCE, half_width=3.0)
    return images + np.array(CENTRE), mu


def run(flux, **kw):
    # Unit noise: the SNR map is the flux map.
    return pf.find_positions(flux, flux, None, PS, light_centre=CENTRE, **kw)


def keys(positions):
    return {(round(p[0], 2), round(p[1], 2)) for p in positions}


def near(positions, point, tol=0.11):
    return any(np.hypot(p[0] - point[0], p[1] - point[1]) <= tol for p in positions)


def test_clean_quad_is_kept(quad):
    images, _ = quad
    result = run(render(images, [10.0] * 4))
    assert result.status == "keep" and result.review_reason == ""
    assert len(result.positions) == 4
    assert all(near(result.positions, p) for p in images)
    assert result.s_final <= 0.1 and result.threshold == pf.FLOOR
    assert result.n_rounds == 1


def test_weak_counter_image_is_added_only_where_predicted(quad):
    images, _ = quad
    # The fourth image is weak (SNR 1.5); a second weak peak sits where the
    # model predicts nothing.
    stray = (-2.0, 2.2)
    flux = render(images, [10.0, 10.0, 10.0, 1.5]) + render([stray], [1.5])
    cand, weak = pf.flux_candidates(flux, flux, PS, CENTRE)
    assert len(cand) == 3 and len(weak) == 2

    result = run(flux)
    assert result.status == "add" and result.review_reason == ""
    assert len(result.positions) == 4
    assert near(result.positions, images[3])
    assert [a["reason"] for a in result.added] == ["predicted_weak"]
    assert 1.0 <= result.added[0]["snr"] < 2.0
    # No SNR < 2 position without a model prediction behind it.
    assert not near(result.positions, stray, tol=0.5)
    assert result.n_rounds == 2


def test_unresolved_neighbouring_peaks_are_one_candidate():
    # Diagonal neighbours (0.14") are both 4-neighbour maxima but one PSF FWHM
    # cannot resolve them; peaks 0.3" apart are two candidates.
    flux = np.zeros((N_PIX, N_PIX))
    flux[10, 10], flux[11, 11] = 10.0, 9.0
    flux[10, 40], flux[13, 40] = 10.0, 9.0
    cand, _ = pf.flux_candidates(flux, flux, PS, CENTRE)
    assert [(p.flux) for p in cand] == [10.0, 10.0, 9.0]
    assert pf.pixel_to_arcsec(11, 11, N_PIX, N_PIX, PS) not in [[p.y, p.x] for p in cand]


def test_nucleus_seed_is_cut(quad):
    images, _ = quad
    seed = np.vstack([images, [CENTRE[0] + 0.1, CENTRE[1]]])
    result = run(render(images, [10.0] * 4), seed=seed)
    assert [(d["index"], d["reason"]) for d in result.dropped] == [(4, "central_cut")]
    assert result.status == "drop" and len(result.positions) == 4


def test_unpredicted_near_centre_seed_is_dropped(quad):
    # Outside the 0.15" cut, so only the model can reject it.
    images, _ = quad
    seed = np.vstack([images, [CENTRE[0] + 0.3, CENTRE[1] + 0.1]])
    result = run(render(images, [10.0] * 4), seed=seed)
    assert [(d["index"], d["reason"]) for d in result.dropped] == [(4, "unpredicted")]
    assert result.status == "drop" and len(result.positions) == 4
    assert result.s_final <= 0.1


def test_neighbour_seed_is_dropped(quad):
    images, _ = quad
    neighbour = [CENTRE[0] + 2.6, CENTRE[1] + 1.9]
    seed = np.vstack([images[:2], [neighbour], images[2:]])
    result = run(render(images, [10.0] * 4), seed=seed)
    assert [(d["index"], d["reason"]) for d in result.dropped] == [(2, "unpredicted")]
    assert result.status == "drop"
    assert keys(result.positions) == keys(np.delete(seed, 2, 0))


@pytest.mark.parametrize("missing", [0, 1, 2, 3])
def test_bright_predicted_image_over_empty_sky_goes_to_review(quad, missing):
    images, _ = quad
    shown = np.delete(images, missing, 0)
    result = run(render(shown, [10.0] * 3))
    assert result.status == "review"
    assert result.review_reason == "unseen_bright_image"
    assert result.n_unseen_bright >= 1
    # The tile is flagged, the positions are not invented.
    assert keys(result.positions) <= {(round(p[0], 2), round(p[1], 2)) for p in result.seed}


def test_one_sided_arc_without_a_counter_image_goes_to_review():
    arc = [[CENTRE[0] + 1.0, CENTRE[1] + 0.3], [CENTRE[0] + 0.9, CENTRE[1] + 0.7]]
    result = run(render(arc, [10.0, 10.0]))
    assert result.status == "review"
    assert result.review_reason == "unseen_bright_image"


def test_faint_one_sided_arc_is_not_penalised_for_faint_counter_images():
    # At SNR 3 the fold pair's counter-images are predicted below the SNR 2
    # floor: empty sky there is no evidence against the set.
    arc = [[CENTRE[0] + 1.0, CENTRE[1] + 0.3], [CENTRE[0] + 0.9, CENTRE[1] + 0.7]]
    result = run(render(arc, [3.0, 3.0]))
    assert result.review_reason != "unseen_bright_image"


def test_unplaceable_set_empties_to_review():
    # Three peaks on one ray from the centre: no SIE + shear places them.
    ray = [[CENTRE[0], CENTRE[1] + r] for r in (1.0, 1.8, 2.6)]
    result = run(render(ray, [10.0, 9.0, 8.0]))
    assert result.status == "review" and result.review_reason == "empty"
    assert result.positions == [] and result.threshold is None
    assert len(result.dropped) == 3


def test_fewer_than_two_seeds_is_n_lt_2(quad):
    images, _ = quad
    seed = np.array([images[0], [CENTRE[0] + 0.05, CENTRE[1]]])
    result = run(render(images, [10.0] * 4), seed=seed)
    assert result.status == "n_lt_2" and result.review_reason == "n_lt_2"
    assert result.threshold is None


def test_no_candidates_returns_nothing():
    result = run(np.zeros((N_PIX, N_PIX)))
    assert result.positions == [] and result.status == "n_lt_2"


def test_convergence_cap(quad):
    images, _ = quad
    flux = render(images, [10.0, 10.0, 10.0, 1.5])
    # The weak counter-image needs a second round to confirm the set.
    assert run(flux).n_rounds == 2
    capped = run(flux, max_rounds=1)
    assert capped.n_rounds == 1
    assert capped.status == "review" and capped.review_reason == "no_converge"


def test_implausible_model_goes_to_review(quad, monkeypatch):
    images, _ = quad
    monkeypatch.setattr(pf, "J_MAX", 0.0)
    result = run(render(images, [10.0] * 4))
    assert result.status == "review" and result.review_reason == "implausible_J"


def test_over_cap_goes_to_review(quad, monkeypatch):
    images, _ = quad
    # With a zero cap every threshold is over it.
    monkeypatch.setattr(pf, "threshold_from", lambda s: (pf.CAP, True))
    result = run(render(images, [10.0] * 4))
    assert result.status == "review" and result.review_reason == "over_cap"


def test_flux_seed_is_ordered_by_flux_and_capped(quad):
    images, mu = quad
    amps = 3.0 * np.abs(mu)
    result = run(render(images, amps), n_positions=3)
    assert len(result.positions) == 3
    assert all(near(result.positions, p) for p in images[np.argsort(-amps)[:3]])


# ---------------------------------------------------------------------------
# positions_meta.json contract
# ---------------------------------------------------------------------------

PHASE1_KEYS = {
    "version",
    "light_centre",
    "n_raw",
    "positions_used",
    "dropped",
    "flag_index",
    "s_min_all",
    "s_min",
    "J",
    "threshold",
    "factor",
    "floor",
    "cap",
    "status",
    "review_reason",
    "params",
}


def test_meta_keeps_the_phase1_keys_and_adds_the_finder_fields(quad):
    images, _ = quad
    meta = pf.meta_from_result(run(render(images, [10.0, 10.0, 10.0, 1.5])))
    assert PHASE1_KEYS <= set(meta)
    assert {"method", "finder_version", "s_final", "n_rounds", "added", "predicted"} <= set(meta)
    assert meta["method"] == "finder" and meta["version"] == pf.META_VERSION
    assert meta["status"] == "add" and len(meta["positions_used"]) == 4
    assert meta["threshold"] == pf.FLOOR and set(meta["params"]) == set(pf.PARAM_NAMES)


def test_util_reads_the_finder_meta(quad, tmp_path):
    import util

    images, _ = quad
    pf.write_meta(tmp_path, pf.meta_from_result(run(render(images, [10.0] * 4))))
    found, lh = util.positions_likelihood_list_from_meta(tmp_path)
    assert found and len(lh) == 1 and len(lh[0].positions) == 4
    assert lh[0].threshold == pytest.approx(pf.FLOOR)

    empty = run(render([[CENTRE[0], CENTRE[1] + r] for r in (1.0, 1.8, 2.6)], [10, 9, 8]))
    pf.write_meta(tmp_path, pf.meta_from_result(empty))
    assert util.positions_likelihood_list_from_meta(tmp_path) == (True, None)


# ---------------------------------------------------------------------------
# Writer parity: segmentation.py and util.py call the same finder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scene", ["quad", "weak", "arc"])
def test_segmentation_and_util_writers_agree(quad, scene):
    import util

    sys.path.insert(0, str(PROJECT_ROOT / "preprocess"))
    import segmentation

    images, mu = quad
    if scene == "quad":
        flux = render(images, 3.0 * np.abs(mu))
    elif scene == "weak":
        flux = render(images, [10.0, 10.0, 10.0, 1.5])
    else:
        flux = render([[CENTRE[0] + 1.0, CENTRE[1] + 0.3], [CENTRE[0] + 0.9, CENTRE[1] + 0.7]], [3, 3])
    noise = np.full_like(flux, 0.5)
    lens_flux = render([CENTRE], [100.0], sigma=3.0)

    via_util = util._compute_positions_from_source_flux(
        source_flux=flux, noise_map=noise, pixel_scale=PS, lens_flux=lens_flux
    )
    snr = pf.snr_map_from(flux, noise)
    via_seg = segmentation.compute_positions(
        flux, snr, N_PIX, N_PIX, PS, lens_flux=lens_flux
    )
    assert via_seg == via_util
    assert len(via_util) >= 2
    # Both default to the brightest lens-flux pixel as the light centre.
    assert pf.light_centre_from_maps(lens_flux, PS) == pytest.approx(CENTRE)


# ---------------------------------------------------------------------------
# Gate: finder path on a synthetic tile, phase 1 path on the exemplars
# ---------------------------------------------------------------------------


def _write_tile(root, name, flux, noise, lens_flux, positions):
    from astropy.io import fits

    d = Path(root) / name
    (d / "segmentation").mkdir(parents=True)
    vis = lens_flux + flux
    psf = np.zeros((11, 11))
    psf[5, 5] = 1.0
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(vis.astype(np.float32), name="VIS_BGSUB"),
            fits.ImageHDU(psf.astype(np.float32), name="VIS_PSF"),
            fits.ImageHDU(noise.astype(np.float32), name="VIS_RMS"),
        ]
    ).writeto(d / f"{name}.fits")
    fits.writeto(d / "segmentation" / "source_flux.fits", flux.astype(np.float32))
    fits.writeto(d / "segmentation" / "lens_flux.fits", lens_flux.astype(np.float32))
    (d / "info.json").write_text('{"pixel_scale": 0.1, "mask_radius": 3.0, "mask_centre": [0.0, 0.0]}')
    values = [[float(y), float(x)] for y, x in positions]
    (d / "positions.json").write_text(
        '{"type": "instance", "arguments": {"values": {"type": "ndarray", "array": %s}}}'
        % json_dumps(values)
    )
    return d


def json_dumps(v):
    import json

    return json.dumps(v)


def test_gate_runs_the_finder_seeded_from_positions_json(quad, tmp_path):
    import json

    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
    import positions_gate as gate

    images, _ = quad
    # Seed = the pixelised quad + a neighbour; the fourth image is weak on the map
    # and missing from the seed.
    pix = np.array([pf.pixel_to_arcsec(*pf.arcsec_to_pixel(y, x, N_PIX, N_PIX, PS), N_PIX, N_PIX, PS) for y, x in images])
    neighbour = [CENTRE[0] + 2.6, CENTRE[1] + 1.9]
    seed = np.vstack([pix[:3], [neighbour]])
    flux = render(images, [10.0, 10.0, 10.0, 1.5]) * 0.5
    noise = np.full_like(flux, 0.5)
    lens_flux = render([CENTRE], [100.0], sigma=3.0)
    d = _write_tile(tmp_path, "TileSynthetic", flux, noise, lens_flux, seed)

    meta = gate.gate_tile(d)
    assert meta["method"] == "finder" and meta["version"] == gate.GATE_VERSION
    assert PHASE1_KEYS <= set(meta)
    assert meta["status"] == "modify"
    assert [(x["index"], x["reason"]) for x in meta["dropped"]] == [(3, "unpredicted")]
    assert [a["reason"] for a in meta["added"]] == ["predicted_weak"]
    assert len(meta["positions_used"]) == 4 and near(meta["positions_used"], pix[3])
    assert meta["threshold"] == pf.FLOOR
    assert json.loads((d / gate.META_NAME).read_text())["status"] == "modify"

    # --no-finder keeps the phase 1 behaviour on the same tile.
    phase1 = gate.gate_tile(d, finder=False)
    assert phase1["method"] == "gate" and "added" not in phase1

    # The segmentation-map inputs match the segmentation writer's SNR map.
    maps = gate.finder_maps_from_dataset(d)
    np.testing.assert_allclose(maps["snr_map"], pf.snr_map_from(flux, noise), rtol=1e-5, atol=1e-6)


def test_gate_exemplars_without_maps_keep_the_phase1_verdicts(tmp_path):
    # The five output_locked exemplar fixtures ship no segmentation maps, so the
    # gate keeps the phase 1 path on them (tests/test_positions_gate.py pins the
    # verdicts); one is re-run here through the default (finder-enabled) call.
    import json
    import shutil

    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
    import positions_gate as gate

    fixtures = Path(__file__).parent / "fixtures" / "positions_gate"
    for tile in sorted(p.name for p in fixtures.iterdir()):
        assert gate.finder_maps_from_dataset(fixtures / tile) is None
    tile = "Tile102012028RA0853629092626DECNEG0587974678810"
    shutil.copytree(fixtures / tile, tmp_path / tile)
    centre = json.loads((fixtures / tile / "light_centre.json").read_text())["light_centre"]
    meta = gate.gate_tile(tmp_path / tile, light_centre=centre, write=False)
    assert meta["method"] == "gate" and meta["version"] == gate.GATE_VERSION
    assert (meta["status"], meta["dropped"][0]["index"], meta["dropped"][0]["reason"]) == (
        "drop",
        2,
        "outlier_unique",
    )
