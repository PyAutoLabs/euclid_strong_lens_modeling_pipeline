"""
Tests for ``positions_finder.py``: the production gate-based writer
(``find_positions_gate``: SNR >= 3 peaks, the phase 1 gate steps, the pair
floor), the seeded gate path that shares it, and the non-production diagnostic
reconcile loop (``find_positions``) and forward solver (``solve_images``).

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


@pytest.mark.parametrize("scene", ["quad", "weak", "arc", "faint"])
def test_segmentation_and_util_writers_agree(quad, scene):
    import util

    sys.path.insert(0, str(PROJECT_ROOT / "preprocess"))
    import segmentation

    images, mu = quad
    if scene == "quad":
        flux = render(images, 3.0 * np.abs(mu))
    elif scene == "weak":
        flux = render(images, [10.0, 10.0, 10.0, 1.5])
    elif scene == "faint":
        # The fourth image at SNR 2.5 (noise 0.5): no walk-down picks it up.
        flux = render(images, [10.0, 10.0, 10.0, 1.25])
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
    if scene == "arc":
        # The two arc blobs are 0.41" apart (< MIN_SEPARATION): both writers
        # collapse them to one candidate.
        assert len(via_util) == 1
    else:
        assert len(via_util) >= 2
    # Both are the production writer, not the diagnostic loop.
    full = util._positions_result_from_source_flux(flux, noise, PS, lens_flux=lens_flux)
    assert full.method == "gate" and full.positions == via_util
    if scene == "faint":
        assert len(via_util) == 3 and not near(via_util, images[3])
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


# ---------------------------------------------------------------------------
# Production writer: find_positions_gate (SNR >= 3 peaks + gate + pair floor)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def double():
    images, mu = pf.solve_images(DOUBLE_PARAMS, DOUBLE_SOURCE, half_width=3.0)
    return images + np.array(CENTRE), mu


def write_gate(flux, **kw):
    # Unit noise: the SNR map is the flux map.
    return pf.find_positions_gate(flux, flux, light_centre=CENTRE, pixel_scale=PS, **kw)


def test_writer_selects_the_snr3_peaks_of_a_clean_quad(quad):
    images, _ = quad
    result = write_gate(render(images, [10.0] * 4))
    assert result.method == "gate"
    assert result.status == "keep" and result.review_reason == ""
    assert len(result.positions) == 4 and all(near(result.positions, p) for p in images)
    assert result.s_final <= 0.1 and result.threshold == pf.FLOOR
    assert result.snr == pytest.approx([10.0] * 4, abs=0.5)


def test_writer_has_no_snr_walk_down(quad, double):
    images, _ = quad
    # The quad's fourth image at SNR 2.5 is not a candidate...
    result = write_gate(render(images, [10.0, 10.0, 10.0, 2.5]))
    assert len(result.positions) == 3 and not near(result.positions, images[3])
    assert all(near(result.positions, p) for p in images[:3])
    cands = pf.gate_candidates(render(images, [10.0, 10.0, 10.0, 2.5]), render(images, [10.0, 10.0, 10.0, 2.5]), PS, CENTRE)
    assert all(p.snr >= pf.SNR_MIN for p in cands)
    # ...and a double whose counter-image sits at SNR 2.5 is never completed:
    # one candidate, penalty off (n_lt_2, no threshold).
    dimg, _ = double
    one = write_gate(render(dimg, [10.0, 2.5]))
    assert len(one.positions) == 1 and not near(one.positions, dimg[1])
    assert one.status == "n_lt_2" and one.threshold is None


def test_writer_excludes_the_light_centre_disc(quad):
    images, _ = quad
    nucleus = [CENTRE[0] + 0.1, CENTRE[1]]
    flux = render(images, [10.0] * 4)
    flux[pf.arcsec_to_pixel(*nucleus, N_PIX, N_PIX, PS)] = 50.0
    cands = pf.gate_candidates(flux, flux, PS, CENTRE)
    assert not near([p.yx for p in cands], nucleus, tol=0.05)
    result = write_gate(flux)
    assert not near(result.positions, nucleus, tol=0.14)
    assert len(result.positions) == 4 and result.status == "keep"


def test_writer_merges_peaks_within_one_psf_fwhm():
    flux = np.zeros((N_PIX, N_PIX))
    flux[10, 10], flux[11, 11] = 10.0, 9.0
    flux[10, 40], flux[13, 40] = 10.0, 9.0
    # The merge alone (de-duplication off): 0.14" merges, 0.3" does not.
    cands = pf.gate_candidates(flux, flux, PS, CENTRE, min_separation=0.0)
    assert [p.flux for p in cands] == [10.0, 10.0, 9.0]


def _point_flux(points_snrs):
    flux = np.zeros((N_PIX, N_PIX))
    for (y, x), v in points_snrs:
        flux[pf.arcsec_to_pixel(y, x, N_PIX, N_PIX, PS)] = v
    return flux


def test_candidates_same_arc_peaks_within_min_separation_keep_the_brighter():
    # Two SNR >= 3 peaks 0.4" apart on one arc: only the brighter survives.
    flux = _point_flux([((1.05, -0.05), 8.0), ((1.05, 0.35), 6.0)])
    cands = pf.gate_candidates(flux, flux, PS, CENTRE)
    assert pf.MIN_SEPARATION == 0.7
    assert [p.flux for p in cands] == [8.0]
    assert near([p.yx for p in cands], (1.05, -0.05), tol=0.01)


def test_candidates_counter_image_beats_fainter_same_arc_peaks():
    # Three same-arc peaks 0.3" apart, all brighter than a counter-image 1.4"
    # away: the arc keeps one slot and the counter-image is kept ahead of the
    # other two arc peaks.
    arc = [((0.75, -0.35), 12.0), ((0.75, -0.05), 11.0), ((0.75, 0.25), 10.0)]
    counter = ((-0.65, -0.05), 4.0)
    flux = _point_flux(arc + [counter])
    cands = pf.gate_candidates(flux, flux, PS, CENTRE, n_positions=2)
    assert [p.flux for p in cands] == [12.0, 4.0]
    assert near([p.yx for p in cands], counter[0], tol=0.01)
    # Without the de-duplication the cap fills with the arc.
    raw = pf.gate_candidates(flux, flux, PS, CENTRE, n_positions=2, min_separation=0.0)
    assert [p.flux for p in raw] == [12.0, 11.0]


def test_candidates_cap_applies_after_de_duplication():
    # Six peaks >= 0.7" apart plus one 0.3" from the brightest: de-dup drops the
    # close one, then the cap keeps the brightest n of the survivors.
    pts = [((1.05, -1.05), 20.0), ((1.05, -0.75), 19.0), ((1.05, 0.95), 18.0),
           ((-0.95, -1.05), 17.0), ((-0.95, 0.95), 16.0), ((0.05, 1.95), 15.0),
           ((0.05, -2.05), 14.0)]
    flux = _point_flux(pts)
    all_c = pf.gate_candidates(flux, flux, PS, CENTRE, n_positions=10)
    assert [p.flux for p in all_c] == [20.0, 18.0, 17.0, 16.0, 15.0, 14.0]
    capped = pf.gate_candidates(flux, flux, PS, CENTRE)
    assert [p.flux for p in capped] == [20.0, 18.0, 17.0, 16.0]
    assert len(capped) == pf.N_POSITIONS


def test_writer_caps_at_the_brightest_n_positions(quad):
    images, mu = quad
    amps = 3.0 * np.abs(mu) + 3.0
    result = write_gate(render(images, amps), n_positions=3)
    assert len(result.positions) == 3
    assert all(near(result.positions, p) for p in images[np.argsort(-amps)[:3]])


def test_pair_floor_drops_a_2_sigma_partner(double):
    dimg, _ = double
    result = pf.gate_result(dimg, CENTRE, snrs=[10.0, 2.0])
    assert result.status == "review" and result.review_reason == "pair_floor"
    assert result.positions == [] and result.threshold is None
    assert [(d["index"], d["reason"]) for d in result.dropped] == [(1, "pair_floor")]
    meta = pf.meta_from_result(result)
    assert meta["positions_used"] == [] and meta["added"] == []


def test_pair_with_both_above_the_floor_is_kept(double):
    dimg, _ = double
    result = pf.gate_result(dimg, CENTRE, snrs=[10.0, 3.0])
    assert result.status == "keep" and result.review_reason == ""
    assert len(result.positions) == 2 and result.threshold == pf.FLOOR
    # And through the writer on a flux map.
    written = write_gate(render(dimg, [10.0, 3.0]))
    assert written.status == "keep" and len(written.positions) == 2


def test_pair_floor_applies_after_an_outlier_drop(double):
    # A nucleus-like inner position (0.25" from the centre, outside the cut)
    # forces the gate's outlier drop; the surviving pair then meets the floor.
    # Without SNRs (no maps) the phase 1 verdict stands.
    dimg, _ = double
    seed = np.vstack([dimg, [[CENTRE[0] + 0.25, CENTRE[1]]]])
    base = pf.gate_positions(seed, CENTRE)
    assert base["status"] == "drop" and base["dropped"][0]["reason"] == "outlier_geom_inner"
    no_maps = pf.gate_result(seed, CENTRE, snrs=None)
    assert no_maps.status == "drop" and no_maps.positions == base["positions_used"]
    bright = pf.gate_result(seed, CENTRE, snrs=[10.0, 5.0, 1.0])
    assert bright.status == "drop" and len(bright.positions) == 2
    faint = pf.gate_result(seed, CENTRE, snrs=[10.0, 2.0, 1.0])
    assert faint.status == "review" and faint.review_reason == "pair_floor"
    assert [d["reason"] for d in faint.dropped] == ["outlier_geom_inner", "pair_floor"]
    assert faint.positions == []


def test_writer_meta_keeps_the_phase1_contract(quad, tmp_path):
    import util

    images, _ = quad
    meta = pf.meta_from_result(write_gate(render(images, [10.0] * 4)))
    assert PHASE1_KEYS <= set(meta)
    assert meta["method"] == "gate" and meta["version"] == pf.META_VERSION == "2.1"
    assert meta["added"] == [] and "rounds" not in meta and "predicted" not in meta
    pf.write_meta(tmp_path, meta)
    found, lh = util.positions_likelihood_list_from_meta(tmp_path)
    assert found and len(lh[0].positions) == 4 and lh[0].threshold == pytest.approx(pf.FLOOR)


# ---------------------------------------------------------------------------
# Seeded gate path (positions_gate.gate_tile) vs the unseeded writer
# ---------------------------------------------------------------------------


def _gate_module():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "tools"))
    import positions_gate as gate

    return gate


COMPARED = ("positions_used", "status", "review_reason", "threshold", "s_min", "s_min_all", "flag_index")


@pytest.mark.parametrize("scene", ["quad", "neighbour"])
def test_seeded_and_unseeded_paths_agree_when_positions_json_is_the_peak_set(quad, tmp_path, scene):
    gate = _gate_module()
    images, _ = quad
    points, amps = list(images), [5.0, 5.0, 5.0, 5.0]
    if scene == "neighbour":
        points, amps = list(images[:3]) + [[CENTRE[0] + 2.6, CENTRE[1] + 1.9]], [5.0, 5.0, 5.0, 4.0]
    flux = render(points, amps) * 0.5
    noise = np.full_like(flux, 0.5)
    lens_flux = render([CENTRE], [100.0], sigma=3.0)
    d = _write_tile(tmp_path, "TileSeeded", flux, noise, lens_flux, [[0.0, 0.0], [1.0, 1.0]])
    centre = gate.light_centre_from_dataset(d)
    snr = pf.snr_map_from(flux, noise)
    peaks = pf.gate_candidates(flux, snr, PS, centre)
    # positions.json := the SNR >= 3 peak set, brightest first.
    d2 = _write_tile(tmp_path, "TileSeeded2", flux, noise, lens_flux, [p.yx for p in peaks])

    seeded = gate.gate_tile(d2, light_centre=centre, write=False)
    unseeded = pf.meta_from_result(pf.find_positions_gate(flux, snr, light_centre=centre, pixel_scale=PS))
    for k in COMPARED:
        assert seeded[k] == unseeded[k], k
    assert seeded["method"] == unseeded["method"] == "gate"
    assert seeded["version"] == gate.GATE_VERSION == "2.1"
    if scene == "neighbour":
        assert seeded["status"] == "drop" and len(seeded["positions_used"]) == 3
    else:
        assert seeded["status"] == "keep" and len(seeded["positions_used"]) == 4


def test_gate_tile_applies_the_pair_floor_from_the_snr_map(double, tmp_path):
    import json

    gate = _gate_module()
    dimg, _ = double
    pix = [pf.pixel_to_arcsec(*pf.arcsec_to_pixel(y, x, N_PIX, N_PIX, PS), N_PIX, N_PIX, PS) for y, x in dimg]
    noise = np.full((N_PIX, N_PIX), 0.5)
    lens_flux = render([CENTRE], [100.0], sigma=3.0)
    # Partner at SNR 2.0: review, no positions.
    d = _write_tile(tmp_path, "TilePairFaint", render(dimg, [5.0, 1.0]), noise, lens_flux, pix)
    meta = gate.gate_tile(d)
    assert meta["status"] == "review" and meta["review_reason"] == "pair_floor"
    assert meta["positions_used"] == [] and meta["threshold"] is None
    assert meta["dropped"][-1]["reason"] == "pair_floor" and meta["dropped"][-1]["index"] == 1
    assert json.loads((d / gate.META_NAME).read_text())["review_reason"] == "pair_floor"
    # Partner at SNR 4: kept.
    d = _write_tile(tmp_path, "TilePairBright", render(dimg, [5.0, 2.0]), noise, lens_flux, pix)
    meta = gate.gate_tile(d)
    assert meta["status"] == "keep" and len(meta["positions_used"]) == 2
    assert meta["added"] == [] and meta["method"] == "gate"
    # The segmentation-map inputs match the segmentation writer's SNR map.
    maps = gate.finder_maps_from_dataset(d)
    np.testing.assert_allclose(maps["snr_map"], pf.snr_map_from(render(dimg, [5.0, 2.0]), noise), rtol=1e-5, atol=1e-6)
