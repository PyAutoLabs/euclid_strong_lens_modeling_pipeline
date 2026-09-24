"""
Multiple-image positions for the ``vis_lp`` positions penalty.

One pure numpy / scipy module (no PyAutoLens import) shared by the three places
that decide a tile's multiple-image positions:

- ``preprocess/segmentation.py::compute_positions`` (the canonical
  ``positions.json`` writer),
- ``util._compute_positions_from_source_flux`` (the ``load_vis_dataset``
  fallback when a tile ships no ``positions.json``),
- ``scripts/tools/positions_gate.py`` (the pre-submit gate on an existing
  ``positions.json``).

PRODUCTION PATH (the gate-based writer, :func:`find_positions_gate`)
    1. **candidates** (:func:`gate_candidates`) -- local maxima of
       ``segmentation/source_flux.fits`` with SNR >= ``SNR_MIN`` (3) outside a
       ``CENTRAL_RADIUS`` (0.15", one VIS PSF FWHM) disc around the light
       centre (the brightest lens-flux pixel, else the mask centre), a peak
       within 0.15" of a brighter one merged into it, the brightest
       ``n_positions`` kept. There is no SNR walk-down of any kind (the old
       writers' walk-down to SNR 0 was the defect that put nuclei,
       companions and neighbours into ``positions.json``);
    2. **phase 1 gate** (:func:`gate_positions`) -- central cut, the
       fixed-centre SIE + external shear quick fit (:func:`quick_fit`,
       ``vis_lp``'s model space), the leave-one-out outlier drop
       (:func:`outlier_drop`), the plausibility flag and the threshold
       ``T = min(max(2 s_final, 0.3"), 0.5")`` with its review reasons;
    3. **pair floor** (:func:`apply_pair_floor`) -- a final set of exactly
       two positions needs both peak SNRs >= ``SNR_PAIR`` (3); otherwise the
       fainter is dropped and the tile goes to review (``pair_floor``) with
       no positions.

    Steps 2-3 are :func:`gate_result`; the gate runs it on ``positions.json``
    (SNRs read off the segmentation SNR map), the writers on the step 1 set, so
    seeded and unseeded paths agree whenever ``positions.json`` equals the
    SNR >= 3 peak set. :func:`meta_from_result` writes the
    ``positions_meta.json`` contract ``util.load_vis_dataset`` reads.

NON-PRODUCTION DIAGNOSTICS
    :func:`find_positions` (a compute / fit / solve / reconcile loop that adds
    model-predicted weak counter-images and drops positions a forward solve
    cannot place) and :func:`solve_images` (a numpy stand-in for
    ``al.PointSolver``) are kept for the witness script
    (``scripts/tools/positions_finder_witness.py --reconcile``) and tooling;
    nothing in the production path calls them. On the 2026-09-24 10-lens
    calibration sample the loop's nucleus removals all came from the central
    cut and the outlier drop, its reviews from the gate's own conditions, its
    model-guided counter-image search never fired and its only unique
    contribution was adding predicted images (one likely a contaminant), while
    unseeded it modified 3/5 good tiles. The gate-based path on the same sample:
    the good five unchanged, the nucleus cut on Tile102014701, Tile102008208
    and Tile102022005 to ``pair_floor`` review, Tile102012741 and
    Tile102023528 to review.

    The research behind the numbers (the 2026-09-23 census of 14,032 DR1 tiles
    and the 2026-09-24 method study) is in the ``euclid_dr1`` project under
    ``inspect/positions_census/research/`` (``SYNTHESIS.md``,
    ``B_final_method.md``).

All positions are PyAutoLens ``(y, x)`` arcsec (y up, x right, origin at the
frame centre, half-pixel offset convention).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize

FINDER_VERSION = "2.1"

# Hard cut around the light centre (one VIS PSF FWHM); also the solver's
# central-image exclusion and the candidate exclusion disc.
CENTRAL_RADIUS = 0.15
# A set is "traceable" when s_min <= D_OUT (the vis_lp default threshold).
D_OUT = 0.2
# A leave-one-out candidate wins on cost when it beats the runner-up by DJ.
DJ = 4.0
# T = min(max(FACTOR * s_final, FLOOR), CAP).
FACTOR = 2.0
FLOOR = 0.3
CAP = 0.5

# Quick-fit model space (vis_lp: Isothermal + ExternalShear, config/priors).
TE_MAX = 8.0
G_MAX = 0.3
E_MAX = 0.95
# Plausibility cost J: source-plane scatter, ell_comps prior, weak shear prior.
SIG_S = 0.05
SIG_E = 0.3
SIG_G = 0.1
# A seed already tracing the set this well skips the minimax polish.
EASY = 0.02
# The census Nelder-Mead fit (~3 CPU-s) is only run when the least-squares/SLSQP
# s_min falls where an optimiser miss could change a decision: below 0.15" the
# threshold sits at the floor and the set passes whatever NM finds; above 0.75"
# no plausible miss (the worst the research saw was 0.37" vs 0.16") reaches the
# 0.2" pass line or the 0.25" floor-to-cap band.
NM_BAND = (0.15, 0.75)

PARAM_NAMES = ("einstein_radius", "ell_comps_0", "ell_comps_1", "gamma_1", "gamma_2")

# Reconcile-loop (diagnostic) parameters, from the original phase 2 plan.
N_POSITIONS = 4
SNR_FLOOR = 2.0
SNR_WEAK = 1.0
R_MATCH = 0.3
MAX_ROUNDS = 5
# Solver defaults.
SOLVE_STEP = 0.05
SOLVE_REFINE = 0.01
SOLVE_TOL = 0.02
SOLVE_DEDUP = 0.1
# A refined minimum is an image only when the polish reaches a true root of the
# lens equation: near a degenerate (ring-like) model a long chain of grid minima
# sits below SOLVE_TOL without being images.
SOLVE_ROOT_TOL = 1e-4
# Reconcile judgement calls (not fixed by the plan; see find_positions).
# Source-plane match: an observed position tracing within S_MATCH of beta may
# claim a solved image up to R_CLAIM away, not only R_MATCH (a source-plane
# offset d moves a highly magnified image by up to |mu| d along the arc).
S_MATCH = 0.5 * D_OUT
R_CLAIM = 1.0
J_MAX = 30.0
MAX_IMAGES = 6

META_NAME = "positions_meta.json"
# positions_meta.json schema version, shared by the gate and the segmentation
# writer (1.0 = phase 1 gate, 2.0 = reconcile-loop fields added, 2.1 = the
# gate-based production writer with the pair floor).
META_VERSION = "2.1"


# ---------------------------------------------------------------------------
# Ray tracing (numpy SIE + external shear, PyAutoLens conventions)
# ---------------------------------------------------------------------------


def source_positions(positions: np.ndarray, params: Sequence[float]) -> np.ndarray:
    """
    Trace image-plane positions to the source plane through an SIE + shear.

    ``positions`` are (y, x) arcsec *relative to the mass centre*; ``params`` is
    ``(einstein_radius, ell_comps_0, ell_comps_1, gamma_1, gamma_2)`` in the
    PyAutoGalaxy parameterisation (``al.mp.Isothermal`` + ``al.mp.ExternalShear``).
    Reproduces ``al.Tracer.traced_grid_2d_list_from(...)[-1]`` up to a constant
    shift (the shear centre), which cancels in every pairwise separation.
    """
    te, e0, e1, g1, g2 = params[:5]
    y, x = positions[:, 0], positions[:, 1]
    f = min(np.hypot(e0, e1), 0.999)
    q = min(max((1 - f) / (1 + f), 1e-4), 0.99999)
    ang = 0.5 * np.arctan2(e0, e1)
    c, s = np.cos(ang), np.sin(ang)
    xr = x * c + y * s
    yr = -x * s + y * c
    sq = np.sqrt(1 - q * q)
    psi = np.sqrt(q * q * xr * xr + yr * yr) + 1e-12
    fac = 2.0 * (te / (1 + q)) * q / sq
    ay = fac * np.arctanh(np.clip(sq * yr / psi, -0.999999, 0.999999))
    ax = fac * np.arctan(sq * xr / psi)
    ax2 = ax * c - ay * s
    ay2 = ax * s + ay * c
    sy = -g1 * y + g2 * x
    sx = g1 * x + g2 * y
    return np.stack([y - ay2 - sy, x - ax2 - sx], 1)


def _pairwise(beta: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(len(beta), 1)
    return np.hypot(beta[iu[0], 0] - beta[iu[1], 0], beta[iu[0], 1] - beta[iu[1], 1])


def max_separation(positions: np.ndarray, params: Sequence[float]) -> float:
    """Maximum pairwise source-plane separation (the ``PositionsLH`` statistic)."""
    return float(_pairwise(source_positions(positions, params)).max())


# ---------------------------------------------------------------------------
# Quick fit
# ---------------------------------------------------------------------------


def _bounds() -> Tuple[np.ndarray, np.ndarray]:
    ell = E_MAX / np.sqrt(2) * 1.3
    lo = np.array([0.0, -ell, -ell, -G_MAX, -G_MAX])
    hi = np.array([TE_MAX, ell, ell, G_MAX, G_MAX])
    return lo, hi


def _starts(positions: np.ndarray) -> List[np.ndarray]:
    """quickfit3 starts (3 ellipticities x {1, 0.6} r_mean) + a thetaE grid (0.5, 2 r_mean)."""
    r = np.hypot(positions[:, 0], positions[:, 1]).mean()
    out = []
    for e in ((0.0, 0.0), (0.3, 0.0), (0.0, 0.3)):
        for te in (r, 0.6 * r):
            out.append(np.array([te, e[0], e[1], 0.0, 0.0]))
    for te in (0.5 * r, 2.0 * r):
        out.append(np.array([te, 0.0, 0.0, 0.0, 0.0]))
    return out


def _resid(p, positions, shear_prior):
    b = source_positions(positions, p)
    r = [((b - b.mean(0)) / SIG_S).ravel(), [p[1] / SIG_E, p[2] / SIG_E]]
    if shear_prior:
        r.append([p[3] / SIG_G, p[4] / SIG_G])
    return np.concatenate(r)


def _chi2_fit(positions, shear_prior, seeds):
    lo, hi = _bounds()
    sols = []
    for s in seeds:
        s = np.clip(s, lo + 1e-9, hi - 1e-9)
        res = least_squares(
            _resid,
            s,
            args=(positions, shear_prior),
            bounds=(lo, hi),
            method="trf",
            max_nfev=150,
        )
        sols.append((2 * res.cost, res.x))
    sols.sort(key=lambda t: t[0])
    return sols


def _sep_fit(positions, seeds):
    """SLSQP minimax: min t s.t. every pairwise source-plane distance <= t."""
    lo, hi = _bounds()
    best = (np.inf, None)
    for x0 in seeds:
        z0 = np.append(x0, _pairwise(source_positions(positions, x0)).max())
        cons = [
            {"type": "ineq", "fun": lambda z: z[-1] - _pairwise(source_positions(positions, z[:-1]))},
            {"type": "ineq", "fun": lambda z: E_MAX - np.hypot(z[1], z[2])},
        ]
        bnds = list(zip(lo, hi)) + [(0, None)]
        res = minimize(
            lambda z: z[-1],
            z0,
            method="SLSQP",
            constraints=cons,
            bounds=bnds,
            options=dict(maxiter=200, ftol=1e-9),
        )
        p = np.clip(res.x[:-1], lo, hi)
        s = max_separation(positions, p)
        s0 = max_separation(positions, x0)
        if s0 < s:
            s, p = s0, x0
        if s < best[0]:
            best = (s, p)
    return float(best[0]), best[1]


def _nm_objective(p, positions, soft):
    te, e0, e1, g1, g2 = p
    pen = 0.0
    pen += max(0, -te) * 100 + max(0, te - TE_MAX) * 100
    pen += max(0, np.hypot(e0, e1) - E_MAX) * 100
    pen += max(0, abs(g1) - G_MAX) * 100 + max(0, abs(g2) - G_MAX) * 100
    te = min(max(te, 0), TE_MAX)
    b = source_positions(
        positions, (te, e0, e1, np.clip(g1, -G_MAX, G_MAX), np.clip(g2, -G_MAX, G_MAX))
    )
    d = _pairwise(b)
    if soft:
        return np.sqrt((d**2).mean()) + pen
    return d.max() + pen


def _nm_fit(positions):
    """The 2026-09-23 census fit (census/trace.py): multi-start Nelder-Mead."""
    lo, hi = _bounds()
    r = np.hypot(positions[:, 0], positions[:, 1])
    best = (np.inf, None)
    for e in ((0, 0), (0.3, 0), (0, 0.3), (-0.3, 0), (0, -0.3)):
        for te in (r.mean(), np.median(r), 0.5 * r.mean()):
            s = (te, e[0], e[1], 0.0, 0.0)
            res = minimize(
                _nm_objective,
                s,
                args=(positions, True),
                method="Nelder-Mead",
                options=dict(maxiter=400, xatol=1e-4, fatol=1e-5),
            )
            res2 = minimize(
                _nm_objective,
                res.x,
                args=(positions, False),
                method="Nelder-Mead",
                options=dict(maxiter=400, xatol=1e-4, fatol=1e-5),
            )
            # Evaluate the exact statistic inside the real bounds (the NM penalty
            # can leave it marginally outside).
            p = np.array(res2.x)
            p[0] = np.clip(p[0], 0, TE_MAX)
            p[3:5] = np.clip(p[3:5], -G_MAX, G_MAX)
            if np.hypot(p[1], p[2]) > E_MAX:
                continue
            sep = max_separation(positions, p)
            if sep < best[0]:
                best = (sep, p)
            if best[0] < 0.1:
                return float(best[0]), best[1]
    return float(best[0]), best[1]


def quick_fit(positions, centre=(0.0, 0.0), nseed: int = 3) -> Dict:
    """
    Fit a fixed-centre SIE + shear to a set of positions (``vis_lp``'s mass model).

    Parameters
    ----------
    positions
        (N, 2) array of (y, x) arcsec image-plane positions, N >= 2.
    centre
        (y, x) mass centre, fixed (the light centre ``vis_lp`` pins its mass to).
    nseed
        Number of best least-squares solutions polished by the minimax fit.

    Returns
    -------
    dict
        ``s_min`` (minimum achievable max pairwise source-plane separation),
        ``params`` (the parameters reaching it), ``J`` (plausibility cost) and
        ``params_J`` (its parameters).
    """
    pos = np.asarray(positions, float) - np.asarray(centre, float)
    if len(pos) < 2:
        raise ValueError("quick_fit needs at least two positions")
    starts = _starts(pos)
    # Uniform shear prior (the real fit's) for the separation seeds.
    sols_np = _chi2_fit(pos, shear_prior=False, seeds=starts)
    seeds = [x for _, x in sols_np[:nseed]]
    ss = [max_separation(pos, x) for x in seeds]
    k = int(np.argmin(ss))
    if ss[k] < EASY:
        s_min, p_sep = float(ss[k]), seeds[k]
    else:
        s_min, p_sep = _sep_fit(pos, seeds)
        if NM_BAND[0] < s_min <= NM_BAND[1]:
            s_nm, p_nm = _nm_fit(pos)
            if s_nm < s_min:
                s_min, p_sep = s_nm, p_nm
    sols = _chi2_fit(pos, shear_prior=True, seeds=seeds[:2] + [starts[0]])
    J, p_J = sols[0]
    return dict(
        s_min=float(s_min),
        params=[float(v) for v in p_sep],
        J=float(J),
        params_J=[float(v) for v in p_J],
    )


def threshold_from(
    s_final: float, factor: float = FACTOR, floor: float = FLOOR, cap: float = CAP
) -> Tuple[float, bool]:
    """``T = min(max(factor * s, floor), cap)``; the bool is True when ``factor * s > cap`` (review)."""
    raw = factor * float(s_final)
    return float(min(max(raw, floor), cap)), bool(raw > cap)


# ---------------------------------------------------------------------------
# Central cut and one-image outlier drop (the phase 1 gate's steps 1 and 3)
# ---------------------------------------------------------------------------


def central_cut(positions, centre, r: float = CENTRAL_RADIUS) -> Tuple[List[int], List[int]]:
    """Return ``(kept_indices, cut_indices)``: positions within ``r`` of ``centre`` are cut."""
    pos = np.asarray(positions, float).reshape(-1, 2)
    dist = np.hypot(pos[:, 0] - centre[0], pos[:, 1] - centre[1])
    kept = [i for i in range(len(pos)) if dist[i] >= r]
    cut = [i for i in range(len(pos)) if dist[i] < r]
    return kept, cut


def leave_one_out(positions, centre) -> List[Dict]:
    """``quick_fit`` of every set with one position removed (index-aligned)."""
    pos = np.asarray(positions, float)
    return [quick_fit(np.delete(pos, i, 0), centre, nseed=2) for i in range(len(pos))]


def outlier_drop(positions, centre, fit: Dict, loo: Optional[List[Dict]] = None) -> Dict:
    """
    Decide whether one position must be dropped for the set to trace (step 3).

    Returns a dict with ``action`` (``keep`` | ``drop`` | ``review``), ``index``
    (the local index dropped, or None), ``reason`` (``unique`` | ``J`` |
    ``geom_inner`` | ``geom_outer`` for a drop; ``no_single_drop`` | ``n2_fail`` |
    ``ambiguous`` for review), ``s_final`` and ``loo`` (the leave-one-out fits, or
    None when they were not needed).
    """
    pos = np.asarray(positions, float)
    n = len(pos)
    out = dict(action="keep", index=None, reason="", s_final=fit["s_min"], loo=loo)
    if fit["s_min"] <= D_OUT:
        return out
    if n < 3:
        out.update(action="review", reason="n2_fail")
        return out
    if loo is None:
        loo = leave_one_out(pos, centre)
    out["loo"] = loo
    sd = np.array([f["s_min"] for f in loo])
    Jd = np.array([f["J"] for f in loo])
    cand = [i for i in range(n) if sd[i] <= D_OUT]
    drop, why = None, ""
    if not cand:
        out.update(action="review", reason="no_single_drop")
        return out
    if len(cand) == 1:
        drop, why = cand[0], "unique"
    else:
        order = sorted(cand, key=lambda i: Jd[i])
        if Jd[order[1]] - Jd[order[0]] >= DJ:
            drop, why = order[0], "J"
        else:
            rl = np.hypot(pos[:, 0] - centre[0], pos[:, 1] - centre[1])
            near = int(np.argmin(rl))
            far = int(np.argmax(rl))

            def med(i):
                return float(np.median(np.delete(rl, i)))

            if near in cand and (rl[near] < 0.3 or rl[near] < 0.6 * med(near)):
                drop, why = near, "geom_inner"
            elif far in cand and rl[far] > 1.5 * med(far):
                drop, why = far, "geom_outer"
    if drop is None:
        out.update(action="review", reason="ambiguous")
        return out
    out.update(action="drop", index=int(drop), reason=why, s_final=float(sd[drop]))
    return out


# ---------------------------------------------------------------------------
# Light centre
# ---------------------------------------------------------------------------


def brightest_sub_pixel_centre(img: np.ndarray, ps: float, centre, half: float = 0.3, box: int = 2):
    """numpy port of ``Array2D.brightest_sub_pixel_coordinate_in_region_from``."""
    ny, nx = img.shape
    cy, cx = centre
    region = (cy - half, cy + half, cx - half, cx + half)

    def to_pixel(y, x):
        return int(-y / ps + (ny - 1) / 2 + 0.5), int(x / ps + (nx - 1) / 2 + 0.5)

    py_min, _ = to_pixel(region[1] - ps / 2.0, 0.0)
    py_max, _ = to_pixel(region[0] + ps / 2.0, 0.0)
    _, px_min = to_pixel(0.0, region[2] + ps / 2.0)
    _, px_max = to_pixel(0.0, region[3] - ps / 2.0)
    py_min, px_min = max(0, py_min), max(0, px_min)
    py_max, px_max = min(ny - 1, py_max), min(nx - 1, px_max)
    sub = img[py_min : py_max + 1, px_min : px_max + 1]
    rr, cc = np.argwhere(sub == np.max(sub))[0]
    # The library round-trips the pixel through scaled coordinates; same pixel.
    y, x = py_min + rr, px_min + cc
    y0, y1 = max(0, y - box), min(ny, y + box + 1)
    x0, x1 = max(0, x - box), min(nx, x + box + 1)
    w = img[y0:y1, x0:x1]
    yi, xi = np.meshgrid(range(y0, y1), range(x0, x1), indexing="ij")
    sy = np.sum(w * yi) / np.sum(w)
    sx = np.sum(w * xi) / np.sum(w)
    return float(-ps * (sy - (ny - 1) / 2)), float(ps * (sx - (nx - 1) / 2))


def pixel_to_arcsec(row: int, col: int, ny: int, nx: int, pixel_scale: float) -> List[float]:
    """(row, col) to PyAutoLens ``[y, x]`` arcsec (half-pixel offset convention)."""
    return [(ny / 2 - 0.5 - row) * pixel_scale, (col - nx / 2 + 0.5) * pixel_scale]


def arcsec_to_pixel(y: float, x: float, ny: int, nx: int, pixel_scale: float) -> Tuple[int, int]:
    """Inverse of :func:`pixel_to_arcsec` (nearest pixel)."""
    return int(round(ny / 2 - 0.5 - y / pixel_scale)), int(round(x / pixel_scale + nx / 2 - 0.5))


def lens_flux_peak(lens_flux: np.ndarray, pixel_scale: float) -> Tuple[float, float]:
    """The brightest ``lens_flux`` pixel in arcsec (``load_vis_dataset``'s mask centre)."""
    ny, nx = lens_flux.shape
    r, c = np.unravel_index(int(np.nanargmax(lens_flux)), lens_flux.shape)
    y, x = pixel_to_arcsec(r, c, ny, nx, pixel_scale)
    return float(y), float(x)


def light_centre_from_maps(
    lens_flux: Optional[np.ndarray],
    pixel_scale: float,
    image: Optional[np.ndarray] = None,
    mask_centre=(0.0, 0.0),
) -> Tuple[float, float]:
    """
    The light centre ``vis_lp`` fixes its mass centre to, from in-memory maps.

    The mask centre is the ``lens_flux`` peak (else ``mask_centre``); with the VIS
    ``image`` it is refined exactly as ``util.load_vis_dataset`` refines
    ``dataset_centre`` (brightest pixel in a +/-0.3" box, 5x5 centroid). Without
    the image it is the brightest lens-flux pixel itself.
    """
    centre = tuple(float(v) for v in mask_centre)
    if lens_flux is not None and np.any(np.isfinite(lens_flux)):
        if image is None or np.shape(lens_flux) == np.shape(image):
            centre = lens_flux_peak(np.asarray(lens_flux, np.float32), pixel_scale)
    if image is not None:
        return brightest_sub_pixel_centre(np.asarray(image, float), pixel_scale, centre)
    return centre


# ---------------------------------------------------------------------------
# Forward solver (numpy stand-in for al.PointSolver; non-production diagnostic)
# ---------------------------------------------------------------------------


def _jacobian_det(theta: np.ndarray, params, h: float = 1e-4) -> np.ndarray:
    """det of d beta / d theta (central differences) at each (y, x) in ``theta``."""
    dy = np.array([h, 0.0])
    dx = np.array([0.0, h])
    by = (source_positions(theta + dy, params) - source_positions(theta - dy, params)) / (2 * h)
    bx = (source_positions(theta + dx, params) - source_positions(theta - dx, params)) / (2 * h)
    # A = [[dby/dy, dby/dx], [dbx/dy, dbx/dx]]
    return by[:, 0] * bx[:, 1] - bx[:, 0] * by[:, 1]


def solve_images(
    params: Sequence[float],
    beta,
    half_width: float,
    step: float = SOLVE_STEP,
    refine: float = SOLVE_REFINE,
    tol: float = SOLVE_TOL,
    centre_exclusion: float = CENTRAL_RADIUS,
    dedup: float = SOLVE_DEDUP,
    root_tol: float = SOLVE_ROOT_TOL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    All image-plane positions of source position ``beta`` through an SIE + shear.

    A non-production diagnostic (used by the :func:`find_positions` reconcile
    loop and by tests/tooling); the production writer never forward-solves.

    Coordinates are (y, x) arcsec relative to the mass centre (as
    :func:`source_positions`). The residual ``|beta(theta) - beta|`` is evaluated
    on a ``step`` grid over ``[-half_width, half_width]^2``; every 3x3 local
    minimum is refined on a ``refine`` grid over the surrounding ``+/- 1.5 step``;
    a refined minimum with residual below ``tol`` is polished by a bounded
    least-squares step and accepted when the polish reaches a root of the lens
    equation (residual below ``root_tol``; near a degenerate ring-like model a
    chain of grid minima sits below ``tol`` without being images), then
    de-duplicated within ``dedup``. Images within
    ``centre_exclusion`` of the centre (the central, demagnified image) are
    dropped.

    Returns
    -------
    (positions, magnifications)
        ``(N, 2)`` image positions and ``(N,)`` signed magnifications
        ``1 / det(d beta / d theta)``, ordered by decreasing ``|mu|``.
    """
    beta = np.asarray(beta, float).reshape(2)
    n = int(np.floor(half_width / step))
    ax = np.arange(-n, n + 1) * step
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    grid = np.stack([yy.ravel(), xx.ravel()], 1)
    res = np.hypot(*(source_positions(grid, params) - beta).T).reshape(yy.shape)

    # 3x3 local minima (edges padded with +inf so border minima count too).
    pad = np.pad(res, 1, constant_values=np.inf)
    is_min = np.ones_like(res, bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                is_min &= res <= pad[1 + di : 1 + di + res.shape[0], 1 + dj : 1 + dj + res.shape[1]]
    rows, cols = np.nonzero(is_min)

    m = int(round(1.5 * step / refine))
    fine = np.arange(-m, m + 1) * refine
    fy, fx = np.meshgrid(fine, fine, indexing="ij")
    offsets = np.stack([fy.ravel(), fx.ravel()], 1)

    found: List[np.ndarray] = []
    for r, c in zip(rows, cols):
        # Coarse minima far from any image are not worth refining.
        if res[r, c] > max(10 * tol, 2 * step):
            continue
        local = np.array([ax[r], ax[c]]) + offsets
        lres = np.hypot(*(source_positions(local, params) - beta).T)
        t0 = local[int(np.argmin(lres))]
        if float(lres.min()) >= tol:
            continue
        sol = least_squares(
            lambda t: source_positions(t[None, :], params)[0] - beta,
            t0,
            bounds=(t0 - 2 * step, t0 + 2 * step),
            xtol=1e-10,
            ftol=1e-10,
            max_nfev=50,
        )
        t = sol.x
        if np.hypot(*(source_positions(t[None, :], params)[0] - beta)) >= root_tol:
            continue
        if np.hypot(*t) < centre_exclusion:
            continue
        if any(np.hypot(*(t - s)) < dedup for s in found):
            continue
        found.append(t)
    if not found:
        return np.zeros((0, 2)), np.zeros(0)
    pos = np.array(found)
    with np.errstate(divide="ignore"):
        mu = 1.0 / _jacobian_det(pos, params)
    order = np.argsort(-np.abs(mu))
    return pos[order], mu[order]


# ---------------------------------------------------------------------------
# Compute (flux-map candidates)
# ---------------------------------------------------------------------------


def find_local_maxima(flux: np.ndarray) -> List[Tuple[float, int, int]]:
    """``(value, row, col)`` for every interior pixel brighter than its 4 neighbours, brightest first."""
    f = np.asarray(flux, float)
    if f.ndim != 2 or min(f.shape) < 3:
        return []
    c = f[1:-1, 1:-1]
    m = (c > f[:-2, 1:-1]) & (c > f[2:, 1:-1]) & (c > f[1:-1, :-2]) & (c > f[1:-1, 2:])
    rr, cc = np.nonzero(m)
    out = [(float(c[r, k]), int(r + 1), int(k + 1)) for r, k in zip(rr, cc)]
    out.sort(reverse=True)
    return out


@dataclass
class Peak:
    """One source-flux local maximum."""

    y: float
    x: float
    flux: float
    snr: float

    @property
    def yx(self) -> Tuple[float, float]:
        return (self.y, self.x)


def flux_candidates(
    source_flux: np.ndarray,
    snr_map: Optional[np.ndarray],
    pixel_scale: float,
    light_centre,
    snr_floor: float = SNR_FLOOR,
    snr_weak: float = SNR_WEAK,
    central_radius: float = CENTRAL_RADIUS,
) -> Tuple[List[Peak], List[Peak]]:
    """
    ``(candidates, weak)``: local maxima with ``SNR >= snr_floor`` and with
    ``snr_weak <= SNR < snr_floor``, both outside ``central_radius`` of the light
    centre and of any brighter maximum, brightest first. Without an SNR map
    every maximum is a candidate.
    """
    ny, nx = source_flux.shape
    cand, weak = [], []
    accepted: List[Tuple[float, float]] = []
    for v, r, c in find_local_maxima(source_flux):
        y, x = pixel_to_arcsec(r, c, ny, nx, pixel_scale)
        if np.hypot(y - light_centre[0], x - light_centre[1]) < central_radius:
            continue
        # Two peaks closer than one PSF FWHM are one unresolved image: keep the
        # brighter (maxima arrive brightest first).
        if any(np.hypot(y - a, x - b) < central_radius for a, b in accepted):
            continue
        accepted.append((y, x))
        snr = np.inf if snr_map is None else float(snr_map[r, c])
        if not np.isfinite(snr) and snr_map is not None:
            continue
        p = Peak(float(y), float(x), float(v), snr)
        if snr_map is None or snr >= snr_floor:
            cand.append(p)
        elif snr >= snr_weak:
            weak.append(p)
    return cand, weak


def snr_map_from(source_flux: np.ndarray, noise_map: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """``source_flux / noise_map`` where the noise is positive, else 0 (the segmentation writer's map)."""
    if noise_map is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(noise_map > 0, source_flux / noise_map, 0.0)


# ---------------------------------------------------------------------------
# Production path: the gate-based writer
# ---------------------------------------------------------------------------

# The production writer's candidate floor and the pair floor (source-flux SNR).
SNR_MIN = 3.0
SNR_PAIR = 3.0
# Step 4 (list only): plausibility flag when one drop lowers J by at least DJ_FLAG.
DJ_FLAG = 10.0


def gate_candidates(
    source_flux: np.ndarray,
    snr_map: Optional[np.ndarray],
    pixel_scale: float,
    light_centre,
    n_positions: int = N_POSITIONS,
    snr_min: float = SNR_MIN,
    central_radius: float = CENTRAL_RADIUS,
    merge_radius: float = CENTRAL_RADIUS,
) -> List[Peak]:
    """
    The production writer's candidate set: local maxima of ``source_flux`` with
    ``SNR >= snr_min`` outside ``central_radius`` of the light centre, a peak
    within ``merge_radius`` (one VIS PSF FWHM) of a brighter kept peak merged
    into it, brightest (by flux) first, capped at ``n_positions``.

    There is no SNR walk-down of any kind: a peak below ``snr_min`` is never a
    candidate. Without an SNR map every maximum passes the floor (SNR ``inf``).
    """
    source_flux = np.asarray(source_flux, float)
    ny, nx = source_flux.shape
    out: List[Peak] = []
    for v, r, c in find_local_maxima(source_flux):
        if snr_map is None:
            snr = np.inf
        else:
            snr = float(snr_map[r, c])
            if not np.isfinite(snr) or snr < snr_min:
                continue
        y, x = pixel_to_arcsec(r, c, ny, nx, pixel_scale)
        if np.hypot(y - light_centre[0], x - light_centre[1]) < central_radius:
            continue
        if any(np.hypot(y - p.y, x - p.x) < merge_radius for p in out):
            continue
        out.append(Peak(float(y), float(x), float(v), snr))
    return out[:n_positions]


def plausibility_flag(fit: Dict, loo: List[Dict]) -> Optional[int]:
    """
    Step 4 (list only): the local index whose removal lowers ``J`` by >= ``DJ_FLAG``
    and beats every other removal by >= ``DJ``, else None.
    """
    if not loo or len(loo) < 3:
        return None
    Jd = np.array([f["J"] for f in loo])
    i = int(np.argmin(Jd))
    if fit["J"] - Jd[i] >= DJ_FLAG and np.sort(Jd)[1] - Jd[i] >= DJ:
        return i
    return None


def gate_positions(positions, centre, version: Optional[str] = None) -> Dict:
    """
    The phase 1 gate on one position set; returns the ``positions_meta.json``
    dict (without the tile-level keys). ``positions`` are raw (y, x) arcsec,
    ``centre`` the light centre.

    Steps: central cut (``CENTRAL_RADIUS``), quick fit, one-image outlier drop
    (``outlier_drop``), plausibility flag, threshold
    ``T = min(max(2 s_final, 0.3"), 0.5")`` with ``over_cap`` review.
    """
    pos_raw = np.asarray(positions, float).reshape(-1, 2)
    centre = (float(centre[0]), float(centre[1]))
    kept, cut = central_cut(pos_raw, centre)
    dropped = [
        dict(index=i, y=_r4(pos_raw[i, 0]), x=_r4(pos_raw[i, 1]), reason="central_cut")
        for i in cut
    ]
    meta = dict(
        version=META_VERSION if version is None else version,
        light_centre=[_r4(centre[0]), _r4(centre[1])],
        n_raw=len(pos_raw),
        positions_used=[],
        dropped=dropped,
        flag_index=None,
        s_min_all=None,
        s_min=None,
        J=None,
        threshold=None,
        factor=FACTOR,
        floor=FLOOR,
        cap=CAP,
        status="keep",
        review_reason="",
        params=None,
    )
    used_idx = list(kept)
    if len(used_idx) < 2:
        meta.update(
            positions_used=[[_r4(v) for v in pos_raw[i]] for i in used_idx],
            status="n_lt_2",
            review_reason="n_lt_2",
        )
        return meta

    pos = pos_raw[used_idx]
    fit = quick_fit(pos, centre)
    meta["s_min_all"] = _r4(fit["s_min"])
    decision = outlier_drop(pos, centre, fit)
    final_fit = fit
    status, reason, flag_index = "keep", "", None
    if decision["action"] == "review":
        status, reason = "review", decision["reason"]
    elif decision["action"] == "drop":
        j = decision["index"]
        raw_i = used_idx[j]
        dropped.append(
            dict(
                index=raw_i,
                y=_r4(pos_raw[raw_i, 0]),
                x=_r4(pos_raw[raw_i, 1]),
                reason=f"outlier_{decision['reason']}",
            )
        )
        final_fit = decision["loo"][j]
        used_idx = [i for i in used_idx if i != raw_i]
    elif len(pos) >= 3 and fit["J"] >= DJ_FLAG:
        loo = decision["loo"] or leave_one_out(pos, centre)
        k = plausibility_flag(fit, loo)
        if k is not None:
            flag_index = used_idx[k]
            status, reason = "flag", "plausibility"

    threshold, over_cap = threshold_from(final_fit["s_min"])
    if over_cap and status != "review":
        status, reason = "review", "over_cap"
    if status == "keep" and dropped:
        status = "drop"

    meta.update(
        positions_used=[[_r4(v) for v in pos_raw[i]] for i in used_idx],
        dropped=dropped,
        flag_index=flag_index,
        s_min=_r4(final_fit["s_min"]),
        J=_r4(final_fit["J"]),
        threshold=_r4(threshold),
        status=status,
        review_reason=reason,
        params={k: _r4(v) for k, v in zip(PARAM_NAMES, final_fit["params"])},
    )
    return meta


def _used_indices(meta: Dict) -> List[int]:
    """Raw indices of ``meta['positions_used']`` (in order): those not dropped."""
    gone = {d["index"] for d in meta["dropped"]}
    return [i for i in range(meta["n_raw"]) if i not in gone]


def apply_pair_floor(meta: Dict, snrs: Optional[Sequence[Optional[float]]], snr_pair: float = SNR_PAIR) -> Dict:
    """
    The pair floor, applied after the gate steps (in place; returns ``meta``).

    A final set of exactly two positions needs both peak SNRs >= ``snr_pair``:
    otherwise the fainter (lower SNR) is dropped (reason ``pair_floor``), fewer
    than two positions remain, and the tile goes to review
    (``review_reason: pair_floor``) with no positions and no threshold.
    ``snrs`` is index-aligned with the raw positions; ``None`` (no SNR map), or
    an unknown SNR for either position, leaves the set as the gate left it.
    """
    used = meta["positions_used"]
    if snrs is None or meta["status"] == "n_lt_2" or len(used) != 2:
        return meta
    idx = _used_indices(meta)
    s = [snrs[i] for i in idx]
    if any(v is None or np.isnan(v) for v in s):
        return meta
    k = int(np.argmin(s))
    if s[k] >= snr_pair:
        return meta
    meta["dropped"].append(
        dict(index=idx[k], y=used[k][0], x=used[k][1], reason="pair_floor", snr=_r4(s[k]))
    )
    meta.update(
        positions_used=[],
        flag_index=None,
        threshold=None,
        status="review",
        review_reason="pair_floor",
    )
    return meta


def snr_at(snr_map: Optional[np.ndarray], positions, pixel_scale: float) -> Optional[List[Optional[float]]]:
    """The SNR-map value at each (y, x) position's pixel (None off the frame); None without a map."""
    if snr_map is None:
        return None
    return [_snr_near(snr_map, float(y), float(x), pixel_scale, radius_pix=0) for y, x in np.asarray(positions, float).reshape(-1, 2)]


def gate_result(positions, light_centre, snrs=None, version: Optional[str] = None) -> "FinderResult":
    """
    The phase 1 gate plus the pair floor on one position set, as a
    :class:`FinderResult` (``method: "gate"``). The seeded path
    (``scripts/tools/positions_gate.py``, positions from ``positions.json``)
    and the unseeded writer (:func:`find_positions_gate`, positions from the
    flux map) both end here.
    """
    pos = np.asarray(positions, float).reshape(-1, 2)
    meta = apply_pair_floor(gate_positions(pos, light_centre, version), snrs)
    snr_list = None if snrs is None else [_r4(v) if v is not None and np.isfinite(v) else None for v in snrs]
    return FinderResult(
        positions=meta["positions_used"],
        threshold=meta["threshold"],
        s_final=meta["s_min"],
        n_rounds=0,
        dropped=meta["dropped"],
        review_reason=meta["review_reason"],
        params=meta["params"],
        status=meta["status"],
        J=meta["J"],
        s_seed=meta["s_min_all"],
        light_centre=tuple(meta["light_centre"]),
        seed=[[_r4(v) for v in p] for p in pos],
        method="gate",
        flag_index=meta["flag_index"],
        snr=snr_list,
    )


def find_positions_gate(
    source_flux: np.ndarray,
    snr_map: Optional[np.ndarray],
    light_centre=None,
    pixel_scale: float = 0.1,
    n_positions: int = N_POSITIONS,
    lens_flux: Optional[np.ndarray] = None,
    mask_centre=(0.0, 0.0),
) -> "FinderResult":
    """
    The production multiple-image finder (the gate-based writer).

    1. candidates (:func:`gate_candidates`): source-flux local maxima with
       ``SNR >= 3`` outside the 0.15" light-centre disc, peaks within 0.15" of a
       brighter one merged, the brightest ``n_positions`` kept -- no SNR
       walk-down;
    2. the phase 1 gate on that set (:func:`gate_positions`): quick fit,
       leave-one-out outlier drop, plausibility flag, threshold
       ``T = min(max(2 s, 0.3"), 0.5")`` and review reasons;
    3. the pair floor (:func:`apply_pair_floor`).

    ``light_centre`` is the fixed mass centre (``vis_lp``'s ``dataset_centre``);
    by default the brightest ``lens_flux`` pixel, else ``mask_centre``.
    ``snr_map=None`` treats every maximum as passing the floor.
    """
    source_flux = np.asarray(source_flux, float)
    if light_centre is None:
        light_centre = light_centre_from_maps(lens_flux, pixel_scale, mask_centre=mask_centre)
    lc = (float(light_centre[0]), float(light_centre[1]))
    snr = None if snr_map is None else np.asarray(snr_map, float)
    cand = gate_candidates(source_flux, snr, pixel_scale, lc, n_positions=n_positions)
    positions = [p.yx for p in cand]
    snrs = None if snr is None else [p.snr for p in cand]
    return gate_result(positions, lc, snrs)


# ---------------------------------------------------------------------------
# Reconcile loop (non-production diagnostic)
# ---------------------------------------------------------------------------


@dataclass
class FinderResult:
    """
    Outcome of the production writer (:func:`find_positions_gate` /
    :func:`gate_result`, ``method == "gate"``) or of the diagnostic reconcile
    loop (:func:`find_positions`, ``method == "finder"``).

    ``positions`` are the final (y, x) arcsec positions; ``status`` is ``keep``
    (the seed set unchanged), ``drop`` (positions dropped), ``flag``
    (plausibility flag, gate only), ``add`` / ``modify`` (loop only: positions
    added, or added and dropped), ``review`` or ``n_lt_2``. ``review_reason`` is
    ``""`` or, for the gate, one of ``n2_fail``, ``no_single_drop``,
    ``ambiguous``, ``over_cap``, ``pair_floor`` (``plausibility`` with
    ``flag``); for the loop one of ``no_converge``, ``empty``,
    ``implausible_J``, ``unseen_bright_image``, ``over_cap``. ``snr`` (gate
    only) is the peak SNR of each ``seed`` position, None without an SNR map.
    """

    positions: List[List[float]]
    threshold: Optional[float]
    s_final: Optional[float]
    n_rounds: int
    added: List[Dict] = field(default_factory=list)
    dropped: List[Dict] = field(default_factory=list)
    review_reason: str = ""
    params: Optional[Dict[str, float]] = None
    status: str = "keep"
    J: Optional[float] = None
    s_seed: Optional[float] = None
    light_centre: Tuple[float, float] = (0.0, 0.0)
    seed: List[List[float]] = field(default_factory=list)
    predicted: List[Dict] = field(default_factory=list)
    n_unseen_bright: int = 0
    history: List[Dict] = field(default_factory=list)
    method: str = "finder"
    flag_index: Optional[int] = None
    snr: Optional[List[Optional[float]]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _r4(v):
    return None if v is None else round(float(v), 4)


def _key(p) -> Tuple[float, float]:
    return (round(float(p[0]), 4), round(float(p[1]), 4))


def _wide_starts(pos_rel: np.ndarray) -> List[np.ndarray]:
    """A deterministic 75-start grid: thetaE x ell_comps x shear, for :func:`_candidate_models`."""
    r = np.hypot(pos_rel[:, 0], pos_rel[:, 1]).mean()
    ells = ((0.0, 0.0), (0.3, 0.0), (-0.3, 0.0), (0.0, 0.3), (0.0, -0.3))
    shears = ((0.0, 0.0), (0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1))
    return [
        np.array([te, e[0], e[1], g[0], g[1]])
        for te in (0.8 * r, r, 1.2 * r)
        for e in ells
        for g in shears
    ]


def _candidate_models(
    pos_rel: np.ndarray, fit: Dict, wide: bool = False
) -> List[Tuple[float, List[float]]]:
    """
    ``(cost, params)`` models that trace ``pos_rel`` (max pairwise source-plane
    separation <= ``D_OUT``), lowest plausibility cost first: the quick fit's
    ``J`` and ``s_min`` models plus the shear-prior chi-squared solution from every
    quick-fit start. A set of two or three positions under-constrains the five
    SIE + shear parameters, so these can predict different image counts. When
    none traces, the ``s_min`` model alone. ``wide`` instead returns the
    solutions from the :func:`_wide_starts` grid (no shear prior in the fit)
    whose cost is below ``J_MAX``: the search for a plausible model that does
    not predict an unseen bright image.
    """
    if wide:
        models = [x for _, x in _chi2_fit(pos_rel, shear_prior=False, seeds=_wide_starts(pos_rel))]
    else:
        models = [fit["params_J"], fit["params"]]
        models += [x for _, x in _chi2_fit(pos_rel, shear_prior=True, seeds=_starts(pos_rel))]
    out, keys = [], set()
    for m in models:
        m = [float(v) for v in m]
        k = tuple(round(v, 3) for v in m)
        if k in keys or max_separation(pos_rel, m) > D_OUT:
            continue
        keys.add(k)
        cost = float(np.sum(_resid(np.array(m), pos_rel, True) ** 2))
        if wide and cost >= J_MAX:
            continue
        out.append((cost, m))
    if wide:
        out.sort(key=lambda t: t[0])
        return out
    out.sort(key=lambda t: t[0])
    if not out:
        out = [(float("inf"), [float(v) for v in fit["params"]])]
    return out


def _snr_near(snr_map, y, x, pixel_scale, radius_pix: int = 1) -> Optional[float]:
    """Max SNR within ``radius_pix`` of (y, x); None when off the frame or with no SNR map."""
    if snr_map is None:
        return None
    ny, nx = snr_map.shape
    r, c = arcsec_to_pixel(y, x, ny, nx, pixel_scale)
    if not (0 <= r < ny and 0 <= c < nx):
        return None
    sub = snr_map[max(0, r - radius_pix) : r + radius_pix + 1, max(0, c - radius_pix) : c + radius_pix + 1]
    sub = sub[np.isfinite(sub)]
    return float(sub.max()) if sub.size else None


def _match(
    obs: np.ndarray, delta: np.ndarray, pred: np.ndarray, r_match: float
) -> Tuple[Dict[int, int], List[int]]:
    """
    Greedy one-to-one matching of observed to predicted images, nearest pairs
    first. A pair is allowed within ``r_match``, or within ``R_CLAIM`` when the
    observed position traces within ``S_MATCH`` of beta (``delta``). Returns
    ``({obs_index: pred_index}, kept)``: ``kept`` are the matched observed
    indices (each predicted image vouches for at most one position).
    """
    out, used = {}, set()
    if len(obs) and len(pred):
        d = np.hypot(obs[:, None, 0] - pred[None, :, 0], obs[:, None, 1] - pred[None, :, 1])
        pairs = sorted(
            (d[i, j], i, j)
            for i in range(len(obs))
            for j in range(len(pred))
            if d[i, j] <= r_match or (delta[i] <= S_MATCH and d[i, j] <= R_CLAIM)
        )
        for _, i, j in pairs:
            if i in out or j in used:
                continue
            out[i] = j
            used.add(j)
    kept = [i for i in range(len(obs)) if i in out]
    return out, kept


def _reconcile(obs, lc, params, model_rel, pool, snr, pixel_scale, half_width, r_match):
    """
    Solve one model and reconcile it with the observed set: returns a dict with
    the solved images (``pred``, ``mu``, frame coordinates), ``kept`` observed
    indices, the pool peaks to ``add`` and the count of bright predicted images
    over empty sky (``n_unseen``).
    """
    rel = obs - lc
    beta = source_positions(model_rel, params).mean(0)
    pred_rel, mu = solve_images(params, beta, half_width)
    pred = pred_rel + lc
    delta = np.hypot(*(source_positions(rel, params) - beta).T)
    match, kept = _match(obs, delta, pred, r_match)
    claimed = set(match.values())
    taken = {_key(p) for p in obs}
    # "Bright": the predicted image's expected SNR, |mu| times the lowest
    # observed SNR per unit |mu| among the matched images (point-image flux
    # scales with |mu|), reaches the candidate floor.
    snr_per_mu = None
    if snr is not None and match:
        ratios = []
        for i, j in match.items():
            s_obs = _snr_near(snr, obs[i, 0], obs[i, 1], pixel_scale, radius_pix=0)
            if s_obs is not None and np.isfinite(mu[j]) and abs(mu[j]) > 0:
                ratios.append(s_obs / abs(mu[j]))
        snr_per_mu = min(ratios) if ratios else None
    add, n_unseen = [], 0
    for j in range(len(pred)):
        if j in claimed:
            continue
        py, px = pred[j]
        near = [p for k, p in pool.items() if k not in taken and np.hypot(p.y - py, p.x - px) <= r_match]
        if near:
            best = max(near, key=lambda p: (p.snr, p.flux))
            taken.add(_key(best.yx))
            add.append(best)
            continue
        s = _snr_near(snr, py, px, pixel_scale)
        if (
            s is not None
            and s < SNR_WEAK
            and snr_per_mu is not None
            and abs(mu[j]) * snr_per_mu >= SNR_FLOOR
        ):
            n_unseen += 1
    return dict(pred=pred, pred_rel=pred_rel, mu=mu, kept=kept, add=add, n_unseen=n_unseen, beta=beta)


def find_positions(
    source_flux: np.ndarray,
    snr_map: Optional[np.ndarray],
    lens_flux: Optional[np.ndarray] = None,
    pixel_scale: float = 0.1,
    seed=None,
    light_centre=None,
    mask_centre=(0.0, 0.0),
    n_positions: int = N_POSITIONS,
    half_width: Optional[float] = None,
    r_match: float = R_MATCH,
    max_rounds: int = MAX_ROUNDS,
) -> FinderResult:
    """
    The compute / fit / solve / reconcile loop -- a NON-PRODUCTION diagnostic.

    Not called by ``preprocess/segmentation.py``, ``util.py`` or
    ``scripts/tools/positions_gate.py`` (they use :func:`find_positions_gate`);
    kept for the witness script's ``--reconcile`` comparison and tooling. See
    the module docstring for why it is not the production path.

    Parameters
    ----------
    source_flux, snr_map
        The segmentation source-flux map and its SNR map (``source_flux /
        VIS_RMS``, :func:`snr_map_from`); ``snr_map=None`` treats every maximum
        as a candidate (no weak peaks, no empty-sky test).
    lens_flux
        The segmentation lens-flux map; its brightest pixel is the light centre
        when ``light_centre`` is not given (fallback ``mask_centre``).
    seed
        (N, 2) seed positions (the gate path: the tile's ``positions.json``).
        Seeds within ``CENTRAL_RADIUS`` of the light centre are cut first.
        Default: the ``n_positions`` brightest candidates.
    light_centre
        The fixed mass centre (``vis_lp``'s ``dataset_centre``).
    half_width
        Solver half-width about the light centre (default: the frame half-size).

    Judgement calls not fixed by the plan (recorded in the issue's witness
    report):

    - an untraceable set of three or more is modelled through the phase 1
      gate's one-image outlier drop (:func:`outlier_drop`) -- the fit to the
      traceable subset is solved, and the left-out position is judged by the
      reconcile step;
    - a set of two or three positions under-constrains the model, so the model
      solved is chosen among the models that trace it (:func:`_candidate_models`):
      fewest bright predicted images over empty sky first, then lowest
      plausibility cost, widening to a 75-start grid (cost < ``J_MAX``) only
      when the quick fit's own models all predict one;
    - matching is one-to-one, with the claim radius widened to ``R_CLAIM``
      for a position tracing within ``S_MATCH`` of beta, because a small
      source-plane offset moves a highly magnified image a long way along its
      arc;
    - candidate peaks closer than ``CENTRAL_RADIUS`` (one VIS PSF FWHM) to a
      brighter candidate are not resolved images and are suppressed;
    - an unmatched predicted image takes the highest-SNR candidate within
      ``r_match`` before a weak one;
    - "bright" means an expected SNR >= ``SNR_FLOOR``: ``|mu|`` times the
      lowest observed SNR per ``|mu|`` among the matched images;
    - a repeated set ends the loop as ``no_converge``; a set emptied by the
      reconcile step goes to review with no positions (``empty``), while fewer
      than two seeds after the central cut keep the phase 1 ``n_lt_2`` status;
    - with a flux-derived seed the output is ordered by flux and capped at
      ``n_positions``.
    """
    source_flux = np.asarray(source_flux, float)
    ny, nx = source_flux.shape
    if light_centre is None:
        light_centre = light_centre_from_maps(lens_flux, pixel_scale, mask_centre=mask_centre)
    lc = np.array([float(light_centre[0]), float(light_centre[1])])
    if half_width is None:
        half_width = 0.5 * min(ny, nx) * pixel_scale
    snr = None if snr_map is None else np.asarray(snr_map, float)

    cand, weak = flux_candidates(source_flux, snr, pixel_scale, lc)
    pool = {_key(p.yx): p for p in weak}
    pool.update({_key(p.yx): p for p in cand})

    dropped: List[Dict] = []
    added: List[Dict] = []
    if seed is None:
        from_flux = True
        seed_arr = np.array([p.yx for p in cand[:n_positions]], float).reshape(-1, 2)
        seed_idx = list(range(len(seed_arr)))
    else:
        from_flux = False
        seed_arr = np.asarray(seed, float).reshape(-1, 2)
        seed_idx = []
        for i, p in enumerate(seed_arr):
            if np.hypot(*(p - lc)) < CENTRAL_RADIUS:
                dropped.append(dict(index=i, y=_r4(p[0]), x=_r4(p[1]), reason="central_cut", round=0))
            else:
                seed_idx.append(i)

    # Current set: list of (y, x, seed index or None).
    current = [(float(seed_arr[i, 0]), float(seed_arr[i, 1]), i) for i in seed_idx]
    result = FinderResult(
        positions=[], threshold=None, s_final=None, n_rounds=0,
        light_centre=(_r4(lc[0]), _r4(lc[1])),
        seed=[[_r4(v) for v in p] for p in seed_arr],
    )

    seen = [frozenset(_key(p[:2]) for p in current)]
    fit = None
    pred_rel, mu = np.zeros((0, 2)), np.zeros(0)
    n_unseen = 0
    converged = False
    reason = ""
    for rnd in range(1, max_rounds + 1):
        result.n_rounds = rnd
        if len(current) < 2:
            reason = "empty"
            break
        obs = np.array([p[:2] for p in current])
        rel = obs - lc
        fit = quick_fit(obs, lc)
        if rnd == 1:
            result.s_seed = fit["s_min"]
        # An untraceable set (s_min > D_OUT, n >= 3) is modelled by the gate's
        # one-image outlier drop: the model solved is the fit to the traceable
        # subset, and the left-out position is then judged by the reconcile
        # step like any other. Without a single-drop decision the full-set fit
        # is used as it is.
        model_rel, model_fit, loo_index = rel, fit, None
        if fit["s_min"] > D_OUT and len(rel) >= 3:
            decision = outlier_drop(rel, (0.0, 0.0), fit)
            if decision["action"] == "drop":
                loo_index = decision["index"]
                model_rel = np.delete(rel, loo_index, 0)
                model_fit = decision["loo"][loo_index]
        # Bright predicted images over empty sky count against a model: among
        # the models that trace the set, the one with the fewest is solved
        # (ties: lowest plausibility cost).
        # If none of the quick fit's models avoids one, a wider start grid is
        # searched for a plausible (cost < J_MAX) model that does.
        best = None
        for wide in (False, True):
            if wide and (best[0][0] == 0 or model_fit["s_min"] > D_OUT):
                break
            for cost, params in _candidate_models(model_rel, model_fit, wide=wide):
                rec = _reconcile(obs, lc, params, model_rel, pool, snr, pixel_scale, half_width, r_match)
                score = (rec["n_unseen"], cost)
                if best is None or score < best[0]:
                    best = (score, params, rec)
                if rec["n_unseen"] == 0:
                    break
        _, params, rec = best
        pred_rel, mu, n_unseen = rec["pred_rel"], rec["mu"], rec["n_unseen"]
        new = [current[i] for i in rec["kept"]]
        round_drop = [current[i] for i in range(len(current)) if i not in rec["kept"]]
        round_add = rec["add"]
        new += [(p.y, p.x, None) for p in round_add]
        pred = rec["pred"]

        result.history.append(
            dict(
                round=rnd,
                n=len(current),
                s_min=_r4(fit["s_min"]),
                J=_r4(fit["J"]),
                n_pred=int(len(pred)),
                loo_index=loo_index,
                dropped=[[_r4(p[0]), _r4(p[1])] for p in round_drop],
                added=[[_r4(p.y), _r4(p.x)] for p in round_add],
                n_unseen_bright=n_unseen,
            )
        )
        if not round_drop and not round_add:
            converged = True
            break
        for p in round_drop:
            dropped.append(
                dict(index=p[2], y=_r4(p[0]), x=_r4(p[1]), reason="unpredicted", round=rnd)
            )
        for p in round_add:
            added.append(
                dict(
                    y=_r4(p.y), x=_r4(p.x), snr=_r4(p.snr) if np.isfinite(p.snr) else None,
                    reason="predicted_peak" if p.snr >= SNR_FLOOR else "predicted_weak",
                    round=rnd,
                )
            )
        current = new
        key = frozenset(_key(p[:2]) for p in current)
        if key in seen:
            reason = "no_converge"
            break
        seen.append(key)
    else:
        reason = "no_converge"

    # A set the reconcile step emptied keeps no positions at all.
    if len(current) < 2 and len(seed_idx) >= 2:
        for p in current:
            dropped.append(dict(index=p[2], y=_r4(p[0]), x=_r4(p[1]), reason="empty_set", round=result.n_rounds))
        current = []

    # With a flux-derived seed the output is ordered by flux and capped at
    # n_positions (the faintest positions over the cap are dropped).
    capped = False
    if from_flux:
        current.sort(key=lambda p: -pool[_key(p[:2])].flux if _key(p[:2]) in pool else 0.0)
        for p in current[n_positions:]:
            capped = True
            dropped.append(
                dict(index=p[2], y=_r4(p[0]), x=_r4(p[1]), reason="n_positions_cap", round=result.n_rounds)
            )
        current = current[:n_positions]

    # A position dropped and later re-added (or added and later dropped) is
    # reported by its final state only, once; "dropped" lists seeds only.
    final_keys = {_key(p[:2]) for p in current}
    dropped = [
        d for d in dropped
        if (d["index"] is not None or d["reason"] == "n_positions_cap")
        and _key((d["y"], d["x"])) not in final_keys
    ]
    seen_add, uniq = set(), []
    for a in added:
        k = _key((a["y"], a["x"]))
        if k in final_keys and k not in seen_add:
            seen_add.add(k)
            uniq.append(a)
    added = uniq

    result.positions = [[_r4(p[0]), _r4(p[1])] for p in current]
    result.added, result.dropped = added, dropped
    result.n_unseen_bright = n_unseen
    result.predicted = [
        dict(y=_r4(p[0] + lc[0]), x=_r4(p[1] + lc[1]), mu=_r4(m)) for p, m in zip(pred_rel, mu)
    ]

    if len(current) < 2:
        if len(seed_idx) < 2:
            # Fewer than two seeds survive the central cut: the phase 1
            # penalty-off status.
            result.status, result.review_reason = "n_lt_2", "n_lt_2"
        else:
            # The reconcile step emptied the set: review, with no positions.
            result.status, result.review_reason = "review", "empty"
            result.positions = []
        return result

    # The final fit (the loop's last fit when it converged on this set).
    obs = np.array([p[:2] for p in current])
    if not (converged and fit is not None) or capped:
        fit = quick_fit(obs, lc)
    result.s_final = _r4(fit["s_min"])
    result.J = _r4(fit["J"])
    result.params = {k: _r4(v) for k, v in zip(PARAM_NAMES, fit["params"])}
    threshold, over_cap = threshold_from(fit["s_min"])
    result.threshold = _r4(threshold)

    if not reason:
        if fit["J"] >= J_MAX or len(pred_rel) > MAX_IMAGES:
            reason = "implausible_J"
        elif n_unseen > 0:
            reason = "unseen_bright_image"
        elif over_cap:
            reason = "over_cap"
    result.review_reason = reason
    if reason:
        result.status = "review"
    elif dropped and added:
        result.status = "modify"
    elif dropped:
        result.status = "drop"
    elif added:
        result.status = "add"
    return result


# ---------------------------------------------------------------------------
# positions_meta.json
# ---------------------------------------------------------------------------


def positions_sha(dataset_dir) -> str:
    """Short sha256 of a tile's ``positions.json`` (the gate's cache key)."""
    return hashlib.sha256((Path(dataset_dir) / "positions.json").read_bytes()).hexdigest()[:16]


def meta_from_result(result: FinderResult, version: str = META_VERSION) -> Dict:
    """
    The ``positions_meta.json`` dict (without the tile-level keys) for a result.

    Keeps every key of the phase 1 gate contract (``positions_used``,
    ``threshold`` and ``status`` are what ``util.load_vis_dataset`` reads;
    ``s_min_all`` is the seed set's ``s_min``, ``s_min`` the final set's).

    For the production writer (``method: "gate"``) it adds ``method``,
    ``finder_version``, ``s_final``, ``snr`` (peak SNR of each raw position)
    and ``added`` (always empty: the writer never adds a position); ``dropped``
    entries carry the raw ``index`` and a ``reason`` (``central_cut`` |
    ``outlier_<why>`` | ``pair_floor``).

    For the diagnostic reconcile loop (``method: "finder"``) it adds
    ``finder_version``, ``s_final``, ``n_rounds``, ``added``, ``predicted``,
    ``n_unseen_bright`` and ``rounds``.
    """
    if result.method == "gate":
        return dict(
            version=version,
            light_centre=list(result.light_centre),
            n_raw=len(result.seed),
            positions_used=result.positions,
            dropped=result.dropped,
            flag_index=result.flag_index,
            s_min_all=result.s_seed,
            s_min=result.s_final,
            J=result.J,
            threshold=result.threshold,
            factor=FACTOR,
            floor=FLOOR,
            cap=CAP,
            status=result.status,
            review_reason=result.review_reason,
            params=result.params,
            method="gate",
            finder_version=FINDER_VERSION,
            s_final=result.s_final,
            snr=result.snr,
            added=[],
        )
    return dict(
        version=version,
        method="finder",
        light_centre=list(result.light_centre),
        n_raw=len(result.seed),
        positions_used=result.positions,
        dropped=result.dropped,
        flag_index=None,
        s_min_all=_r4(result.s_seed),
        s_min=result.s_final,
        J=result.J,
        threshold=result.threshold if result.status != "n_lt_2" else None,
        factor=FACTOR,
        floor=FLOOR,
        cap=CAP,
        status=result.status,
        review_reason=result.review_reason,
        params=result.params,
        finder_version=FINDER_VERSION,
        s_final=result.s_final,
        n_rounds=result.n_rounds,
        added=result.added,
        predicted=result.predicted,
        n_unseen_bright=result.n_unseen_bright,
        rounds=result.history,
    )


def write_meta(dataset_dir, meta: Dict) -> Path:
    """Write ``meta`` as the tile's ``positions_meta.json``."""
    path = Path(dataset_dir) / META_NAME
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path
