"""
Shared pure numpy / scipy lens-model core for the ``vis_lp`` positions work.

Holds the numpy SIE + external shear tracer, the fixed-centre quick fit, the
threshold rule ``T = min(max(2 s, 0.3"), 0.5")``, the phase 1 gate's central
cut and one-image outlier drop, the light-centre port, and
:func:`solve_images`, a numpy forward solver (grid search + refinement, a
stand-in for ``al.PointSolver``) returning the image positions and
magnifications of a source position. No PyAutoLens import. The research behind
the numbers is in the ``euclid_dr1`` project under
``inspect/positions_census/research/`` (``SYNTHESIS.md``, ``B_final_method.md``).

All positions are PyAutoLens ``(y, x)`` arcsec.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize

FINDER_VERSION = "1.0"

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

# Finder (compute / reconcile) parameters, from the approved plan.
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
MU_BRIGHT_FRAC = 0.5
# Source-plane match: an observed position tracing within S_MATCH of beta is
# predicted even when its solved image is further than R_MATCH away (a
# source-plane offset d moves a highly magnified image by up to |mu| d along the
# arc); it claims the nearest solved image within R_CLAIM.
S_MATCH = 0.5 * D_OUT
R_CLAIM = 1.0
J_MAX = 30.0
MAX_IMAGES = 6

META_NAME = "positions_meta.json"


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
# Forward solver (numpy stand-in for al.PointSolver)
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
