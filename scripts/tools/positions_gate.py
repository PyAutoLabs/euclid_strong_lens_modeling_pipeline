"""
Pre-submit positions gate for the ``vis_lp`` positions penalty.

WHAT THIS DOES
    For every tile it reads the raw ``positions.json`` (never modified) and the
    light centre that ``vis_lp`` fixes its mass centre to, then

    1. **central cut** — removes positions within ``CENTRAL_RADIUS`` (0.15", one
       VIS PSF FWHM) of the light centre: those are the lens nucleus, not a
       lensed image;
    2. **quick fit** — fits a singular isothermal ellipsoid with its centre fixed
       at the light centre plus external shear, in exactly ``vis_lp``'s model
       space (``einstein_radius`` in [0, 8], ``gamma_1``/``gamma_2`` in
       [-0.3, 0.3], ``|ell_comps| < 0.95``), and returns

       - ``s_min``: the smallest achievable maximum pairwise source-plane
         separation of the positions (exactly the statistic ``al.PositionsLH``
         penalises), the minimum over a least-squares + SLSQP minimax fit, the
         census Nelder-Mead starts and an Einstein-radius start grid;
       - ``J``: a plausibility cost, the source-plane chi-squared (sigma 0.05")
         plus the ``ell_comps`` N(0, 0.3) prior and a weak shear term;

    3. **outlier drop** — if the set cannot be traced (``s_min > 0.2"``) and it
       has at least three positions, leaves each position out in turn and drops
       the single one that makes it traceable (cost / geometry tie-break), or
       sends the tile to review;
    4. **plausibility flag** — a traceable set where dropping one position lowers
       ``J`` by at least 10 is flagged (listed only, nothing is dropped);
    5. **threshold** — ``T = min(max(2 s_final, 0.3"), 0.5")``; a tile whose
       ``2 s_final`` exceeds the cap goes to review.

    The result is written next to ``positions.json`` as ``positions_meta.json``,
    which ``util.load_vis_dataset`` reads (positions used + threshold). A batch
    run also writes ``positions_review.csv`` (review, flag and ``n_lt_2`` tiles)
    and ``positions_submit.txt`` (tiles to submit; review tiles are held back
    unless ``--include-review``).

    The research behind every number here (the 2026-09-23 census of 14,032 DR1
    tiles and the 2026-09-24 method study) is in the ``euclid_dr1`` project under
    ``inspect/positions_census/research/`` (``SYNTHESIS.md``, ``B_final_method.md``,
    ``A_central_radius.md``); this module is the production port of its
    ``quickfit3.py`` / ``trace.py`` / ``eval_sample.py`` prototype.

    Pure numpy / scipy / astropy: no PyAutoLens import, so it runs in seconds per
    tile on a login node or a small CPU array job.

USAGE
    python scripts/tools/positions_gate.py --tile dataset/<sample>/<tile>
    python scripts/tools/positions_gate.py --root dataset/<sample> --nproc 8
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize

GATE_VERSION = "1.0"

# Step 1: hard cut around the light centre (one VIS PSF FWHM).
CENTRAL_RADIUS = 0.15
# Step 3: a set is "traceable" when s_min <= D_OUT (the vis_lp default threshold).
D_OUT = 0.2
# Step 3: a leave-one-out candidate wins on cost when it beats the runner-up by DJ.
DJ = 4.0
# Step 4: plausibility flag when one drop lowers J by at least DJ_FLAG.
DJ_FLAG = 10.0
# Step 5: T = min(max(FACTOR * s_final, FLOOR), CAP).
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

META_NAME = "positions_meta.json"
REVIEW_CSV = "positions_review.csv"
SUBMIT_TXT = "positions_submit.txt"

PARAM_NAMES = ("einstein_radius", "ell_comps_0", "ell_comps_1", "gamma_1", "gamma_2")


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


# ---------------------------------------------------------------------------
# Gate steps
# ---------------------------------------------------------------------------


def central_cut(positions, centre, r: float = CENTRAL_RADIUS) -> Tuple[List[int], List[int]]:
    """Return ``(kept_indices, cut_indices)``: positions within ``r`` of ``centre`` are cut."""
    pos = np.asarray(positions, float).reshape(-1, 2)
    dist = np.hypot(pos[:, 0] - centre[0], pos[:, 1] - centre[1])
    kept = [i for i in range(len(pos)) if dist[i] >= r]
    cut = [i for i in range(len(pos)) if dist[i] < r]
    return kept, cut


def threshold_from(
    s_final: float, factor: float = FACTOR, floor: float = FLOOR, cap: float = CAP
) -> Tuple[float, bool]:
    """``T = min(max(factor * s, floor), cap)``; the bool is True when ``factor * s > cap`` (review)."""
    raw = factor * float(s_final)
    return float(min(max(raw, floor), cap)), bool(raw > cap)


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


# ---------------------------------------------------------------------------
# Whole-tile gate
# ---------------------------------------------------------------------------


def _r4(v):
    return None if v is None else round(float(v), 4)


def gate_positions(positions, centre) -> Dict:
    """
    Run the full gate on one position set; returns the ``positions_meta.json`` dict
    (without the tile-level keys). ``positions`` are raw (y, x) arcsec, ``centre``
    the light centre.
    """
    pos_raw = np.asarray(positions, float).reshape(-1, 2)
    centre = (float(centre[0]), float(centre[1]))
    kept, cut = central_cut(pos_raw, centre)
    dropped = [
        dict(index=i, y=_r4(pos_raw[i, 0]), x=_r4(pos_raw[i, 1]), reason="central_cut")
        for i in cut
    ]
    meta = dict(
        version=GATE_VERSION,
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


def read_positions(dataset_dir) -> Optional[np.ndarray]:
    """The raw ``positions.json`` values as an (N, 2) array, or None if absent."""
    path = Path(dataset_dir) / "positions.json"
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    values = payload
    if isinstance(payload, dict):
        values = payload.get("arguments", {}).get("values", {}).get("array", payload)
    return np.asarray(values, float).reshape(-1, 2)


def _positions_sha(dataset_dir) -> str:
    return hashlib.sha256((Path(dataset_dir) / "positions.json").read_bytes()).hexdigest()[:16]


def light_centre_from_dataset(dataset_dir, image_tag: str = "_BGSUB") -> Tuple[float, float]:
    """
    The light centre ``util.load_vis_dataset`` returns as ``dataset_centre``.

    Replicates it without PyAutoLens: the mask centre is the peak of
    ``segmentation/lens_flux.fits`` (else ``info.json`` ``mask_centre``, else the
    frame centre); the brightest VIS pixel inside a +/-0.3" box around it is
    refined by a flux-weighted centroid over the 5x5 pixels around it
    (``Array2D.brightest_sub_pixel_coordinate_in_region_from(box_size=2)``).
    """
    from astropy.io import fits

    d = Path(dataset_dir)
    name = d.name
    with open(d / "info.json") as f:
        info = json.load(f)
    ps = float(info["pixel_scale"])
    with fits.open(d / f"{name}.fits") as hdul:
        names = [h.name for h in hdul]
        vis_index = None
        for tag in (image_tag, "_FLUX", "_BGSUB"):
            bands = [n[: -len(tag)].lower() for n in names if n.endswith(tag)]
            if "vis" in bands:
                vis_index = bands.index("vis")
                break
        if vis_index is None:
            raise ValueError(f"{name}: no VIS image HDU")
        img = np.asarray(hdul[vis_index * 3 + 1].data, float)
    ny, nx = img.shape
    mask_centre = None
    lf_path = d / "segmentation" / "lens_flux.fits"
    if lf_path.exists():
        lf = np.asarray(fits.getdata(lf_path), np.float32)
        if lf.shape == img.shape:
            pr, pc = np.unravel_index(int(np.nanargmax(lf)), lf.shape)
            mask_centre = ((ny / 2 - 0.5 - pr) * ps, (pc - nx / 2 + 0.5) * ps)
    if mask_centre is None:
        mask_centre = tuple(info.get("mask_centre") or (0.0, 0.0))
    return brightest_sub_pixel_centre(img, ps, mask_centre)


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


def gate_tile(dataset_dir, light_centre=None, write: bool = True) -> Optional[Dict]:
    """
    Gate one tile directory and (by default) write its ``positions_meta.json``.

    Returns None (and writes nothing) when the tile has no ``positions.json`` or
    fewer than two raw positions: ``util.load_vis_dataset`` then keeps today's
    behaviour (its source-flux fallback).
    """
    d = Path(dataset_dir)
    positions = read_positions(d)
    if positions is None or len(positions) < 2:
        return None
    if light_centre is None:
        light_centre = light_centre_from_dataset(d)
    t0 = time.process_time()
    meta = gate_positions(positions, light_centre)
    meta = dict(tile=d.name, positions_sha=_positions_sha(d), **meta)
    meta["cpu_s"] = round(time.process_time() - t0, 2)
    if write:
        with open(d / META_NAME, "w") as f:
            json.dump(meta, f, indent=2)
    return meta


def _gate_one(args):
    d, force = args
    d = Path(d)
    try:
        if not force and (d / META_NAME).exists() and (d / "positions.json").exists():
            with open(d / META_NAME) as f:
                old = json.load(f)
            if old.get("version") == GATE_VERSION and old.get("positions_sha") == _positions_sha(d):
                return d.name, old, None
        return d.name, gate_tile(d), None
    except Exception as exc:  # one broken tile must not stop a batch
        return d.name, None, repr(exc)


REVIEW_COLUMNS = (
    "tile",
    "status",
    "review_reason",
    "n_raw",
    "n_used",
    "s_min_all",
    "s_min",
    "J",
    "threshold",
    "dropped",
    "flag_index",
)


def gate_batch(
    root,
    out_csv=None,
    include_review: bool = False,
    nproc: int = 1,
    force: bool = False,
    tiles: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """
    Gate every tile directory under ``root``; write ``positions_review.csv`` and
    ``positions_submit.txt`` (next to ``out_csv``, default inside ``root``).

    Tiles are skipped from the submit list when their status is ``review``
    unless ``include_review``; tiles without a sidecar (no/one raw position) are
    submitted as before. An unchanged ``positions.json`` with a current-version
    sidecar is not re-fitted unless ``force``.
    """
    root = Path(root)
    names = sorted(tiles) if tiles else sorted(p.name for p in root.iterdir() if p.is_dir())
    jobs = [(root / n, force) for n in names if (root / n).is_dir()]
    if nproc > 1:
        from multiprocessing import Pool

        with Pool(nproc) as pool:
            results = list(pool.imap(_gate_one, jobs, chunksize=1))
    else:
        results = [_gate_one(j) for j in jobs]

    out_csv = Path(out_csv) if out_csv else root / REVIEW_CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows, submit = [], []
    for name, meta, err in results:
        if err is not None:
            rows.append(dict(tile=name, status="error", review_reason=err))
            continue
        if meta is None:
            submit.append(name)
            continue
        if meta["status"] != "review" or include_review:
            submit.append(name)
        if meta["status"] in ("review", "flag", "n_lt_2"):
            rows.append(
                dict(
                    tile=name,
                    status=meta["status"],
                    review_reason=meta["review_reason"],
                    n_raw=meta["n_raw"],
                    n_used=len(meta["positions_used"]),
                    s_min_all=meta["s_min_all"],
                    s_min=meta["s_min"],
                    J=meta["J"],
                    threshold=meta["threshold"],
                    dropped=";".join(f"{x['index']}:{x['reason']}" for x in meta["dropped"]),
                    flag_index=meta["flag_index"],
                )
            )
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(out_csv.parent / SUBMIT_TXT, "w") as f:
        f.write("".join(n + "\n" for n in submit))
    return [dict(tile=n, meta=m, error=e) for n, m, e in results]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Positions gate: 0.15\" central cut, fixed-centre SIE+shear quick fit, "
            "one-image outlier drop and a per-tile PositionsLH threshold, written "
            "to positions_meta.json beside each tile's (untouched) positions.json."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--tile", help="one tile directory (containing positions.json)")
    src.add_argument("--root", help="a sample directory whose subdirectories are tiles")
    parser.add_argument("--tiles-file", help="with --root: only the tiles named in this file")
    parser.add_argument("--out-csv", help=f"review CSV path (default <root>/{REVIEW_CSV})")
    parser.add_argument(
        "--include-review",
        action="store_true",
        help=f"list review tiles in {SUBMIT_TXT} too (held back by default)",
    )
    parser.add_argument("--nproc", type=int, default=1, help="worker processes (default 1)")
    parser.add_argument("--force", action="store_true", help="re-fit even if the sidecar is current")
    args = parser.parse_args(argv)

    if args.tile:
        meta = gate_tile(args.tile)
        if meta is None:
            print(f"{args.tile}: no positions.json or < 2 positions; no sidecar written")
            return 0
        print(json.dumps(meta, indent=2))
        return 0

    tiles = None
    if args.tiles_file:
        tiles = Path(args.tiles_file).read_text().split()
    t0 = time.time()
    results = gate_batch(
        args.root,
        out_csv=args.out_csv,
        include_review=args.include_review,
        nproc=args.nproc,
        force=args.force,
        tiles=tiles,
    )
    counts: Dict[str, int] = {}
    for r in results:
        key = "error" if r["error"] else ("no_sidecar" if r["meta"] is None else r["meta"]["status"])
        counts[key] = counts.get(key, 0) + 1
    print(f"gated {len(results)} tiles in {time.time() - t0:.0f} s: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
