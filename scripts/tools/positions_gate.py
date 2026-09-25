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

    6. **pair floor** (phase 2) — when the tile ships its segmentation maps
       (``segmentation/source_flux.fits`` and the VIS RMS map), a final set of
       exactly two positions needs both peak source-flux SNRs >= 3; otherwise
       the fainter is dropped (``pair_floor``) and the tile goes to review with
       no positions.

    Steps 1-6 are ``positions_finder.gate_result``, the same call that ends
    the production positions writer (``positions_finder.find_positions_gate``,
    used by ``preprocess/segmentation.py`` and the ``util`` fallback), so
    existing and new tiles share one logic. Extra sidecar keys over phase 1:
    ``method`` (``gate``), ``finder_version``, ``s_final``, ``snr`` and
    ``added`` (always empty).

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
    tile on a login node or a small CPU array job. The tracer, quick fit,
    threshold rule and light-centre port are shared with the segmentation writer
    and the ``util`` fallback through ``positions_finder.py`` at the package root.

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

# The tracer, quick fit, threshold rule and light centre live in the shared
# pure-numpy finder module at the package root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from positions_finder import (  # noqa: E402,F401
    CAP,
    CENTRAL_RADIUS,
    META_VERSION,
    D_OUT,
    DJ,
    E_MAX,
    EASY,
    FACTOR,
    FLOOR,
    G_MAX,
    META_NAME,
    NM_BAND,
    PARAM_NAMES,
    SIG_E,
    SIG_G,
    SIG_S,
    TE_MAX,
    _bounds,
    _chi2_fit,
    _nm_fit,
    _nm_objective,
    _pairwise,
    _resid,
    _sep_fit,
    _starts,
    brightest_sub_pixel_centre,
    DJ_FLAG,
    apply_pair_floor,
    central_cut,
    gate_result,
    leave_one_out,
    max_separation,
    meta_from_result,
    outlier_drop,
    plausibility_flag,
    quick_fit,
    snr_at,
    snr_map_from,
    source_positions,
    threshold_from,
)
from positions_finder import gate_positions as _pf_gate_positions  # noqa: E402

# 2.1: phase 1 steps + the pair floor, shared with the production writer
# (2.0 was the reconcile-loop finder path, now a diagnostic only).
GATE_VERSION = META_VERSION

REVIEW_CSV = "positions_review.csv"
SUBMIT_TXT = "positions_submit.txt"


# ---------------------------------------------------------------------------
# Whole-tile gate
# ---------------------------------------------------------------------------


def _r4(v):
    return None if v is None else round(float(v), 4)


def gate_positions(positions, centre) -> Dict:
    """
    Run the phase 1 gate (steps 1-5) on one position set; returns the
    ``positions_meta.json`` dict (without the tile-level keys). ``positions``
    are raw (y, x) arcsec, ``centre`` the light centre. The steps live in
    ``positions_finder.gate_positions``, shared with the production writer;
    :func:`gate_tile` adds the pair floor.
    """
    return _pf_gate_positions(positions, centre, GATE_VERSION)


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


def finder_maps_from_dataset(dataset_dir, image_tag: str = "_BGSUB") -> Optional[Dict]:
    """
    The finder's inputs for one tile, or None when the tile has no usable
    ``segmentation/source_flux.fits``.

    Returns ``source_flux``, ``snr_map`` (``source_flux / VIS_RMS`` where the RMS
    is positive, as ``preprocess/segmentation.py`` builds it), ``lens_flux`` (or
    None) and ``pixel_scale``.
    """
    from astropy.io import fits

    d = Path(dataset_dir)
    sf_path = d / "segmentation" / "source_flux.fits"
    if not sf_path.exists() or not (d / "info.json").exists():
        return None
    with open(d / "info.json") as f:
        ps = float(json.load(f)["pixel_scale"])
    source_flux = np.asarray(fits.getdata(sf_path), np.float32)
    noise = None
    with fits.open(d / f"{d.name}.fits") as hdul:
        names = [h.name for h in hdul]
        for tag in (image_tag, "_FLUX", "_BGSUB"):
            bands = [n[: -len(tag)].lower() for n in names if n.endswith(tag)]
            if "vis" in bands:
                noise = np.asarray(hdul[bands.index("vis") * 3 + 3].data, np.float32)
                break
    if noise is None or noise.shape != source_flux.shape:
        return None
    lens_flux = None
    lf_path = d / "segmentation" / "lens_flux.fits"
    if lf_path.exists():
        lens_flux = np.asarray(fits.getdata(lf_path), np.float32)
        if lens_flux.shape != source_flux.shape:
            lens_flux = None
    return dict(
        source_flux=source_flux,
        snr_map=snr_map_from(source_flux, noise),
        lens_flux=lens_flux,
        pixel_scale=ps,
    )


def gate_tile(dataset_dir, light_centre=None, write: bool = True) -> Optional[Dict]:
    """
    Gate one tile directory and (by default) write its ``positions_meta.json``.

    Runs the phase 1 steps 1-5 above on the raw ``positions.json`` and then
    the pair floor (``positions_finder.apply_pair_floor``: a final pair needs
    both peak SNRs >= 3 on the segmentation SNR map, else review with no
    positions). The same ``positions_finder.gate_result`` call ends the
    production writer, so a tile whose ``positions.json`` equals its SNR >= 3
    peak set gets the same verdict either way. Without the segmentation maps
    the pair floor cannot be judged and the phase 1 verdict stands.

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
    maps = finder_maps_from_dataset(d)
    snrs = None if maps is None else snr_at(maps["snr_map"], positions, maps["pixel_scale"])
    result = gate_result(positions, light_centre, snrs, GATE_VERSION)
    meta = meta_from_result(result, GATE_VERSION)
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
        if not (d / "positions.json").exists() and (d / META_NAME).exists():
            # The segmentation writer left no positions (review / pair_floor):
            # its sidecar still decides the tile.
            with open(d / META_NAME) as f:
                old = json.load(f)
            if old.get("version") == GATE_VERSION:
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
