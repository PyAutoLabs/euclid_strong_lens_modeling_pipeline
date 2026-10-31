"""
Positions witness (phase-1 research, read-only): for the most asymmetric local
lenses, draw VIS twice and overlay the multiple images.

- left: the pipeline frame (array as loaded, row 0 at the top, origin=upper),
  with ``positions.json`` (native arcsec) placed through the native pixel map;
- right: north-up east-left (FITS origin=lower) with the same positions placed
  through the same pixel map, and the fit's ``wcs.json`` lensed-image RA / Dec
  placed independently through astropy ``world2pix``.

If the inputs share the pipeline frame, the crosses sit on the arcs in both
panels and the circles (sky route) sit on the crosses (native route).

Usage:
    python scripts/research/sky_orientation/positions_overlay.py \
        --output <euclid_dr1>/output/dr1_sep1 --dataset <dataset>/dr1_sep1 --out <dir> [--n 8]
"""
import argparse
import glob
import json
import os
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

PS = 0.1


def native_to_pixel(yx, shape):
    """(row, col) 0-based array pixel of native (y, x) arcsec."""
    yx = np.atleast_2d(yx)
    return shape[0] / 2 - 0.5 - yx[:, 0] / PS, yx[:, 1] / PS + shape[1] / 2 - 0.5


def positions_from(path):
    with open(path) as f:
        return np.array(json.load(f)["arguments"]["values"]["array"], dtype=float)


def listed(wj, key):
    """A wcs.json list value (stored as {"type": "list", "values": [...]})."""
    v = wj.get(key, [])
    return list(v["values"]) if isinstance(v, dict) else list(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cands = []
    for zp in sorted(glob.glob(os.path.join(a.output, "*", "initial_lens_model", "vis_lp", "*.zip"))):
        tile = zp.split(os.sep)[-4]
        pos_path = os.path.join(a.dataset, tile, "positions.json")
        if not os.path.exists(pos_path):
            continue
        pos = positions_from(pos_path)
        # Asymmetry: how far the image centroid sits from the cutout centre.
        cands.append((float(np.hypot(*pos.mean(axis=0))), tile, zp, pos))
    cands.sort(reverse=True)

    summary = []
    for _, tile, zp, pos in cands[: a.n]:
        fpath = [q for q in glob.glob(os.path.join(a.dataset, tile, "*.fits")) if "segmentation" not in q][0]
        with fits.open(fpath) as h:
            vis, header = h["VIS_BGSUB"].data.astype(float), h["VIS_BGSUB"].header
        wcs = WCS(header)
        with zipfile.ZipFile(zp) as z:
            wj = json.loads(z.read("files/wcs.json"))["arguments"]
        r, c = native_to_pixel(pos, vis.shape)
        ra, dec = listed(wj, "lensed_source_image_ra_deg"), listed(wj, "lensed_source_image_dec_deg")
        sx, sy = wcs.wcs_world2pix(np.array(ra), np.array(dec), 0) if ra else ([], [])
        my, mx = np.array(listed(wj, "lensed_source_image_y_arcsec")), np.array(listed(wj, "lensed_source_image_x_arcsec"))
        mr, mc = native_to_pixel(np.c_[my, mx], vis.shape) if len(my) else ([], [])
        d = float(np.max(np.hypot(np.asarray(sx) - mc, np.asarray(sy) - mr))) if len(my) else float("nan")
        summary.append((tile, len(pos), d))

        lo, hi = np.percentile(vis, [1, 99.5])
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
        for k, (title, origin) in enumerate([("pipeline frame (origin=upper)\n+y south, +x west", "upper"),
                                             ("north up, east left (origin=lower)", "lower")]):
            ax[k].imshow(vis, origin=origin, cmap="gray", vmin=lo, vmax=hi)
            ax[k].plot(c, r, "x", color="orange", ms=10, mew=2, label="positions.json (native)")
            if k == 1 and len(sx):
                ax[k].plot(sx, sy, "o", mfc="none", mec="cyan", ms=12, mew=1.5, label="wcs.json RA/Dec (astropy)")
            ax[k].set_title(title, fontsize=9)
            ax[k].set_xticks([])
            ax[k].set_yticks([])
        ax[1].annotate("", xy=(0.9, 0.95), xytext=(0.9, 0.78), xycoords="axes fraction",
                       arrowprops=dict(arrowstyle="->", color="w"))
        ax[1].text(0.9, 0.96, "N", color="w", transform=ax[1].transAxes, ha="center", va="bottom")
        ax[1].annotate("", xy=(0.73, 0.78), xytext=(0.9, 0.78), xycoords="axes fraction",
                       arrowprops=dict(arrowstyle="->", color="w"))
        ax[1].text(0.71, 0.78, "E", color="w", transform=ax[1].transAxes, ha="right", va="center")
        ax[1].legend(loc="lower left", fontsize=7)
        fig.suptitle(tile, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"positions_{tile}.png"), dpi=110)
        plt.close(fig)

    with open(os.path.join(a.out, "positions_overlay_summary.txt"), "w") as f:
        f.write("tile n_positions max_pixel_offset_wcs_json_sky_vs_native\n")
        for t, n, d in summary:
            f.write(f"{t} {n} {d:.4f}\n")
    print(f"{len(summary)} figures -> {a.out}")
    for t, n, d in summary:
        print(f"  {t} n={n} sky-vs-native max offset {d:.4f} px")


if __name__ == "__main__":
    main()
