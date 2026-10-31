"""
Scan every cutout's VIS_BGSUB WCS for rotation, parity or scale outliers
(phase-1 research, read-only). Headers only; no data is read.

Usage:
    python scripts/research/sky_orientation/wcs_scan.py --dataset <dataset>/dr1_sep1 --out <csv>
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conversions as cv  # noqa: E402

PS_DEG = 0.1 / 3600.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    rows, flags = [], Counter()
    for tile_dir in sorted(glob.glob(os.path.join(a.dataset, "*"))):
        paths = [q for q in glob.glob(os.path.join(tile_dir, "*.fits")) if "segmentation" not in q]
        if not paths:
            flags["no_fits"] += 1
            continue
        try:
            with fits.open(paths[0]) as h:
                hd = h["VIS_BGSUB"].header
                shape = (hd["NAXIS2"], hd["NAXIS1"])
            cd = cv.cd_matrix_from(hd)
        except Exception as e:  # noqa: BLE001 — record and move on
            flags[f"unreadable:{type(e).__name__}"] += 1
            rows.append(dict(tile=os.path.basename(tile_dir), flag="unreadable"))
            continue
        rot = np.degrees(np.arctan2(cd[1, 0], cd[1, 1]))
        scale = np.sqrt(abs(np.linalg.det(cd)))
        flag = []
        if not cv.is_mirrored(cd):
            flag.append("parity")
        if abs(rot) > 1e-3:
            flag.append("rotated")
        if abs(scale / PS_DEG - 1) > 1e-6:
            flag.append("scale")
        if cd[0, 0] >= 0 or cd[1, 1] <= 0:
            flag.append("cd_sign")
        if shape != (100, 100):
            flag.append("shape")
        flags["ok" if not flag else "+".join(flag)] += 1
        rows.append(dict(
            tile=os.path.basename(tile_dir), cd1_1=cd[0, 0], cd1_2=cd[0, 1], cd2_1=cd[1, 0],
            cd2_2=cd[1, 1], rotation_deg=rot, scale_arcsec=scale * 3600, ny=shape[0], nx=shape[1],
            ctype1=hd.get("CTYPE1"), flag="+".join(flag) or "ok",
        ))

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} tiles scanned -> {a.out}")
    for k, v in flags.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
