"""
Phase-1 Witness (1) on real DR1 lenses: read-only, no search.

For each local ``vis_lp`` result it checks:

- frame: the fitted DATA array in the result zip equals the cutout's VIS_BGSUB
  HDU byte for byte (so the fit saw the FITS in native row order, no flip);
- centre: the lens light centre converted by ``conversions.east_north_from``
  plus the cutout WCS lands within a pixel of the stored ``wcs.json`` crval
  (which goes through astropy ``wcs_pix2world`` on the FITS pixel);
- angles: for lens light, lens mass and external shear, the analytic PA
  (``pa_from_native_angle`` on the fitted parameter) agrees with an independent
  pixel-route PA (second moments of a model map, read through the CD matrix);
  for the light also against the second moments of the VIS data itself.

Usage:
    python scripts/research/sky_orientation/witness_real.py \
        --output <euclid_dr1>/output/dr1_sep1 --dataset <dataset>/dr1_sep1 \
        --out <report dir> [--n 20]
"""
import argparse
import csv
import glob
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conversions as cv  # noqa: E402

import autolens as al  # noqa: E402
from autonerves.dictable import from_json  # noqa: E402

PS = 0.1


def load_zip(zip_path, tmp):
    with zipfile.ZipFile(zip_path) as z:
        for name in ("files/tracer.json", "files/wcs.json", "image/dataset.fits"):
            z.extract(name, tmp)
    tracer = from_json(os.path.join(tmp, "files/tracer.json"))
    with open(os.path.join(tmp, "files/wcs.json")) as f:
        wcs_json = json.load(f)["arguments"]
    data = fits.getdata(os.path.join(tmp, "image/dataset.fits"), extname="DATA")
    return tracer, wcs_json, data


def cutout_from(dataset_dir, tile):
    fits_path = [
        p for p in glob.glob(os.path.join(dataset_dir, tile, "*.fits"))
        if "segmentation" not in p
    ][0]
    with fits.open(fits_path) as h:
        return h["VIS_BGSUB"].data.astype(float), h["VIS_BGSUB"].header


def shear_image(shear, grid, sigma=0.3):
    """Lensed image of a round Gaussian source through the shear alone."""
    traced = np.asarray(grid) - np.asarray(shear.deflections_yx_2d_from(grid=grid))
    return np.exp(-0.5 * (traced ** 2).sum(axis=1) / sigma ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    zips = sorted(glob.glob(os.path.join(a.output, "*", "initial_lens_model", "vis_lp", "*.zip")))
    rows = []
    for zip_path in zips[: a.n]:
        tile = Path(zip_path).parents[2].name
        with tempfile.TemporaryDirectory() as tmp:
            tracer, wj, data = load_zip(zip_path, tmp)
        vis, header = cutout_from(a.dataset, tile)
        cd = cv.cd_matrix_from(header)
        wcs = WCS(header)
        shape = data.shape
        grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=PS)
        lens = tracer.galaxies[0]
        shear = tracer.fields[0].shear if getattr(tracer, "fields", None) else None

        # Frame: the fitted array is the FITS array in native order.
        same_frame = bool(np.array_equal(data.astype(np.float32), vis.astype(np.float32)))
        flipped_frame = bool(np.array_equal(data.astype(np.float32), vis[::-1].astype(np.float32)))

        # Centre: analytic east/north from the image centre vs stored crval.
        cy, cx = lens.bulge.centre
        ra0, dec0 = wcs.wcs_pix2world(shape[1] / 2 + 0.5, shape[0] / 2 + 0.5, 1)
        east, north = cv.east_north_from(cy, cx, cd, PS)
        ra = ra0 + east / 3600.0 / np.cos(np.radians(dec0))
        dec = dec0 + north / 3600.0
        d_centre = 3600.0 * np.hypot(
            (ra - wj["crval_ra_deg"]) * np.cos(np.radians(dec0)), dec - wj["crval_dec_deg"]
        )

        row = dict(
            tile=tile, same_frame=same_frame, flipped_frame=flipped_frame,
            centre_y=cy, centre_x=cx, centre_east=float(east), centre_north=float(north),
            centre_vs_wcs_json_arcsec=float(d_centre),
        )

        # Lens light: the brightest MGE group's angle (groups share ell_comps).
        light_model = np.asarray(lens.bulge.image_2d_from(grid=grid)).reshape(shape)
        groups = {}
        for prof in lens.bulge.profile_list:
            flux = float(np.asarray(prof.image_2d_from(grid=grid)).sum())
            groups.setdefault(tuple(prof.ell_comps), 0.0)
            groups[tuple(prof.ell_comps)] += flux
        light_ec = max(groups, key=groups.get)
        q_l, phi_l = cv.q_phi_from_ell_comps(light_ec)
        pa_l = cv.pa_from_native_angle(phi_l, cd, PS)
        pa_l_pix, q_l_pix = cv.moments_pa(light_model, cd, PS, (cy, cx), radius=1.0)
        pa_l_dat, q_l_dat = cv.moments_pa(vis, cd, PS, (cy, cx), radius=0.6)
        row.update(
            light_q=q_l, light_phi_native=phi_l, light_pa=float(pa_l),
            light_pa_model_moments=float(pa_l_pix), light_q_model_moments=q_l_pix,
            light_pa_data_moments=float(pa_l_dat), light_q_data_moments=q_l_dat,
        )

        # Lens mass: convergence moments within a circular aperture.
        q_m, phi_m = cv.q_phi_from_ell_comps(lens.mass.ell_comps)
        pa_m = cv.pa_from_native_angle(phi_m, cd, PS)
        kappa = np.asarray(lens.mass.convergence_2d_from(grid=grid)).reshape(shape)
        pa_m_pix, _ = cv.moments_pa(kappa, cd, PS, lens.mass.centre, radius=1.0)
        row.update(mass_q=q_m, mass_phi_native=phi_m, mass_pa=float(pa_m),
                   mass_pa_moments=float(pa_m_pix))

        if shear is not None:
            g, phi_s = cv.phi_from_shear(*shear.gamma_1_2 if hasattr(shear, "gamma_1_2")
                                         else (shear.gamma_1, shear.gamma_2))
            pa_s = cv.pa_from_native_angle(phi_s, cd, PS)
            img = shear_image(shear, grid).reshape(shape)
            pa_s_pix, _ = cv.moments_pa(img, cd, PS)
            row.update(shear_mag=g, shear_phi_native=phi_s, shear_pa=float(pa_s),
                       shear_stretch_pa_moments=float(pa_s_pix))
        rows.append(row)

    with open(os.path.join(a.out, "witness_real.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} lenses -> {a.out}/witness_real.csv")


if __name__ == "__main__":
    main()
