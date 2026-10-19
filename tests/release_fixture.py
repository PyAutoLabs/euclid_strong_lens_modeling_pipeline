"""
Synthetic project tree for the release-packaging tests
(``tests/test_csv_snapshot.py``, ``tests/test_package_release.py``).

``make_project(root)`` writes, under ``root``:

- ``manifest.csv`` (v5 schema): 5 included tiles (T1..T5) and one ``no_model``
  row (an A/B object with no tile); ``objects.csv`` (v5 objects: those six plus
  one grade C object outside the manifest) and ``uncovered.csv`` (the
  ``no_model`` object).
- ``output/dr1_sep1/<tile>/initial_lens_model/{vis_lp,vis_pix}/<hash>.zip`` and
  ``output_sed/dr1_sep1/<tile>/sersic_lens_model/{vis,nir_y}/<hash>.zip`` result
  zips with ``.completed`` and small summary / latent / wcs JSONs. T2 has no
  vis_pix result; T3 has no SED results; T1 also has a NEWER vis_pix result
  modified after ``CUTOFF`` (must be ignored).
- ``dataset/dr1_sep1/<tile>/``: ``<tile>.fits`` (VIS + NIR_Y BGSUB HDUs),
  ``segmentation/*.fits``, ``positions.json``, ``positions_meta.json``,
  ``info.json``, plus a ``rgb.jpg`` that no tier ships. T4 has no ``info.json``.
- ``inspect/cat/<tile>/``: the per-lens catalogue folders of T1..T4 (T5 absent),
  with every full-tier product except: T2 lacks ``lens_mass.csv`` and the source
  reconstruction CSV, T3 lacks ``fit_sersic.png``, ``fit_multi_wavelength.png``,
  ``magnitudes.csv``. ``model.fits`` / ``pre_psf.fits`` carry VIS and NIR_Y HDU
  pairs. Root tables ``lens_mass.csv``, ``magnitudes.csv``, ``lens_index.csv``.
"""

import csv
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

TILES = ["T1", "T2", "T3", "T4", "T5"]
UIDS = {t: f"EUCL-DR1-L00000{i}" for i, t in enumerate(TILES, 1)}
CUTOFF = time.time() - 3600  # results older than an hour are "before the cutoff"

FULL_FILES = [
    "astrometric_offsets.csv", "convergence.fits", "coolest.json", "coolest_sersic.json",
    "deflections.fits", "fit_multi_wavelength.png", "fit_sersic.png", "lens_mass.csv",
    "lens_sersic.csv", "magnitudes.csv", "model.fits", "potential.fits", "pre_psf.fits",
    "rgb.png", "segmentation.png", "source_sersic.csv", "vis_lp_fit.png",
    "vis_lp_image_with_positions.png", "vis_pix_fit.png",
    "vis_pix_source_plane_reconstruction_0.csv", "wcs_vis_lp.json", "wcs_vis_pix.json",
    "witt_wynne.csv", "witt_wynne.in",
]
CATALOGUE_MISSING = {
    "T2": {"lens_mass.csv", "vis_pix_source_plane_reconstruction_0.csv"},
    "T3": {"fit_sersic.png", "fit_multi_wavelength.png", "magnitudes.csv"},
}
SEGMENTATION = ["lens_binary", "lens_flux", "source_binary", "source_flux"]


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _summary(prefix_values):
    median = dict(prefix_values)
    return {
        "median_pdf_sample": {"kwargs": median},
        "max_log_likelihood_sample": {"kwargs": {k: v + 0.01 for k, v in median.items()}},
        "values_at_sigma_1": {k: [v - 0.1, v + 0.1] for k, v in median.items()},
        "errors_at_sigma_3": {k: [0.3, 0.3] for k, v in median.items()},
        "model": {"huge": "dropped"},
    }


def _result_zip(path: Path, summary=None, latent=None, wcs=None, mtime=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(".completed", "")
        if summary is not None:
            archive.writestr("files/samples_summary.json", json.dumps(summary))
        if latent is not None:
            archive.writestr("files/latent/latent_summary.json", json.dumps(latent))
        if wcs is not None:
            archive.writestr("files/wcs.json", json.dumps(wcs))
    stamp = mtime if mtime is not None else CUTOFF - 600
    os.utime(path, (stamp, stamp))


def _fits(path: Path, names):
    from astropy.io import fits

    hdus = [fits.PrimaryHDU()] + [fits.ImageHDU(np.zeros((4, 4), dtype="f4"), name=n) for n in names]
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _results(root: Path, i: int, tile: str):
    base = root / "output" / "dr1_sep1" / tile / "initial_lens_model"
    wcs = {"crval_ra_deg": 10.0 + i, "crval_dec_deg": -5.0, "crpix_x": 51, "crpix_y": 51,
           "source_clumps": [{"x_arcsec": 0.1, "y_arcsec": 0.2}]}
    mass = {"galaxies.lens.mass.einstein_radius": 1.0 + i / 10,
            "galaxies.lens.mass.centre.centre_0": 0.01, "fields.shear.gamma_1": 0.02}
    _result_zip(base / "vis_lp" / f"lp{i}.zip", _summary(mass), None, wcs)
    if tile != "T2":
        _result_zip(base / "vis_pix" / f"pix{i}.zip", _summary(mass),
                    _summary({"effective_einstein_radius": 1.1, "magnification": 5.0}), wcs)
    if tile == "T1":
        # Finished after the cutoff: a snapshot at CUTOFF must not see it.
        _result_zip(base / "vis_pix" / "pix_late.zip",
                    _summary({"galaxies.lens.mass.einstein_radius": 99.0}), None, wcs,
                    mtime=time.time())
    if tile != "T3":
        sed = root / "output_sed" / "dr1_sep1" / tile / "sersic_lens_model"
        light = {"galaxies.lens.bulge.sersic_index": 4.0, "galaxies.source.bulge.sersic_index": 1.0,
                 "galaxies.lens.bulge.effective_radius": 0.5}
        flux = {"total_lens_flux_mujy": 10.0, "total_lensed_source_flux_mujy": 2.0,
                "total_source_flux_mujy": 0.4, "magnification": 5.0,
                "dataset_model.grid_offset.grid_offset_0": 0.01}
        _result_zip(sed / "vis" / f"vis{i}.zip", _summary(light), _summary(flux), wcs)
        _result_zip(sed / "nir_y" / f"y{i}.zip", _summary(light), _summary(flux), wcs)


def make_project(root: Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [(t, UIDS[t], "A" if i % 2 else "B", 2.5, "sep1", "True", "included", "")
            for i, t in enumerate(TILES)]
    rows += [("", "EUCL-DR1-L000099", "B", 2.1, "", "False", "no_model", "no FITS made")]
    _write_csv(root / "manifest.csv",
               ["tile", "unique_identifier", "grade", "score", "cutout_batch", "include_in_catalogue", "reason",
                "note"], rows)
    # T1 was modelled on an earlier cutout than the one v5 recommends.
    objects = [(UIDS[t], 2.5, g, 10.0 + i, -5.0, t, ("sep28:" if t == "T1" else "sep1:") + t, "sep1:" + t)
               for i, (t, _u, g, *_r) in enumerate(rows[:5])]
    objects += [("EUCL-DR1-L000090", 0.5, "C", 1.0, 1.0, "TC", "", ""),
                ("EUCL-DR1-L000099", 2.1, "B", 2.0, 2.0, "TX", "v5:TX_v5", "")]
    _write_csv(root / "objects.csv",
               ["Unique_Identifier", "SWAP_Score", "SWAP_Grade", "RA_obj", "DEC_obj", "id_str_for_cutout",
                "fits_to_use", "fits_modelled_20261011"], objects)
    _write_csv(root / "uncovered.csv",
               ["unique_identifier", "grade", "score", "RA_obj", "DEC_obj", "id_str_for_cutout", "tile", "reason",
                "note"],
               [("EUCL-DR1-L000099", "B", 2.1, 2.0, 2.0, "TX", "", "no_model", "no FITS made")])

    for i, tile in enumerate(TILES, 1):
        _results(root, i, tile)
        data = root / "dataset" / "dr1_sep1" / tile
        (data / "segmentation").mkdir(parents=True)
        _fits(data / f"{tile}.fits", ["VIS_BGSUB", "NIR_Y_BGSUB", "VIS_PSF"])
        for name in SEGMENTATION:
            _fits(data / "segmentation" / f"{name}.fits", ["SEG"])
        (data / "positions.json").write_text("[[0.5, 0.5]]")
        (data / "positions_meta.json").write_text("{}")
        (data / "rgb.jpg").write_bytes(b"\xff\xd8 not shipped")
        if tile != "T4":
            (data / "info.json").write_text('{"pixel_scales": 0.1}')

        if tile == "T5":
            continue
        lens = root / "inspect" / "cat" / tile
        lens.mkdir(parents=True)
        for name in FULL_FILES:
            if name in CATALOGUE_MISSING.get(tile, set()):
                continue
            if name in ("model.fits", "pre_psf.fits"):
                _fits(lens / name, ["VIS_GALAXY_0", "VIS_GALAXY_1", "NIR_Y_GALAXY_0", "NIR_Y_GALAXY_1"])
            elif name.endswith(".fits"):
                _fits(lens / name, [name.split(".")[0].upper()])
            elif name.endswith(".png"):
                (lens / name).write_bytes(b"\x89PNG" + bytes(200 * i))
            else:
                (lens / name).write_text(f"{tile},{name}\n" * i)
    cat = root / "inspect" / "cat"
    (cat / "_build").mkdir()
    (cat / "_build" / "build.env").write_text("CATALOGUE=cat\nGIT_REVISION=abc1234\n")
    for name in ("lens_mass.csv", "magnitudes.csv", "lens_index.csv"):
        (cat / name).write_text("lens_name,x\nT1,1\n")
    return root
