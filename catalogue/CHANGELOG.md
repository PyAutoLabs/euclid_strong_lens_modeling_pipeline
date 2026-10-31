# ECLIPSE DR1 catalogue changelog

Changes that affect how the released files are read. The most recent is first.

## DR1 v1.0: sky orientation (2026-10-31)

The pipeline loads each cut-out FITS without flipping its rows. For the north-up DR1 cut-outs
this means the fitted coordinates have **+y pointing south and +x pointing west**. The fits are
all valid, because every input was read the same way. However, any direction you read straight
off a fit is the mirror image of its direction on the sky. This release makes every product
usable on the sky without that caveat.

- **`coolest.json` and `coolest_sersic.json` were regenerated (values changed).** Earlier
  releases wrote them in the pipeline frame, so `phi` was minus the sky position angle and
  `center_y` had the wrong sign. They are now rebuilt from each fit's stored best-fit model,
  converted to the frame COOLEST expects (+y north, +x west). Only `phi`, `phi_ext` and
  `center_y` change. Every other value is identical to before, which was checked on every file.
  Each regenerated file carries `meta.frame`. The originals and their sha256 sums are kept with
  the release build.
- **The model FITS now carry a WCS.** This covers `model.fits`, `pre_psf.fits`,
  `convergence.fits`, `potential.fits` and `deflections.fits`. Each image HDU gets its cut-out
  band's celestial WCS, and the zoomed mass-map grids get the correct pixel offset. Only the
  headers changed, not the data. `SKYFRAME` records the pipeline frame, and `DEFLNOTE` notes
  that `DEFLECTIONS_Y > 0` points south.
- **New sky-frame columns in the CSV tables.** For the mass, lens light and source light there
  are `*_east_offset` and `*_north_offset` (arcsec, with their credible intervals) and
  `*_pa_east_of_north` (degrees, [0, 180)). There is also `mass_shear_pa_east_of_north`. Every
  existing column is unchanged.
- **Unchanged:** `wcs_vis_lp.json`, `wcs_vis_pix.json`, `wcs.csv`, `positions*.json` and
  `info.json`. Their RA and Dec were always computed through the cut-out's FITS WCS, so they
  were already correct.

To convert a fitted angle φ (degrees, counter-clockwise from +x) yourself: the position angle
east of north is (90° − φ) mod 180°, east is −x and north is −y. The conventions of COOLEST,
lenstronomy and herculens are compared in `docs/sky_orientation/conventions.md` in the pipeline
repository.
