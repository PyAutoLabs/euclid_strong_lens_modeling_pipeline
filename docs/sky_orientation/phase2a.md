# Sky orientation, phase 2a: making the catalogue products sky-frame

Issue: PyAutoLabs/euclid_strong_lens_modeling_pipeline#115. Human decisions (2026-10-31):
regenerate the COOLEST templates, add a WCS to the model FITS. No lens model is refit. The
user-facing summary is in `catalogue/CHANGELOG.md`.

## What changed

| piece | where | what |
|---|---|---|
| tracer mirror | `scripts/tools/sky_frame.py` | `plus_y_north_from(tracer)`: centres y → −y, `ell_comps[0]` → −, `gamma_2` → −, `multipole_comps[0]` → −. Raises on any explicit angle it does not know. |
| new fits | `util.coolest_json_from(pixel_wcs=)` | `save_results` passes the dataset WCS. A +y-south WCS (every DR1 cut-out) mirrors the tracer before export; `meta.frame` records which frame the file is in. |
| existing products | `scripts/tools/regenerate_coolest.py` | Rebuilds `coolest.json` / `coolest_sersic.json` of a bundle from each fit's `files/tracer.json`. It runs as stage 1b of `build_inspection_bundle.sh` (details below). |
| model FITS WCS | `scripts/tools/add_fits_wcs.py` | Header-only: the cut-out band's celestial WCS on every image HDU, with CRPIX shifted for zoomed grids. It is idempotent and runs as stage 3b. |
| sky columns | `scripts/tools/csv_snapshot/sky_columns.py` | Appends the `*_east_offset`, `*_north_offset`, `*_pa_east_of_north` and `mass_shear_pa_east_of_north` columns. Run by `build.py` after assemble and before package. Existing bytes are kept, including the `\n` terminator. |
| docstring | `scripts/tools/build_wcs_package.py` | +y points towards decreasing FITS row: south. |

### How `regenerate_coolest.py` works

- It first checks that the *unmirrored* export reproduces the stored template. A lens that fails
  this check is left untouched and reported as `input_mismatch`.
- It snapshots each original with its sha256 into `<bundle>_coolest_regeneration/`, a sibling of
  the bundle rather than a folder inside it, because the bundle stages treat every sub-directory
  as a lens.
- It writes one timestamped report per run. The manifest is append-only.
- A rerun reports `already` for every file it has done.

## Evidence

- **Physics (`tests/test_sky_frame.py`).** The mirrored tracer, evaluated on the mirrored grid,
  gives the mirrored deflections and identical images to 1e-10. The test covers a PowerLaw, a
  multipole, a shear and Sersic light.
- **COOLEST (`tests/test_sky_frame.py`, `tests/test_coolest_output.py`).** A lens at
  PA 30° exports `phi` = 30 and a north-positive `center_y`. The pipeline function does the
  same when given the DR1 WCS.
- **Local pilot.** The tool was run on all 113 local `vis_pix` templates
  (`Science/scratch/sky_orientation_dr1/p2a_pilot`):
  - 113/113 passed the input check and were regenerated.
  - `phi` and `phi_ext` flipped (|after + before| mod 180 ≤ 1.4e-14) and `center_y` flipped,
    while `center_x` is unchanged.
  - All 339 non-oriented values are identical.
  - A rerun reported 113 × `already`.
- **FITS WCS.** On a real `tracer.fits` (102×102) and `galaxy_images.fits` (100×100), each array
  pixel's WCS sky position equals the cut-out's sky position for the same arcsec point to 0".
  The real RAL bundle files keep the `PIXSCAY`/`ORIGIN` keys the tool reads.

## Not yet done

- **Bundle run on RAL.** Run the two stages over the full `dr1_sep1` bundle, then merge and
  package, during the 2c re-render campaign. That run produces the release-scale snapshot and
  diff report the JSON procedure requires.
- **CHANGELOG in the release.** Shipping `catalogue/CHANGELOG.md` inside the release zips is
  part of 2b (docs).
