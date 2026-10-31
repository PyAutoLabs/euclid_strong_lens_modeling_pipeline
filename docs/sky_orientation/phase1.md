# Sky orientation, phase 1: what frame DR1 is in, and how to convert it

Issue: PyAutoLabs/euclid_strong_lens_modeling_pipeline#113 (2026-10-31). Research only: no lens
model was refit, no library source changed, and no JSON was modified.

## The short answer

- Every DR1 fit is in one consistent frame: the cutout FITS read with no row flip. For the north-up
  DR1 cutouts this means **+y is south and +x is west**. Every input (data, noise, PSF, mask,
  segmentation, `positions.json`) shares that frame, so no fit is invalid.
- Scalar results (Einstein radius, axis ratios, Sersic index, radii, fluxes, magnifications,
  evidences) need no conversion.
- Oriented results convert with these rules (DR1 WCS: CD1_1 = -0.1"/px, CD2_2 = +0.1"/px, no
  rotation). Here φ is the PyAutoGalaxy angle, counter-clockwise from +x.

| quantity | sky frame |
|---|---|
| centre / offset (y, x) | east = -x, north = -y; RA/Dec through the cutout WCS |
| ellipse or shear angle φ | PA east of north = (90° - φ) mod 180° |
| ell_comps (e0, e1) in a +y-north, +x-west frame | (-e0, +e1) |
| shear (γ1, γ2) in a +y-north, +x-west frame | (+γ1, -γ2) |

  The general version for any CD matrix is `scripts/research/sky_orientation/conversions.py`.
- The RA/Dec already stored in each fit's `files/wcs.json` are **correct**: they go through the FITS
  pixel with astropy, so they are right whichever way the frame points.
- The COOLEST files written by the pipeline (`files/coolest.json`) are **mirrored** for DR1: phi
  comes out as -PA and center_y has the wrong sign. Our COOLEST export assumes +y is north.
  `conventions.md` gives the full table for COOLEST, lenstronomy and herculens and the correction.

## Evidence

All outputs are in `Science/scratch/sky_orientation_dr1/`.

1. **Frame of the fitted data.** For every local `vis_lp` fit, the DATA array stored in the result
   correlates with the cutout's VIS_BGSUB HDU as read (0.98–0.997) far better than with its row
   flip (0.74–0.97). It is not byte-identical: pixel ratios scatter by ~0.3 around 1, so the
   cutout was regenerated after the fits. That is worth knowing, but it does not change the frame.
2. **Positions** (`positions/`, 8 most asymmetric local lenses). The `positions.json` crosses sit on
   the arcs in the pipeline frame and, through the same pixel map, in the north-up frame. The
   `wcs.json` lensed-image RA/Dec, placed back through astropy `world2pix`, land on the native
   positions to 0.0000 px on all 8.
3. **Real-lens angles and centres** (`witness_real.csv`, all 113 local `vis_lp` fits).
   - Lens centre converted by the rule + WCS vs stored `wcs.json` crval: max 0.009".
   - Mass PA from the rule vs second moments of the fitted convergence read through the CD matrix
     (108 lenses with q < 0.85): median 0.8°, 95th percentile 4.0°, max 6.6°. Control: the
     +y-north rule (φ - 90°) gives a median 38° off.
   - Shear PA from the rule vs the stretch direction of a round source lensed by the fitted shear
     alone: identical to 1e-13° on all 113. The PyAutoGalaxy shear angle is the stretch direction.
   - Lens light PA from the rule (brightest MGE group) vs moments of the VIS data: median 15°
     (83 lenses with q < 0.8). The control gives 44°. The agreement is looser because arcs and
     the second MGE group contaminate the moments. The model light vs the data, both read through
     the CD matrix, agree to a median 6.7°.
4. **Simulated lens** (`tests/test_sky_orientation.py`, 10 tests pass). An elliptical Gaussian is
   drawn directly in sky coordinates with astropy at PA 10°, 30°, 75°, 120° and 160°. It is written
   to a FITS with the DR1 WCS, loaded with `al.Array2D.from_fits` (the DR1 way) and fitted by least
   squares (no search). The fit gives centre (y, x) = (-north, -east), and the conversion recovers
   PA to < 0.5° and RA/Dec to < 0.01". Round trips also pass for rotated and unmirrored CD matrices.
5. **Model FITS headers.** The result FITS (`image/dataset.fits`, `image/fit.fits`, …) carry
   **no WCS** at all, so none is mislabelled. A WCS will have to be added if the release wants
   them to be self-describing (phase-2 decision).
6. **Cutout WCS scan** (`wcs_scan_dr1_sep1.csv`): see "WCS scan" below.
7. **Other codes**: `conventions.md` (COOLEST, lenstronomy 1.14.3, herculens 0.3.0, with citations;
   numeric check `scripts/research/sky_orientation/conventions_check.py`, "ALL OK").
   - lenstronomy, given the DR1 WCS as `transform_pix2angle`, is the PyAutoLens frame rotated by
     180°. Its (e1, e2) = (ell_comps[1], ell_comps[0]), and its shear components are unchanged.
   - herculens behaves the same given that matrix. With its default +ps·I matrix it is the
     COOLEST frame instead.

## WCS scan

This is the VIS_BGSUB header of every local `dr1_sep1` cutout (15,034 tile folders, read on the
laptop rather than RAL: same files, headers only).

- **15,032 / 15,032 pass every check:** CD1_1 = -0.1"/px and CD2_2 = +0.1"/px exactly, with no
  rotation, no parity outlier and shape 100x100.
- **2 tile folders contain no cutout FITS** (`no_fits`), so the scan skipped them.

The conversion rules above therefore hold for every DR1 cutout. `conversions.py` stays general
(any CD matrix) for future samples.

## Open items for the human (phase-2 inputs)

- The pipeline's `coolest.json` files are mirrored for DR1. Options: (a) leave them byte-identical
  and add corrected COOLEST values as catalogue columns or new files (the default under the JSON
  rule); or (b) regenerate them. Option (b) changes JSON, so it needs your explicit go-ahead and the
  full snapshot/diff/CHANGELOG procedure.
- Add a WCS to the release model FITS, or document that they have none.
- The DR1 cutouts were regenerated after the local fits (the DATA comparison in point 1). Confirm
  this is expected (v5 cutouts) before the re-render uses the current cutouts as the backdrop.
- `scripts/tools/build_wcs_package.py` (the docstring that says "+y points along increasing FITS
  row") exists only on the unpushed `ral-path-v5-catalogue` branch, so it is not fixed here. The fix
  is to say +y points towards decreasing FITS row, i.e. south. The `positions_finder.py` docstring
  is fixed here.
- Phase 3: our COOLEST interop needs an explicit orientation input (or the CD2_2 sign). See the
  verdict in `conventions.md`.
