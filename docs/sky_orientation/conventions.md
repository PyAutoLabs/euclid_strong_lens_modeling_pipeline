# Orientation conventions: PyAutoLens vs COOLEST vs lenstronomy vs herculens (Euclid DR1)

Sources read: PyAutoLabs checkouts; installed `herculens` 0.3.0 and `coolest` 0.1.11
(`/home/jammy/venv/PyAuto/lib/python3.12/site-packages/`); lenstronomy 1.14.3 sdist downloaded and
read from the lenstronomy 1.14.3 sdist (read only, never imported). Lenstronomy
paths below are relative to `lenstronomy-1.14.3/lenstronomy/`. COOLEST docs:
https://coolest.readthedocs.io/en/latest/conventions.html .

Numeric check: `scripts/research/sky_orientation/conventions_check.py`
(run with `/home/jammy/venv/PyAuto/bin/python -I`; all checks pass, "ALL OK").

DR1 cutout: CD1_1 = -0.1"/px, CD2_2 = +0.1"/px, no rotation. So FITS column index increasing = RA
decreasing = **west**, FITS row index increasing = Dec increasing = **north**; numpy row 0 = first FITS
row = southern edge.

---

## 1. Per-code conventions

### PyAutoLens / PyAutoGalaxy / PyAutoArray
* **FITS load, no flip**: `organs/PyAutoNerves/autonerves/fitsable.py:294` (`values = hdu.data.astype("float")`),
  called from `array/PyAutoArray/autoarray/structures/arrays/uniform_2d.py:898`.
* **Pixel -> arcsec**: `array/PyAutoArray/autoarray/structures/grids/grid_2d_util.py:279-291`
  (`(indices - centres) * [-1, +1] * pixel_scales`) with centre `((ny-1)/2, (nx-1)/2)` from
  `geometry/geometry_util.py:268-290, 319-325`. Hence `y = ((ny-1)/2 - row) ps`, `x = (col - (nx-1)/2) ps`
  (matches the brief). Row 0 is the top (+y max).
* **DR1 consequence**: +x = increasing column = **west**; +y = decreasing row = **south**. Verified
  numerically against astropy WCS (check 1). The native frame (x=W, y=S) is a y-mirror of the sky
  (as displayed north-up/east-left, frame (W, N)).
* **Angle**: `angle` = major-axis direction, degrees counter-clockwise from +x in the (x, y) frame
  (`galaxy/PyAutoGalaxy/autogalaxy/profiles/geometry_profiles.py:293-298`, `convert.py:53-76`). Verified:
  Isothermal convergence peaks along `angle` (check 2).
* **ell_comps** (`convert.py:60-61, 74-76`): **ell_comps[0] = f sin 2phi, ell_comps[1] = f cos 2phi**,
  f = (1-q)/(1+q). Note the order: index 0 is the *sine* term (opposite to lenstronomy's e1).
* **Shear** (`convert.py:193-218`; `profiles/mass/sheets/external_shear.py:54-89, 138-166`):
  gamma_1 = g cos 2phi_g, gamma_2 = g sin 2phi_g, phi_g ccw from +x = stretch direction. Deflection
  numerically equals lenstronomy's formula in the same frame: alpha_x = g1 x + g2 y, alpha_y = g2 x - g1 y
  (check 3).
* **Centre**: `(y, x)` arcsec in the frame above.

### COOLEST (standard + `coolest` 0.1.11 API)
* Docs (conventions page): "x axis positively increasing to the right, y axis positively increasing
  upwards"; position angles "measured counter-clockwise from the positive y axis ... similar to the
  East-of-North convention", range (-90, 90] for 180-degree-symmetric profiles (ellipses, shear);
  q = b/a; radii along the intermediate axis sqrt(ab). Pixel origin and RA/Dec mapping are *not*
  stated on the page (flagged).
* Code fixes the sky mapping: `template/classes/observation.py:69` (commented) "RA is opposite to x"
  -> **+x = west, +y = north**; `api/util.py:49-72` pixel (0,0) = bottom-left, matrix = +ps * I (x grows
  with column, y with row); `api/plotting.py:22` `origin='lower'`; `template/classes/grid.py:73-80`
  field of view "bottom side of the lower pixel". So a DR1 FITS array used as-is is already in the
  COOLEST frame (column = west, row = north).
* **PA**: `api/profiles/util.py:12-13` `eastofnorth2normalradians(phi) = phi - 90` (ccw-from-+x angle,
  equivalent mod 180 to phi + 90). With x=W, y=N, ccw from +y goes toward -x = east, so phi =
  **PA east of north**. Verified: COOLEST Sersic peaks at PA on sky (check 2).
* **Ellipticity**: stored as (q, phi) only; no e1/e2.
* **Shear**: (gamma_ext, phi_ext), `api/profiles/mass.py:170-192`: g1 = g cos 2(phi_ext-90),
  g2 = g sin 2(phi_ext-90) in (x=W, y=N); potential `g/2 r^2 cos 2(theta - phi_ext_)` -> phi_ext is the
  stretch direction, east of north.
* **Centre**: `center_x, center_y` in (W, N) arcsec.

### lenstronomy 1.14.3 (read, not run)
* **Pixel -> angle**: `Util/util.py:333-360` `grid_from_coordinate_transform`:
  `ra = M00*i + M01*j + ra_at_xy_0`, `dec = M10*i + M11*j + dec_at_xy_0` with i = column, j = row of the
  numpy image (no flip; `util.py:117-134` array2image is a plain reshape). `Data/coord_transforms.py:14-30`
  documents `transform_pix2angle` as pixel->(ra, dec) and `ra_at_xy_0` as "ra coordinate at pixel (0,0)".
  Default grid `util.py:~272-330` `make_grid_with_coordtransform(inverse=True)` -> matrix
  `[[-ps, 0], [0, ps]]`, docstring "if true sets East as left". So x = relative RA, **+x = east, +y = north**.
* **DR1 WCS**: transform_pix2angle = 3600 * [[CD1_1, CD1_2], [CD2_1, CD2_2]] = [[-0.1, 0], [0, 0.1]]
  (identical to the lenstronomy default). Lenstronomy has no CD-matrix helper in source (no `CD1_1`
  string found); this mapping is the standard use and was checked numerically against astropy (check 1).
* **PA**: phi ccw from +x (= from east toward north). `Util/param_util.py:77-86` phi_q2_ellipticity:
  **e1 = f cos 2phi, e2 = f sin 2phi**; inverse `param_util.py:92-99` phi = atan2(e2, e1)/2. Major axis
  along phi (`param_util.py:183-202` rotation x_ = cos x + sin y with sqrt(1-e) on x_).
* **Shear**: `param_util.py:51-59, 64-70` g1 = g cos 2phi, g2 = g sin 2phi; `LensModel/Profiles/shear.py:34-48`
  alpha_x = g1 x + g2 y, alpha_y = g2 x - g1 y.
* **Centre**: `center_x, center_y` = (RA offset, Dec offset) = (east, north).

### herculens 0.3.0
* Same machinery as lenstronomy: `Coordinates/pixel_grid.py:27-45`, `Coordinates/coord_transforms.py:68-88`,
  `Util/util.py:138-155`; e1/e2 `Util/param_util.py:15-27` (e1 = f cos 2phi, e2 = f sin 2phi), shear
  `param_util.py:79-98`, `MassModel/Profiles/shear.py:40-55` (same deflection as lenstronomy).
* **Difference**: herculens' *default* `transform_pix2angle` is `+ps * I` (`pixel_grid.py:37-43`) — i.e.
  the COOLEST frame (+x = west, +y = north) on a FITS-ordered array. If you pass the DR1 WCS matrix it is
  identical to lenstronomy. Two columns are given in the table.
* Note: herculens SIE/NIE `hessian` crashes with the installed JAX (`nie.py:249` `jnp.clip(a_max=...)`
  removed); the check used herculens Sersic for the ellipse direction instead.

---

## 2. Frame relations (DR1-loaded PyAutoLens native frame x_A = W, y_A = S)

| frame | +x | +y | relation to (x_A, y_A) | type |
|---|---|---|---|---|
| sky offsets | east | north | (E, N) = (-x_A, -y_A) | — |
| COOLEST / herculens default | west | north | (x_A, -y_A) | mirror |
| lenstronomy / herculens with DR1 WCS | east | north | (-x_A, -y_A) | 180° rotation |

Derivations: a direction at angle phi_A (ccw from +x_A) has components (W, S) = (cos phi_A, sin phi_A),
so (E, N) = (-cos phi_A, -sin phi_A). PA (east of north) satisfies (E, N) = (sin PA, cos PA), giving
sin PA = -cos phi_A, cos PA = -sin phi_A -> PA = -90 - phi_A ≡ **90 - phi_A (mod 180)**.
* COOLEST mirror: angle ccw from +x_C is -phi_A; COOLEST phi (ccw from +y_C) = -phi_A - 90 ≡ 90 - phi_A = PA.
* lenstronomy 180° rotation: phi_L = phi_A + 180 ≡ phi_A (mod 180) -> spin-2 components unchanged.
* herculens default mirror: phi_H = -phi_A -> sin-2phi terms flip sign.

## 3. Conversion table (DR1-loaded PyAutoLens quantity -> other codes)

Input: centre (y_A, x_A); angle phi_A = 0.5 atan2(ell_comps[0], ell_comps[1]); q; f = (1-q)/(1+q);
shear (g1_A, g2_A), g = sqrt(g1²+g2²), phi_gA = 0.5 atan2(g2_A, g1_A).

| quantity | sky | COOLEST | lenstronomy (DR1 WCS) = herculens (DR1 WCS) | herculens (default +ps·I) |
|---|---|---|---|---|
| centre | E = -x_A, N = -y_A | center_x = x_A, **center_y = -y_A** | center_x = -x_A, center_y = -y_A | center_x = x_A, center_y = -y_A |
| ellipse angle | PA = 90 - phi_A (mod 180) | phi = 90 - phi_A, in (-90, 90] | phi_L = phi_A (rad) | phi_H = -phi_A |
| q | q | q (radii -> sqrt(q)·a, unchanged) | q | q |
| e1, e2 | — | (q, phi only) | **e1 = ell_comps[1], e2 = ell_comps[0]** | e1 = ell_comps[1], e2 = -ell_comps[0] |
| shear | PA_shear = 90 - phi_gA (stretch dir.) | gamma_ext = g, **phi_ext = 90 - phi_gA** | gamma1 = g1_A, gamma2 = g2_A | gamma1 = g1_A, gamma2 = -g2_A |

Inverse (COOLEST -> PyAutoLens DR1 frame): phi_A = 90 - phi, centre (y_A, x_A) = (-center_y, center_x),
phi_gA = 90 - phi_ext. (The map phi -> 90 - phi is its own inverse.)

Worked numbers (check script): PA = 30°, q = 0.6 -> phi_A = 60°, ell_comps = (0.2165, -0.125);
COOLEST phi = 30; lenstronomy (e1, e2) = (-0.125, 0.2165); herculens default (-0.125, -0.2165).

## 4. Verdict on our COOLEST export

`galaxy/PyAutoGalaxy/autogalaxy/interop/coolest/conventions.py` assumes the PyAutoLens frame has +x right
= west and +y up = north (docstring `conventions.py:84-85`), i.e. it maps `phi = angle - 90`
(`:26`) and `center = (x, y)` (`:91`). Both are used for ellipses (`q_phi_from_ell_comps`, `:48-59`) and
shear (`interop/coolest/mass.py:277-295`).

* **For DR1-loaded data (+y south, +x west): WRONG.** The PA comes out mirrored
  (export gives phi = phi_A - 90 = -PA; correct is +PA) and center_y has the wrong sign. The external
  shear phi_ext is likewise mirrored (check: export phi_ext = -56.6° vs correct +56.6°; COOLEST's own
  ExternalShear deflection with the exported value does not reproduce the PyAutoLens deflection, with the
  corrected value it does). q, theta_E, radii, gamma_ext, and the observation grid size are unaffected.
  **Correction**: `phi = 90 - angle` (normalise to (-90, 90]), `center = (x, -y)`, `phi_ext = 90 - angle_shear`;
  import inverse `angle = 90 - phi`, `centre = (-center_y, center_x)`. Equivalently, flip the frame once
  (y -> -y, ell_comps[0] -> -ell_comps[0], gamma_2 -> -gamma_2) and then apply the existing functions.
  Any pixel data written into COOLEST FITS should stay in FITS order (row 0 at the bottom), which is what
  DR1 FITS already are.
* **For data whose +y IS north (and +x west, e.g. the array flipped with `np.flipud` on load, or a
  north-up/east-left simulation): CORRECT.** Then (x_A, y_A) = (x_C, y_C), and phi = angle - 90 ≡
  angle + 90 mod 180 = PA (check: export ellipse phi = 30 = PA; export phi_ext reproduces the deflection).
  The round trip `ell_comps -> (q, phi) -> ell_comps` is exact in both cases (the bug is a frame choice,
  not an inconsistency).
* Suggestion: the frame is a property of the data, not the profile, so the export should take an explicit
  orientation flag (e.g. `y_points_north: bool`, or derived from the dataset's WCS CD2_2 sign) rather than
  a fixed rule. If the sky-orientation work makes PyAutoLens flip DR1 on load, the current export becomes
  correct without change.

## 5. Uncertainties

* COOLEST's docs page does not explicitly say x = west; that rests on "similar to East-of-North" plus the
  code comment `observation.py:69` ("RA is opposite to x") and the API using +ps on both axes with
  origin='lower'. The numeric check of COOLEST's own Sersic and ExternalShear profiles is consistent with
  it.
* lenstronomy source has no WCS/CD helper; "transform_pix2angle = 3600·CD" is standard practice, not a
  lenstronomy function. Lenstronomy itself was not executed; its formulas were re-implemented from the
  cited lines (herculens, which copies them, was executed).
* The checks assume a cutout with no rotation (CD1_2 = CD2_1 = 0) and small fields (cos Dec applied for
  east offsets). For rotated WCS the lenstronomy matrix carries the rotation; the PyAutoLens/COOLEST
  frames would need an extra rotation by the WCS position angle.
* The PyAutoLens checkout imported is the local source tree (not a released wheel).
