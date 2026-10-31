"""Numeric check of orientation conventions: PyAutoLens (DR1-loaded), sky, COOLEST,
lenstronomy (re-implemented formulas, cited), herculens (installed)."""
import os, sys, tempfile
for p in ["organs/PyAutoNerves", "fit/PyAutoFit", "array/PyAutoArray",
          "galaxy/PyAutoGalaxy", "lens/PyAutoLens"]:
    sys.path.insert(0, os.path.join("/home/jammy/Code/PyAutoLabs", p))
os.chdir(tempfile.mkdtemp())
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import autoarray as aa
import autogalaxy as ag
from autogalaxy.interop.coolest import conventions as conv
from coolest.api.profiles import mass as cmass, light as clight
from coolest.api import util as cutil
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Util import param_util as hpu
from herculens.MassModel.Profiles.shear import Shear as HShear
from herculens.LightModel.Profiles.sersic import Sersic as HSersic

ok = True
def check(name, a, b, tol=1e-6):
    global ok
    a, b = np.atleast_1d(a).astype(float), np.atleast_1d(b).astype(float)
    good = np.allclose(a, b, atol=tol)
    ok &= good
    print(f"[{'OK ' if good else 'BAD'}] {name}: {a} vs {b}")

def mod180(a):  # normalise to (-90, 90]
    a = (a + 90.0) % 180.0 - 90.0
    return np.where(a <= -90.0, a + 180.0, a)

# ---------------- 1. DR1-like FITS: CD1_1=-0.1"/px, CD2_2=+0.1"/px ----------
n = 11; ps = 0.1
hdr = fits.Header()
hdr["CTYPE1"], hdr["CTYPE2"] = "RA---TAN", "DEC--TAN"
hdr["CRPIX1"] = hdr["CRPIX2"] = (n + 1) / 2  # 1-based centre pixel
hdr["CRVAL1"], hdr["CRVAL2"] = 150.0, 2.0
hdr["CD1_1"], hdr["CD2_2"] = -ps / 3600, ps / 3600
hdr["CD1_2"] = hdr["CD2_1"] = 0.0
w = WCS(hdr)
row, col = 2, 8   # numpy index of FITS data (row 0 = first FITS row, at the bottom)
ra, dec = w.wcs_pix2world([[col, row]], 0)[0]
east = (ra - 150.0) * np.cos(np.deg2rad(2.0)) * 3600
north = (dec - 2.0) * 3600
print(f"pixel row={row} col={col}: east={east:.4f}\" north={north:.4f}\"")

# PyAutoArray grid for the unflipped array (as Array2D.from_fits gives)
grid = aa.Grid2D.uniform(shape_native=(n, n), pixel_scales=ps).native
yA, xA = grid[row, col]
print(f"PyAutoArray (y,x) = ({yA:.4f}, {xA:.4f})")
check("PyAutoArray x = west = -east", xA, -east, 1e-3)
check("PyAutoArray y = south = -north", yA, -north, 1e-3)

# lenstronomy/herculens with the WCS as transform_pix2angle (arcsec), origin at centre
M = np.array([[hdr["CD1_1"], hdr["CD1_2"]], [hdr["CD2_1"], hdr["CD2_2"]]]) * 3600
c = (n - 1) / 2
ra0, dec0 = -(M @ np.array([c, c]))
pgL = PixelGrid(n, n, transform_pix2angle=M, ra_at_xy_0=ra0, dec_at_xy_0=dec0)
xL, yL = pgL.map_pix2coord(col, row)   # lenstronomy/herculens (x=col, y=row)
check("lenstronomy/herculens(WCS) x = east = -xA", xL, -xA, 1e-6)
check("lenstronomy/herculens(WCS) y = north = -yA", yL, -yA, 1e-6)
# herculens / COOLEST default (positive diagonal)
pgH = PixelGrid(n, n, transform_pix2angle=ps * np.eye(2), ra_at_xy_0=-c * ps, dec_at_xy_0=-c * ps)
xH, yH = pgH.map_pix2coord(col, row)
check("herculens-default / COOLEST x = west = xA", xH, xA)
check("herculens-default / COOLEST y = north = -yA", yH, -yA)
coords = cutil.get_coordinates_from_regular_grid((-n*ps/2, n*ps/2), (-n*ps/2, n*ps/2), n, n)
xC, yC = coords.pixel_to_radec(col, row)
check("COOLEST api x,y = (xA, -yA)", [xC, yC], [xA, -yA])

# ---------------- 2. Ellipse with known PA east of north ----------------
PA = 30.0; q = 0.6
# direction of major axis on sky: (east, north) = (sin PA, cos PA)
uE, uN = np.sin(np.deg2rad(PA)), np.cos(np.deg2rad(PA))
# PyAutoLens frame (x=west, y=south): (x, y) = (-uE, -uN)
phiA = np.rad2deg(np.arctan2(-uN, -uE))
phiA_pred = mod180(90.0 - PA)
check("phi_A = 90 - PA (mod 180)", mod180(phiA), phiA_pred)
ellA = ag.convert.ell_comps_from(axis_ratio=q, angle=phiA)
# check AG major axis direction numerically on its convergence
prof = ag.mp.Isothermal(centre=(0.0, 0.0), ell_comps=ellA, einstein_radius=1.0)
th = np.deg2rad(np.arange(0, 180, 0.25))
pts = aa.Grid2DIrregular(np.stack([np.sin(th), np.cos(th)], 1))   # (y, x) at r=1
kap = np.asarray(prof.convergence_2d_from(grid=pts))
check("AG major axis (max kappa direction) == phi_A", np.rad2deg(th[np.argmax(kap)]), mod180(phiA) % 180, 0.3)

# COOLEST: correct value = PA; our export gives phi_coolest_from(phi_A)
q_exp, phi_exp = conv.q_phi_from_ell_comps(ellA)
print(f"COOLEST correct phi = PA = {PA}; our export gives {phi_exp:.3f}")
check("export phi == -PA (sign flipped) for DR1 data", phi_exp, mod180(-PA))
# check COOLEST's own profile: major axis at PA in its (x=west, y=north) frame
cs = clight.Sersic()
thC = np.deg2rad(np.arange(0, 180, 0.25))
sb = cs.evaluate_surface_brightness(np.cos(thC), np.sin(thC), phi=PA, q=q, theta_eff=1.0, n=1.0)
dirC = np.rad2deg(thC[np.argmax(sb)])   # ccw from +x(west) in COOLEST frame
# convert to PA: unit (x=west, y=north) -> east = -x
PA_from_coolest = np.rad2deg(np.arctan2(-np.cos(np.deg2rad(dirC)), np.sin(np.deg2rad(dirC))))
check("COOLEST Sersic major axis at PA east of north", mod180(PA_from_coolest), mod180(PA), 0.3)

# lenstronomy (param_util.phi_q2_ellipticity, re-implemented) in WCS frame (x=east, y=north)
phiL = np.arctan2(uN, uE)
e1L = (1 - q) / (1 + q) * np.cos(2 * phiL); e2L = (1 - q) / (1 + q) * np.sin(2 * phiL)
check("lenstronomy(WCS) (e1,e2) = (ell_comps[1], ell_comps[0])", [e1L, e2L], [ellA[1], ellA[0]])
# herculens SIE in WCS frame: check major axis along (uE, uN)
xs, ys = np.cos(th), np.sin(th)
kH = np.asarray(HSersic().function(xs, ys, 1.0, 1.0, 1.0, e1L, e2L))
check("herculens Sersic(WCS) major axis at phi_L", np.rad2deg(th[np.argmax(kH)]), np.rad2deg(phiL) % 180, 0.3)
# herculens default frame (x=west, y=north) = COOLEST frame
phiHd = np.arctan2(uN, -uE)
e1Hd, e2Hd = hpu.phi_q2_ellipticity(phiHd, q)
check("herculens-default (e1,e2) = (ell_comps[1], -ell_comps[0])", [e1Hd, e2Hd], [ellA[1], -ellA[0]])

# ---------------- 3. Shear ----------------
g1A, g2A = 0.03, 0.07
sh = ag.mp.ExternalShear(gamma_1=g1A, gamma_2=g2A)
P = np.array([[0.3, 0.8], [-0.5, 0.2]])  # (y, x)
defl = np.asarray(sh.deflections_yx_2d_from(grid=aa.Grid2DIrregular(P)))
yP, xP = P[:, 0], P[:, 1]
check("AG shear alpha_x = g1 x + g2 y (lenstronomy formula in AG frame)", defl[:, 1], g1A * xP + g2A * yP)
check("AG shear alpha_y = g2 x - g1 y", defl[:, 0], g2A * xP - g1A * yP)
# lenstronomy/herculens (WCS frame) = 180 deg rotation -> (g1, g2) unchanged
fx, fy = HShear().derivatives(-xP, -yP, g1A, g2A)
check("herculens Shear(WCS): alpha_L = -alpha_A with same (g1,g2)", [fx, fy], [-defl[:, 1], -defl[:, 0]])
# herculens default / COOLEST frame (x=xA, y=-yA): (g1, -g2)
fx, fy = HShear().derivatives(xP, -yP, g1A, -g2A)
check("herculens Shear(default frame): (g1, -g2)", [fx, fy], [defl[:, 1], -defl[:, 0]])
# COOLEST ExternalShear: correct phi_ext = 90 - phi_gamma_A
mag, angA = ag.convert.shear_magnitude_and_angle_from(gamma_1=g1A, gamma_2=g2A)
phi_ext_correct = mod180(90.0 - angA)
fx, fy = cmass.ExternalShear().deflection(xP, -yP, gamma_ext=mag, phi_ext=phi_ext_correct)
check("COOLEST ExternalShear(phi_ext = 90 - phi_A) reproduces AG deflection", [fx, fy], [defl[:, 1], -defl[:, 0]])
phi_ext_export = conv.phi_coolest_from(float(angA))
fx2, fy2 = cmass.ExternalShear().deflection(xP, -yP, gamma_ext=mag, phi_ext=phi_ext_export)
print(f"export phi_ext={phi_ext_export:.3f} vs correct {phi_ext_correct:.3f}; "
      f"export deflection matches? {np.allclose([fx2, fy2], [defl[:, 1], -defl[:, 0]])}")
# COOLEST shear frame with +y north (export's assumption): export correct
fx3, fy3 = cmass.ExternalShear().deflection(xP, yP, gamma_ext=mag, phi_ext=phi_ext_export)
check("if +y_A IS north (x west): export phi_ext correct", [fx3, fy3], [defl[:, 1], defl[:, 0]])
# and the ellipse for +y-north data: export phi = phi_A - 90 == PA
phiA_n = mod180(PA + 90.0)  # direction (west=-sinPA, north=cosPA) ccw from +x(west)
check("if +y_A IS north: export ellipse phi == PA", conv.phi_coolest_from(phiA_n), mod180(PA))

# round trips
ell_rt = conv.ell_comps_from_q_phi(*conv.q_phi_from_ell_comps(ellA))
check("export round trip ell_comps", ell_rt, ellA)

print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
