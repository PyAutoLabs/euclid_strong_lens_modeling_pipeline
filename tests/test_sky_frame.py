"""
``sky_frame.plus_y_north_from`` is the conversion every regenerated DR1 COOLEST
template goes through, so it is tested physically, not by parameter: the
mirrored tracer evaluated on the mirrored grid must give the mirrored
deflections and the same images, and its COOLEST export must carry the sky
position angle and the north-positive centre.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

al = pytest.importorskip("autolens")

from scripts.tools import sky_frame  # noqa: E402

DR1_CD = np.array([[-0.1 / 3600.0, 0.0], [0.0, 0.1 / 3600.0]])


def tracer_from():
    lens = al.Galaxy(
        redshift=0.5,
        bulge=al.lp.Sersic(centre=(0.05, -0.12), ell_comps=(0.15, -0.08), intensity=1.0,
                           effective_radius=0.6, sersic_index=3.0),
        mass=al.mp.PowerLaw(centre=(0.04, -0.1), ell_comps=(0.2, 0.1), einstein_radius=1.1, slope=2.1),
        multipole=al.mp.PowerLawMultipole(centre=(0.04, -0.1), einstein_radius=1.1, slope=2.1,
                                          m=4, multipole_comps=(0.03, -0.02)),
    )
    source = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(centre=(0.2, 0.15), ell_comps=(-0.1, 0.25), intensity=0.5,
                           effective_radius=0.2, sersic_index=1.0),
    )
    shear = al.mp.ExternalShear(gamma_1=0.04, gamma_2=-0.06)
    return al.Tracer(galaxies=[lens, source, al.Galaxy(redshift=0.5, shear=shear)])


def mirror_grid(grid):
    values = np.asarray(grid).copy()
    values[:, 0] *= -1.0
    return al.Grid2DIrregular(values=values)


def test__mirrored_tracer__deflections_and_images_are_the_mirror_image():
    tracer = tracer_from()
    mirrored = sky_frame.plus_y_north_from(tracer)
    grid = al.Grid2DIrregular(values=np.random.default_rng(1).uniform(-2, 2, size=(200, 2)))

    d = np.asarray(tracer.deflections_yx_2d_from(grid=grid))
    d_m = np.asarray(mirrored.deflections_yx_2d_from(grid=mirror_grid(grid)))
    np.testing.assert_allclose(d_m[:, 0], -d[:, 0], atol=1e-10)
    np.testing.assert_allclose(d_m[:, 1], d[:, 1], atol=1e-10)

    image = np.asarray(tracer.image_2d_from(grid=grid))
    image_m = np.asarray(mirrored.image_2d_from(grid=mirror_grid(grid)))
    np.testing.assert_allclose(image_m, image, rtol=1e-10)


def test__mirror_twice_is_identity_and_input_untouched():
    tracer = tracer_from()
    before = tracer.galaxies[0].mass.ell_comps
    twice = sky_frame.plus_y_north_from(sky_frame.plus_y_north_from(tracer))
    assert tracer.galaxies[0].mass.ell_comps == before
    assert twice.galaxies[0].mass.ell_comps == pytest.approx(before)
    assert twice.galaxies[0].mass.centre == pytest.approx(tracer.galaxies[0].mass.centre)
    assert twice.galaxies[2].shear.gamma_2 == pytest.approx(tracer.galaxies[2].shear.gamma_2)


def test__unknown_explicit_angle_raises():
    class Odd:
        def __init__(self):
            self.position_angle = 30.0

    with pytest.raises(ValueError, match="explicit angle"):
        sky_frame.plus_y_north_from(Odd())


def test__dr1_is_mirrored_and_rotated_wcs_raises():
    assert sky_frame.native_is_mirrored(DR1_CD)
    with pytest.raises(ValueError):
        sky_frame.native_is_mirrored(np.array([[-1e-5, 1e-6], [1e-6, 1e-5]]))


def test__coolest_export_of_mirrored_tracer_carries_the_sky_pa(tmp_path):
    pytest.importorskip("coolest")
    q_true, pa_true = 0.6, 30.0
    # Native DR1 frame: PA = 90 - phi, so phi = 60; centre 0.3" north, 0.2" east.
    phi = 90.0 - pa_true
    ell_comps = al.convert.ell_comps_from(axis_ratio=q_true, angle=phi)
    lens = al.Galaxy(redshift=0.5, mass=al.mp.Isothermal(centre=(-0.3, -0.2), ell_comps=ell_comps,
                                                          einstein_radius=1.0))
    tracer = al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])
    dataset = SimpleNamespace(shape_native=(100, 100), pixel_scales=(0.1, 0.1))

    path = al.interop.coolest.to_coolest(
        galaxies=sky_frame.plus_y_north_from(tracer), file_path=tmp_path / "c", dataset=dataset,
        on_unsupported="skip",
    )
    with open(path) as f:
        params = json.load(f)["lensing_entities"][0]["mass_model"][0]["parameters"]
    value = lambda k: params[k]["point_estimate"]["value"]  # noqa: E731
    assert value("phi") == pytest.approx(pa_true, abs=1e-6)
    assert value("q") == pytest.approx(q_true, abs=1e-6)
    assert value("center_y") == pytest.approx(0.3, abs=1e-9)  # north positive
    assert value("center_x") == pytest.approx(-0.2, abs=1e-9)  # COOLEST +x is west


def test__pa_east_of_north_dr1_rule():
    for phi in (0.0, 15.0, 60.0, 135.0):
        assert sky_frame.pa_east_of_north(phi, DR1_CD) == pytest.approx((90.0 - phi) % 180.0)
    east, north = sky_frame.east_north_from(0.3, -0.2, DR1_CD)
    assert (east, north) == pytest.approx((0.2, -0.3))
