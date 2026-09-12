"""
Known-answer tests for the ``files/wcs.json`` record ``util.wcs_dict_from``
builds and ``util.AnalysisImaging.save_results`` writes: the fitted lens light
centre on the sky, and the lensed source's multiple images — the lens equation
solved for the source light centre with ``al.PointSolver`` — in the image plane
and on the sky.

The fixture is the committed simulated dataset
``dataset/simulated/euclid_dr1_like/`` and its ``truth.json``. Two of its
records are independent known answers here:

* ``truth["positions"]`` — the multiple images ``scripts/simulator.py`` solved
  for the truth tracer with the same ``al.PointSolver`` settings
  (``util.LENSED_SOURCE_PIXEL_SCALE_PRECISION`` /
  ``util.LENSED_SOURCE_MAGNIFICATION_THRESHOLD``), so the images a fit records
  for that tracer must be the same points;
* the simulated cut-out's own TAN WCS (``scripts/simulator.py::
  image_wcs_header_from``: north-up, ``CD1_1 < 0``, ``CD2_2 > 0``, reference
  pixel at the frame centre), against which every sky value is checked by
  hand — the small-angle offsets from the header's ``CRVAL``, not a round trip
  through the code under test.

These tests are deliberately **JAX-free** (``use_jax=False`` everywhere, as
``AGENTS.md`` requires) and run no non-linear search; the one real-mode fit
that proves ``save_results`` actually *writes* the record is
``test_latent_run_level.py`` (``slow``).
"""

import inspect
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import util  # noqa: E402


SIMULATED_SAMPLE = "simulated"
SIMULATED_DATASET = "euclid_dr1_like"
SIMULATED_PATH = PROJECT_ROOT / "dataset" / SIMULATED_SAMPLE / SIMULATED_DATASET

# The image positions are solved to `LENSED_SOURCE_PIXEL_SCALE_PRECISION`
# arcsec; in practice they replay `truth.json` bit for bit.
POSITION_ABS = 2.0 * util.LENSED_SOURCE_PIXEL_SCALE_PRECISION

# The small-angle offsets from CRVAL hold to second order in the angle: the
# gnomonic projection's cross term at 1.5" and declination -51 is ~1.5e-9 deg
# (5 micro-arcsec), so this tolerance is a few times that and still a hundred
# times below the 1.4e-6 deg of a 0.005" solver step.
SKY_ABS_DEG = 1e-8

# The keys `wcs_dict_from` always writes, in the order it writes them.
WCS_KEYS = (
    "crpix_x",
    "crpix_y",
    "crval_ra_deg",
    "crval_dec_deg",
    "source_centre_y_arcsec",
    "source_centre_x_arcsec",
    "lensed_source_image_y_arcsec",
    "lensed_source_image_x_arcsec",
    "lensed_source_image_ra_deg",
    "lensed_source_image_dec_deg",
)
SOURCE_KEYS = WCS_KEYS[4:]
IMAGE_KEYS = WCS_KEYS[6:]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def push_config():
    from autolens import conf

    conf.instance.push(
        new_path=PROJECT_ROOT / "config", output_path=PROJECT_ROOT / "output"
    )


@pytest.fixture(scope="session")
def truth():
    with open(SIMULATED_PATH / "truth.json") as f:
        return json.load(f)


def _profile_from(entry):
    """
    Rebuild one profile from its ``truth.json`` record, filtering the recorded
    parameters against the constructor signature (``truth.json`` also records
    derived parameters such as ``Isothermal.slope``).
    """
    import autolens as al

    cls = {
        "Sersic": al.lp.Sersic,
        "Isothermal": al.mp.Isothermal,
        "ExternalShear": al.mp.ExternalShear,
    }[entry["type"]]

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}

    return cls(
        **{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in entry["parameters"].items()
            if key in accepted
        }
    )


@pytest.fixture(scope="session")
def truth_galaxies(truth):
    """The truth galaxies in ``truth.json`` order: the lens first, the source last."""
    import autolens as al

    return [
        al.Galaxy(
            redshift=galaxy["redshift"],
            **{
                profile_name: _profile_from(entry)
                for profile_name, entry in galaxy["profiles"].items()
            },
        )
        for galaxy in truth["model"].values()
    ]


@pytest.fixture(scope="session")
def euclid_dataset():
    return util.load_vis_dataset(SIMULATED_DATASET, sample_name=SIMULATED_SAMPLE)


@pytest.fixture(scope="session")
def wcs_dict(truth_galaxies, euclid_dataset):
    """The record for the truth tracer — solved once for the module."""
    import autolens as al

    return util.wcs_dict_from(
        tracer=al.Tracer(galaxies=truth_galaxies),
        data=euclid_dataset.dataset.data,
        pixel_wcs=euclid_dataset.pixel_wcs,
    )


def _header_crval(euclid_dataset):
    return (
        float(euclid_dataset.header["CRVAL1"]),
        float(euclid_dataset.header["CRVAL2"]),
    )


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


def test_the_record_has_one_schema(wcs_dict):
    assert tuple(wcs_dict) == WCS_KEYS


def test_the_record_is_json_serialisable(wcs_dict):
    """
    ``save_results`` hands the dict to ``output_to_json``; a NumPy scalar or
    array left inside would raise there, after the search has finished.
    """
    round_trip = json.loads(json.dumps(wcs_dict))

    assert round_trip == wcs_dict


# ---------------------------------------------------------------------------
# The lens light centre (the keys the DR1 chain already read)
# ---------------------------------------------------------------------------


def test_crval_is_the_lens_light_centre_on_the_sky(wcs_dict, truth, euclid_dataset):
    """
    The truth lens light sits at the image-plane origin, which the simulator
    made the WCS reference pixel — so the lens centre on the sky *is* the
    header's ``CRVAL``, and the origin's WCS pixel is its ``CRPIX``.
    """
    lens = next(iter(truth["model"].values()))
    assert lens["profiles"]["bulge"]["parameters"]["centre"] == [0.0, 0.0]

    crval_ra_deg, crval_dec_deg = _header_crval(euclid_dataset)

    assert wcs_dict["crval_ra_deg"] == pytest.approx(crval_ra_deg, abs=SKY_ABS_DEG)
    assert wcs_dict["crval_dec_deg"] == pytest.approx(crval_dec_deg, abs=SKY_ABS_DEG)
    assert wcs_dict["crpix_x"] == pytest.approx(float(euclid_dataset.header["CRPIX1"]))
    assert wcs_dict["crpix_y"] == pytest.approx(float(euclid_dataset.header["CRPIX2"]))


# ---------------------------------------------------------------------------
# The lensed source
# ---------------------------------------------------------------------------


def test_the_source_centre_is_the_source_light_centre(wcs_dict, truth):
    source = list(truth["model"].values())[-1]
    centre_y, centre_x = source["profiles"]["bulge"]["parameters"]["centre"]

    assert wcs_dict["source_centre_y_arcsec"] == pytest.approx(centre_y)
    assert wcs_dict["source_centre_x_arcsec"] == pytest.approx(centre_x)


def test_the_lensed_source_images_are_the_simulators_positions(wcs_dict, truth):
    """
    ``truth["positions"]`` is what ``scripts/simulator.py`` solved for this
    tracer with the same solver settings, and what every fit of the dataset
    reads back as its positions constraint — the record must name the same
    points, as a set (the solver's output order is not part of its contract).
    """
    recorded = sorted(
        zip(
            wcs_dict["lensed_source_image_y_arcsec"],
            wcs_dict["lensed_source_image_x_arcsec"],
        )
    )
    expected = sorted((y, x) for y, x in truth["positions"])

    assert len(recorded) == len(expected) == 4, (
        "the simulated source is quadruply imaged; "
        f"recorded {len(recorded)} images against {len(expected)} in truth.json"
    )
    assert np.asarray(recorded) == pytest.approx(np.asarray(expected), abs=POSITION_ABS)


def test_the_lensed_source_sky_positions_follow_the_header(wcs_dict, euclid_dataset):
    """
    Checked by hand against the simulated header rather than through the code
    under test. ``scripts/simulator.py`` writes a north-up TAN WCS with the
    reference pixel at the frame centre, and PyAutoArray loads the FITS
    without a row flip, so PyAutoLens's positive ``y`` is the FITS row-1
    direction — south — and positive ``x`` is the FITS column direction, west
    under ``CD1_1 < 0``. Hence, to first order in the angle::

        dec = CRVAL2 - y / 3600
        (ra - CRVAL1) * cos(dec) = -x / 3600

    with the lens at the reference pixel. The sign of ``y`` is the point of the
    test: a record that put the images north of where the light is would pass
    every round-trip check and mislead every cross-match.
    """
    header = euclid_dataset.header
    assert float(header["CD1_1"]) < 0.0 and float(header["CD2_2"]) > 0.0
    assert float(header["CD1_2"]) == 0.0 and float(header["CD2_1"]) == 0.0

    crval_ra_deg, crval_dec_deg = _header_crval(euclid_dataset)

    for y, x, ra, dec in zip(*(wcs_dict[key] for key in IMAGE_KEYS)):
        assert dec == pytest.approx(crval_dec_deg - y / 3600.0, abs=SKY_ABS_DEG)
        assert (ra - crval_ra_deg) * np.cos(np.radians(dec)) == pytest.approx(
            -x / 3600.0, abs=SKY_ABS_DEG
        )


def test_the_source_centre_reads_an_mge_basis(truth_galaxies):
    """
    ``vis_lp``'s source is one MGE ``Basis``, whose ``centre`` is the centre
    its Gaussians share — the centre the lens equation is solved for.
    """
    import autolens as al

    basis = al.lp_basis.Basis(
        profile_list=[
            al.lp.Gaussian(centre=(0.08, 0.12), sigma=sigma) for sigma in (0.1, 0.2)
        ]
    )
    tracer = al.Tracer(
        galaxies=[truth_galaxies[0], al.Galaxy(redshift=1.0, bulge=basis)]
    )

    assert util.source_centre_from(tracer=tracer) == (0.08, 0.12)


def test_a_source_without_a_light_centre_writes_null(truth_galaxies, euclid_dataset):
    """
    A pixelized source (``vis_pix``, the Delaunay stages) has no light profile,
    so there is no centre to solve for: the six source keys are ``null`` and
    the four lens keys are written exactly as before.
    """
    import autolens as al

    tracer = al.Tracer(galaxies=[truth_galaxies[0], al.Galaxy(redshift=1.0)])

    wcs_dict = util.wcs_dict_from(
        tracer=tracer,
        data=euclid_dataset.dataset.data,
        pixel_wcs=euclid_dataset.pixel_wcs,
    )

    assert tuple(wcs_dict) == WCS_KEYS
    assert all(wcs_dict[key] is None for key in SOURCE_KEYS)
    assert wcs_dict["crval_ra_deg"] == pytest.approx(
        _header_crval(euclid_dataset)[0], abs=SKY_ABS_DEG
    )


def test_a_solver_failure_is_logged_and_written_as_null(
    truth_galaxies, euclid_dataset, monkeypatch, caplog
):
    """
    ``save_results`` runs after the search has finished; a raise there would
    lose the fit to its own record. The four image keys are ``null``, the
    source centre and the lens keys are still written, and the failure is on
    the log.
    """
    import autolens as al

    def raise_(**kwargs):
        raise RuntimeError("no triangles")

    monkeypatch.setattr(util, "lensed_source_image_positions_from", raise_)

    with caplog.at_level(logging.WARNING, logger="util"):
        wcs_dict = util.wcs_dict_from(
            tracer=al.Tracer(galaxies=truth_galaxies),
            data=euclid_dataset.dataset.data,
            pixel_wcs=euclid_dataset.pixel_wcs,
        )

    assert tuple(wcs_dict) == WCS_KEYS
    assert all(wcs_dict[key] is None for key in IMAGE_KEYS)
    assert wcs_dict["source_centre_y_arcsec"] == pytest.approx(0.08)
    assert "no triangles" in caplog.text
