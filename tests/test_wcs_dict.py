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

The pixelized leg builds the ``vis_pix`` stage as a zero-free-parameter model
on the same dataset (the ``Delaunay`` mesh mirrored from
``test_compute_latent_variable.py``'s ``pixelized_source_model`` fixture, at the
same quarter size) and fits it once, so the clumps read off the mapper have the
truth source and the truth images to be checked against.

The on-disk shape is asserted through ``al.output_to_json`` + ``al.from_json``,
the pair ``save_results`` and the aggregator actually use — never ``json.dumps``:
PyAutoFit's envelope drops ``None``-valued keys, which is why the record's
contract is *absent when unavailable* and every test here asserts absence, not
``None``.

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
from pixelized_model import pixelized_model_and_adapt_images_from  # noqa: E402


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
LENS_KEYS = ("crpix_x", "crpix_y", "crval_ra_deg", "crval_dec_deg", "source_model")

# The keys present once the source was located: the solved position and its
# images (the `al.PointSolver` route).
CENTRE_KEYS = ("source_centre_y_arcsec", "source_centre_x_arcsec")
IMAGE_KEYS = (
    "lensed_source_image_y_arcsec",
    "lensed_source_image_x_arcsec",
    "lensed_source_image_ra_deg",
    "lensed_source_image_dec_deg",
)
SOURCE_KEYS = CENTRE_KEYS + IMAGE_KEYS

# The keys of one `source_clumps` entry (the mapper route), in written order.
CLUMP_KEYS = (
    "peak_y_arcsec",
    "peak_x_arcsec",
    "peak_value",
    "mesh_pixels",
    "image_y_arcsec",
    "image_x_arcsec",
    "image_ra_deg",
    "image_dec_deg",
)

# The mapper route reports the brightest *data pixel* of each image region of
# the source's model image, so it is quantised to the 0.1" pixel grid and reads
# the peak of an extended (PSF-convolved, mesh-smoothed) image rather than the
# point-source position: measured 0.10-0.13" from `truth["positions"]`.
MAPPER_IMAGE_ABS = 0.2

# The solver route on the clump's peak mesh pixel, which sits a few
# milli-arcsec from the truth source centre: measured within 0.035" of
# `truth["positions"]`.
SOLVER_FROM_PEAK_ABS = 0.1

# The clump peak is the centre of a mesh pixel, so it is the truth source centre
# to within the mesh spacing there: measured 0.004".
PEAK_ABS = 0.05



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


def _analysis_from(euclid_dataset, adapt_images=None):
    """The pipeline analysis as the fits build it, NumPy, no RGB plot."""
    return util.AnalysisImaging(
        dataset=euclid_dataset.dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=None,
        use_jax=False,
        dataset_main_path=euclid_dataset.dataset_main_path,
        title_prefix="VIS",
        plot_rgb=False,
        skip_rgb_plot=True,
        psf_lowest_resolution=euclid_dataset.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=euclid_dataset.psf_lowest_resolution_fwhm,
        pixel_wcs=euclid_dataset.pixel_wcs,
        magzero=euclid_dataset.magzero,
    )


@pytest.fixture(scope="session")
def pixelized_fit(truth_galaxies, euclid_dataset):
    """
    The ``vis_pix`` stage fitted once at zero free parameters: the truth lens
    as an instance, the source a ``Delaunay`` ``Pixelization`` (the builder in
    ``tests/pixelized_model.py``, which says why the mesh travels in
    ``AdaptImages``).
    """
    model, adapt_images = pixelized_model_and_adapt_images_from(
        lens=truth_galaxies[0],
        sersic_source=truth_galaxies[-1],
        euclid_dataset=euclid_dataset,
    )

    analysis = _analysis_from(euclid_dataset, adapt_images=adapt_images)

    return analysis.fit_from(instance=model.instance_from_vector([]))


@pytest.fixture(scope="session")
def pixelized_wcs_dict(pixelized_fit, euclid_dataset):
    """The record for the pixelized fit — clumps read and solved once for the module."""
    return util.wcs_dict_from(
        tracer=pixelized_fit.tracer,
        data=euclid_dataset.dataset.data,
        pixel_wcs=euclid_dataset.pixel_wcs,
        fit=pixelized_fit,
    )


def _on_disk(wcs_dict, tmp_path):
    """
    The record as it comes back off disk: written by ``al.output_to_json`` and
    read by ``al.from_json``, the pair ``save_results`` and the aggregator use.
    """
    import autolens as al

    path = tmp_path / "wcs.json"
    al.output_to_json(obj=wcs_dict, file_path=path)
    return al.from_json(file_path=path)


def _sorted_positions(ys, xs):
    return sorted(zip(ys, xs))


def _header_crval(euclid_dataset):
    return (
        float(euclid_dataset.header["CRVAL1"]),
        float(euclid_dataset.header["CRVAL2"]),
    )


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


def test_a_light_profile_source_writes_the_solver_keys(wcs_dict):
    assert tuple(wcs_dict) == LENS_KEYS + SOURCE_KEYS
    assert wcs_dict["source_model"] == "light_profile"
    assert "source_clumps" not in wcs_dict


def test_the_record_survives_the_json_envelope(wcs_dict, pixelized_wcs_dict, tmp_path):
    """
    ``save_results`` writes with ``output_to_json`` and the aggregator reads
    with ``from_json``. PyAutoFit's envelope drops ``None`` values and would
    raise on a NumPy scalar, so the record must contain neither: both records
    must come back off disk equal to what was written, nested clump list
    included.
    """
    assert _on_disk(wcs_dict, tmp_path / "lp") == wcs_dict
    assert _on_disk(pixelized_wcs_dict, tmp_path / "pix") == pixelized_wcs_dict


def test_nothing_is_written_as_null(wcs_dict, pixelized_wcs_dict):
    """A ``None`` would silently vanish on disk; the contract is an absent key."""

    def has_none(value):
        if value is None:
            return True
        if isinstance(value, dict):
            return any(has_none(v) for v in value.values())
        if isinstance(value, list):
            return any(has_none(v) for v in value)
        return False

    assert not has_none(wcs_dict)
    assert not has_none(pixelized_wcs_dict)


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


def test_a_source_with_neither_light_nor_fit_writes_no_source_keys(
    truth_galaxies, euclid_dataset
):
    """
    A bare source galaxy has no light centre, and without a ``fit`` a pixelized
    one has no reconstruction to read: ``source_model`` says ``"none"`` and
    no source key is written — never ``null``.
    """
    import autolens as al

    tracer = al.Tracer(galaxies=[truth_galaxies[0], al.Galaxy(redshift=1.0)])

    wcs_dict = util.wcs_dict_from(
        tracer=tracer,
        data=euclid_dataset.dataset.data,
        pixel_wcs=euclid_dataset.pixel_wcs,
    )

    assert tuple(wcs_dict) == LENS_KEYS
    assert wcs_dict["source_model"] == "none"
    assert wcs_dict["crval_ra_deg"] == pytest.approx(
        _header_crval(euclid_dataset)[0], abs=SKY_ABS_DEG
    )


def test_a_pixelized_source_without_a_fit_writes_no_source_keys(
    pixelized_fit, euclid_dataset
):
    wcs_dict = util.wcs_dict_from(
        tracer=pixelized_fit.tracer,
        data=euclid_dataset.dataset.data,
        pixel_wcs=euclid_dataset.pixel_wcs,
        fit=None,
    )

    assert tuple(wcs_dict) == LENS_KEYS
    assert wcs_dict["source_model"] == "none"


def test_a_solver_failure_is_logged_and_its_keys_left_absent(
    truth_galaxies, euclid_dataset, monkeypatch, caplog
):
    """
    ``save_results`` runs after the search has finished; a raise there would
    lose the fit to its own record. The four image keys are absent, the
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

    assert tuple(wcs_dict) == LENS_KEYS + CENTRE_KEYS
    assert wcs_dict["source_centre_y_arcsec"] == pytest.approx(0.08)
    assert "no triangles" in caplog.text


# ---------------------------------------------------------------------------
# The pixelized source (vis_pix): clumps read off the mapper
# ---------------------------------------------------------------------------


def test_a_pixelized_source_writes_its_clumps_and_the_solver_keys(pixelized_wcs_dict):
    assert tuple(pixelized_wcs_dict) == LENS_KEYS + ("source_clumps",) + SOURCE_KEYS
    assert pixelized_wcs_dict["source_model"] == "pixelized"

    clumps = pixelized_wcs_dict["source_clumps"]

    assert len(clumps) == 1, (
        "the simulated source is one smooth Sersic, which the default threshold "
        f"isolates as one clump; got {len(clumps)}"
    )
    assert tuple(clumps[0]) == CLUMP_KEYS
    assert clumps[0]["mesh_pixels"] >= util.SOURCE_CLUMP_MIN_PIXELS
    assert clumps[0]["peak_value"] > 0.0


def test_the_clump_peak_is_the_truth_source_centre(pixelized_wcs_dict, truth):
    """
    The clump's peak mesh pixel sits on the truth source centre to within the
    mesh spacing there, and it is the position the solver keys are solved for.
    """
    source = list(truth["model"].values())[-1]
    centre_y, centre_x = source["profiles"]["bulge"]["parameters"]["centre"]

    clump = pixelized_wcs_dict["source_clumps"][0]

    assert clump["peak_y_arcsec"] == pytest.approx(centre_y, abs=PEAK_ABS)
    assert clump["peak_x_arcsec"] == pytest.approx(centre_x, abs=PEAK_ABS)
    assert pixelized_wcs_dict["source_centre_y_arcsec"] == clump["peak_y_arcsec"]
    assert pixelized_wcs_dict["source_centre_x_arcsec"] == clump["peak_x_arcsec"]


def test_the_clump_images_off_the_mapper_are_the_truth_images(pixelized_wcs_dict, truth):
    """
    Each of the four image regions the mapper attributes to the clump has its
    brightest model pixel within two pixels of a distinct truth image.
    """
    clump = pixelized_wcs_dict["source_clumps"][0]
    recorded = _sorted_positions(clump["image_y_arcsec"], clump["image_x_arcsec"])
    expected = sorted((y, x) for y, x in truth["positions"])

    assert len(recorded) == 4, (
        "the simulated source is quadruply imaged and the mapper must find "
        f"four image regions; got {len(recorded)}"
    )
    assert np.asarray(recorded) == pytest.approx(np.asarray(expected), abs=MAPPER_IMAGE_ABS)
    assert len(clump["image_ra_deg"]) == len(clump["image_dec_deg"]) == 4


def test_the_solver_keys_for_a_pixelized_source_are_the_truth_images(
    pixelized_wcs_dict, truth
):
    """
    The solver route on the clump peak lands on the truth images to a few
    hundredths of an arcsecond — closer than the mapper route, which is
    quantised to data pixels — so the two routes are checked separately, at
    their own tolerances.
    """
    recorded = _sorted_positions(
        pixelized_wcs_dict["lensed_source_image_y_arcsec"],
        pixelized_wcs_dict["lensed_source_image_x_arcsec"],
    )
    expected = sorted((y, x) for y, x in truth["positions"])

    assert len(recorded) == 4
    assert np.asarray(recorded) == pytest.approx(
        np.asarray(expected), abs=SOLVER_FROM_PEAK_ABS
    )


def test_the_clump_sky_positions_follow_the_header(pixelized_wcs_dict, euclid_dataset):
    """The mapper-route images get the same by-hand sky check as the solver route."""
    crval_ra_deg, crval_dec_deg = _header_crval(euclid_dataset)
    clump = pixelized_wcs_dict["source_clumps"][0]

    for y, x, ra, dec in zip(
        clump["image_y_arcsec"],
        clump["image_x_arcsec"],
        clump["image_ra_deg"],
        clump["image_dec_deg"],
    ):
        assert dec == pytest.approx(crval_dec_deg - y / 3600.0, abs=SKY_ABS_DEG)
        assert (ra - crval_ra_deg) * np.cos(np.radians(dec)) == pytest.approx(
            -x / 3600.0, abs=SKY_ABS_DEG
        )


def test_a_clump_finder_failure_is_logged_and_leaves_only_the_lens_keys(
    pixelized_fit, euclid_dataset, monkeypatch, caplog
):
    """
    With no clumps there is no peak to solve for either, so the record is the
    lens keys plus ``source_model == "pixelized"``, and the failure is logged.
    """

    def raise_(**kwargs):
        raise RuntimeError("no mapper")

    monkeypatch.setattr(util, "pixelized_source_clumps_from", raise_)

    with caplog.at_level(logging.WARNING, logger="util"):
        wcs_dict = util.wcs_dict_from(
            tracer=pixelized_fit.tracer,
            data=euclid_dataset.dataset.data,
            pixel_wcs=euclid_dataset.pixel_wcs,
            fit=pixelized_fit,
        )

    assert tuple(wcs_dict) == LENS_KEYS
    assert wcs_dict["source_model"] == "pixelized"
    assert "no mapper" in caplog.text
