"""
Known-answer tests for the Euclid latent catalogue (``util.LatentEuclid``).

The fixture is the committed simulated dataset
``dataset/simulated/euclid_dr1_like/``, written by ``scripts/simulator.py``
together with a ``truth.json`` that records both

* ``truth["latents"]`` — the pipeline's own 12 latent values evaluated on the
  truth model after the dataset was written back to disk. Replaying that path
  here is a **regression** check: it must reproduce to floating-point noise, and
  it is what catches an accidental change to the latent maths or to
  ``load_vis_dataset``'s mask / over-sampling contract.
* ``truth["bands"]``, ``truth["aperture_lens_flux_mujy"]``,
  ``truth["magnification"]`` and ``truth["einstein_radius"]`` — the
  **independent** known answers, computed by the simulator from the analytic
  input model on the full simulation frame, without going anywhere near
  ``LatentEuclid``. These are the real known-answer side, and they carry real
  tolerances because the two sides differ by documented, physical offsets
  (recorded in ``truth["conventions"]``):

  - the latents integrate over the *masked* grid (``mask_radius = 3"``) while
    the band fluxes integrate the whole 100x100 frame. For the compact source
    this costs a few tenths of a per cent; for the extended Sersic lens light
    it costs ~10 per cent, so the lens-flux tests assert the *ordering* plus a
    loose bound rather than a tight equality;
  - ``magnification.area`` is the full-frame flux ratio, the latent is the same
    ratio on the masked grid (0.1 per cent apart);
  - ``einstein_radius.model_parameter`` is the ``Isothermal`` input parameter,
    the latent is the area-equivalent radius of the tangential critical curve,
    which differs by the ellipticity convention (1 per cent apart).

Every test therefore makes two named assertions: one against the replayed
pipeline value, one against the independent truth block.

These tests are deliberately **JAX-free** (``use_jax=False`` everywhere, as
``AGENTS.md`` requires) and run no non-linear search. The whole module is a
handful of seconds: the dataset is loaded and the latents evaluated once, in
session-scoped fixtures.
"""

import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import util  # noqa: E402


SIMULATED_SAMPLE = "simulated"
SIMULATED_DATASET = "euclid_dr1_like"
SIMULATED_PATH = PROJECT_ROOT / "dataset" / SIMULATED_SAMPLE / SIMULATED_DATASET

# `load_vis_dataset` resolves `dataset/` relative to `util.py`, so the degraded
# copy built by `test_aperture_latents_degrade_when_worst_band_absent` has to
# live inside the repository's own dataset tree. Created and destroyed by the
# `worst_band_absent_latents` fixture.
TMP_SAMPLE = "_pytest_latent"

# The replay must reproduce `truth.json` to floating-point noise; in practice it
# is bit-identical.
REPLAY_REL = 1e-6

# Independent-truth tolerances, justified in the module docstring.
COMPACT_FLUX_REL = 0.004  # source / lensed-source: masked vs full frame
LENS_FLUX_REL = 0.15  # extended lens light: masked vs full frame
APERTURE_REL = 0.005  # aperture photometry: masked vs full frame
MAGNIFICATION_REL = 0.002  # flux ratio, masked vs full frame
EINSTEIN_RADIUS_REL = 0.01  # critical-curve radius vs model parameter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def push_config():
    """
    Point the library at this repository's ``config/`` so ``config/latent.yaml``
    governs which latents are enabled (and therefore how many keys
    ``LatentEuclid.keys`` returns).
    """
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
    Rebuild one light or mass profile from its ``truth.json`` record.

    ``truth.json`` records every parameter it can read off a profile, including
    derived ones the constructor does not take (``Isothermal.slope``,
    ``ExternalShear.centre``), so the recorded parameters are filtered against
    the constructor signature rather than assumed to match it.
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
def truth_model(truth):
    """
    The truth model as a zero-free-parameter ``af.Collection``, in the galaxy
    order ``truth.json`` records it (galaxy 0 is the lens, galaxy -1 the source
    — the order the library latent functions index by).
    """
    import autofit as af
    import autolens as al

    galaxies = {
        name: al.Galaxy(
            redshift=galaxy["redshift"],
            **{
                profile_name: _profile_from(entry)
                for profile_name, entry in galaxy["profiles"].items()
            },
        )
        for name, galaxy in truth["model"].items()
    }

    return af.Collection(
        galaxies=af.Collection(
            **{
                name: af.Model.from_instance(galaxy)
                for name, galaxy in galaxies.items()
            }
        )
    )


def _analysis_from(euclid_dataset):
    """
    The pipeline ``AnalysisImaging`` exactly as ``scripts/simulator.py`` builds
    it when it writes ``truth["latents"]`` — same kwargs, ``use_jax=False``.
    """
    return util.AnalysisImaging(
        dataset=euclid_dataset.dataset,
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
def euclid_dataset():
    return util.load_vis_dataset(SIMULATED_DATASET, sample_name=SIMULATED_SAMPLE)


@pytest.fixture(scope="session")
def analysis(euclid_dataset):
    return _analysis_from(euclid_dataset)


@pytest.fixture(scope="session")
def latents(analysis, truth_model):
    """
    The 12 latent values, keyed by name. The model has zero free parameters, so
    the parameter vector is empty.
    """
    keys = util.LatentEuclid.keys(analysis)
    values = util.LatentEuclid.variables(
        analysis=analysis, parameters=[], model=truth_model
    )
    return {key: float(value) for key, value in zip(keys, values)}


@pytest.fixture(scope="session")
def worst_band_absent_latents(truth_model):
    """
    The 12 latent values on a copy of the simulated dataset whose primary header
    has had ``WORST_BAND`` deleted.

    This is the documented degradation path: without ``WORST_BAND``
    ``load_vis_dataset`` cannot resolve the worst-seeing band, warns, and
    returns ``psf_lowest_resolution = None``; ``LatentEuclid.variables`` then
    trips the ``AttributeError`` branch and the four aperture latents come back
    NaN instead of the whole fit crashing.
    """
    sample_path = PROJECT_ROOT / "dataset" / TMP_SAMPLE
    dataset_path = sample_path / SIMULATED_DATASET

    shutil.rmtree(sample_path, ignore_errors=True)
    try:
        shutil.copytree(SIMULATED_PATH, dataset_path)

        with fits.open(dataset_path / f"{SIMULATED_DATASET}.fits", mode="update") as hd:
            assert "WORST_BAND" in hd[0].header, (
                "fixture precondition: the simulated dataset's primary header "
                "must carry WORST_BAND before this test strips it"
            )
            del hd[0].header["WORST_BAND"]

        degraded = util.load_vis_dataset(SIMULATED_DATASET, sample_name=TMP_SAMPLE)

        assert degraded.psf_lowest_resolution is None, (
            "fixture precondition: load_vis_dataset must degrade "
            "psf_lowest_resolution to None when WORST_BAND is absent"
        )

        degraded_analysis = _analysis_from(degraded)
        keys = util.LatentEuclid.keys(degraded_analysis)
        values = util.LatentEuclid.variables(
            analysis=degraded_analysis, parameters=[], model=truth_model
        )
        yield dict(zip(keys, values))
    finally:
        shutil.rmtree(sample_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_replays(latents, truth, key):
    """
    Assertion 1 of every latent test: the pipeline reproduces the value
    ``scripts/simulator.py`` recorded for this latent on the truth model.
    """
    assert latents[key] == pytest.approx(truth["latents"][key], rel=REPLAY_REL), (
        f"replay of latent '{key}' does not reproduce truth.json's own "
        f"recorded value (rel={REPLAY_REL})"
    )


# ---------------------------------------------------------------------------
# The 12 latents — one test each
# ---------------------------------------------------------------------------


def test_total_lens_flux(latents, truth):
    _assert_replays(latents, truth, "total_lens_flux")

    full_frame = truth["bands"]["VIS"]["lens_flux_counts"]

    assert latents["total_lens_flux"] < full_frame, (
        "known answer: the masked lens flux must be below the full-frame "
        'simulated lens flux (the 3" mask clips the Sersic wings)'
    )
    assert latents["total_lens_flux"] == pytest.approx(
        full_frame, rel=LENS_FLUX_REL
    ), "known answer: masked lens flux within 15% of the simulated full-frame flux"


def test_total_lensed_source_flux(latents, truth):
    _assert_replays(latents, truth, "total_lensed_source_flux")

    assert latents["total_lensed_source_flux"] == pytest.approx(
        truth["bands"]["VIS"]["lensed_source_flux_counts"], rel=COMPACT_FLUX_REL
    ), "known answer: lensed-source counts match the simulated full-frame value"


def test_total_source_flux(latents, truth):
    _assert_replays(latents, truth, "total_source_flux")

    assert latents["total_source_flux"] == pytest.approx(
        truth["bands"]["VIS"]["source_flux_counts"], rel=COMPACT_FLUX_REL
    ), "known answer: source-plane counts match the simulated full-frame value"


def test_total_lens_flux_mujy(latents, truth):
    _assert_replays(latents, truth, "total_lens_flux_mujy")

    full_frame = truth["bands"]["VIS"]["lens_flux_mujy"]

    assert latents["total_lens_flux_mujy"] < full_frame, (
        "known answer: the masked lens flux in uJy must be below the "
        "full-frame simulated lens flux"
    )
    assert latents["total_lens_flux_mujy"] == pytest.approx(
        full_frame, rel=LENS_FLUX_REL
    ), "known answer: masked lens uJy within 15% of the simulated full-frame value"


def test_total_lensed_source_flux_mujy(latents, truth):
    _assert_replays(latents, truth, "total_lensed_source_flux_mujy")

    assert latents["total_lensed_source_flux_mujy"] == pytest.approx(
        truth["bands"]["VIS"]["lensed_source_flux_mujy"], rel=COMPACT_FLUX_REL
    ), "known answer: lensed-source uJy matches the simulated full-frame value"


def test_total_source_flux_mujy(latents, truth):
    _assert_replays(latents, truth, "total_source_flux_mujy")

    assert latents["total_source_flux_mujy"] == pytest.approx(
        truth["bands"]["VIS"]["source_flux_mujy"], rel=COMPACT_FLUX_REL
    ), "known answer: source-plane uJy matches the simulated full-frame value"


def test_magnification(latents, truth):
    _assert_replays(latents, truth, "magnification")

    assert latents["magnification"] == pytest.approx(
        truth["magnification"]["area"], rel=MAGNIFICATION_REL
    ), (
        "known answer: the masked-grid magnification matches the simulator's "
        "independent full-frame image-plane / source-plane flux ratio"
    )


def test_effective_einstein_radius(latents, truth):
    _assert_replays(latents, truth, "effective_einstein_radius")

    assert latents["effective_einstein_radius"] == pytest.approx(
        truth["einstein_radius"]["model_parameter"], rel=EINSTEIN_RADIUS_REL
    ), (
        "known answer: the critical-curve effective Einstein radius is within "
        "1% of the Isothermal einstein_radius the simulation was given"
    )


@pytest.mark.parametrize("multiplier", [1, 2, 3, 4])
def test_total_lens_flux_k_fwhm_mujy(latents, truth, multiplier):
    key = f"total_lens_flux_{multiplier}_fwhm_mujy"

    _assert_replays(latents, truth, key)

    assert latents[key] == pytest.approx(
        truth["aperture_lens_flux_mujy"]["values"][multiplier - 1], rel=APERTURE_REL
    ), (
        f"known answer: the {multiplier} x FWHM aperture latent matches the "
        "simulator's independent full-frame aperture photometry"
    )


# ---------------------------------------------------------------------------
# Aperture behaviour
# ---------------------------------------------------------------------------


def test_aperture_radii_scale_with_fwhm(euclid_dataset, truth, latents):
    """
    ``LatentEuclid.variables`` sizes the four apertures at
    ``fwhm / (pixel_scale * 2) * k`` pixels for ``k = 1, 2, 3, 4`` — i.e. k
    half-FWHM in pixels — using the *worst* band's PSF FWHM. For this dataset
    (worst band NIR_H, FWHM 0.5875") that is 2.9375 / 5.875 / 8.8125 / 11.75 px.
    """
    fwhm = euclid_dataset.psf_lowest_resolution_fwhm

    assert fwhm == pytest.approx(truth["worst_psf_fwhm"]), (
        "the worst-band PSF FWHM read back off the dataset must match the one "
        "the simulator recorded"
    )

    expected = [fwhm / (0.1 * 2.0) * multiplier for multiplier in (1.0, 2.0, 3.0, 4.0)]

    assert expected == pytest.approx(
        truth["aperture_lens_flux_mujy"]["radii_pixels"]
    ), "aperture radii in pixels are fwhm/(0.1*2)*k for k = 1, 2, 3, 4"

    values = [
        latents[f"total_lens_flux_{multiplier}_fwhm_mujy"]
        for multiplier in (1, 2, 3, 4)
    ]

    assert all(
        later > earlier for earlier, later in zip(values, values[1:])
    ), f"aperture fluxes must increase monotonically with radius, got {values}"


def test_aperture_latents_degrade_when_worst_band_absent(
    worst_band_absent_latents, latents
):
    """
    With ``WORST_BAND`` stripped from the primary header the four aperture
    latents degrade to NaN and the eight library latents are unaffected — the
    pipeline loses the Euclid-only columns rather than the whole fit.
    """
    for key in util.LatentEuclid.APERTURE_LATENT_KEYS:
        assert np.isnan(worst_band_absent_latents[key]), (
            f"aperture latent '{key}' must be NaN when WORST_BAND is absent, "
            f"got {worst_band_absent_latents[key]}"
        )

    for key in ("total_lens_flux_mujy", "magnification", "effective_einstein_radius"):
        assert float(worst_band_absent_latents[key]) == pytest.approx(
            latents[key], rel=REPLAY_REL
        ), (
            f"library latent '{key}' must be unchanged by a missing WORST_BAND "
            "header"
        )


def test_magnification_equals_lensed_over_source_flux(latents):
    """
    ``magnification`` is defined as the ratio of the two uJy source fluxes; the
    zero-point cancels, so this is an exact identity within the latent block.
    """
    ratio = latents["total_lensed_source_flux_mujy"] / latents["total_source_flux_mujy"]

    assert latents["magnification"] == pytest.approx(ratio, rel=1e-12), (
        "the magnification latent must equal lensed-source uJy / source uJy "
        "computed from the same latent block"
    )
