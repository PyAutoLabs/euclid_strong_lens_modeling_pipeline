"""
Run-level check: a real fit actually **writes** the Euclid latent catalogue.

``tests/test_compute_latent_variable.py`` proves the latent *values* are right,
but it calls ``LatentEuclid.variables`` directly. That path is ungated, so it
would keep passing even if the pipeline never wrote a latent again. The write
is gated: ``autonerves.test_mode.skip_latents()`` is
``is_test_mode() or PYAUTO_SKIP_LATENTS``, consumed at exactly one line
(``PyAutoFit/autofit/non_linear/search/updater.py``'s
``_compute_latent_samples``), and there is **no override** — no environment
variable forces latents on under test mode. So the only way to prove the
pipeline still writes them is a fit with ``PYAUTO_TEST_MODE`` unset, which is
why this module is marked ``slow`` and runs in its own CI job: a failure here
means "the fit stopped emitting latents", which is a different problem from a
wrong latent value.

The file to assert on is ``files/latent/latent_summary.json``, **not**
``files/latent.csv``: this repository's ``config/output.yaml`` sets
``latent_draw_via_pdf: true``, and on that branch the updater calls only
``save_samples_summary(..., "latent/latent_summary")`` — ``save_latent_samples``
never runs, so no ``latent.csv`` is written.

Two things shape the fit:

* **``af.Drawer``, not a sampler.** It draws uniformly from the priors and does
  no parameter search — the cheapest real search in PyAutoFit — while still
  running the full post-fit updater path, so ``skip_latents()`` and
  ``latent_after_fit`` behave exactly as in production. (The pipeline scripts
  hard-code ``n_live=750`` for Nautilus with no override; that is not a CI fit.)
* **The model is anchored on the truth**, with a single free parameter
  (``einstein_radius``, uniform within 10 per cent of its true value). A Drawer
  over a fully free model draws junk: the source's linear intensity solves to
  exactly zero, ``magnification`` becomes ``0 / 0`` and lands in the summary as
  NaN — and a NaN latent is **dropped from ``latent_summary.json`` entirely**.
  That is worth knowing, and it is why the key-set assertion below is also the
  NaN check: a latent that failed to compute is a missing key, not a NaN value.

Wall time: about 15 s (10 likelihood evaluations plus 10 latent evaluations on
the 100x100 masked simulated VIS image, non-JAX).
"""

import inspect
import json
import shutil
import zipfile
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

TOTAL_DRAWS = 10
LATENT_DRAW_VIA_PDF_SIZE = 5

# The pixelized fit's latents integrate an inversion on every draw and are
# several times dearer than the light-profile fit's; three draws are enough to
# prove the write path without doubling the job's wall time.
PIXELIZED_TOTAL_DRAWS = 3

# The einstein_radius prior is this fraction either side of the true value, so
# every draw is a physically sensible lens and every latent is computable.
EINSTEIN_RADIUS_PRIOR_WIDTH = 0.1


pytestmark = pytest.mark.slow


def _profile_from(entry):
    """
    Rebuild one profile from ``truth.json``, filtering the recorded parameters
    against the constructor signature (``truth.json`` also records derived
    parameters such as ``Isothermal.slope`` that the constructor does not take).
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


def _latent_kwargs_from(summary_path):
    """
    The latent name -> value mapping out of ``latent_summary.json``.

    ``af.Drawer`` is an MLE search, so the summary carries a
    ``max_log_likelihood_sample`` and a null ``median_pdf_sample``; a sampler
    would populate both. Read whichever block is present.
    """
    summary = json.loads(summary_path.read_text())["arguments"]

    for block in ("max_log_likelihood_sample", "median_pdf_sample"):
        sample = summary.get(block)
        if sample is not None:
            return sample["arguments"]["kwargs"]["arguments"]

    raise AssertionError(
        f"{summary_path} carries neither a max_log_likelihood_sample nor a "
        "median_pdf_sample block"
    )


@pytest.fixture(scope="module")
def run_level(tmp_path_factory):
    """
    Run two tiny real-mode fits on the committed simulated dataset — a
    light-profile source and the ``vis_pix`` pixelized source — under one
    pushed config, and return where their output landed.

    Module-scoped so the fits run once for every assertion below;
    ``pytest.MonkeyPatch`` is the supported way to use ``monkeypatch``'s
    undo semantics outside function scope.
    """
    tmp_path = tmp_path_factory.mktemp("latent_run_level")

    with pytest.MonkeyPatch.context() as monkeypatch:
        yield from _fit(tmp_path, monkeypatch)


@pytest.fixture(scope="module")
def latent_summary(run_level):
    """The light-profile fit's ``latent_summary.json`` path and the latent keys."""
    return run_level["light_profile"] / "latent" / "latent_summary.json", run_level["keys"]


def _fit(tmp_path, monkeypatch):
    monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
    monkeypatch.delenv("PYAUTO_SKIP_LATENTS", raising=False)
    monkeypatch.delenv("PYAUTO_SKIP_FIT_OUTPUT", raising=False)

    from autonerves.test_mode import skip_latents

    assert not skip_latents(), (
        "the environment still asks the pipeline to skip latents; this test "
        "cannot prove anything about the write path"
    )

    # A copy of the pipeline's own config with the PDF draw count reduced. The
    # config is pushed rather than mutated in place so the repository's
    # `config/output.yaml` is never touched.
    config_path = tmp_path / "config"
    shutil.copytree(PROJECT_ROOT / "config", config_path)
    output_yaml = config_path / "output.yaml"
    output_yaml.write_text(
        output_yaml.read_text().replace(
            "latent_draw_via_pdf_size : 100",
            f"latent_draw_via_pdf_size : {LATENT_DRAW_VIA_PDF_SIZE}",
        )
    )

    from autolens import conf

    conf.instance.push(new_path=config_path, output_path=tmp_path / "output")
    try:
        import autofit as af
        import autolens as al

        with open(SIMULATED_PATH / "truth.json") as f:
            truth = json.load(f)

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

        model = af.Collection(
            galaxies=af.Collection(
                **{
                    name: af.Model.from_instance(galaxy)
                    for name, galaxy in galaxies.items()
                }
            )
        )

        lens_name = next(iter(truth["model"]))
        einstein_radius = truth["einstein_radius"]["model_parameter"]
        getattr(model.galaxies, lens_name).mass.einstein_radius = af.UniformPrior(
            lower_limit=einstein_radius * (1.0 - EINSTEIN_RADIUS_PRIOR_WIDTH),
            upper_limit=einstein_radius * (1.0 + EINSTEIN_RADIUS_PRIOR_WIDTH),
        )

        assert model.total_free_parameters == 1, (
            "the run-level fit is meant to have exactly one free parameter; "
            f"got {model.total_free_parameters}"
        )

        euclid_dataset = util.load_vis_dataset(
            SIMULATED_DATASET, sample_name=SIMULATED_SAMPLE
        )

        # The `vis_pix` model: the same truth lens with the same one free
        # parameter, and a Delaunay pixelized source (tests/pixelized_model.py).
        # It is the stage that writes `source_clumps` into wcs.json, which no
        # light-profile fit can exercise.
        pixelized_model, adapt_images = pixelized_model_and_adapt_images_from(
            lens=galaxies[lens_name],
            sersic_source=galaxies[list(truth["model"])[-1]],
            euclid_dataset=euclid_dataset,
        )
        pixelized_model.galaxies.lens.mass.einstein_radius = af.UniformPrior(
            lower_limit=einstein_radius * (1.0 - EINSTEIN_RADIUS_PRIOR_WIDTH),
            upper_limit=einstein_radius * (1.0 + EINSTEIN_RADIUS_PRIOR_WIDTH),
        )
        assert pixelized_model.total_free_parameters == 1

        def analysis_from(adapt_images=None):
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

        analysis = analysis_from()

        output_path = tmp_path / "output"
        files = {}

        for name, search_model, search_analysis, total_draws in (
            ("drawer", model, analysis, TOTAL_DRAWS),
            (
                "drawer_pix",
                pixelized_model,
                analysis_from(adapt_images=adapt_images),
                PIXELIZED_TOTAL_DRAWS,
            ),
        ):
            search = af.Drawer(
                path_prefix="latent_run_level",
                name=name,
                total_draws=total_draws,
            )
            search.fit(model=search_model, analysis=search_analysis)

            search_output = output_path / "latent_run_level" / name
            summaries = list(search_output.rglob("latent/latent_summary.json"))

            if not summaries:
                # `config/general.yaml` sets `hpc.hpc_mode: true`, which forces
                # PyAutoFit's `remove_files` (paths/abstract.py): the search zips
                # its output and deletes the unzipped tree, so the files survive
                # only inside `<identifier>.zip`. Read them from there rather
                # than turning HPC mode off, so this test keeps exercising the
                # production config. The zip is extracted *outside* `output/`
                # so the aggregator read below sees the zip, not a sibling dir.
                zips = list(search_output.rglob("*.zip"))
                assert len(zips) == 1, (
                    f"the real-mode fit `{name}` must leave exactly one search "
                    f"output zip; found {zips}"
                )
                extracted = tmp_path / "extracted" / name
                with zipfile.ZipFile(zips[0]) as archive:
                    archive.extractall(extracted)
                summaries = list(extracted.rglob("latent/latent_summary.json"))

            assert len(summaries) == 1, (
                f"the real-mode fit `{name}` must write exactly one "
                f"files/latent/latent_summary.json; found {summaries}"
            )
            files[name] = summaries[0].parent.parent
            assert files[name].name == "files"

        yield {
            "light_profile": files["drawer"],
            "pixelized": files["drawer_pix"],
            "output": output_path,
            "keys": util.LatentEuclid.keys(analysis),
        }
    finally:
        # Restore the repository config for any test module that runs after
        # this one in the same session (`conf.instance` has no pop).
        conf.instance.push(
            new_path=PROJECT_ROOT / "config", output_path=PROJECT_ROOT / "output"
        )


def test_a_real_mode_fit_writes_the_latent_summary(latent_summary):
    summary_path, _ = latent_summary

    assert summary_path.is_file()
    assert summary_path.parent.name == "latent"
    assert summary_path.parent.parent.name == "files"


def test_the_latent_summary_carries_every_latent_key(latent_summary):
    """
    Exactly the 12 keys ``LatentEuclid.keys`` declares.

    This doubles as the NaN check: a latent that fails to compute is not
    written as NaN, it is dropped from the summary — so a missing key is how a
    broken latent presents itself here.
    """
    summary_path, expected_keys = latent_summary

    kwargs = _latent_kwargs_from(summary_path)

    assert len(expected_keys) == 12, (
        "the Euclid latent catalogue is 12 keys (8 library + 4 aperture); "
        f"config/latent.yaml now yields {len(expected_keys)}"
    )
    assert sorted(kwargs) == sorted(expected_keys), (
        "latent_summary.json must carry exactly the keys LatentEuclid declares; "
        f"missing {sorted(set(expected_keys) - set(kwargs))}, "
        f"unexpected {sorted(set(kwargs) - set(expected_keys))}"
    )


def test_no_latent_is_none_nan_or_exactly_zero(latent_summary):
    """
    Zero is the sentinel that matters: a linear light profile whose intensity
    solves to zero yields a zero flux latent and a ``0 / 0`` magnification, and
    it looks like a number rather than a failure.
    """
    summary_path, _ = latent_summary

    kwargs = _latent_kwargs_from(summary_path)

    bad = {
        key: value
        for key, value in kwargs.items()
        if value is None or not np.isfinite(value) or value == 0.0
    }

    assert not bad, f"latent values must be finite and non-zero; got {bad}"


WCS_IMAGE_KEYS = (
    "lensed_source_image_y_arcsec",
    "lensed_source_image_x_arcsec",
    "lensed_source_image_ra_deg",
    "lensed_source_image_dec_deg",
)
CLUMP_IMAGE_KEYS = ("image_y_arcsec", "image_x_arcsec", "image_ra_deg", "image_dec_deg")


def _wcs_dict_from(files_path):
    """
    ``files/wcs.json`` read back the way the aggregator hands it to
    ``catalogue/scripts/magnitudes.py``: `output_to_json` writes PyAutoFit's
    dictable envelope ({"type": "dict", "arguments": ...}, lists as
    {"type": "list", "values": ...}), and `from_json` undoes it.
    """
    import autolens as al

    wcs_path = files_path / "wcs.json"
    assert wcs_path.is_file(), f"save_results must write {wcs_path}"

    return al.from_json(file_path=wcs_path)


def _assert_images(wcs_dict, keys, n_images):
    lengths = {key: len(wcs_dict[key]) for key in keys}
    assert set(lengths.values()) == {n_images}, (
        f"one entry per image in each of the four lists, got {lengths}"
    )
    for key in keys:
        assert np.all(np.isfinite(wcs_dict[key])), f"{key} must be finite"


def test_a_real_mode_fit_writes_the_lensed_source_images_to_wcs_json(run_level):
    """
    ``util.AnalysisImaging.save_results`` writes ``files/wcs.json`` beside the
    latent summary, and since the model here has a light-profile source it
    must carry the lensed source's multiple images: the run-level proof that
    the point solver runs at the end of a real fit and its record survives the
    zip. The values are checked in ``test_wcs_dict.py``; this asserts they are
    written, finite, one per image, and that the source is still quadruply
    imaged at a max-likelihood Einstein radius drawn within 10 per cent of the
    truth's.
    """
    wcs_dict = _wcs_dict_from(run_level["light_profile"])

    for key in ("crval_ra_deg", "crval_dec_deg"):
        assert np.isfinite(wcs_dict[key])

    assert wcs_dict["source_model"] == "light_profile"
    assert "source_clumps" not in wcs_dict

    for key in ("source_centre_y_arcsec", "source_centre_x_arcsec"):
        assert np.isfinite(wcs_dict[key])

    _assert_images(wcs_dict, WCS_IMAGE_KEYS, n_images=4)


def test_a_pixelized_real_mode_fit_writes_its_clumps_to_wcs_json(run_level):
    """
    The ``vis_pix`` leg: at the end of a real pixelized fit ``save_results``
    reads the clumps off the max-likelihood fit's mapper and writes
    ``source_clumps``, then solves the lens equation for the brightest clump's
    peak into the same solver keys the light-profile fit writes. Values are
    checked in ``test_wcs_dict.py``; this proves the write happens on the
    production path — through ``result.max_log_likelihood_fit``, under
    ``hpc_mode``'s zip-and-remove — and that nothing is lost to the envelope.
    """
    wcs_dict = _wcs_dict_from(run_level["pixelized"])

    assert wcs_dict["source_model"] == "pixelized"

    clumps = wcs_dict["source_clumps"]
    assert isinstance(clumps, list) and len(clumps) >= 1, (
        f"a pixelized fit must record at least one clump; got {clumps!r}"
    )
    assert len(clumps) == 1, (
        "the simulated source is one smooth Sersic, which the default threshold "
        f"isolates as one clump; got {len(clumps)}"
    )

    clump = clumps[0]
    assert clump["mesh_pixels"] >= util.SOURCE_CLUMP_MIN_PIXELS
    assert clump["peak_value"] > 0.0
    _assert_images(clump, CLUMP_IMAGE_KEYS, n_images=4)

    assert wcs_dict["source_centre_y_arcsec"] == clump["peak_y_arcsec"]
    assert wcs_dict["source_centre_x_arcsec"] == clump["peak_x_arcsec"]
    _assert_images(wcs_dict, WCS_IMAGE_KEYS, n_images=4)


def test_the_aggregator_reads_both_records_back(run_level):
    """
    The consumer path: ``catalogue/scripts/magnitudes.py`` reads
    ``wcs.json`` as ``agg.values("wcs")`` over an ``Aggregator.from_directory``
    of the output tree, exactly as ``lens_mass.py`` opens it (``completed_only``
    and, under ``remove_files``, ``unzip_temporary``). Both fits' records must
    come back decoded, the nested clump list included, so a producer can add a
    column from them without touching the envelope.
    """
    from autofit.aggregator import Aggregator

    agg = Aggregator.from_directory(
        directory=run_level["output"], completed_only=True, unzip_temporary=True
    )
    wcs_list = list(agg.values("wcs"))

    assert len(wcs_list) == 2, f"two completed fits, got {len(wcs_list)} wcs records"
    by_model = {wcs_dict["source_model"]: wcs_dict for wcs_dict in wcs_list}
    assert set(by_model) == {"light_profile", "pixelized"}

    pixelized = by_model["pixelized"]
    assert isinstance(pixelized["source_clumps"], list)
    assert isinstance(pixelized["source_clumps"][0], dict)
    assert isinstance(pixelized["source_clumps"][0]["image_ra_deg"], list)
    assert isinstance(by_model["light_profile"]["lensed_source_image_ra_deg"], list)
