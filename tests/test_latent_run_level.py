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

TOTAL_DRAWS = 10
LATENT_DRAW_VIA_PDF_SIZE = 5

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
def latent_summary(tmp_path_factory):
    """
    Run a tiny real-mode fit on the committed simulated dataset and return the
    path of the ``latent_summary.json`` it wrote.

    Module-scoped so the fit runs once for the three assertions below;
    ``pytest.MonkeyPatch`` is the supported way to use ``monkeypatch``'s
    undo semantics outside function scope.
    """
    tmp_path = tmp_path_factory.mktemp("latent_run_level")

    with pytest.MonkeyPatch.context() as monkeypatch:
        yield from _fit(tmp_path, monkeypatch)


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

        analysis = util.AnalysisImaging(
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

        search = af.Drawer(
            path_prefix="latent_run_level",
            name="drawer",
            total_draws=TOTAL_DRAWS,
        )
        search.fit(model=model, analysis=analysis)

        summaries = list((tmp_path / "output").rglob("latent/latent_summary.json"))

        assert len(summaries) == 1, (
            "a real-mode fit must write exactly one "
            f"files/latent/latent_summary.json; found {summaries}"
        )

        yield summaries[0], util.LatentEuclid.keys(analysis)
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
