"""
Known-answer tests for the ``files/coolest.json`` template ``util.coolest_json_from``
writes and ``util.AnalysisImaging.save_results`` puts beside ``files/wcs.json``.

The fixture is a hand-built model of exactly the shape this pipeline fits — a
40-Gaussian MGE lens light reduced to three Gaussians, an ``Isothermal`` +
``ExternalShear`` mass, and a Delaunay ``Pixelization`` source — on a 20 x 20,
0.1" mock ``al.Imaging``. That combination is what makes the record worth
asserting: COOLEST has no profile for an MGE ``Basis`` or a ``Pixelization``, so
the template must carry the *whole* mass model and name the two omissions under
``meta.skipped_profiles`` rather than silently exporting a lens with no light
and a source that never existed.

The observation grid is the second known answer: a 20-pixel, 0.1" dataset is a
2" field of view centred on the origin, so ``observation.pixels`` must read
``num_pix_x == 20`` over ``field_of_view_x == [-1.0, 1.0]``. Without it COOLEST
writes a grid of zeros its plotting API cannot render.

The remaining two tests are the guard, not the record: a COOLEST template is a
convenience, ``coolest`` is an optional dependency of PyAutoLens, and neither a
missing package nor a conversion that raises may take a finished fit down with
it. Both must come back as a ``coolest.json:`` warning and no file.

These tests are deliberately **JAX-free** and run no non-linear search (as
``AGENTS.md`` requires); the real-mode fit that proves ``save_results`` actually
*writes* the template is ``test_latent_run_level.py`` (``slow``).
"""

import json
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import util  # noqa: E402


# The mock dataset: 20 pixels of 0.1", so a 2" field of view centred on (0, 0).
SHAPE_NATIVE = (20, 20)
PIXEL_SCALE = 0.1
FIELD_OF_VIEW = [-1.0, 1.0]

# The MGE stand-in. Three Gaussians is enough for the record under test — the
# skipped entry names the `Basis` and how many inner profiles went with it.
GAUSSIANS = 3

SEARCH_NAME = "initial_lens_model"
DATASET_NAME = "euclid_dr1_like"


@pytest.fixture(scope="session", autouse=True)
def push_config():
    from autolens import conf

    conf.instance.push(
        new_path=PROJECT_ROOT / "config", output_path=PROJECT_ROOT / "output"
    )


@pytest.fixture(scope="module")
def tracer():
    """
    The pipeline's ``vis_pix`` model shape: an MGE ``Basis`` lens light,
    an ``Isothermal`` + ``ExternalShear`` mass, a Delaunay ``Pixelization``
    source.
    """
    import autolens as al

    lens = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_basis.Basis(
            profile_list=[
                al.lp.Gaussian(
                    centre=(0.0, 0.0), sigma=0.1 * (i + 1), intensity=1.0 / (i + 1)
                )
                for i in range(GAUSSIANS)
            ]
        ),
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=1.2
        ),
        shear=al.mp.ExternalShear(gamma_1=0.03, gamma_2=-0.02),
    )

    source = al.Galaxy(
        redshift=1.0,
        pixelization=al.Pixelization(
            mesh=al.mesh.Delaunay(pixels=100),
            regularization=al.reg.Constant(coefficient=1.0),
        ),
    )

    return al.Tracer(galaxies=[lens, source])


@pytest.fixture(scope="module")
def dataset():
    """A blank 20 x 20, 0.1" ``al.Imaging`` — only its pixel grid is read."""
    import numpy as np

    import autolens as al

    return al.Imaging(
        data=al.Array2D.no_mask(values=np.ones(SHAPE_NATIVE), pixel_scales=PIXEL_SCALE),
        noise_map=al.Array2D.no_mask(
            values=np.ones(SHAPE_NATIVE), pixel_scales=PIXEL_SCALE
        ),
    )


def _write(tracer, dataset, tmp_path):
    return util.coolest_json_from(
        tracer=tracer,
        dataset=dataset,
        file_path=tmp_path / "coolest",
        search_name=SEARCH_NAME,
        dataset_name=DATASET_NAME,
    )


@pytest.fixture(scope="module")
def template(tracer, dataset, tmp_path_factory):
    """The written template, read back as plain JSON — written once."""
    path = _write(tracer, dataset, tmp_path_factory.mktemp("coolest"))

    assert path is not None, "the template must be written for a supported model"

    with open(path) as f:
        return json.load(f)


def _mass_profile_types(template):
    """
    Every mass profile of the template, across entities: COOLEST puts external
    shear in its own ``MassField`` rather than on the galaxy, so the mass model
    is only complete when both entity kinds are read.
    """
    return sorted(
        profile["type"]
        for entity in template["lensing_entities"]
        for profile in entity.get("mass_model", [])
    )


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_the_observation_grid_is_the_datasets_pixel_grid(template):
    """
    ``dataset=`` stamps the template with the cut-out's own grid. 20 pixels of
    0.1" is a 2" field of view centred on the origin; a template written
    without it has a grid of zeros COOLEST's plotting API cannot render.
    """
    pixels = template["observation"]["pixels"]

    assert pixels["num_pix_x"] == SHAPE_NATIVE[1]
    assert pixels["num_pix_y"] == SHAPE_NATIVE[0]
    assert pixels["field_of_view_x"] == FIELD_OF_VIEW
    assert pixels["field_of_view_y"] == FIELD_OF_VIEW


def test_the_full_mass_model_is_exported(template):
    """
    The mass model is what a COOLEST consumer wants out of DR1, and unlike the
    light it is fully analytic — so both profiles must be there, the
    ``Isothermal`` as the galaxy's ``SIE`` and the shear as its own
    ``MassField``.
    """
    assert _mass_profile_types(template) == ["ExternalShear", "SIE"]


def test_the_unsupported_profiles_are_named_rather_than_dropped(template):
    """
    ``on_unsupported="skip"`` exports what COOLEST can represent, but a reader
    must never mistake the template for the whole model: the MGE ``Basis``
    (with its Gaussian count) and the Delaunay ``Pixelization`` are both listed.
    """
    skipped = template["meta"]["skipped_profiles"]
    profiles = [entry["profile"] for entry in skipped]

    assert profiles == [f"Basis(Gaussian x {GAUSSIANS})", "Pixelization(Delaunay)"]
    assert [entry["galaxy"] for entry in skipped] == ["galaxy_0", "galaxy_1"]

    # The skipped light is skipped, not silently exported under another name.
    assert all(not entity.get("light_model") for entity in template["lensing_entities"])


def test_the_template_records_where_it_came_from(template):
    """The metadata a catalogue reader needs to trace a template to its fit."""
    meta = template["meta"]

    assert meta["pipeline"] == util.COOLEST_PIPELINE_NAME
    assert meta["search"] == SEARCH_NAME
    assert meta["dataset"] == DATASET_NAME


def test_the_template_is_a_map_template(template):
    """The model is an inferred maximum log likelihood one, not a mock."""
    assert template["mode"] == "MAP"


# ---------------------------------------------------------------------------
# The guard — a record must never fail a fit
# ---------------------------------------------------------------------------


def test_a_missing_coolest_package_is_a_warning_not_a_failure(
    tracer, dataset, tmp_path, monkeypatch, caplog
):
    """
    ``coolest`` is an optional dependency (``pip install autolens[coolest]``).
    Without it ``to_coolest`` raises ``ImportError`` before writing anything,
    which must surface as a ``coolest.json:`` warning naming the install.

    Every ``coolest`` module is dropped from ``sys.modules`` first: a submodule
    another test already imported would otherwise satisfy the import from cache.
    """
    for name in [
        name
        for name in list(sys.modules)
        if name == "coolest" or name.startswith("coolest.")
    ]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "coolest", None)

    with caplog.at_level(logging.WARNING):
        assert _write(tracer, dataset, tmp_path) is None

    messages = [record.getMessage() for record in caplog.records]

    assert any(
        message.startswith("coolest.json:") and "autolens[coolest]" in message
        for message in messages
    ), messages
    assert not (tmp_path / "coolest.json").exists()


def test_a_conversion_that_raises_is_a_warning_not_a_failure(
    tracer, dataset, tmp_path, monkeypatch, caplog
):
    """
    Anything else ``to_coolest`` raises — an unconvertible profile, a broken
    cosmology, a read-only output directory — is logged and the fit stands.
    """
    import autolens as al

    def raise_it(**kwargs):
        raise ValueError("a profile could not be converted")

    monkeypatch.setattr(al.interop.coolest, "to_coolest", raise_it)

    with caplog.at_level(logging.WARNING):
        assert _write(tracer, dataset, tmp_path) is None

    messages = [record.getMessage() for record in caplog.records]

    assert any(
        message.startswith("coolest.json:")
        and "a profile could not be converted" in message
        for message in messages
    ), messages
    assert not (tmp_path / "coolest.json").exists()
