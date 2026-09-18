"""Fields API ownership and free/fixed chaining across the full SLaM stages."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_model  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def push_config():
    from autolens import conf

    conf.instance.push(
        new_path=PROJECT_ROOT / "config", output_path=PROJECT_ROOT / "output"
    )


@pytest.fixture
def capture_search(monkeypatch):
    import autofit as af

    results = []

    class CaptureSearch:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, model, analysis, **kwargs):
            result = SimpleNamespace(
                model=model,
                instance=model.instance_from_prior_medians(),
            )
            results.append(result)
            return result

    monkeypatch.setattr(af, "Nautilus", CaptureSearch)
    return results


def _settings():
    return SimpleNamespace(search_dict={}, fit_dict={})


def _source_lp_result(capture_search):
    import autofit as af
    import autolens as al

    result = full_model.source_lp(
        settings_search=_settings(),
        analysis=object(),
        lens_bulge=af.Model(al.lp.Sersic),
        mass=af.Model(al.mp.Isothermal),
        shear=af.Model(al.mp.ExternalShear),
        source_bulge=af.Model(al.lp.Sersic),
    )
    assert result.model.fields.prior_count == 2
    assert not hasattr(result.model.galaxies.lens, "shear")
    return result


def test_source_pixelized_stages_refit_then_fix_the_field(capture_search, monkeypatch):
    import autolens as al

    source_lp_result = _source_lp_result(capture_search)
    monkeypatch.setattr(
        al.util.chaining,
        "mass_from",
        lambda mass, mass_result, unfix_mass_centre: mass,
    )

    source_pix_1 = full_model.source_pix_1(
        settings_search=_settings(),
        analysis=object(),
        source_lp_result=source_lp_result,
        mesh_init=al.mesh.Delaunay(pixels=9),
        regularization_init=al.reg.Constant(coefficient=1.0),
    )

    assert source_pix_1.model.fields is source_lp_result.model.fields
    assert source_pix_1.model.fields.prior_count == 2

    source_pix_2 = full_model.source_pix_2(
        settings_search=_settings(),
        analysis=object(),
        source_lp_result=source_lp_result,
        source_pix_result_1=source_pix_1,
        mesh=al.mesh.Delaunay(pixels=9),
        regularization=al.reg.Constant(coefficient=1.0),
    )

    assert isinstance(source_pix_2.model.fields, al.MassField)
    assert not any(path[0] == "fields" for path in source_pix_2.model.paths)
    assert source_pix_2.model.fields.shear.gamma_1 == pytest.approx(
        source_pix_1.instance.fields.shear.gamma_1
    )
    assert source_pix_2.model.fields.shear.gamma_2 == pytest.approx(
        source_pix_1.instance.fields.shear.gamma_2
    )


@pytest.mark.parametrize("reset_shear_prior", (False, True))
def test_mass_total_chains_or_resets_the_whole_field(
    capture_search, monkeypatch, reset_shear_prior
):
    import autofit as af
    import autolens as al

    source_result = _source_lp_result(capture_search)
    light_model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                bulge=al.lp.Sersic(intensity=1.0),
                disk=None,
            )
        )
    )
    light_result = SimpleNamespace(instance=light_model.instance_from_prior_medians())

    monkeypatch.setattr(
        al.util.chaining,
        "mass_from",
        lambda mass, mass_result, unfix_mass_centre: mass,
    )
    monkeypatch.setattr(
        al.util.chaining,
        "source_from",
        lambda result: result.model.galaxies.source,
    )

    result = full_model.mass_total(
        settings_search=_settings(),
        analysis=object(),
        source_result_for_lens=source_result,
        source_result_for_source=source_result,
        light_result=light_result,
        mass=af.Model(al.mp.PowerLaw),
        reset_shear_prior=reset_shear_prior,
    )

    assert result.model.fields.prior_count == 2
    assert not hasattr(result.model.galaxies.lens, "shear")
    if reset_shear_prior:
        assert result.model.fields is not source_result.model.fields
        assert set(result.model.fields.priors) != set(source_result.model.fields.priors)
    else:
        assert result.model.fields is source_result.model.fields
