"""A capped SED band records its status and lets the chain advance."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import autofit as af
import pytest

from scripts import lens_model_waveband as waveband


@pytest.mark.parametrize("partial_posterior", [True, False])
def test_capped_band_then_healthy_band(tmp_path, monkeypatch, capsys, partial_posterior):
    settings = SimpleNamespace(search_dict={"path_prefix": "sample"}, fit_dict={})
    nir_y = tmp_path / "nir_y" / "fit_id"
    nir_j = tmp_path / "nir_j" / "fit_id"
    reason = "n_like_max" if partial_posterior else "timeout"
    stopped = Mock(paths=SimpleNamespace(output_path=nir_y))
    if partial_posterior:
        stopped.fit.return_value.samples.samples_info = {"termination_reason": reason}
    else:
        stopped.fit.side_effect = af.exc.SearchBudgetExceeded(reason)

    healthy = Mock(paths=SimpleNamespace(output_path=nir_j))
    healthy.fit.return_value.samples.samples_info = {}
    factory = Mock(side_effect=[stopped, healthy, healthy])
    monkeypatch.setattr(af, "Nautilus", factory)

    for band in ("nir_y", "nir_j"):
        waveband._fit_band_with_limits("tile", band, "model", "analysis", settings, 5000)

    assert [call.kwargs["name"] for call in factory.call_args_list] == ["nir_y", "nir_j"]
    for call in factory.call_args_list:
        assert call.kwargs["timeout"] == 1800
        assert call.kwargs["n_like_max"] == 10_000
        assert call.kwargs["n_batch"] == 50
    marker = nir_y.with_suffix(".waveband_capped.json")
    assert json.loads(marker.read_text()) == {
        "reason": reason, "band": "nir_y", "dataset": "tile",
        "timeout_s": 1800, "n_like_max": 10_000,
        "capped": True, "has_posterior": partial_posterior,
    }
    assert f"CAPPED BAND: tile nir_y ({reason})" in capsys.readouterr().out
    assert not nir_y.exists()  # A marker must not shadow a zip-only fit.
    assert not nir_j.with_suffix(".waveband_capped.json").exists()

    # A successful retry for exactly the same fit clears its earlier cap.
    healthy.paths.output_path = nir_y
    waveband._fit_band_with_limits("tile", "nir_y", "model", "analysis", settings, 5000)
    recovery = json.loads(marker.read_text())
    assert recovery["capped"] is False
    assert recovery["reason"] is None
    assert recovery["has_posterior"] is True


def test_unrelated_search_failure_is_visible(tmp_path, monkeypatch):
    search = Mock(paths=SimpleNamespace(output_path=tmp_path / "id"))
    search.fit.side_effect = RuntimeError("invalid dataset")
    monkeypatch.setattr(af, "Nautilus", Mock(return_value=search))
    settings = SimpleNamespace(search_dict={}, fit_dict={})
    with pytest.raises(RuntimeError, match="invalid dataset"):
        waveband._fit_band_with_limits("tile", "nir_y", None, None, settings, 5000)
    assert not list(tmp_path.iterdir())
