"""The pre-fit visualizer writes RGB output for both Euclid image deliveries."""

from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

import util


@pytest.mark.parametrize(
    ("names", "expected_titles"),
    [
        (["rgb.jpg"], ["RGB", "RGB Masked"]),
        (
            ["rgb_0.jpg", "rgb_1.jpg"],
            ["RGB 0", "RGB 1", "RGB 0 Masked", "RGB 1 Masked"],
        ),
        (
            ["rgb.jpg", "rgb_0.jpg", "rgb_1.jpg"],
            ["RGB", "RGB Masked"],
        ),
    ],
)
def test_pre_fit_rgb_output(tmp_path: Path, monkeypatch, names, expected_titles):
    for name in names:
        Image.new("RGB", (12, 10), (80, 120, 160)).save(tmp_path / name)

    class BaseVisualizer:
        def visualize_before_fit(self, **kwargs):
            pass

    monkeypatch.setattr(util.al, "VisualizerImaging", BaseVisualizer)

    calls = []
    real_subplot = util.subplot_rgb

    def record_subplot(**kwargs):
        calls.append(kwargs)
        real_subplot(**kwargs)

    monkeypatch.setattr(util, "subplot_rgb", record_subplot)

    analysis = SimpleNamespace(
        kwargs={"dataset_main_path": tmp_path},
        dataset=SimpleNamespace(
            pixel_scales=(0.1, 0.1),
            mask=SimpleNamespace(origin=(0.0, 0.0), circular_radius=0.4),
        ),
    )
    image_path = tmp_path / "output"
    paths = SimpleNamespace(image_path=image_path)

    util.VisualizerImaging.visualize_before_fit(
        analysis=analysis, paths=paths, model=None
    )

    assert (image_path / "rgb.png").is_file()
    assert len(calls) == 1
    assert calls[0]["titles"] == expected_titles
    assert len(calls[0]["arrays"]) == len(expected_titles)
    assert calls[0]["output_filename"] == "rgb"
