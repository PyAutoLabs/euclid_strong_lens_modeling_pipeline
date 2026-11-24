"""SED cap sidecars remain visible in both waveband catalogue products."""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_catalogue_latent_columns import (
    CRVAL_RA_DEG,
    SAMPLE,
    _run_producer,
    _write_result,
    pipeline_config,  # noqa: F401
)
from test_catalogue_parity import expected_header_for


STAGE = "sersic_lens_model"
BAND = "nir_h"


def _cap(output_path, lens, result_hash, **status):
    marker = output_path / SAMPLE / lens / STAGE / BAND
    marker.mkdir(parents=True, exist_ok=True)
    path = marker / f"{result_hash}.waveband_capped.json"
    path.write_text(json.dumps({
        "reason": "n_like_max", "dataset": lens, "band": BAND, **status
    }))
    return path


def _read_csv(path, producer):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert ",".join(reader.fieldnames) == expected_header_for(producer)
        return list(reader)


def _complete(lens):
    import autofit as af
    import autolens as al

    model = af.Collection(dataset_model=af.Model(al.DatasetModel))
    model.dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(-2.0, 2.0)
    model.dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(-2.0, 2.0)
    return Path(
        _write_result(
            model,
            path_prefix=Path(SAMPLE) / lens,
            name=BAND,
            unique_tag=STAGE,
            offset=0.5,
            files={"wcs": {"crval_ra_deg": CRVAL_RA_DEG}},
        )
    )


@pytest.mark.parametrize("producer", ["magnitudes", "astrometric_offsets"])
def test_cap_only_result_writes_a_per_lens_qa_row(
    producer, pipeline_config, tmp_path, monkeypatch, capsys
):
    _cap(pipeline_config, "TileA", "capped_result")
    inspect = tmp_path / "inspect"
    _run_producer(producer, monkeypatch, pipeline_config, inspect)

    master = _read_csv(inspect / f"{producer}.csv", producer)
    assert master == _read_csv(inspect / "TileA" / f"{producer}.csv", producer)
    assert len(master) == 1
    assert {key: value for key, value in master[0].items() if value} == {
        "id": "capped_result",
        "lens_name": "TileA",
        "waveband": BAND,
        "capped": "True",
    }
    assert "waveband_capped (n_like_max); no usable posterior" in capsys.readouterr().out


@pytest.mark.parametrize("producer", ["magnitudes", "astrometric_offsets"])
def test_archived_cap_retains_measurements_and_backwards_compatible_winners(
    producer, pipeline_config, tmp_path, monkeypatch, capsys
):
    archived = _complete("TileCapped")
    shutil.make_archive(str(archived), "zip", root_dir=archived)
    shutil.rmtree(archived)
    _cap(pipeline_config, "TileCapped", archived.name)

    earlier = _complete("TileEarlier")
    os.utime(earlier, (1_000_000, 1_000_000))
    _cap(pipeline_config, "TileEarlier", "recent_incomplete")

    _complete("TileRecovered")
    stale = _cap(pipeline_config, "TileRecovered", "stale_incomplete")
    os.utime(stale, (1_000_000, 1_000_000))

    incomplete = _complete("TileNoSummary")
    (incomplete / "files/samples_summary.json").unlink()
    _cap(pipeline_config, "TileNoSummary", incomplete.name)

    inspect = tmp_path / "inspect"
    _run_producer(producer, monkeypatch, pipeline_config, inspect)
    rows = _read_csv(inspect / f"{producer}.csv", producer)
    assert len(rows) == 4
    by_lens = {row["lens_name"]: row for row in rows}
    assert by_lens["TileCapped"]["capped"] == "True"
    assert float(by_lens["TileCapped"]["crval_ra_deg"]) == pytest.approx(CRVAL_RA_DEG)
    assert by_lens["TileEarlier"]["capped"] == "False"
    assert by_lens["TileRecovered"]["capped"] == "False"
    assert by_lens["TileNoSummary"]["capped"] == "True"
    assert by_lens["TileNoSummary"]["crval_ra_deg"] == ""

    log = capsys.readouterr().out
    assert "recent_incomplete: waveband_capped (n_like_max); retained completed result=" in log
    assert "stale_incomplete" not in log



@pytest.mark.parametrize("producer", ["magnitudes", "astrometric_offsets"])
def test_completed_metadata_overrides_or_replaces_the_cap_sidecar(
    producer, pipeline_config, tmp_path, monkeypatch, capsys
):
    recovered = _complete("TileRecovered")
    recovered_info = recovered / "files" / "samples_info.json"
    recovered_info.write_text(json.dumps({"total_samples": 20}))
    _cap(pipeline_config, "TileRecovered", recovered.name)

    capped = _complete("TileCapped")
    capped_info = capped / "files" / "samples_info.json"
    capped_info.write_text(json.dumps({"termination_reason": "timeout", "total_samples": 10}))
    shutil.make_archive(str(capped), "zip", root_dir=capped)
    shutil.rmtree(capped)

    inspect = tmp_path / "inspect"
    _run_producer(producer, monkeypatch, pipeline_config, inspect)
    rows = _read_csv(inspect / f"{producer}.csv", producer)
    assert {row["lens_name"]: row["capped"] for row in rows} == {
        "TileRecovered": "False",
        "TileCapped": "True",
    }
    assert all(row["crval_ra_deg"] for row in rows)

    log = capsys.readouterr().out
    assert "waveband_capped (timeout)" in log
    assert "lens=TileRecovered" not in log



@pytest.mark.parametrize("producer", ["magnitudes", "astrometric_offsets"])
def test_synced_receipts_with_samples_csv_disabled(
    producer, pipeline_config, tmp_path, monkeypatch, capsys
):
    recovered = _complete("TileRecovered")
    _cap(pipeline_config, "TileRecovered", recovered.name, has_posterior=False)

    cleared = _complete("TileCleared")
    _cap(
        pipeline_config, "TileCleared", cleared.name,
        capped=False, reason=None, has_posterior=True,
    )

    older = _complete("TileEarlier")
    os.utime(older, (1_000_000, 1_000_000))
    _cap(pipeline_config, "TileEarlier", "new_retry", has_posterior=False)

    _cap(pipeline_config, "TileCapped", "no_posterior", has_posterior=False)

    inspect = tmp_path / "inspect"
    _run_producer(producer, monkeypatch, pipeline_config, inspect)
    rows = _read_csv(inspect / f"{producer}.csv", producer)
    assert {row["lens_name"]: row["capped"] for row in rows} == {
        "TileRecovered": "False",
        "TileCleared": "False",
        "TileEarlier": "False",
        "TileCapped": "True",
    }
    assert {
        row["lens_name"] for row in rows if not row["crval_ra_deg"]
    } == {"TileCapped"}

    log = capsys.readouterr().out
    assert "lens=TileRecovered" not in log
    assert "lens=TileCleared" not in log
    assert "new_retry: waveband_capped (n_like_max); retained completed result=" in log
    assert "no_posterior: waveband_capped (n_like_max); no usable posterior" in log


def test_post_build_coverage_excludes_only_capped_rows_without_measurements(tmp_path):
    (tmp_path / "magnitudes.csv").write_text(
        "lens_name,waveband,capped,lens_flux_max_lh\n"
        "TileA,vis,False,1.0\n"
        "TileA,nir_h,True,\n"
        "TileA,nir_j,True,0.0\n"
        "TileA,ext_i,False,2.0\n"
    )
    (tmp_path / "band_inventory.csv").write_text(
        "tile_name,completed_bands,fully_complete\n"
        "TileA,vis;nir_h;nir_j,True\n"
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts" / "tools" / "catalogue_post_build_check.py"
    )
    out = tmp_path / "report"
    subprocess.run(
        [sys.executable, str(script), "--catalogue_dir", str(tmp_path),
         "--out_dir", str(out)],
        check=True,
    )
    with (out / "magnitude_rows_short.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [{
            "tile_name": "TileA",
            "completed_band_count": "3",
            "magnitude_band_count": "3",
            "missing_magnitude_bands": "nir_h",
            "qa_fully_complete": "True",
        }]
