"""
The catalogue contract of ``catalogue/scripts/witt_wynne.py``.

Stage 7 of the inspection bundle projects each lens's maximum-likelihood mass
model onto a Witt-Wynne SIEP and writes a ``witt_wynne.in`` file plus one
``witt_wynne.csv`` row per lens. The numerics of that projection are pinned by
``tests/test_witt_wynne_util.py``; what is pinned **here** is everything the
catalogue depends on and a fit cannot tell you:

1. **The column contract.** ``CSV_COLUMNS`` is the header the bundle promises.
   A reordering or a silent extra column produces a catalogue that still builds
   and no longer parses.
2. **Where the source position comes from.** ``source_centre_from_wcs`` is the
   whole of the ``wcs.json`` route, including the two vintages of that file and
   the ``(None, "none")`` that sends a lens to the recompute fallback.
3. **The row.** ``row_from`` has to fill four image slots in **arrival order**,
   leave unused slots blank, and turn a sentinel model into a full-width row of
   blanks rather than a crash or a dropped lens.
4. **The defaults.** ``--search_name=vis_pix`` and ``--projection=caustic`` are
   what makes this producer describe the same tracer as ``lens_mass.csv``.
5. **The wiring.** The bundle script has to actually call the producer, and its
   stage echoes have to be a consecutive ``1..9`` after the renumber.

6. **The empty-query skip.** ``dataset_names_from`` lists every directory under
   ``output/<sample>/``, so a lens with results from another stage only, or one
   whose fit is still in flight, reaches the producer with an empty aggregator
   query. That has to be a per-lens skip, because the bundle runs under
   ``set -e``.

JAX-free and fit-free: no non-linear search runs and no fit is read. Every test
but the empty-query one imports only the ``numpy`` ``witt_wynne_util`` already
needs; that one runs ``main()`` over a temporary output tree with no results in
it, which imports ``autolens`` but reads nothing.
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "catalogue" / "scripts"))

_SPEC = importlib.util.spec_from_file_location(
    "_witt_wynne_under_test",
    PROJECT_ROOT / "catalogue" / "scripts" / "witt_wynne.py",
)
witt_wynne = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(witt_wynne)

import witt_wynne_util  # noqa: E402  (loaded through the same path insert)


BUNDLE_SCRIPT = PROJECT_ROOT / "scripts" / "build_inspection_bundle.sh"

TOTAL_STAGES = 10


def _model():
    """
    A projected model whose source is **inside** the caustic, so it solves to
    four images with four different arrival times — the case the row's slot
    ordering has to get right.
    """
    return witt_wynne_util.WittWynne(
        centre=(0.0, 0.0),
        source=(0.04, 0.03),
        e=7.965893e-02,
        b=1.187194,
        pa_deg=112.790997,
        d_ol=878.6177660230832,
        d_ls=686.4784027811058,
        z_lens=0.5,
        z_source=1.5,
        h=0.6774,
    )


def _sentinel():
    """
    The model ``witt_wynne_util`` returns for a lens it cannot project: NaN
    parameters, ``valid=False`` and a reason. The source and the distances
    survive, so the row still identifies the lens.
    """
    nan = float("nan")
    return witt_wynne_util.WittWynne(
        centre=(nan, nan),
        source=(0.04, 0.03),
        e=nan,
        b=nan,
        pa_deg=nan,
        d_ol=878.6177660230832,
        d_ls=686.4784027811058,
        z_lens=0.5,
        z_source=1.5,
        h=0.6774,
        valid=False,
        reason="no Isothermal/PowerLaw-family mass profile in the tracer",
    )


def _row(model, **kwargs):
    images = model.zeroed().images()
    return witt_wynne.row_from(
        lens_name="Tile000",
        model=model,
        images=images,
        projection=kwargs.pop("projection", "caustic"),
        source_rule=kwargs.pop("source_rule", "brightest_clump_peak"),
        search_name=kwargs.pop("search_name", "vis_pix"),
        **kwargs,
    )


def _column(row, name):
    return row[witt_wynne.CSV_COLUMNS.index(name)]


"""
__The Column Contract__
"""


def test_csv_columns_are_the_published_header_in_order():
    assert witt_wynne.CSV_COLUMNS == [
        "lens_name",
        "projection",
        "valid",
        "reason",
        "b_arcsec",
        "e_gravlens",
        "pa_deg_E_of_N",
        "source_dx_arcsec",
        "source_dy_arcsec",
        "source_rule",
        "n_images",
        "image1_x",
        "image2_x",
        "image3_x",
        "image4_x",
        "image1_y",
        "image2_y",
        "image3_y",
        "image4_y",
        "mag1",
        "mag2",
        "mag3",
        "mag4",
        "lag1_days",
        "lag2_days",
        "lag3_days",
        "lag4_days",
        "z_lens",
        "z_source",
        "redshift_source",
        "d_ol_hinv_mpc",
        "d_ls_hinv_mpc",
        "h",
        "mass_profile",
        "search_name",
    ]


def test_lens_name_is_first_because_the_per_tile_split_groups_on_it():
    # catalogue_util.write_per_tile_csv groups the master CSV on `lens_name`;
    # a row without it in the first column lands in no lens folder at all.
    assert witt_wynne.CSV_COLUMNS[0] == "lens_name"
    assert len(witt_wynne.CSV_COLUMNS) == len(set(witt_wynne.CSV_COLUMNS))


"""
__The Source Position Read Out Of wcs.json__
"""


def test_source_centre_from_wcs_reads_a_light_profile_centre():
    centre, rule = witt_wynne.source_centre_from_wcs(
        {
            "crval_ra_deg": 13.5,
            "source_model": "light_profile",
            "source_centre_y_arcsec": 0.21,
            "source_centre_x_arcsec": -0.13,
        }
    )

    assert centre == (0.21, -0.13)
    assert rule == "light_profile_centre"


def test_source_centre_from_wcs_reads_the_brightest_clump_peak():
    centre, rule = witt_wynne.source_centre_from_wcs(
        {
            "source_model": "pixelized",
            "source_clump_rule": "percentile",
            "source_clumps": [
                {"peak_y_arcsec": 0.0974, "peak_x_arcsec": -0.1079, "peak_value": 0.39},
                {"peak_y_arcsec": 0.0770, "peak_x_arcsec": -0.0160, "peak_value": 0.13},
            ],
        }
    )

    # The first clump is the brightest: `pixelized_source_clumps_from` orders
    # them that way, and the peak is the brightest mesh pixel, not the mean.
    assert centre == (0.0974, -0.1079)
    assert rule == "brightest_clump_peak"


@pytest.mark.parametrize(
    "wcs_dict",
    [
        {},
        None,
        # The pre-2026-09-12 vintage: the four WCS numbers and nothing else.
        {
            "crpix_x": 50.5,
            "crpix_y": 50.5,
            "crval_ra_deg": 13.5,
            "crval_dec_deg": -70.2,
        },
        # Nothing was reconstructed, so the envelope wrote an empty list.
        {"source_model": "none", "source_clumps": []},
        # A light-profile source whose centre key the JSON envelope dropped.
        {"source_model": "light_profile"},
    ],
)
def test_source_centre_from_wcs_returns_none_when_there_is_no_source(wcs_dict):
    assert witt_wynne.source_centre_from_wcs(wcs_dict) == (None, "none")


"""
__The Row__
"""


def test_row_from_a_valid_model_is_exactly_one_cell_per_column():
    row = _row(_model())

    assert len(row) == len(witt_wynne.CSV_COLUMNS)
    assert _column(row, "lens_name") == "Tile000"
    assert _column(row, "valid") == "True"
    assert _column(row, "n_images") == "4"
    assert _column(row, "projection") == "caustic"
    assert _column(row, "source_rule") == "brightest_clump_peak"
    assert _column(row, "search_name") == "vis_pix"
    assert _column(row, "redshift_source") == "placeholder"
    assert float(_column(row, "b_arcsec")) == pytest.approx(1.187194)
    assert float(_column(row, "e_gravlens")) == pytest.approx(0.07965893)
    assert float(_column(row, "pa_deg_E_of_N")) == pytest.approx(112.791, abs=1e-3)


def test_row_from_fills_the_image_slots_in_arrival_order():
    row = _row(_model())

    lags = [float(_column(row, f"lag{i}_days")) for i in range(1, 5)]

    # Ascending, and referred to the leading image, so the first lag is zero.
    assert lags == sorted(lags)
    assert lags[0] == pytest.approx(0.0)
    assert lags[-1] > lags[0]

    # The solver returns the images in its own quartic-root order, which is
    # *not* the arrival order — this is the re-sort actually happening.
    _, _, _, _ = _model().zeroed().images()
    x_solver, y_solver, _, _ = _model().zeroed().images()
    unsorted_lags = list(_model().zeroed().lags(x_solver, y_solver))
    assert unsorted_lags != sorted(unsorted_lags)

    # Every image cell of a 4-image row is filled.
    for i in range(1, 5):
        assert _column(row, f"image{i}_x") != ""
        assert _column(row, f"image{i}_y") != ""
        assert _column(row, f"mag{i}") != ""


def test_row_from_leaves_unused_image_slots_blank():
    # The same lens with the source moved well outside the caustic: two images.
    model = witt_wynne_util.WittWynne(
        centre=(0.0, 0.0),
        source=(0.5, 0.4),
        e=7.965893e-02,
        b=1.187194,
        pa_deg=112.790997,
        d_ol=878.6177660230832,
        d_ls=686.4784027811058,
        z_lens=0.5,
        z_source=1.5,
        h=0.6774,
    )

    row = _row(model)

    assert _column(row, "n_images") == "2"
    assert len(row) == len(witt_wynne.CSV_COLUMNS)

    for i in (1, 2):
        assert _column(row, f"image{i}_x") != ""
        assert _column(row, f"lag{i}_days") != ""

    for i in (3, 4):
        assert _column(row, f"image{i}_x") == ""
        assert _column(row, f"image{i}_y") == ""
        assert _column(row, f"mag{i}") == ""
        assert _column(row, f"lag{i}_days") == ""


def test_row_from_a_sentinel_is_blank_numbers_and_a_reason_not_a_dropped_lens():
    row = _row(_sentinel(), source_rule="none")

    assert len(row) == len(witt_wynne.CSV_COLUMNS)
    assert _column(row, "lens_name") == "Tile000"
    assert _column(row, "valid") == "False"
    assert _column(row, "reason")

    # A sentinel's e/b/PA are NaN, so they are written blank, never "nan".
    for name in ("b_arcsec", "e_gravlens", "pa_deg_E_of_N"):
        assert _column(row, name) == ""

    # Its solve is degenerate, so the verdict is the -1 sentinel and every
    # image, magnification and lag cell is blank.
    assert _column(row, "n_images") == str(witt_wynne_util.N_IMAGES_SENTINEL)
    for i in range(1, 5):
        assert _column(row, f"image{i}_x") == ""
        assert _column(row, f"image{i}_y") == ""
        assert _column(row, f"mag{i}") == ""
        assert _column(row, f"lag{i}_days") == ""

    # The fiducial cosmology it would have used is still recorded.
    assert float(_column(row, "z_lens")) == pytest.approx(0.5)
    assert float(_column(row, "d_ol_hinv_mpc")) > 0.0

    assert "nan" not in ",".join(row).lower()


def test_row_from_zero_centres_so_no_sky_position_reaches_the_catalogue():
    offset = witt_wynne_util.WittWynne(
        centre=(-0.2, 0.3),
        source=(-0.16, 0.33),
        e=7.965893e-02,
        b=1.187194,
        pa_deg=112.790997,
        d_ol=878.6177660230832,
        d_ls=686.4784027811058,
        z_lens=0.5,
        z_source=1.5,
        h=0.6774,
    )

    row = _row(offset)

    # The source offset is written relative to the lens, which the .in file
    # also does (`zero_centre=True`), so the two can never disagree.
    assert float(_column(row, "source_dx_arcsec")) == pytest.approx(0.04)
    assert float(_column(row, "source_dy_arcsec")) == pytest.approx(0.03)


"""
__The Defaults__
"""


def test_parse_args_defaults_match_the_stage_the_rest_of_the_catalogue_reports(
    monkeypatch,
):
    monkeypatch.setattr(sys, "argv", ["witt_wynne.py"])

    args = witt_wynne.parse_args()

    assert args.sample == "q1_walsmley"
    assert args.output_path == "output"
    assert args.inspect_dir is None
    assert args.unique_tag == "initial_lens_model"
    assert args.search_name == "vis_pix"
    assert args.z_lens == 0.5
    assert args.z_source == 1.0
    assert args.projection == "caustic"
    assert args.caustic_pixel_scale == 0.05


def test_parse_args_rejects_an_unknown_projection(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["witt_wynne.py", "--projection=astroid"])

    with pytest.raises(SystemExit):
        witt_wynne.parse_args()


def test_parse_args_rejects_redshifts_that_are_not_ordered(monkeypatch):
    for argv in [
        ["witt_wynne.py", "--z_lens=1.0", "--z_source=0.5"],
        ["witt_wynne.py", "--z_lens=0.5", "--z_source=0.5"],
        ["witt_wynne.py", "--z_lens=-0.5"],
    ]:
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit):
            witt_wynne.parse_args()


"""
__The Empty Query__

`dataset_names_from` lists every directory under `output/<sample>/`, so a lens
with only another stage's results, or one whose fit is still running, reaches
the per-lens loop with an empty aggregator query. `next(iter(...))` on one of
those raises `StopIteration`, which the per-lens guard did not catch and which
would abort stage 7 -- and with it every later stage, because the bundle runs
under `set -e`.
"""


def test_a_lens_with_no_completed_fit_is_skipped_not_raised(
    tmp_path, monkeypatch, capsys
):
    output_path = tmp_path / "output"
    inspect_path = tmp_path / "inspect"

    (output_path / "no_fit_sample" / "lens_0000").mkdir(parents=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "witt_wynne.py",
            "--sample=no_fit_sample",
            f"--output_path={output_path}",
            f"--inspect_dir={inspect_path}",
        ],
    )

    witt_wynne.main()

    out = capsys.readouterr().out

    assert "skipping lens_0000: no completed initial_lens_model/vis_pix fit" in out
    assert "no lenses projected; no witt_wynne.csv written" in out
    assert not (inspect_path / "witt_wynne.csv").exists()


"""
__The Bundle Wiring__
"""


def test_the_bundle_script_runs_the_producer():
    script = BUNDLE_SCRIPT.read_text()

    assert "catalogue/scripts/witt_wynne.py" in script


def test_the_bundle_stage_echoes_are_a_consecutive_run_of_ten():
    script = BUNDLE_SCRIPT.read_text()

    stages = [int(n) for n in re.findall(r"\[(\d+)/%d" % TOTAL_STAGES, script)]

    # Every stage announces itself, and the SED stages announce themselves
    # again in each skip branch, so compare the *set*.
    assert sorted(set(stages)) == list(range(1, TOTAL_STAGES + 1))

    # No stale `[n/8]` marker survived the renumber.
    assert not re.findall(r"\[\d+/%d[,\]]" % (TOTAL_STAGES - 1), script)
