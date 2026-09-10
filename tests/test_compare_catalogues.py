"""
The row-level catalogue comparator's verdicts.

``scripts/tools/compare_catalogues.py`` is the tool that decides whether a
rebuilt DR1 catalogue is the same catalogue as the June reference. Nothing else
in this repository reads catalogue *values*: ``tests/test_catalogue_parity.py``
compares producer headers out of the syntax tree, and
``tests/test_catalogue_latent_columns.py`` runs the producers and asserts their
cells are non-blank. So the comparator is the only thing standing between a
plausible-looking rebuild and a science ruling, and its verdicts are what this
module pins.

Five cases, one per rule the comparator implements, each built from a pair of
tiny CSVs written here rather than from a real bundle:

* ``test_identical_catalogues_pass`` — the same numbers on both sides is a PASS
  and exit code 0. Without this the other four cases could all be satisfied by a
  comparator that fails everything.
* ``test_einstein_radius_outside_three_sigma_fails`` — one tile moved far
  outside the combined 3σ fails, **and the failing tile is named in the report**.
  A comparator that fails without saying which row is not usable.
* ``test_swapped_mge_bases_pass`` — ``ell_comps_0`` and ``ell_comps_1``
  exchanged is a PASS, because ``order_bases=True`` deliberately re-labels the
  bases; comparing label to label rather than set to set would fail every
  ordered rebuild.
* ``test_blank_latent_cell_fails`` — a blank ``effective_einstein_radius`` cell
  fails. This is the ``latent.`` prefix regression, which is written blank
  rather than raised.
* ``test_three_sigma_equal_to_one_sigma_fails`` — 3σ bounds equal to the 1σ
  bounds fail. This is the 2026-09-10 ``row.py:102`` regression, which shipped a
  published DR1 catalogue whose latent error bars were understated threefold.

The comparator imports no PyAuto library and runs no fit, so these are pure-CSV
tests: fast, and safe in the ``not slow`` suite.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent

TILE_A = "Tile102005065RA0135279431487DECNEG0701599765928"
TILE_B = "Tile102007299RA0702283866574DECNEG0660415308762"

# The value bases the synthetic lens_mass.csv carries: the two elliptical
# components and the Einstein radius are sampled model parameters, and
# `effective_einstein_radius` is the latent whose blank cells this comparator
# was written to catch.
BASES = ("ell_comps_0", "ell_comps_1", "einstein_radius", "effective_einstein_radius")

VALUE_SUFFIXES = ("_lower_1_sigma", "_upper_1_sigma", "_lower_3_sigma", "_upper_3_sigma")


@pytest.fixture(scope="module")
def comparator():
    """
    ``scripts/tools/compare_catalogues.py`` loaded by path: it is a script run
    with ``python scripts/tools/...``, not an importable package member, so it
    is loaded the way ``tests/test_catalogue_latent_columns.py`` loads a
    producer.
    """
    path = PROJECT_ROOT / "scripts" / "tools" / "compare_catalogues.py"
    spec = importlib.util.spec_from_file_location("_compare_catalogues_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns():
    """
    The header of the synthetic ``lens_mass.csv``: the two label columns, then
    five columns per base, in the order ``AggregateCSV`` writes them.
    """
    header = ["id", "lens_name"]
    for base in BASES:
        header.append(base)
        header += [f"{base}{suffix}" for suffix in VALUE_SUFFIXES]
    return header


def _row(lens_name, values, *, blank=(), collapsed=()):
    """
    One CSV row from ``{base: (median, sigma)}``.

    The four bound columns are built from the median and the sigma, so a row is
    self-consistent by construction and a test that wants an *in*consistent one
    has to ask for it: ``blank`` empties every column of a base (the latent
    prefix bug) and ``collapsed`` writes the 3σ bounds equal to the 1σ bounds
    (the ``row.py:102`` bug).
    """
    row = {"id": f"{lens_name}_id", "lens_name": lens_name}
    for base, (median, sigma) in values.items():
        if base in blank:
            row[base] = ""
            for suffix in VALUE_SUFFIXES:
                row[f"{base}{suffix}"] = ""
            continue
        row[base] = repr(median)
        row[f"{base}_lower_1_sigma"] = repr(median - sigma)
        row[f"{base}_upper_1_sigma"] = repr(median + sigma)
        if base in collapsed:
            row[f"{base}_lower_3_sigma"] = repr(median - sigma)
            row[f"{base}_upper_3_sigma"] = repr(median + sigma)
        else:
            row[f"{base}_lower_3_sigma"] = repr(median - 3 * sigma)
            row[f"{base}_upper_3_sigma"] = repr(median + 3 * sigma)
    return row


def _write_catalogue(directory, rows):
    """
    Write ``lens_mass.csv`` into ``directory`` — the one product the comparator
    requires on both sides.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "lens_mass.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_columns())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return directory


def _baseline_values(offset=0.0):
    """
    A self-consistent set of values, with a sigma small enough that a 0.5 shift
    is many sigma away and large enough that the 3σ interval is not degenerate.
    """
    return {
        "ell_comps_0": (0.10 + offset, 0.01),
        "ell_comps_1": (-0.20 + offset, 0.01),
        "einstein_radius": (1.50 + offset, 0.02),
        "effective_einstein_radius": (1.40 + offset, 0.02),
    }


def _run(comparator, directory_a, directory_b, tmp_path, extra=()):
    """
    Run the comparator's ``main`` over the two directories and return
    ``(exit_code, report_text)``. The report is read back from ``--report``
    rather than captured off stdout, because the file is what a rebuild keeps.
    """
    report = tmp_path / "report.md"
    code = comparator.main(
        [str(directory_a), str(directory_b), "--report", str(report), *extra]
    )
    return code, report.read_text()


def _status_of(report, check_name):
    """
    The status the summary table records for one check.
    """
    for line in report.splitlines():
        if line.startswith(f"| {check_name} |"):
            return line.split("|")[2].strip()
    raise AssertionError(f"no '{check_name}' row in the report summary:\n{report}")


def test_identical_catalogues_pass(comparator, tmp_path, capsys):
    """
    Two identical catalogues are a PASS with exit code 0.

    This is the control: every other case here asserts a FAIL, and a comparator
    that returned FAIL unconditionally would satisfy all of them.
    """
    rows = [
        _row(TILE_A, _baseline_values()),
        _row(TILE_B, _baseline_values(offset=0.3)),
    ]
    directory_a = _write_catalogue(tmp_path / "a", rows)
    directory_b = _write_catalogue(tmp_path / "b", rows)

    code, report = _run(comparator, directory_a, directory_b, tmp_path)

    assert "RESULT: PASS" in report
    assert code == 0
    assert _status_of(report, "lens_mass: tile identity (exact)") == "PASS"
    assert _status_of(report, "lens_mass: declared quantities (combined 3σ)") == "PASS"
    assert _status_of(report, "lens_mass: MGE ell_comps (set swap)") == "PASS"
    assert _status_of(report, "lens_mass: latent completeness (A only)") == "PASS"


def test_einstein_radius_outside_three_sigma_fails(comparator, tmp_path):
    """
    One tile whose ``effective_einstein_radius`` has moved far outside the
    combined 3σ fails, and the report names that tile.

    ``effective_einstein_radius`` is the gated quantity; the other tile is left
    identical so the failure cannot be blamed on the whole table.
    """
    good = _row(TILE_A, _baseline_values())

    values_b = _baseline_values(offset=0.3)
    moved = dict(values_b)
    # 0.5 against a 0.02 sigma on each side is z ≈ 18, far outside the 3 the
    # comparator allows.
    moved["effective_einstein_radius"] = (
        values_b["effective_einstein_radius"][0] + 0.5,
        values_b["effective_einstein_radius"][1],
    )

    directory_a = _write_catalogue(tmp_path / "a", [good, _row(TILE_B, moved)])
    directory_b = _write_catalogue(tmp_path / "b", [good, _row(TILE_B, values_b)])

    code, report = _run(comparator, directory_a, directory_b, tmp_path)

    assert "RESULT: FAIL" in report
    assert code == 1
    assert _status_of(report, "lens_mass: declared quantities (combined 3σ)") == "FAIL"
    assert TILE_B in report, "the failing tile must be named, not just counted"

    # The tile that agrees is not dragged into the failure.
    outside_section = report.split("### lens_mass: declared quantities")[1].split("###")[0]
    assert TILE_A not in outside_section


def test_swapped_mge_bases_pass(comparator, tmp_path):
    """
    ``ell_comps_0`` and ``ell_comps_1`` exchanged between the two catalogues is
    a PASS: ``order_bases=True`` re-labels the bases on purpose, so the pair is
    compared as a set.

    The rest of the row is left alone, so a comparator that passed this by
    ignoring ``ell_comps`` altogether would still have to fail the other cases.
    """
    values = _baseline_values()
    swapped = dict(values)
    swapped["ell_comps_0"], swapped["ell_comps_1"] = (
        values["ell_comps_1"],
        values["ell_comps_0"],
    )

    directory_a = _write_catalogue(tmp_path / "a", [_row(TILE_A, swapped)])
    directory_b = _write_catalogue(tmp_path / "b", [_row(TILE_A, values)])

    code, report = _run(comparator, directory_a, directory_b, tmp_path)

    assert _status_of(report, "lens_mass: MGE ell_comps (set swap)") == "PASS"
    assert "RESULT: PASS" in report
    assert code == 0

    # And the swap is reported as a swap, not as an identity match: a build
    # where every row needed the permutation is worth seeing.
    section = report.split("### lens_mass: MGE ell_comps")[1].split("###")[0]
    assert "| ell_comps | 1 | 0 | 1 | 0 |" in section


def test_blank_latent_cell_fails(comparator, tmp_path):
    """
    A blank ``effective_einstein_radius`` in catalogue A fails the
    latent-completeness check — the ``latent.`` prefix regression, which
    ``add_variable`` writes as an empty cell rather than raising.
    """
    values = _baseline_values()
    directory_a = _write_catalogue(
        tmp_path / "a", [_row(TILE_A, values, blank=("effective_einstein_radius",))]
    )
    directory_b = _write_catalogue(tmp_path / "b", [_row(TILE_A, values)])

    code, report = _run(comparator, directory_a, directory_b, tmp_path)

    assert _status_of(report, "lens_mass: latent completeness (A only)") == "FAIL"
    assert "RESULT: FAIL" in report
    assert code == 1
    assert "blank" in report


def test_three_sigma_equal_to_one_sigma_fails(comparator, tmp_path):
    """
    3σ bounds equal to the 1σ bounds fail: no sampler produces that posterior,
    and the DR1 catalogue published on 2026-09-10 did exactly this, understating
    every latent error bar by a factor of three.

    The medians are identical on both sides, so nothing but the collapsed bounds
    can be what fails.
    """
    values = _baseline_values()
    directory_a = _write_catalogue(
        tmp_path / "a", [_row(TILE_A, values, collapsed=("effective_einstein_radius",))]
    )
    directory_b = _write_catalogue(tmp_path / "b", [_row(TILE_A, values)])

    code, report = _run(comparator, directory_a, directory_b, tmp_path)

    assert _status_of(report, "lens_mass: latent completeness (A only)") == "FAIL"
    assert _status_of(report, "lens_mass: declared quantities (combined 3σ)") == "PASS"
    assert "RESULT: FAIL" in report
    assert code == 1
    assert "3σ not outside 1σ" in report


def test_tiles_filter_restricts_the_comparison(comparator, tmp_path):
    """
    ``--tiles`` scopes every check to the named tiles, which is how a rebuild in
    progress is compared against the full reference without the tiles it has not
    reached yet counting against it.
    """
    values = _baseline_values()
    moved = dict(values)
    moved["effective_einstein_radius"] = (values["effective_einstein_radius"][0] + 0.5, 0.02)

    directory_a = _write_catalogue(
        tmp_path / "a", [_row(TILE_A, values), _row(TILE_B, moved)]
    )
    directory_b = _write_catalogue(
        tmp_path / "b", [_row(TILE_A, values), _row(TILE_B, values)]
    )

    code, report = _run(comparator, directory_a, directory_b, tmp_path, extra=[f"--tiles={TILE_A}"])

    assert "RESULT: PASS" in report
    assert code == 0
    assert TILE_B not in report
