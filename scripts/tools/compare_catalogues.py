"""
Euclid Pipeline: Row-Level Catalogue Comparator
===============================================

Diff two built catalogues row by row and say, with a single PASS/FAIL, whether
the second reproduces the first.

``tests/test_catalogue_parity.py`` compares producer *headers*: it reads the
``add_variable`` calls out of the producers' syntax trees and checks the column
names they would write against stored DR1 fixtures. It runs nothing and reads no
values, so a rebuild that emits the right header full of wrong — or blank —
numbers passes it. This script is the other half: it reads the numbers.

The question it answers is the DR1 rebuild's: the ``euclid_dr1_prelim`` fits are
being repeated under ordered MGE bases and a low-disk output configuration, and
the rebuilt catalogue has to be shown to be the same catalogue as the June
reference at ``catalogue/catalogue/dr1_prelim_grade_ab_catalogue_csvs_20260623/``
before any science is read off it.

Usage
-----
::

    python scripts/tools/compare_catalogues.py <catalogue_dir_A> <catalogue_dir_B>
    python scripts/tools/compare_catalogues.py A B --tiles Tile102005065...,Tile102007299...
    python scripts/tools/compare_catalogues.py A B --report parity.md

Each directory is a bundle as ``scripts/build_inspection_bundle.sh`` writes one:
a ``lens_mass.csv`` at its root, and optionally ``lens_sersic.csv``,
``source_sersic.csv`` and ``magnitudes.csv`` beside it. ``lens_mass.csv`` is
required on both sides; every other product is compared when both sides have it
and reported as absent when they do not.

A is the catalogue under test and B the reference, and the two are not
symmetric: the latent-completeness checks below run on **A only**, because they
ask whether the new build produced usable numbers, not whether the old one did.

__What Counts As Agreement__

Two independent nested-sampling runs of the same model on the same data are not
bitwise reproducible, so most of this is a tolerance question rather than an
equality question. Four rules, one per kind of quantity:

**Tile identity — exact.** Rows are keyed by ``lens_name`` (and, for
``magnitudes.csv``, by ``(lens_name, waveband)``, which is the row identity that
producer writes). The key is compared as a string, with no normalisation: a tile
present on one side only is reported and fails the check. Only the tiles common
to both sides are carried into the value checks, so a partial rebuild is
compared on what it built rather than being drowned in absences — but the
identity check still records, and fails on, the difference.

**Astrometry — exact.** Any of ``crval_ra_deg``, ``crval_dec_deg``, ``ra`` and
``dec`` present on both sides must be equal as a float, with no tolerance: the
position of a lens on the sky is read from its WCS, not sampled, so a rerun that
moves it has read the wrong cut-out. Note that the DR1 ``lens_mass.csv`` carries
no astrometry column — its ``centre_0``/``centre_1`` are arcsecond offsets within
the cut-out, which *are* sampled — so for a lens_mass-only bundle this check is
reported as skipped. The tile name itself encodes the RA and Dec, and the exact
key match above is therefore already an exact astrometry match at tile
resolution.

**Values — combined 3σ.** For a value column ``x``, each side publishes
``x_lower_1_sigma`` and ``x_upper_1_sigma``, so::

    sigma = (x_upper_1_sigma - x_lower_1_sigma) / 2

is that side's 1σ half-width, and the two agree when::

    |a - b| <= 3 * sqrt(sigma_a**2 + sigma_b**2)

which is the "combined 3σ" of the issue. The ``_lower_3_sigma`` / ``_upper_3_sigma``
columns are not used for the tolerance — they are the posterior's own asymmetric
3σ bounds rather than a scale, and combining two asymmetric intervals has no
single defensible reading — but they *are* used by the latent-completeness
checks below.

The number this reduces to is reported for every quantity as::

    z = |a - b| / sqrt(sigma_a**2 + sigma_b**2)

and its median and maximum across the compared tiles are the rerun scatter a
science ruling quotes: z < 3 is the pass condition, a median z near 1 means the
two runs differ by about their own quoted error, and a median z near 0 would mean
they are far more alike than their error bars — which for independent sampling
runs is its own kind of surprise.

**MGE ``ell_comps`` — up to a set swap.** ``order_bases=True`` deliberately
re-labels which basis is which, so comparing ``ell_comps_0`` against
``ell_comps_0`` compares two labels rather than two numbers. Each group of
``…ell_comps_N`` columns is therefore compared as an unordered set: the pair
agrees if *some* permutation of A's entries matches B's within the combined 3σ
rule above. With two entries that is the identity and the swap, and the check
records which of the two matched.

__Which Quantities Gate The Result__

The issue's tolerances name tile identity, astrometry, ``effective_einstein_radius``
and the per-band magnitudes, and the ``ell_comps`` set swap. Those, plus the
latent-completeness checks, are the **gated** checks: they decide PASS/FAIL and
the exit code.

Every other value column — the sampled mass and light parameters, ``centre``,
``einstein_radius``, ``shear_gamma_*``, ``effective_radius``, ``sersic_index`` —
is measured and reported under the same z statistic as **informational**. They
are the rerun scatter of quantities nobody declared a tolerance for, which is
worth reading and is not worth failing a build over.

__Latent Completeness (Catalogue A Only)__

Three checks that need only one catalogue, all of them pins on regressions this
pipeline has actually shipped:

1. **No blank latent cell.** A latent whose ``add_variable`` argument misses is
   written *blank*, not raised — the ``latent.`` prefix bug that emptied
   ``effective_einstein_radius`` and the whole of ``magnitudes.csv`` while
   leaving the header at full DR1 width.

2. **3σ strictly outside 1σ.** ``lower_3_sigma < lower_1_sigma`` and
   ``upper_3_sigma > upper_1_sigma`` for every latent of every row. The
   2026-09-10 ``row.py:102`` regression published a DR1 catalogue whose latent
   3σ bounds were *equal* to its 1σ bounds, which is not a posterior any sampler
   produces and which silently understates every latent error bar by a factor of
   three.

3. **max_lh differs from the median somewhere.** Where a latent carries a
   ``_max_lh`` column, the maximum-likelihood value and the median of the
   posterior must differ for at least one (row, latent): they are computed from
   different samples, so a table where they agree everywhere is one where one of
   them was copied from the other. ``lens_mass.csv`` carries no ``_max_lh``
   column at all, so for a lens_mass-only bundle this check reports as skipped
   rather than failing.

A latent column is a value column whose base name is not one of the model
parameters listed in ``MODEL_PARAMETER_BASES`` below. The list is short and
closed because the catalogue's model columns are: a genuinely new model
parameter must be added to it, and until it is, it is checked as a latent — which
only ever means it is *also* required to be non-blank and to have real 3σ
bounds.

__Output__

A markdown report on stdout, and to ``--report`` when given: one table per check
with ``n_exact`` / ``n_within`` / ``n_outside`` counts and the tiles that fell
outside, the z-statistic summary, and a final ``RESULT: PASS`` or
``RESULT: FAIL`` line. Exit status is 1 on FAIL, 0 on PASS, so it can gate a
rebuild from a shell script.

No fit is run and no PyAuto library is imported: this reads CSVs.
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

# The products a bundle can hold, and the columns that identify a row in each.
# `lens_mass` is required; the rest are compared when both sides carry them.
PRODUCTS = {
    "lens_mass": ("lens_name",),
    "lens_sersic": ("lens_name",),
    "source_sersic": ("lens_name",),
    "magnitudes": ("lens_name", "waveband"),
}

REQUIRED_PRODUCT = "lens_mass"

# The five flavours `AggregateCSV` writes per variable. Stripping any of them
# off a column name leaves the variable's base name; a column carrying none of
# them is either the bare median or a label.
VALUE_SUFFIXES = (
    "_max_lh",
    "_lower_1_sigma",
    "_upper_1_sigma",
    "_lower_3_sigma",
    "_upper_3_sigma",
)

# Columns that identify a row rather than measuring anything.
LABEL_COLUMNS = ("id", "lens_name", "waveband")

# Astrometry is compared exactly. `crval_ra_deg` is the only one the DR1
# producers write today (in `magnitudes.csv`); the others are named so a future
# producer that adds them is covered without an edit here.
ASTROMETRY_COLUMNS = ("crval_ra_deg", "crval_dec_deg", "ra", "dec")

# Sampled model parameters. Everything else with a `_lower_1_sigma` column is a
# latent — see "__Latent Completeness__" in the module docstring.
MODEL_PARAMETER_BASES = frozenset(
    {
        "centre_0",
        "centre_1",
        "ell_comps_0",
        "ell_comps_1",
        "einstein_radius",
        "shear_gamma_1",
        "shear_gamma_2",
        "effective_radius",
        "sersic_index",
    }
)

# The quantities whose agreement the issue declares a tolerance for, and which
# therefore decide PASS/FAIL. Everything else is measured and reported without
# gating. `ell_comps` is gated too, but through the set-swap check rather than
# this list.
GATED_BASES = frozenset(
    {
        "effective_einstein_radius",
        "lens_flux",
        "lens_flux_1_fwhm",
        "lens_flux_2_fwhm",
        "lens_flux_3_fwhm",
        "lens_flux_4_fwhm",
        "lensed_source_flux",
        "source_flux",
        "magnification",
    }
)

SIGMA_MULTIPLE = 3.0

# A z statistic needs a scale. When both sides publish a zero 1σ half-width the
# ratio is undefined, so the pair is compared exactly instead and reported in
# its own count rather than being silently passed or silently failed.
ZERO_SIGMA = 0.0


def parse_args(argv=None):
    """
    Two catalogue directories, an optional tile filter and an optional report
    path. A is the catalogue under test, B the reference.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare two built Euclid catalogues row by row: tile identity and "
            "astrometry exactly, values within a combined 3 sigma, MGE "
            "ell_comps up to a set swap, plus latent-completeness checks on A."
        )
    )
    parser.add_argument(
        "catalogue_a",
        metavar="catalogue_dir_A",
        help="Directory of the catalogue under test (holds lens_mass.csv, ...).",
    )
    parser.add_argument(
        "catalogue_b",
        metavar="catalogue_dir_B",
        help="Directory of the reference catalogue.",
    )
    parser.add_argument(
        "--tiles",
        metavar="t1,t2,...",
        default=None,
        help=(
            "Comma-separated lens_name values to restrict the comparison to. "
            "Default: every tile the two catalogues have in common."
        ),
    )
    parser.add_argument(
        "--report",
        metavar="out.md",
        default=None,
        help="Also write the markdown report to this path.",
    )
    return parser.parse_args(argv)


def read_product(directory: Path, product: str):
    """
    ``(fieldnames, rows)`` for one product of a bundle, or ``None`` when that
    product is not in the directory. Rows are plain dicts of strings — nothing
    is coerced here, because a blank cell is a finding rather than a zero.
    """
    path = directory / f"{product}.csv"
    if not path.is_file():
        return None

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def row_key(row, key_columns):
    """
    The identity of a row: the tuple of its key columns, as strings.
    """
    return tuple((row.get(column) or "").strip() for column in key_columns)


def key_label(key):
    """
    A key rendered for a report table: the bare tile name for a one-column key,
    ``tile/band`` for the two-column ``magnitudes`` key.
    """
    return "/".join(key)


def base_name(column):
    """
    The variable a column belongs to: ``einstein_radius_lower_1_sigma`` and
    ``einstein_radius`` both belong to ``einstein_radius``.

    Suffixes are stripped by exact match, so ``lens_flux_1_fwhm_lower_1_sigma``
    resolves to ``lens_flux_1_fwhm`` rather than to ``lens_flux``.
    """
    for suffix in VALUE_SUFFIXES:
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return column


def value_bases(fieldnames):
    """
    The value variables of a header, in header order: every base that carries
    both a ``_lower_1_sigma`` and an ``_upper_1_sigma`` column, which is what
    makes a sigma — and so a tolerance — available for it.
    """
    present = set(fieldnames)
    bases = []
    for column in fieldnames:
        base = base_name(column)
        if base in bases or base in LABEL_COLUMNS:
            continue
        if f"{base}_lower_1_sigma" in present and f"{base}_upper_1_sigma" in present:
            bases.append(base)
    return bases


def ell_comps_groups(bases):
    """
    Group the ``…ell_comps_N`` bases by their prefix, so each group is one
    profile's set of elliptical components.

    Returns ``{prefix: [base, ...]}`` ordered by ``N``. The DR1 producers write
    a single unprefixed group of two, ``ell_comps_0`` and ``ell_comps_1``; the
    grouping is written generally so a producer that emits several ordered MGE
    bases is compared per profile rather than all in one pool.
    """
    groups = {}
    for base in bases:
        parts = base.rsplit("ell_comps_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        groups.setdefault(parts[0], []).append((int(parts[1]), base))

    return {
        prefix: [base for _index, base in sorted(entries)]
        for prefix, entries in groups.items()
    }


def as_float(value):
    """
    ``float(value)`` or ``None`` for a blank or unparseable cell. A blank cell is
    the shape of the latent bug, so it is never silently a zero.
    """
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return None if math.isnan(result) else result


def sigma_of(row, base):
    """
    The 1σ half-width ``(upper_1_sigma - lower_1_sigma) / 2``, or ``None`` when
    either bound is missing.

    ``abs`` guards the case of a producer writing the bounds the other way
    round: a negative scale would invert every tolerance silently.
    """
    lower = as_float(row.get(f"{base}_lower_1_sigma"))
    upper = as_float(row.get(f"{base}_upper_1_sigma"))
    if lower is None or upper is None:
        return None
    return abs(upper - lower) / 2.0


def z_statistic(value_a, sigma_a, value_b, sigma_b):
    """
    ``|a - b| / sqrt(sigma_a**2 + sigma_b**2)``, or ``None`` when the combined
    scale is zero — in which case there is nothing to normalise by and the pair
    has to be judged on exact equality instead.
    """
    scale = math.sqrt(sigma_a**2 + sigma_b**2)
    if scale <= ZERO_SIGMA:
        return None
    return abs(value_a - value_b) / scale


class Check:
    """
    One check's verdict: a name, a status, the markdown table it renders, and
    whether it gates the overall result.

    ``status`` is one of ``PASS``, ``FAIL``, ``SKIPPED`` (the inputs could not
    support the check — a missing product or a missing column) and ``INFO`` (a
    measurement that deliberately does not gate).
    """

    def __init__(self, name, status, lines, gating=True, note=None):
        self.name = name
        self.status = status
        self.lines = lines
        self.gating = gating
        self.note = note

    @property
    def failed(self):
        return self.gating and self.status == "FAIL"

    def render(self):
        out = [f"### {self.name} — {self.status}", ""]
        if self.note:
            out += [self.note, ""]
        out += self.lines
        out += [""]
        return out


def markdown_table(header, rows):
    """
    A markdown table, or a single italic line when there are no rows — an empty
    table renders as a header with nothing under it, which reads as a bug.
    """
    if not rows:
        return ["_no rows_"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def format_number(value, digits=4):
    """
    A number for a report cell; ``n/a`` for a missing one.
    """
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def check_identity(product, rows_a, rows_b, key_columns, tile_filter):
    """
    Tile identity, exactly, plus the reference coverage.

    Two questions live here and they are not the same question, so they are two
    checks:

    * **Identity gates.** Every row of A must exist in B under an exactly equal
      key. A row A produced that the reference has never heard of is an identity
      error — a mis-parsed tile name, a dataset directory renamed, a lens fitted
      from the wrong cut-out — and it fails.
    * **Coverage does not gate.** Rows of B that A has not produced are counted
      and reported, because the ordinary case is a rebuild in progress: five
      tiles of a 2990-tile reference is a valid comparison of five tiles, not a
      2985-way failure. The coverage fraction is the number a ruling reads to
      decide whether the rebuild is finished; it is not the number that decides
      whether it is *correct*.

    Returns the two checks and the common keys, in A's order and restricted by
    ``--tiles``, which are the rows every value check below compares.
    """
    keys_a = [row_key(row, key_columns) for row in rows_a]
    keys_b = {row_key(row, key_columns) for row in rows_b}

    if tile_filter is not None:
        keys_a = [key for key in keys_a if key[0] in tile_filter]
        keys_b = {key for key in keys_b if key[0] in tile_filter}

    set_a = set(keys_a)
    only_a = sorted(set_a - keys_b)
    only_b = sorted(keys_b - set_a)
    common = [key for key in keys_a if key in keys_b]

    identity_lines = markdown_table(
        ["quantity", "count"],
        [
            ["rows in A", len(keys_a)],
            ["n_exact (A rows matched in B)", len(set(common))],
            ["n_outside (in A only)", len(only_a)],
        ],
    )
    if only_a:
        identity_lines += [""]
        identity_lines += markdown_table(
            ["row in A with no counterpart in B"], [[key_label(key)] for key in only_a[:20]]
        )
        if len(only_a) > 20:
            identity_lines += ["", f"_… {len(only_a) - 20} further rows not listed_"]

    identity = Check(
        f"{product}: tile identity (exact)",
        "FAIL" if only_a else "PASS",
        identity_lines,
        note=(
            "Keys are compared as strings with no normalisation. Only the rows "
            "common to both sides are carried into the value checks below."
        ),
    )

    coverage_lines = markdown_table(
        ["quantity", "count"],
        [
            ["distinct rows in B (reference)", len(keys_b)],
            ["reproduced by A", len(set(common))],
            ["not yet in A", len(only_b)],
            [
                "coverage",
                f"{100.0 * len(set(common)) / len(keys_b):.1f}%" if keys_b else "n/a",
            ],
        ],
    )
    if only_b:
        coverage_lines += [""]
        coverage_lines += markdown_table(
            ["row in B not in A"], [[key_label(key)] for key in only_b[:20]]
        )
        if len(only_b) > 20:
            coverage_lines += ["", f"_… {len(only_b) - 20} further rows not listed_"]

    coverage = Check(
        f"{product}: reference coverage",
        "INFO" if only_b else "PASS",
        coverage_lines,
        gating=False,
        note=(
            "How much of the reference this build has reproduced. Informational: "
            "a partial rebuild is a valid comparison of the rows it did build."
        ),
    )

    return identity, coverage, common


def check_astrometry(product, index_a, index_b, common):
    """
    Astrometry, exactly. Skipped when neither catalogue publishes one of
    ``ASTROMETRY_COLUMNS`` — which is the case for a ``lens_mass``-only bundle.
    """
    columns = [
        column
        for column in ASTROMETRY_COLUMNS
        if any(column in row for row in (index_a.get(key) for key in common) if row)
        and any(column in row for row in (index_b.get(key) for key in common) if row)
    ]

    if not common:
        return Check(
            f"{product}: astrometry (exact)",
            "SKIPPED",
            ["_no rows in common_"],
        )

    if not columns:
        return Check(
            f"{product}: astrometry (exact)",
            "SKIPPED",
            [
                "_no astrometry column in this product; the exact tile-name match "
                "is the astrometric identity at tile resolution_"
            ],
        )

    table = []
    outside = []
    for column in columns:
        n_exact = 0
        n_outside = 0
        for key in common:
            value_a = as_float(index_a[key].get(column))
            value_b = as_float(index_b[key].get(column))
            if value_a is not None and value_b is not None and value_a == value_b:
                n_exact += 1
            else:
                n_outside += 1
                outside.append((column, key_label(key), index_a[key].get(column), index_b[key].get(column)))
        table.append([column, n_exact, n_outside])

    lines = markdown_table(["column", "n_exact", "n_outside"], table)
    if outside:
        lines += [""]
        lines += markdown_table(
            ["column", "row", "A", "B"], [list(entry) for entry in outside[:20]]
        )
        if len(outside) > 20:
            lines += ["", f"_… {len(outside) - 20} further rows not listed_"]

    return Check(
        f"{product}: astrometry (exact)",
        "FAIL" if outside else "PASS",
        lines,
    )


def compare_base(index_a, index_b, common, base):
    """
    The combined-3σ comparison of one value base across the common rows.

    Returns ``(n_within, n_outside, n_missing, outside, z_values)`` where
    ``outside`` is the list of ``(row, a, b, z)`` that failed and ``z_values``
    every z that could be formed. A row where either side is blank, or where
    both sigmas are zero and the values differ, counts as outside: a blank is
    exactly the failure this comparator exists to catch, so it is never a skip.
    """
    n_within = 0
    outside = []
    n_missing = 0
    z_values = []

    for key in common:
        row_a, row_b = index_a[key], index_b[key]
        value_a = as_float(row_a.get(base))
        value_b = as_float(row_b.get(base))
        sigma_a = sigma_of(row_a, base)
        sigma_b = sigma_of(row_b, base)

        if value_a is None or value_b is None or sigma_a is None or sigma_b is None:
            n_missing += 1
            outside.append((key, row_a.get(base), row_b.get(base), None))
            continue

        z = z_statistic(value_a, sigma_a, value_b, sigma_b)
        if z is None:
            # No scale on either side: the only defensible test is equality.
            if value_a == value_b:
                n_within += 1
            else:
                outside.append((key, value_a, value_b, None))
            continue

        z_values.append(z)
        if z <= SIGMA_MULTIPLE:
            n_within += 1
        else:
            outside.append((key, value_a, value_b, z))

    return n_within, len(outside), n_missing, outside, z_values


def _values_check(product, index_a, index_b, common, bases, title, gating):
    """
    The shared body of the gated and informational value checks: one table row
    per base with its counts and its z summary, then the rows that fell outside.
    """
    if not common or not bases:
        return Check(
            f"{product}: {title}",
            "SKIPPED",
            ["_no rows or no columns to compare_"],
            gating=gating,
        )

    table = []
    all_outside = []
    for base in bases:
        n_within, n_outside, n_missing, outside, z_values = compare_base(
            index_a, index_b, common, base
        )
        table.append(
            [
                base,
                len(common),
                n_within,
                n_outside,
                n_missing,
                format_number(statistics.median(z_values)) if z_values else "n/a",
                format_number(max(z_values)) if z_values else "n/a",
            ]
        )
        all_outside += [(base, *entry) for entry in outside]

    lines = markdown_table(
        [
            "quantity",
            "n_compared",
            "n_within_3sigma",
            "n_outside",
            "n_missing",
            "median z",
            "max z",
        ],
        table,
    )
    lines += [
        "",
        "_z = |a − b| / sqrt(sigma_a² + sigma_b²); the pass condition is z ≤ 3. "
        "`n_missing` counts rows where a value or a sigma was blank on one side, "
        "which is counted as outside._",
    ]

    if all_outside:
        lines += [""]
        lines += markdown_table(
            ["quantity", "row", "A", "B", "z"],
            [
                [base, key_label(key), format_number(as_float(str(a)) if a is not None else None), format_number(as_float(str(b)) if b is not None else None), format_number(z)]
                for base, key, a, b, z in all_outside[:20]
            ],
        )
        if len(all_outside) > 20:
            lines += ["", f"_… {len(all_outside) - 20} further rows not listed_"]

    status = "FAIL" if all_outside else "PASS"
    if not gating:
        status = "INFO" if all_outside else "PASS"

    return Check(f"{product}: {title}", status, lines, gating=gating)


def check_gated_values(product, index_a, index_b, common, bases):
    """
    The declared-tolerance quantities: ``effective_einstein_radius`` and the
    per-band magnitudes. These decide the result.
    """
    gated = [base for base in bases if base in GATED_BASES]
    return _values_check(
        product,
        index_a,
        index_b,
        common,
        gated,
        "declared quantities (combined 3σ)",
        gating=True,
    )


def check_informational_values(product, index_a, index_b, common, bases):
    """
    Every other value column, measured under the same statistic and reported
    without gating — the rerun scatter of quantities no tolerance was declared
    for. ``ell_comps`` is excluded: it has its own set-swap check.
    """
    ell_comps = {base for group in ell_comps_groups(bases).values() for base in group}
    rest = [base for base in bases if base not in GATED_BASES and base not in ell_comps]
    return _values_check(
        product,
        index_a,
        index_b,
        common,
        rest,
        "other quantities (informational, combined 3σ)",
        gating=False,
    )


def _permutations(items):
    """
    Every ordering of ``items``. Written out rather than imported so the cost of
    a large group is visible: the groups here have two entries.
    """
    if len(items) <= 1:
        return [list(items)]
    result = []
    for index, item in enumerate(items):
        rest = list(items[:index]) + list(items[index + 1 :])
        for tail in _permutations(rest):
            result.append([item] + tail)
    return result


def check_ell_comps(product, index_a, index_b, common, bases):
    """
    ``ell_comps`` up to a set swap: the group agrees when *some* permutation of
    A's entries matches B's within the combined 3σ rule.

    ``n_exact`` counts the rows that matched in the identity order (the labels
    agree) and ``n_within`` the rows that needed a permutation (the labels were
    swapped, which ``order_bases=True`` is entitled to do). Both pass; the split
    is reported because a build where every row needed the swap is telling you
    something about the ordering.
    """
    groups = ell_comps_groups(bases)
    if not common or not groups:
        return Check(
            f"{product}: MGE ell_comps (set swap)",
            "SKIPPED",
            ["_no ell_comps columns in this product_"],
        )

    table = []
    outside = []
    for prefix, group in sorted(groups.items()):
        label = f"{prefix}ell_comps" if prefix else "ell_comps"
        n_identity = 0
        n_swapped = 0
        n_outside = 0

        for key in common:
            row_a, row_b = index_a[key], index_b[key]
            values_a = [(as_float(row_a.get(base)), sigma_of(row_a, base)) for base in group]
            values_b = [(as_float(row_b.get(base)), sigma_of(row_b, base)) for base in group]

            if any(value is None or sigma is None for value, sigma in values_a + values_b):
                n_outside += 1
                outside.append((label, key_label(key), "blank cell", ""))
                continue

            matched_order = None
            for order in _permutations(list(range(len(group)))):
                agreed = True
                for position, index in enumerate(order):
                    value_a, sigma_a = values_a[position]
                    value_b, sigma_b = values_b[index]
                    z = z_statistic(value_a, sigma_a, value_b, sigma_b)
                    if z is None:
                        if value_a != value_b:
                            agreed = False
                            break
                    elif z > SIGMA_MULTIPLE:
                        agreed = False
                        break
                if agreed:
                    matched_order = order
                    break

            if matched_order is None:
                n_outside += 1
                outside.append(
                    (
                        label,
                        key_label(key),
                        ", ".join(format_number(value) for value, _ in values_a),
                        ", ".join(format_number(value) for value, _ in values_b),
                    )
                )
            elif matched_order == list(range(len(group))):
                n_identity += 1
            else:
                n_swapped += 1

        table.append([label, len(common), n_identity, n_swapped, n_outside])

    lines = markdown_table(
        ["group", "n_compared", "n_exact (same order)", "n_within (swapped)", "n_outside"],
        table,
    )
    if outside:
        lines += [""]
        lines += markdown_table(
            ["group", "row", "A", "B"], [list(entry) for entry in outside[:20]]
        )
        if len(outside) > 20:
            lines += ["", f"_… {len(outside) - 20} further rows not listed_"]

    return Check(
        f"{product}: MGE ell_comps (set swap)",
        "FAIL" if outside else "PASS",
        lines,
    )


def latent_bases(fieldnames):
    """
    The latent columns of a header: every value base that is not a sampled model
    parameter.
    """
    return [base for base in value_bases(fieldnames) if base not in MODEL_PARAMETER_BASES]


def check_latent_completeness(product, fieldnames, rows):
    """
    Catalogue A's latents must be present, must carry real 3σ bounds, and must
    not have had their max-likelihood value copied from their median. See
    "__Latent Completeness__" in the module docstring for why each of the three
    is a pin on a shipped regression.
    """
    bases = latent_bases(fieldnames)
    if not rows or not bases:
        return Check(
            f"{product}: latent completeness (A only)",
            "SKIPPED",
            ["_no rows, or no latent columns in this product_"],
        )

    present = set(fieldnames)
    table = []
    findings = []
    any_max_lh_column = False
    max_lh_differs = False

    for base in bases:
        columns = [base] + [
            f"{base}{suffix}" for suffix in VALUE_SUFFIXES if f"{base}{suffix}" in present
        ]
        n_blank = 0
        n_collapsed = 0

        has_max_lh = f"{base}_max_lh" in present
        any_max_lh_column = any_max_lh_column or has_max_lh

        for row in rows:
            key = row.get("lens_name", "?")

            blank = [column for column in columns if as_float(row.get(column)) is None]
            if blank:
                n_blank += 1
                findings.append((base, key, "blank", ", ".join(blank)))
                continue

            lower_1 = as_float(row.get(f"{base}_lower_1_sigma"))
            upper_1 = as_float(row.get(f"{base}_upper_1_sigma"))
            lower_3 = as_float(row.get(f"{base}_lower_3_sigma"))
            upper_3 = as_float(row.get(f"{base}_upper_3_sigma"))

            if None not in (lower_1, upper_1, lower_3, upper_3):
                if not (lower_3 < lower_1 and upper_3 > upper_1):
                    n_collapsed += 1
                    findings.append(
                        (
                            base,
                            key,
                            "3σ not outside 1σ",
                            f"1σ [{format_number(lower_1)}, {format_number(upper_1)}] "
                            f"3σ [{format_number(lower_3)}, {format_number(upper_3)}]",
                        )
                    )

            if has_max_lh:
                median = as_float(row.get(base))
                max_lh = as_float(row.get(f"{base}_max_lh"))
                if median is not None and max_lh is not None and median != max_lh:
                    max_lh_differs = True

        table.append([base, len(rows), n_blank, n_collapsed, "yes" if has_max_lh else "no"])

    lines = markdown_table(
        ["latent", "n_rows", "n_blank_rows", "n_3sigma_not_outside_1sigma", "has max_lh column"],
        table,
    )

    if any_max_lh_column and not max_lh_differs:
        findings.append(
            ("(all latents)", "-", "max_lh == median everywhere", "no latent moved")
        )

    if not any_max_lh_column:
        lines += [
            "",
            "_No `_max_lh` column in this product, so the max_lh-vs-median check "
            "is not applicable here (`lens_mass.csv` publishes medians only)._",
        ]

    if findings:
        lines += [""]
        lines += markdown_table(
            ["latent", "row", "finding", "detail"], [list(entry) for entry in findings[:20]]
        )
        if len(findings) > 20:
            lines += ["", f"_… {len(findings) - 20} further findings not listed_"]

    return Check(
        f"{product}: latent completeness (A only)",
        "FAIL" if findings else "PASS",
        lines,
    )


def compare(directory_a: Path, directory_b: Path, tile_filter=None):
    """
    Run every check over every product the two bundles share, and return the
    list of ``Check``s in report order.
    """
    checks = []

    if read_product(directory_a, REQUIRED_PRODUCT) is None:
        raise SystemExit(f"ERROR: no {REQUIRED_PRODUCT}.csv in {directory_a}")
    if read_product(directory_b, REQUIRED_PRODUCT) is None:
        raise SystemExit(f"ERROR: no {REQUIRED_PRODUCT}.csv in {directory_b}")

    for product, key_columns in PRODUCTS.items():
        table_a = read_product(directory_a, product)
        table_b = read_product(directory_b, product)

        if table_a is None or table_b is None:
            sides = []
            if table_a is None:
                sides.append("A")
            if table_b is None:
                sides.append("B")
            checks.append(
                Check(
                    f"{product}: present in both catalogues",
                    "SKIPPED",
                    [f"_{product}.csv absent from catalogue {' and '.join(sides)}_"],
                )
            )
            continue

        fieldnames_a, rows_a = table_a
        fieldnames_b, rows_b = table_b

        index_a = {row_key(row, key_columns): row for row in rows_a}
        index_b = {row_key(row, key_columns): row for row in rows_b}

        identity, coverage, common = check_identity(
            product, rows_a, rows_b, key_columns, tile_filter
        )
        checks.append(identity)
        checks.append(coverage)

        bases_b = set(value_bases(fieldnames_b))
        bases = [base for base in value_bases(fieldnames_a) if base in bases_b]

        checks.append(check_astrometry(product, index_a, index_b, common))
        checks.append(check_gated_values(product, index_a, index_b, common, bases))
        checks.append(check_ell_comps(product, index_a, index_b, common, bases))
        checks.append(check_latent_completeness(product, fieldnames_a, rows_a))
        checks.append(check_informational_values(product, index_a, index_b, common, bases))

    return checks


def render_report(directory_a, directory_b, tile_filter, checks):
    """
    The whole markdown report: a header naming the two catalogues, a summary
    table of every check's status, then each check in full, then the RESULT
    line.
    """
    failed = [check for check in checks if check.failed]

    lines = [
        "# Catalogue comparison",
        "",
        f"- **A (under test)**: `{directory_a}`",
        f"- **B (reference)**: `{directory_b}`",
        f"- **Tile filter**: {'none' if tile_filter is None else ', '.join(sorted(tile_filter))}",
        "",
        "## Summary",
        "",
    ]
    lines += markdown_table(
        ["check", "status", "gates result"],
        [[check.name, check.status, "yes" if check.gating else "no"] for check in checks],
    )
    lines += ["", "## Checks", ""]
    for check in checks:
        lines += check.render()

    lines += [
        "## Result",
        "",
        f"RESULT: {'FAIL' if failed else 'PASS'}"
        + (f" ({len(failed)} gating check(s) failed)" if failed else ""),
    ]
    return "\n".join(lines), bool(failed)


def main(argv=None):
    args = parse_args(argv)

    directory_a = Path(args.catalogue_a)
    directory_b = Path(args.catalogue_b)

    for directory in (directory_a, directory_b):
        if not directory.is_dir():
            raise SystemExit(f"ERROR: no such catalogue directory: {directory}")

    tile_filter = None
    if args.tiles:
        tile_filter = {tile.strip() for tile in args.tiles.split(",") if tile.strip()}

    checks = compare(directory_a, directory_b, tile_filter=tile_filter)
    report, failed = render_report(directory_a, directory_b, tile_filter, checks)

    print(report)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report + "\n")
        print(f"\nwrote {report_path}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
