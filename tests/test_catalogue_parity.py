"""
Column parity between this pipeline's catalogue producers and the DR1
reference catalogue.

The four DR1 CSV header lines are checked in verbatim under
``tests/data/dr1_headers/``. For each producer under ``catalogue/scripts/``
this module reconstructs the header ``af.AggregateCSV`` will write — from the
producer's own ``add_label_column`` / ``add_variable`` calls and the real
``autofit`` ``Column`` class — and asserts it reproduces the DR1 header
**exactly, order included**.

Why reconstruct rather than run the producers: each producer's column list is
a literal inside ``main()``, and ``main()`` needs a populated ``output/`` tree
and a live ``Aggregator`` to reach ``AggregateCSV``. So the column specs are
read out of the module's syntax tree (``ast`` — exact, not a regex over text),
and the *suffix* expansion is delegated to ``autofit``'s own
``Column.value()``, which is the single place that decides both which suffixes
a set of ``ValueType``s produces and what order they come out in. A change to
either side of that contract breaks this test rather than silently drifting the
catalogue.

JAX-free and sub-second: nothing is imported from the producers, nothing runs.
"""

import ast
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CATALOGUE_SCRIPTS = PROJECT_ROOT / "catalogue" / "scripts"
DR1_HEADERS = Path(__file__).parent / "data" / "dr1_headers"

# The producers whose CSV has a DR1 counterpart. `deblending.py` and
# `multi_wavelength.py` write products DR1 has no CSV for, and
# `catalogue_util.py` is a shared helper.
PRODUCERS = ["lens_mass", "lens_sersic", "source_sersic", "magnitudes"]


class _EmptyRow:
    """
    A ``Row`` stand-in whose lookups all miss.

    ``autofit``'s ``Column.value`` populates one dict entry per emitted column
    whether the lookup hits or misses, so an empty row yields exactly the
    suffixes — in exactly the order — that ``AggregateCSV`` would write, with no
    aggregator, no results and no disk access.
    """

    median_pdf_sample_kwargs: dict = {}
    max_likelihood_kwargs: dict = {}
    values_at_sigma_1_kwargs: dict = {}
    values_at_sigma_3_kwargs: dict = {}


def _column_names_from(name, value_type_names):
    """
    The CSV column names ``AggregateCSV`` emits for one variable, via the real
    ``autofit`` ``Column``.
    """
    from autofit.aggregator.summary.aggregate_csv.column import Column, ValueType

    column = Column(
        argument="unused",
        name=name,
        value_types=[ValueType[value_type] for value_type in value_type_names],
    )

    return [
        f"{column.name}_{suffix}" if suffix else column.name
        for suffix in column.value(_EmptyRow()).keys()
    ]


def _value_type_names(node, environment):
    """
    ``[ValueType.Median, ...]`` / ``(af.ValueType.MaxLogLikelihood, ...)`` /
    a name bound to one of those -> ``["Median", ...]``. ``None`` if the node is
    not a value-type sequence.
    """
    if isinstance(node, ast.Name):
        return environment.get(("value_types", node.id))

    if not isinstance(node, (ast.List, ast.Tuple)):
        return None

    names = []
    for element in node.elts:
        if not isinstance(element, ast.Attribute):
            return None
        names.append(element.attr)

    return names or None


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def _keyword(call, name):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def column_specs_from(path):
    """
    The ordered ``(column name, value type names)`` specs a producer registers
    on its ``AggregateCSV``, read from the module's syntax tree.

    ``value type names`` is ``None`` for a label column (``add_label_column``
    writes exactly one column, unsuffixed).

    Recognised forms, in source order:

    * ``agg_csv.add_label_column(name="lens_name", values=...)``
    * ``agg_csv.add_variable(argument=..., name=..., value_types=...)``
    * ``for argument, name in <list of (argument, name) pairs>:`` with an
      ``add_variable`` call in the body — the loop is unrolled over the literal.
    """
    tree = ast.parse(path.read_text(), filename=str(path))

    environment = {}
    specs = []

    def add_variable_spec(call, name, environment):
        value_types = _value_type_names(_keyword(call, "value_types"), environment)
        assert value_types is not None, (
            f"{path.name}: could not resolve the value_types of the "
            f"add_variable call producing column '{name}'"
        )
        specs.append((name, value_types))

    def visit(statements):
        for statement in statements:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    value_types = _value_type_names(statement.value, environment)
                    if value_types is not None:
                        environment[("value_types", target.id)] = value_types
                    literal = _literal(statement.value)
                    if literal is not None:
                        environment[("literal", target.id)] = literal
                continue

            if isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Call
            ):
                call = statement.value
                function = call.func
                if not isinstance(function, ast.Attribute):
                    continue
                if function.attr == "add_label_column":
                    name = _literal(_keyword(call, "name"))
                    if name is not None:
                        specs.append((name, None))
                elif function.attr == "add_variable":
                    name = _literal(_keyword(call, "name"))
                    if name is not None:
                        add_variable_spec(call, name, environment)
                continue

            if isinstance(statement, ast.For):
                pairs = (
                    environment.get(("literal", statement.iter.id))
                    if isinstance(statement.iter, ast.Name)
                    else None
                )
                if pairs is not None and isinstance(statement.target, ast.Tuple):
                    calls = [
                        node
                        for node in ast.walk(statement)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "add_variable"
                    ]
                    for call in calls:
                        for _argument, name in pairs:
                            add_variable_spec(call, name, environment)
                    continue

            for field in ("body", "orelse", "finalbody"):
                nested = getattr(statement, field, None)
                if nested:
                    visit(nested)

    visit(tree.body)

    return specs


def header_from(specs):
    """
    The CSV header line ``AggregateCSV.save`` writes for these column specs.
    ``Row.dict`` seeds every row with ``id``, so ``id`` is always column 0.
    """
    names = ["id"]

    for name, value_types in specs:
        if value_types is None:
            names.append(name)
        else:
            names.extend(_column_names_from(name, value_types))

    return ",".join(names)


@pytest.mark.parametrize("producer", PRODUCERS)
def test_catalogue_columns_reproduce_the_dr1_header(producer):
    """
    Deliverable: the euclid catalogue is column-for-column the DR1 catalogue.
    A renamed column, a reordered ``add_variable`` block, a dropped parameter or
    a changed ``value_types`` set all break this.
    """
    specs = column_specs_from(CATALOGUE_SCRIPTS / f"{producer}.py")

    assert specs, f"no AggregateCSV columns found in catalogue/scripts/{producer}.py"

    expected = (DR1_HEADERS / f"{producer}.txt").read_text().strip()

    assert header_from(specs) == expected


@pytest.mark.parametrize("producer", PRODUCERS)
def test_dr1_header_fixture_is_a_single_comma_separated_line(producer):
    """
    Guard on the fixtures themselves: a truncated or CRLF-mangled header would
    otherwise make the parity test above assert against nothing.
    """
    text = (DR1_HEADERS / f"{producer}.txt").read_text()

    assert "\r" not in text, f"{producer}.txt must be LF-only"
    assert text.strip().count("\n") == 0, f"{producer}.txt must be one header line"
    assert text.strip().startswith(
        "id,"
    ), f"{producer}.txt must start with the id column"
