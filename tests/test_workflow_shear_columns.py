"""Workflow shear-column helpers accept real AggregateCSV SearchOutput rows."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).parent.parent


def _function_from(relative_path, name):
    """Load one helper without executing the workflow script's top-level query."""
    path = PROJECT_ROOT / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _result_with(instance):
    samples = SimpleNamespace(median_pdf=lambda: instance)
    # AggregateCSV passes a SearchOutput row. Deliberately provide
    # ``median_pdf`` only through ``row.samples`` so a direct call on the row
    # fails this test.
    return SimpleNamespace(samples=samples)


@pytest.mark.parametrize(
    "relative_path,function_name,arguments,expected",
    (
        ("workflow/csv_make.py", "shear_magnitude_from", (), 0.05),
        (
            "workflow/example/csv/lens_mass.py",
            "shear_component_from",
            ("gamma_1",),
            0.03,
        ),
        (
            "workflow/example/csv/lens_mass.py",
            "shear_component_from",
            ("gamma_2",),
            -0.04,
        ),
    ),
)
def test_shear_helpers_read_current_and_legacy_search_outputs(
    relative_path, function_name, arguments, expected
):
    helper = _function_from(relative_path, function_name)
    shear = SimpleNamespace(gamma_1=0.03, gamma_2=-0.04, magnitude=0.05)

    current = SimpleNamespace(fields=SimpleNamespace(shear=shear))
    legacy = SimpleNamespace(
        galaxies=SimpleNamespace(lens=SimpleNamespace(shear=shear)),
    )

    assert helper(_result_with(current), *arguments) == pytest.approx(expected)
    assert helper(_result_with(legacy), *arguments) == pytest.approx(expected)
