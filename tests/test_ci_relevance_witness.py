"""Temporary live witness for PyAutoHeart #298; never merge this branch.

The base already pins the candidate reusable workflow. This PR changes only
this tests/ file, so the old smoke-only filter would skip both runner jobs.
"""
from pathlib import Path
import pytest


def _assert_candidate():
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml"
    assert workflow.read_text().count("smoke-tests.yml@e2f7589c676c2124bacca840fa0acccc0964ac2b") == 2


def test_unit_runner_reaches_tests_only_witness():
    _assert_candidate()


@pytest.mark.slow
def test_slow_runner_reaches_tests_only_witness():
    _assert_candidate()
