"""
Repository invariants the automated runs depend on.

Three things that are true today and that CI must keep true:

1. **Nothing auto-simulates.** Every fitting script in this pipeline reads a
   dataset off disk — the way a user with real Euclid data runs it. Simulation
   lives in exactly one place, ``scripts/simulator.py``, and is something you
   ask for. A script that quietly simulates a dataset when it cannot find one
   turns a broken data path into a green run.
2. **Every script is accounted for.** Each ``*.py`` under the executable trees
   is either on the smoke allowlist (``smoke_tests.txt``) or excluded in
   ``config/build/no_run.yaml`` with a written reason. New scripts otherwise
   join the repository with no automated coverage and nothing says so.
3. **The allowlist is not stale.** A ``smoke_tests.txt`` entry that no longer
   exists on disk is reported by the runner as ``FAIL (listed but not found)``,
   which is a slow way to learn about a rename.

No imports of pipeline code, no dataset, no fit — this module is pure file
inspection and runs in milliseconds.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SMOKE_TESTS = PROJECT_ROOT / "smoke_tests.txt"
NO_RUN = PROJECT_ROOT / "config" / "build" / "no_run.yaml"

# The only file allowed to simulate data.
SIMULATOR = PROJECT_ROOT / "scripts" / "simulator.py"

# Tokens that mean "this file can bring a dataset into being". ``SimulatorImaging``
# is the library entry point; ``should_simulate`` / ``auto_simulate`` are the
# workspace idioms for the simulate-if-missing fallback.
AUTO_SIMULATE_TOKENS = ("SimulatorImaging", "should_simulate", "auto_simulate")

# The trees the release mega-run walks (the runner is pointed at "." for this
# repository, and `find_scripts_in_folder` rglobs — dotted directories
# included) plus the repository root itself.
EXECUTABLE_TREES = (
    "scripts",
    "catalogue/scripts",
    "preprocess",
    "tools",
    "workflow",
    ".github/scripts",
)

# Package markers, never executed by the runner: ``build_util.py`` hard-codes
# ``infra_skip = ["__init__", "README"]``.
INFRA_STEMS = {"__init__"}


def _python_files():
    """
    Every ``*.py`` the automated runs can reach: the executable trees,
    recursively, plus the repository root (non-recursively).
    """
    paths = []

    for tree in EXECUTABLE_TREES:
        paths.extend(sorted((PROJECT_ROOT / tree).rglob("*.py")))

    paths.extend(sorted(PROJECT_ROOT.glob("*.py")))

    return [
        path
        for path in paths
        if "__pycache__" not in path.parts and path.stem not in INFRA_STEMS
    ]


def _all_python_files():
    """
    Every ``*.py`` in the repository, including ``tests/``, minus caches and
    this module (which necessarily names the forbidden tokens).
    """
    return [
        path
        for path in sorted(PROJECT_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts and path != Path(__file__)
    ]


def _smoke_entries():
    """
    The allowlist, parsed the way ``PyAutoHands/autohands/build_util.py``'s
    ``files_from_list`` parses it: one path per line relative to the runner's
    directory argument (``"."`` for this repository); blank lines ignored; a
    line whose first non-whitespace character is ``#`` is a comment, and that is
    the *only* comment form — a trailing ``# reason`` is part of the path.
    """
    entries = []

    for line in SMOKE_TESTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)

    return entries


def _no_run_reasons():
    """
    ``pattern -> reason`` from ``no_run.yaml``, mirroring
    ``PyAutoHands/autohands/result_collector.py``'s ``parse_no_run_reasons``:
    PyYAML strips comments, so the raw lines are read and the inline ``#``
    comment is the reason. This repository uses the flat-list form.
    """
    reasons = {}

    for line in NO_RUN.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        entry = stripped[2:]
        if "#" in entry:
            pattern, reason = entry.split("#", 1)
            reasons[pattern.strip()] = reason.strip()
        else:
            reasons[entry.strip()] = ""

    return reasons


def _matching_no_run_pattern(relative_path, patterns):
    """
    Mirrors ``PyAutoHands/autohands/build_util.py``'s ``should_skip``: a pattern
    containing ``/`` is substring-matched against the path including extension,
    a pattern without one matches a file whose stem equals it.
    """
    path_str = str(relative_path)
    stem = Path(relative_path).stem

    for pattern in patterns:
        if "/" in pattern:
            if pattern in path_str:
                return pattern
        elif stem == pattern:
            return pattern

    return None


# ---------------------------------------------------------------------------
# 1. Nothing auto-simulates
# ---------------------------------------------------------------------------


def test_only_the_simulator_script_simulates():
    """
    ``scripts/simulator.py`` is the single place a dataset is created. Every
    other script must load one. A hit here means a script grew a
    simulate-if-missing fallback, which would let a broken dataset path pass
    smoke on invented data.
    """
    offenders = {}

    for path in _all_python_files():
        if path == SIMULATOR:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [token for token in AUTO_SIMULATE_TOKENS if token in text]
        if hits:
            offenders[str(path.relative_to(PROJECT_ROOT))] = hits

    assert not offenders, (
        "only scripts/simulator.py may simulate data; found simulation tokens "
        f"in: {offenders}"
    )


def test_the_simulator_script_is_the_one_that_simulates():
    """
    The control on the test above: if ``scripts/simulator.py`` stopped using
    ``SimulatorImaging`` the invariant would pass vacuously.
    """
    assert "SimulatorImaging" in SIMULATOR.read_text(), (
        "scripts/simulator.py must use al.SimulatorImaging — otherwise "
        "test_only_the_simulator_script_simulates asserts nothing"
    )


# ---------------------------------------------------------------------------
# 2. Every script is accounted for
# ---------------------------------------------------------------------------


def test_every_script_is_smoked_or_excluded_with_a_reason():
    """
    A script is covered if it is on the smoke allowlist, or excluded in
    ``no_run.yaml`` with a non-empty reason. Both files are policy statements —
    a script in neither is invisible to every automated run.
    """
    smoke_entries = set(_smoke_entries())
    reasons = _no_run_reasons()

    unlisted = []
    reasonless = []

    for path in _python_files():
        relative = path.relative_to(PROJECT_ROOT).as_posix()

        if relative in smoke_entries:
            continue

        pattern = _matching_no_run_pattern(relative, reasons)
        if pattern is None:
            unlisted.append(relative)
        elif not reasons[pattern]:
            reasonless.append(relative)

    assert not unlisted, (
        "these scripts are in neither smoke_tests.txt nor config/build/"
        f"no_run.yaml, so nothing runs or documents them: {unlisted}"
    )
    assert not reasonless, (
        "these scripts are excluded in config/build/no_run.yaml without an "
        f"inline '# reason' comment: {reasonless}"
    )


# ---------------------------------------------------------------------------
# 3. The allowlist is not stale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", _smoke_entries())
def test_smoke_entry_exists(entry):
    """
    ``build_util.execute_scripts_in_folder`` reports a missing allowlist entry
    as ``FAIL (listed but not found)`` — a red CI run for a rename. Catch it
    here instead.
    """
    assert (
        PROJECT_ROOT / entry
    ).is_file(), f"smoke_tests.txt lists '{entry}', which does not exist"
