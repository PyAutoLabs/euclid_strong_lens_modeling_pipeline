"""
Run the pipeline smoke test suite: the scripts listed in `smoke_tests.txt`.

Nothing about discovery, exclusion, environment resolution, per-entry timeouts
or reporting is implemented here. This is a thin shim over PyAutoHands'
`autohands/run_python.py` — the same entry point PyAutoHeart's
workspace-validation uses — so the PR gate and the validation runner cannot
drift apart. The copy is the thing to avoid: the last time this machinery was
duplicated per repo, a local copy of the env resolver silently drifted and left
the PR gate unable to read the release profile.

Two differences from the `autolens_workspace` original this is copied from:

  * The directory argument is `"."`, not `"scripts"`. This repository's
    runnable scripts are not all under `scripts/` — `start_here.py` sits at the
    root and the catalogue producers under `catalogue/scripts/` — so
    `smoke_tests.txt` entries are repository-root-relative.
  * There is no notebook leg. This repository ships no notebooks and no
    `smoke_notebooks.txt`; `.gitignore` excludes `*.ipynb` outright.

`config/build/no_run.yaml` is deliberately NOT applied to the allowlist. It is
policy for the release mega-run; the smoke list is policy for this gate, and a
script legitimately appears in both (PyAutoHands#262). `config/build/
profile_smoke.yaml` IS applied — `run_python.py` finds it relative to the cwd,
which is why `_run` sets `cwd` to the repository root. That profile carries this
repository's `args_default` (`--dataset=... --sample=...`), appended to every
entry.

`--report-dir` is REQUIRED, not cosmetic. `build_util.execute_script` only
records a failure and carries on when a report was built; without one it aborts
at the first failing script, and `run_python.py` only propagates the failure
(`sys.exit(1)`) when a report exists — so a run without it is a vacuously green
gate that also stops at the first break.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
PROJECT = "euclid"

# CI puts PyAutoHands/autohands on PYTHONPATH (PyAutoHeart's reusable
# smoke-tests.yml clones it alongside the dependency chain); for local runs,
# fall back to the sibling checkout.
try:
    import build_util
except ImportError:  # pragma: no cover - local-run fallback
    sys.path.insert(0, str(WORKSPACE.parent / "PyAutoHands" / "autohands"))
    import build_util

AUTOHANDS = Path(build_util.__file__).resolve().parent

SCRIPT_LIST = WORKSPACE / "smoke_tests.txt"
REPORT_DIR = WORKSPACE / "test-results"


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(AUTOHANDS), env.get("PYTHONPATH", "")) if p
    )
    return env


def _run(argv: list[str]) -> int:
    # The shared runners resolve config/build/ relative to the cwd.
    return subprocess.run(argv, cwd=str(WORKSPACE), env=_env()).returncode


def main() -> int:
    if not SCRIPT_LIST.exists():
        # A missing list is a configuration error, not "nothing to do".
        print(f"ERROR: no {SCRIPT_LIST.name} at {SCRIPT_LIST}", file=sys.stderr)
        return 1

    rc = _run(
        [
            sys.executable,
            str(AUTOHANDS / "run_python.py"),
            PROJECT,
            ".",
            "--list",
            str(SCRIPT_LIST),
            "--report-dir",
            str(REPORT_DIR),
        ]
    )

    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
