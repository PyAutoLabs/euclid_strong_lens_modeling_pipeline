"""
Run this repository's pytest suite, one marker selection per invocation.

Reached through PyAutoHeart's reusable smoke-test workflow via its `runner`
input (see .github/workflows/tests.yml), so the dependency-chain checkout,
install epilogue and cache dirs are the same ones the smoke gate uses — there
is no second copy of that ceremony here.

  (no flag)  `pytest -m "not slow"` — the fast, JAX-free unit suite.
  --slow     `pytest -m slow`       — the real-mode run-level latent fit.

The two are separate invocations (and separate CI jobs) on purpose: "a latent
value is wrong" and "the pipeline stopped writing latents at all" are different
failures and should not arrive as one red X.

pytest is run with the repository root as cwd. That is load-bearing twice over:
`pytest.ini` and `tests/` are resolved from there, and the pipeline's own
`config/` — which `conf.instance.push` and `config/latent.yaml` drive — is
resolved relative to it too.

pytest reports every failing test in a run (it does not stop at the first), so
no report directory is needed here; the exit code is pytest's own.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Run the `slow` marker (the real-mode fit) instead of the fast suite.",
    )
    args = parser.parse_args()

    marker = "slow" if args.slow else "not slow"

    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-m", marker, "tests"],
        cwd=str(WORKSPACE),
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
