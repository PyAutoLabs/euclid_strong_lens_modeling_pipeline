"""
The ``--stage`` fail-fast guard in ``scripts/initial_lens_model.py``.

``--stage vis_pix`` exists so the CPU route can run the two searches as two
processes: ``vis_lp`` with JAX, then ``vis_pix`` with the Numba sparse operator
and a multiprocessing pool, which a process that has initialised JAX cannot
fork. The second process must therefore *load* the ``vis_lp`` result rather than
produce it — and the failure mode that matters is the silent one: with no
completed ``vis_lp`` on disk, ``search.fit`` would simply run the whole
light-profile fit again, in the process configured for the pixelized stage, and
the user would only find out hours later.

So ``fit`` checks ``search.paths.is_complete`` — PyAutoFit's own predicate, the
``.completed`` marker that ``paths.completed()`` writes, and the same condition
``AbstractSearch.fit`` short-circuits on — before running ``vis_lp``, and raises
if it is not set.

These tests run **no search**. The first stops at the guard (that is the whole
point of a fail-fast guard: it fires before any fitting), and the second never
gets as far as loading the dataset. They run under the CI test-mode environment
with ``conf`` and ``output/`` redirected into ``tmp_path``, so nothing is
written into the repository tree.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import initial_lens_model  # noqa: E402


EXAMPLE_SAMPLE = "q1_walsmley"
EXAMPLE_DATASET = "102018665_NEG570040238507752998"


# The environment CI runs the smoke suite under. `fit` reads
# `PYAUTO_OUTPUT_DIR` and joins it onto the repository root; an absolute path
# replaces that root outright (pathlib), which is how `output/` is redirected
# into `tmp_path` here.
TEST_MODE_ENV = {
    "PYAUTO_TEST_MODE": "2",
    "PYAUTO_SKIP_FIT_OUTPUT": "1",
    "PYAUTO_SKIP_VISUALIZATION": "1",
    "PYAUTO_SKIP_CHECKS": "1",
    "PYAUTO_DISABLE_JAX": "1",
    "JAX_ENABLE_X64": "True",
}


@pytest.fixture
def isolated_output(monkeypatch, tmp_path):
    """
    Run ``fit`` under the CI test-mode environment with its output tree inside
    ``tmp_path``, then restore the repository's own ``conf`` push — ``fit``
    pushes ``conf.instance`` and there is no pop, so a later test module in the
    same session would otherwise inherit ``tmp_path``.
    """
    for key, value in TEST_MODE_ENV.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setenv("PYAUTO_OUTPUT_DIR", str(tmp_path / "output"))

    yield tmp_path

    from autolens import conf

    conf.instance.push(
        new_path=PROJECT_ROOT / "config", output_path=PROJECT_ROOT / "output"
    )


def test_stage_vis_pix_without_a_completed_vis_lp_fails_fast(isolated_output):
    with pytest.raises(RuntimeError) as exc_info:
        initial_lens_model.fit(
            dataset_name=EXAMPLE_DATASET,
            sample_name=EXAMPLE_SAMPLE,
            stage="vis_pix",
            use_cpu=True,
        )

    message = str(exc_info.value)

    # The message has to be actionable: the path that was looked for, and the
    # command that produces it.
    assert "--stage vis_lp" in message
    assert "vis_lp" in message
    assert EXAMPLE_DATASET in message
    assert EXAMPLE_SAMPLE in message


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError) as exc_info:
        initial_lens_model.fit(
            dataset_name=EXAMPLE_DATASET,
            sample_name=EXAMPLE_SAMPLE,
            stage="bogus",
        )

    assert "bogus" in str(exc_info.value)


class _Sentinel(Exception):
    """Raised in place of loading a dataset, to stop ``fit`` past the guards."""


def test_skip_pix_true_still_means_stage_vis_lp(monkeypatch, isolated_output):
    """
    ``skip_pix=True`` — the deprecated keyword the chain scripts still pass —
    is resolved to ``stage="vis_lp"`` *before* the stage is validated, so it
    overrides whatever ``stage`` holds instead of colliding with it.

    Proven without running a search: the dataset load, the first thing ``fit``
    does after the guards, is replaced with a sentinel. Reaching it means both
    guards passed.
    """
    import inspect

    import util

    signature = inspect.signature(initial_lens_model.fit)

    assert signature.parameters["stage"].default == "all"
    assert signature.parameters["skip_pix"].default is None

    def _no_dataset(*args, **kwargs):
        raise _Sentinel()

    monkeypatch.setattr(util, "load_vis_dataset", _no_dataset)

    with pytest.raises(_Sentinel):
        initial_lens_model.fit(
            dataset_name=EXAMPLE_DATASET,
            sample_name=EXAMPLE_SAMPLE,
            stage="bogus",
            skip_pix=True,
        )
