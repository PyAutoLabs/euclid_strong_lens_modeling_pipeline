#!/usr/bin/env bash
# Workspace-owned install epilogue for the reusable Smoke Tests workflow
# (PyAutoHeart/.github/workflows/smoke-tests.yml). Runs with cwd at the
# checkout root (the dependency chain is cloned beside `workspace/`) and
# receives PYTHON_VERSION. Everything that differs per workspace lives
# here; the ceremony lives in the reusable workflow.
set -e

pip install ./PyAutoNerves ./PyAutoFit ./PyAutoArray ./PyAutoGalaxy ./PyAutoLens
# NOTE: no jax pin here, deliberately. One lived on this line until it was
# found to be vestigial: #82 added `jax<0.7 jaxlib<0.7` solely to keep
# `tensorflow-probability==0.25.0` importable, and #184 removed that
# dependency when the stack moved to tfp-nightly. jax's supported range is
# owned by autonerves' base dependencies (jax>=0.7.0,<0.12.0, PyAutoLens#702)
# and is installed by the line above -- do not restate it here, it drifts.
# The assertion at the end of this script checks the resolved version.
pip install "./PyAutoArray[optional]" "./PyAutoGalaxy[optional]" "./PyAutoLens[optional]"
# NOTE: do NOT `pip install tensorflow-probability==0.25.0` here. The stable
# release crashes at import under the resolved modern JAX
# (`jax.interpreters.xla.pytype_aval_mappings` was removed), which broke the
# JAX Matern-kernel (delaunay_mge) likelihood path. The working modified-Bessel
# dependency is `tfp-nightly`, pinned by `PyAutoArray[optional]` above.
# The [optional] re-resolution above can upgrade autonerves to the
# stale PyPI release (setuptools_scm reports the local copy as
# 1.0.dev0 from the shallow checkout). Pin the local source one
# last time so site-packages has skip_latents() and other recent
# autonerves APIs available at import time.
pip install --force-reinstall --no-deps ./PyAutoNerves

# Assert the resolved jax rather than inferring it from a green smoke run. This
# install previously landed on a supported jax only because the [optional]
# re-resolution above happened to undo the pin removed here -- correct by line
# ordering, not by constraint. Reordering these lines now fails loudly at install
# time instead of silently dropping the smoke suite onto an unsupported jax.
python - <<'JAXCHECK'
import jax

major, minor = (int(part) for part in jax.__version__.split(".")[:2])
assert (0, 7) <= (major, minor) < (0, 12), (
    f"resolved jax {jax.__version__} is outside autonerves' supported range "
    "(>=0.7.0,<0.12.0)"
)
print(f"resolved jax {jax.__version__}")
JAXCHECK
