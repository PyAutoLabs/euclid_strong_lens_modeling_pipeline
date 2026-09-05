"""
Diagnostic: does a JAX-initialised process poison PyAutoFit's forked pool?
=========================================================================

The CPU route of ``scripts/initial_lens_model.py`` runs its two searches in two
separate processes rather than one, and this script is the control test that
measures why. Both stages live in one script — ``vis_lp`` builds its analysis
with ``use_jax=True`` and ``vis_pix`` builds a second analysis with
``use_jax=False`` and hands it to a multiprocessing ``Nautilus`` pool — so if
one process could do both, the two-process split would be unnecessary
bookkeeping. PyAutoFit builds every pool with the ``fork`` start method
(``autofit/non_linear/parallel/context.py::fork_context``, pinned because
``forkserver`` corrupts model instances, PyAutoFit#1437) and documents at
``autofit/non_linear/search/nest/nautilus/search.py`` L370-380 that a forked
child of a JAX-initialised parent "deadlocks in XLA compilation when the
likelihood touches JAX"; the same file at L240 routes any analysis with
``analysis._use_jax`` to the serial ``fit_x1_cpu`` path, so a JAX analysis never
reaches a pool at all. What is *not* documented is the case the pipeline
actually hits: a parent that initialised JAX for an earlier, finished stage,
then forks a pool whose likelihood is pure numpy/numba. This script measures
exactly that, by running two legs and reporting PASS / HANG / ERROR for each:

  control       one process: evaluate a ``vis_lp``-style ``use_jax=True``
                likelihood (which initialises XLA in-process), then run a
                ``vis_pix``-style ``use_jax=False`` pooled Nautilus fit.
  control_real  the same in one process, but driving the pipeline's own
                ``scripts/initial_lens_model.py::fit`` — ``stage="vis_lp"``
                with ``use_cpu=False``, then ``stage="vis_pix"`` with
                ``use_cpu=True`` — so the measured path is the caller's real
                invocation and the pixelized stage consumes the actual
                JAX-produced ``vis_lp`` result.
  subprocess    the pipeline's proposed fix: the JAX stage runs in a child
                process launched by ``subprocess.run``, so the parent never
                imports JAX, and the parent then runs the same pooled fit.

``forkserver`` / ``spawn`` are deliberately NOT measured as a fix: PyAutoFit
pins ``fork`` for correctness (PyAutoFit#1437, cited above), so switching the
start method is not an option this pipeline may take, and a leg that measured
it would answer a question nobody can act on.

Run it (from the project root, on any machine with the example dataset)::

    python hpc/diagnostics/jax_fork_control.py --leg all \
        --output /tmp/jax_fork_control

    python hpc/diagnostics/jax_fork_control.py --leg control --cores 4 \
        --n_like_max 300 --timeout 600

Each leg runs in its own child process launched by this script's driver, so a
leg that deadlocks is killed at ``--timeout`` and recorded as HANG instead of
hanging the driver. Per-leg wall time, outcome and environment (start method,
cores, whether ``jax`` is in ``sys.modules``, the JAX backend platform, Python
and library versions) are printed to stdout and written to
``<output>/results.json``.

This is a diagnostic, never part of CI. It runs a real (if tiny) Nautilus fit,
takes minutes, and one of its two legs is *expected* to hang — none of which
belongs in an automated test suite. Its output is evidence for a design
decision, not a pass/fail gate.

**Questions:** contact James Nightingale on the Euclid Consortium Slack.
"""

import argparse
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SAMPLE = "q1_walsmley"
DEFAULT_DATASET = "102018665_NEG570040238507752998"

# Sampler settings deliberately far below the pipeline's own. The question is
# whether the pool starts and keeps running, not what the posterior looks like,
# so every leg is sized to finish in minutes when it does not deadlock.
N_LIVE = 50
N_BATCH = 8
HILBERT_PIXELS = 200
EDGE_PIXELS_TOTAL = 30


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------


def jax_state() -> dict:
    """
    Whether JAX is imported, and whether it has actually *initialised* a
    backend — measured without initialising one.

    These are two different things, and the difference is the whole
    measurement. ``import jax`` is cheap and inert; it is
    ``jax._src.xla_bridge`` handing out a backend (the first compiled call)
    that spins up XLA and its threads, and that is the state a fork is said to
    be unsafe from. ``jax.default_backend()`` cannot be used to ask, because
    asking initialises the default backend — so the private ``_backends``
    registry is read directly instead, and an empty registry means nothing has
    been initialised. If a future JAX moves that registry the value comes back
    as ``"unknown"`` rather than a wrong answer.
    """
    state = {
        "jax_in_sys_modules": "jax" in sys.modules,
        "jax_backends_initialised": None,
    }

    if "jax" in sys.modules:
        try:
            from jax._src import xla_bridge

            state["jax_backends_initialised"] = sorted(xla_bridge._backends.keys())
        except Exception:  # pragma: no cover - diagnostic only
            state["jax_backends_initialised"] = "unknown"

    return state


def env_dict(cores: int, output_path: Path = None, extra: dict = None) -> dict:
    """
    The facts each leg records about the process it ran in.

    ``jax_backends_initialised`` is the load-bearing one: it says whether the
    process that is about to fork has XLA running inside it (see
    :func:`jax_state`).
    """
    info = {
        "python": sys.version.split()[0],
        "start_method": multiprocessing.get_start_method(),
        "cores": cores,
        "autofit_version": None,
        "autolens_version": None,
        "PYAUTO_DISABLE_JAX": os.environ.get("PYAUTO_DISABLE_JAX"),
        "PYAUTO_TEST_MODE": os.environ.get("PYAUTO_TEST_MODE"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
    }
    info.update(jax_state())

    for name in ("autofit", "autolens"):
        module = sys.modules.get(name)
        if module is not None:
            info[f"{name}_version"] = getattr(module, "__version__", None)
        else:
            # Read the metadata rather than importing: the `subprocess` leg
            # records its environment before it has imported anything, and an
            # import here would change the very process being measured.
            try:
                from importlib.metadata import version

                info[f"{name}_version"] = version(name)
            except Exception:
                info[f"{name}_version"] = None

    if extra is not None:
        info.update(extra)

    # Written at the fork point rather than at the end of the leg, so that a
    # leg which never returns still leaves its environment on disk for the
    # driver to fold into results.json.
    if output_path is not None:
        (output_path / "leg.json").write_text(json.dumps(info, indent=2))

    return info


def import_autolens_pulls_jax() -> bool:
    """
    Import ``autolens`` and report whether that alone put ``jax`` into
    ``sys.modules``.

    This is the fact the two-process split stands or falls on. If merely
    importing the library initialised JAX, then *every* pipeline process would
    be a JAX process and no arrangement of stages could give the pool a clean
    parent — the fix would have to be a different start method, which
    PyAutoFit#1437 rules out. Both legs record it.
    """
    import autolens  # noqa: F401

    return "jax" in sys.modules


# ---------------------------------------------------------------------------
# The two stages, built to mirror scripts/initial_lens_model.py
# ---------------------------------------------------------------------------


def _push_config(output_path: Path):
    """
    Point the config at the project's own ``config/`` and the results at
    ``output_path``, exactly as ``scripts/initial_lens_model.py`` does — except
    that the output path is always given explicitly, so a diagnostic run can
    never write into the project's real ``output/`` tree.
    """
    from autolens import conf

    conf.instance.push(new_path=PROJECT_ROOT / "config", output_path=output_path)


def _load(dataset: str, sample: str):
    sys.path.insert(0, str(PROJECT_ROOT))
    import util

    return util, util.load_vis_dataset(dataset, sample_name=sample)


def jax_stage(dataset: str, sample: str, output_path: Path) -> float:
    """
    The ``vis_lp`` stage, reduced to the one thing that matters here: build the
    MGE-lens-light + SIE-mass + MGE-source analysis with ``use_jax=True`` and
    evaluate its likelihood once. That single call traces and compiles the
    likelihood, so XLA is initialised in this process when it returns.

    The model mirrors ``scripts/initial_lens_model.py`` ``fit`` (L128-L228): 2
    bases of 20 Gaussians for the lens light, an ``Isothermal`` with its centre
    fixed to the brightest pixel plus ``ExternalShear``, and 20 Gaussians for
    the source. No search is run — the sampler is not what is being measured.
    """
    _push_config(output_path)

    import autofit as af
    import autolens as al

    util, d = _load(dataset, sample)

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=d.dataset_centre,
    )

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = d.dataset_centre[0]
    mass.centre.centre_1 = d.dataset_centre[1]

    source_bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius,
        total_gaussians=20,
        centre_prior_is_uniform=False,
        centre=d.dataset_centre,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                bulge=lens_bulge,
                mass=mass,
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge),
        )
    )

    analysis = util.AnalysisImaging(
        dataset=d.dataset,
        positions_likelihood_list=d.positions_likelihood_list,
        use_jax=True,
        dataset_main_path=d.dataset_main_path,
        title_prefix="VIS",
        plot_rgb=True,
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        magzero=d.magzero,
    )

    if not analysis._use_jax:
        raise RuntimeError(
            "the JAX stage built an analysis with _use_jax=False — JAX is "
            "missing or PYAUTO_DISABLE_JAX=1 is set, so this leg would not "
            "measure the fork conflict at all."
        )

    log_likelihood = float(
        analysis.log_likelihood_function(model.instance_from_prior_medians())
    )

    if "jax" not in sys.modules:
        raise RuntimeError(
            "the JAX likelihood was evaluated but `jax` is not in sys.modules — "
            "the analysis did not route through JAX, so this leg is void."
        )

    import jax

    print(
        f"[jax stage] log_likelihood={log_likelihood:.4f} "
        f"backend={jax.default_backend()} devices={jax.devices()}",
        flush=True,
    )

    return log_likelihood


def pooled_pix_fit(dataset: str, sample: str, output_path: Path, cores: int, n_like_max: int):
    """
    The ``vis_pix`` stage: a Delaunay pixelized source on the Numba CPU sparse
    operator, ``use_jax=False``, handed to a multiprocessing ``Nautilus``.

    This mirrors ``scripts/initial_lens_model.py`` L410-L572 — the CPU sparse
    operator, the Hilbert image mesh with a ring of zeroed edge points, the
    signal-to-noise over-sampling map, ``reg.AdaptSplit``, and a free mass
    centre — with two deliberate departures, neither of which touches the
    fork question:

    * the adapt image is the dataset's own signal-to-noise map rather than the
      lens-subtracted source image from a converged ``vis_lp`` result, so the
      diagnostic does not have to run a full first search to build its second
      one, and
    * the fixed lens-light MGE instance is drawn from the prior medians rather
      than from a converged ``vis_lp`` fit, and the sampler settings are tiny
      (see ``N_LIVE`` above).

    What is preserved is everything the pool sees: a numpy/numba likelihood, a
    pixelized inversion, and ``number_of_cores > 1``, which is the condition
    under which ``Nautilus.fit_multiprocessing`` builds a forked pool.
    """
    _push_config(output_path)

    import numpy as np

    import autofit as af
    import autolens as al

    util, d = _load(dataset, sample)

    imaging = d.dataset
    mask = imaging.mask

    # scripts/initial_lens_model.py L436-440: --use_cpu selects the Numba CPU
    # sparse operator rather than the JAX one.
    imaging = imaging.apply_sparse_operator_cpu()

    adapt_data = np.asarray(imaging.signal_to_noise_map)
    adapt_data = np.clip(adapt_data, 0.01 * float(np.max(adapt_data)), None)
    adapt_image = al.Array2D(values=adapt_data, mask=mask)

    image_mesh = al.image_mesh.Hilbert(
        pixels=HILBERT_PIXELS, weight_power=3.5, weight_floor=0.01
    )
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=adapt_image
    )
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=d.mask_radius + mask.pixel_scale / 2.0,
        n_points=EDGE_PIXELS_TOTAL,
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict={"('galaxies', 'source')": adapt_image},
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )

    over_sample_size_pixelization = al.Array2D(
        values=np.where(adapt_data > 3.0, 4, 2), mask=mask
    )
    imaging = imaging.apply_over_sampling(
        over_sample_size_lp=imaging.grids.lp.over_sample_size,
        over_sample_size_pixelization=over_sample_size_pixelization,
    )

    analysis = util.AnalysisImaging(
        dataset=imaging,
        adapt_images=adapt_images,
        positions_likelihood_list=d.positions_likelihood_list,
        use_jax=False,
        dataset_main_path=d.dataset_main_path,
        title_prefix="VIS",
        plot_rgb=True,
        psf_lowest_resolution=d.psf_lowest_resolution,
        psf_lowest_resolution_fwhm=d.psf_lowest_resolution_fwhm,
        pixel_wcs=d.pixel_wcs,
        magzero=d.magzero,
    )

    if analysis._use_jax:
        raise RuntimeError(
            "the pixelized analysis has _use_jax=True — it would be routed to "
            "the serial fit_x1_cpu path (nautilus/search.py L240) and no pool "
            "would ever be forked, so this leg is void."
        )

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.UniformPrior(
        lower_limit=d.dataset_centre[0] - 0.1, upper_limit=d.dataset_centre[0] + 0.1
    )
    mass.centre.centre_1 = af.UniformPrior(
        lower_limit=d.dataset_centre[1] - 0.1, upper_limit=d.dataset_centre[1] + 0.1
    )

    # The lens light is a fixed 2x20 Gaussian MGE *instance*, as in
    # scripts/initial_lens_model.py L540-546 where it is the vis_lp maximum
    # likelihood instance. Here it is drawn from the prior medians instead —
    # the light is not fitted either way, so it costs the likelihood the same
    # 40 Gaussian evaluations per call that the pipeline pays.
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=d.mask_radius,
        total_gaussians=20,
        gaussian_per_basis=2,
        centre_prior_is_uniform=True,
        centre=d.dataset_centre,
    ).instance_from_prior_medians()

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                bulge=lens_bulge,
                mass=mass,
                shear=af.Model(al.mp.ExternalShear),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=1.0,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=al.mesh.Delaunay(
                        pixels=HILBERT_PIXELS,
                        zeroed_pixels=EDGE_PIXELS_TOTAL,
                    ),
                    regularization=al.reg.AdaptSplit,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        path_prefix=Path(sample) / dataset if sample is not None else Path(dataset),
        unique_tag="jax_fork_control",
        name="vis_pix",
        n_live=N_LIVE,
        n_batch=N_BATCH,
        n_like_max=n_like_max,
        iterations_per_quick_update=1000000,
        number_of_cores=cores,
        session=None,
    )

    # The real fork point. Everything above — loading the dataset in
    # particular — has already run, so this is the state the pool workers
    # inherit, and it is not the same as the state recorded when the leg
    # started.
    fork_state = jax_state()
    (output_path / "fork_state.json").write_text(json.dumps(fork_state, indent=2))
    print(
        f"[pooled fit] forking pool: cores={cores} "
        f"start_method={multiprocessing.get_start_method()} "
        f"at_fork={json.dumps(fork_state)}",
        flush=True,
    )

    result = search.fit(model=model, analysis=analysis)

    print(
        f"[pooled fit] finished: "
        f"max_log_likelihood={result.samples.max_log_likelihood_sample.log_likelihood:.4f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# The legs
# ---------------------------------------------------------------------------


def leg_control(args, output_path: Path) -> dict:
    """
    One process does both stages. The JAX stage runs first and leaves XLA
    initialised; the pooled numpy/numba fit then forks out of that process.
    """
    pulls_jax = import_autolens_pulls_jax()
    print(f"[control] import autolens -> jax in sys.modules: {pulls_jax}", flush=True)

    print("[control] stage 1: JAX likelihood (initialises XLA in this process)", flush=True)
    jax_stage(args.dataset, args.sample, output_path)

    info = env_dict(
        args.cores, output_path, {"import_autolens_pulls_jax": pulls_jax}
    )
    print(f"[control] environment at fork: {json.dumps(info)}", flush=True)

    print("[control] stage 2: pooled numpy/numba pixelized fit", flush=True)
    pooled_pix_fit(args.dataset, args.sample, output_path, args.cores, args.n_like_max)

    return info


def leg_subprocess(args, output_path: Path) -> dict:
    """
    The proposed fix: the JAX stage runs in a child process, so this process
    never imports JAX and its fork is clean.

    The assertion before and after the ``subprocess.run`` is the whole point of
    the leg — a parent that had quietly pulled JAX in (via ``import autolens``,
    say) would be forking from a JAX-initialised process after all, and the leg
    would be measuring nothing.
    """
    if "jax" in sys.modules:
        raise RuntimeError(
            "`jax` is already in sys.modules before the JAX stage was launched — "
            "this leg cannot demonstrate a clean parent."
        )

    # Import autolens in the parent, exactly as the real pipeline process does,
    # then re-check. If this pulled JAX in, the leg is void and says so.
    pulls_jax = import_autolens_pulls_jax()
    print(
        f"[subprocess] import autolens -> jax in sys.modules: {pulls_jax}", flush=True
    )
    if pulls_jax:
        raise RuntimeError(
            "`import autolens` alone put jax in sys.modules — no pipeline "
            "process can be a clean fork source, so the two-process split "
            "cannot be the fix and this leg proves nothing."
        )

    print(
        f"[subprocess] before launching the JAX stage: "
        f"jax_in_sys_modules={'jax' in sys.modules} "
        f"(autolens imported: {'autolens' in sys.modules})",
        flush=True,
    )

    print("[subprocess] stage 1: JAX likelihood in a child process", flush=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_jax_stage_worker",
            "--dataset",
            args.dataset,
            "--sample",
            args.sample,
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        print(f"    [child] {line}", flush=True)
    if completed.returncode != 0:
        print(completed.stderr[-4000:], flush=True)
        raise RuntimeError(
            f"the JAX stage child exited {completed.returncode}; see stderr above."
        )

    if "jax" in sys.modules:
        raise RuntimeError(
            "`jax` entered sys.modules in the parent while the child ran — the "
            "parent is no longer a clean fork source."
        )

    info = env_dict(
        args.cores, output_path, {"import_autolens_pulls_jax": pulls_jax}
    )
    print(f"[subprocess] environment at fork: {json.dumps(info)}", flush=True)

    print("[subprocess] stage 2: pooled numpy/numba pixelized fit", flush=True)
    pooled_pix_fit(args.dataset, args.sample, output_path, args.cores, args.n_like_max)

    return info


def _load_pipeline_script():
    """
    Import ``scripts/initial_lens_model.py`` as a module, so the leg calls the
    pipeline's own ``fit`` rather than a re-implementation of it.
    """
    import importlib.util

    path = PROJECT_ROOT / "scripts" / "initial_lens_model.py"
    spec = importlib.util.spec_from_file_location("initial_lens_model_diag", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clamp_nautilus(n_live: int, n_like_max: int):
    """
    Make the pipeline's two real searches cheap enough to run as a diagnostic,
    without editing the pipeline.

    ``scripts/initial_lens_model.py`` resolves ``af.Nautilus`` from the
    ``autofit`` module at call time, so replacing that attribute with a
    factory that overrides ``n_live`` and ``n_like_max`` reaches both searches
    and leaves every other argument — ``batch_size``, ``n_batch``,
    ``number_of_cores``, the paths — exactly as the pipeline sets them. A
    factory rather than a subclass, so the search is still an ``af.Nautilus``
    and its config lookups and identifier are unaffected.
    """
    import autofit as af

    original = af.Nautilus

    def clamped(*args, **kwargs):
        kwargs["n_live"] = n_live
        kwargs["n_like_max"] = n_like_max
        return original(*args, **kwargs)

    af.Nautilus = clamped
    return original


def leg_control_real(args, output_path: Path) -> dict:
    """
    The control test run through the pipeline's own entry point.

    ``leg_control`` builds a faithful *imitation* of the two stages; this leg
    calls the real ``scripts/initial_lens_model.py::fit`` twice in one process,
    exactly as a single-process CPU run would:

      1. ``fit(..., use_cpu=False, stage="vis_lp")`` — the real light-profile
         search, under JAX, in this process. The sharpened probe is asserted
         afterwards: if no XLA backend was initialised, this leg would be
         forking from a clean process and would prove nothing.
      2. ``fit(..., use_cpu=True, number_of_cores=cores, stage="vis_pix")`` —
         the real pixelized search, which loads the JAX-produced ``vis_lp``
         result from disk via ``search.fit``'s completed-fit short circuit and
         forks its Numba likelihood pool out of this same, JAX-initialised
         process.

    Output is redirected with ``PYAUTO_OUTPUT_DIR``, which is what the script
    itself reads (L70-73), so the diagnostic never writes into the project's
    ``output/``. The only thing altered is the sampler cost, via
    ``_clamp_nautilus``.
    """
    os.environ["PYAUTO_OUTPUT_DIR"] = str(output_path)

    pulls_jax = import_autolens_pulls_jax()
    print(f"[control_real] import autolens -> jax in sys.modules: {pulls_jax}", flush=True)

    _clamp_nautilus(N_LIVE, args.n_like_max)
    pipeline = _load_pipeline_script()

    print(
        f"[control_real] stage 1: real fit(stage='vis_lp', use_cpu=False) under JAX",
        flush=True,
    )
    pipeline.fit(
        dataset_name=args.dataset,
        sample_name=args.sample,
        iterations_per_quick_update=1000000,
        number_of_cores=1,
        use_cpu=False,
        stage="vis_lp",
    )

    after_vis_lp = jax_state()
    print(f"[control_real] after vis_lp: {json.dumps(after_vis_lp)}", flush=True)
    if not after_vis_lp["jax_backends_initialised"]:
        raise RuntimeError(
            "the real vis_lp stage returned without initialising an XLA "
            "backend — this process is not a JAX-initialised parent, so the "
            "leg cannot test the fork conflict."
        )

    info = env_dict(
        args.cores, output_path, {"import_autolens_pulls_jax": pulls_jax}
    )
    print(f"[control_real] environment at fork: {json.dumps(info)}", flush=True)

    fork_state = jax_state()
    (output_path / "fork_state.json").write_text(json.dumps(fork_state, indent=2))

    print(
        f"[control_real] stage 2: real fit(stage='vis_pix', use_cpu=True, "
        f"number_of_cores={args.cores}) forking from this process",
        flush=True,
    )
    result = pipeline.fit(
        dataset_name=args.dataset,
        sample_name=args.sample,
        iterations_per_quick_update=1000000,
        number_of_cores=args.cores,
        use_cpu=True,
        stage="vis_pix",
    )

    print(
        f"[control_real] finished: max_log_likelihood="
        f"{result.samples.max_log_likelihood_sample.log_likelihood:.4f}",
        flush=True,
    )

    return info


LEGS = {
    "control": leg_control,
    "control_real": leg_control_real,
    "subprocess": leg_subprocess,
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_leg_in_child(leg: str, args) -> dict:
    """
    Run one leg in its own process group and wait up to ``--timeout`` seconds.

    A deadlocked leg is unkillable from inside itself, so the driver owns the
    clock: on timeout the whole process group is killed (the pool's forked
    workers included) and the leg is recorded as HANG. Each leg writes its
    environment to ``<output>/<leg>/leg.json`` before it forks, so that record
    survives even when the leg never returns.
    """
    output_path = Path(args.output) / leg
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "leg.log"

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_run_leg",
        leg,
        "--dataset",
        args.dataset,
        "--sample",
        args.sample,
        "--cores",
        str(args.cores),
        "--n_like_max",
        str(args.n_like_max),
        "--output",
        str(args.output),
    ]

    print(f"\n{'=' * 78}\n=== leg: {leg}\n{'=' * 78}", flush=True)
    print(f"    {' '.join(command)}", flush=True)

    start = time.time()
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()
    wall = time.time() - start

    print(log_path.read_text(), flush=True)

    record = {"leg": leg, "wall_seconds": round(wall, 1), "log": str(log_path)}

    leg_json = output_path / "leg.json"
    if leg_json.exists():
        record.update(json.loads(leg_json.read_text()))

    fork_json = output_path / "fork_state.json"
    if fork_json.exists():
        record.update(
            {f"at_fork_{k}": v for k, v in json.loads(fork_json.read_text()).items()}
        )

    if timed_out:
        record["outcome"] = "HANG"
        record["detail"] = f"killed after --timeout={args.timeout}s"
    elif returncode == 0:
        record["outcome"] = "PASS"
    else:
        tail = log_path.read_text().strip().splitlines()
        record["outcome"] = "ERROR"
        record["detail"] = "\n".join(tail[-12:])

    print(
        f"=== leg {leg}: {record['outcome']} in {record['wall_seconds']}s", flush=True
    )
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--leg",
        choices=["control", "control_real", "subprocess", "all"],
        default="all",
        help="which leg to run (default: all).",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument(
        "--cores",
        type=int,
        default=4,
        help="cores for the pooled vis_pix fit; must exceed 1 or no pool is "
        "forked (default: 4).",
    )
    parser.add_argument("--n_like_max", type=int, default=300)
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="seconds before a leg is killed and recorded as HANG (default: 600).",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "output" / "jax_fork_control"),
        help="directory for per-leg fit output, logs and results.json.",
    )

    # Internal entry points: one leg, or the JAX stage of the subprocess leg.
    parser.add_argument("--_run_leg", choices=list(LEGS), default=None)
    parser.add_argument("--_jax_stage_worker", action="store_true")

    args = parser.parse_args(argv)

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args._jax_stage_worker:
        jax_stage(args.dataset, args.sample, output_root)
        return 0

    if args._run_leg is not None:
        leg = args._run_leg
        output_path = output_root / leg
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            LEGS[leg](args, output_path)
        except Exception:
            traceback.print_exc()
            return 1
        return 0

    if args.cores <= 1:
        parser.error(
            "--cores must be > 1: Nautilus builds no pool at all for a single "
            "core (nautilus/search.py, `if self.number_of_cores <= 1`), so the "
            "legs would fork nothing and measure nothing."
        )
    if os.environ.get("PYAUTO_TEST_MODE"):
        parser.error(
            "PYAUTO_TEST_MODE is set: test mode skips the sampler entirely, so "
            "no pool is created and neither leg measures anything. Unset it."
        )
    if os.environ.get("PYAUTO_DISABLE_JAX") == "1":
        parser.error(
            "PYAUTO_DISABLE_JAX=1 is set: the JAX stage would silently fall "
            "back to numpy and never initialise XLA. Unset it."
        )

    legs = list(LEGS) if args.leg == "all" else [args.leg]
    records = [run_leg_in_child(leg, args) for leg in legs]

    results = {
        "dataset": args.dataset,
        "sample": args.sample,
        "cores": args.cores,
        "n_like_max": args.n_like_max,
        "timeout": args.timeout,
        "driver_python": sys.version.split()[0],
        "legs": records,
    }
    results_path = output_root / "results.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 78}\n=== summary\n{'=' * 78}", flush=True)
    for record in records:
        print(
            f"{record['leg']:<12} {record['outcome']:<6} "
            f"{record['wall_seconds']:>8.1f}s  "
            f"at_fork: jax_imported={record.get('at_fork_jax_in_sys_modules')} "
            f"xla_backends={record.get('at_fork_jax_backends_initialised')}  "
            f"import_autolens_pulls_jax="
            f"{record.get('import_autolens_pulls_jax')}",
            flush=True,
        )
    print(f"\nwritten to {results_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
