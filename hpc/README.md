# Running the pipeline on a cluster

This directory holds everything needed to run `scripts/initial_lens_model.py` (and the
other fitting pipelines) under SLURM: example submission scripts for a GPU route and a
CPU route in `batch_gpu/` and `batch_cpu/`, and the `sync` script that moves code,
datasets, logs and results between your machine and the cluster.

Read this page once to pick a route. Everything else about the fits themselves is in
`start_here.py`; the submission scripts only decide *where* and *how* that script runs.

## Which route?

| Your situation | Route | Script | What one job does |
|---|---|---|---|
| A small subset of lenses (up to a few hundred) and a GPU node is available | **GPU** (recommended) | `batch_gpu/submit_initial_lens_model` | Both stages, `vis_lp` then `vis_pix`, in one process under JAX. Measured on one A100 80GB PCIe with the committed config: 1 h 14 min to 1 h 44 min per lens across two runs on two different nodes (`vis_lp` 30-38 min, `vis_pix` 42-64 min). |
| A large sample and many CPU cores, or no GPU | **Two-stage CPU** | `batch_cpu/submit_initial_lens_model_two_stage` | One submission; `vis_lp` under JAX on the CPU backend, then `vis_pix` with the Numba sparse operator and a process pool, as two consecutive Python processes. Measured on 8 cores with the committed config: 3 h 17 min per lens (26 min `vis_lp`, 2 h 51 min `vis_pix`). |
| As above, but you want different walltime, memory or core counts per stage, or to re-run one stage alone | **Two-stage CPU, two jobs** | `batch_cpu/submit_initial_lens_model_vis_lp` then `batch_cpu/submit_initial_lens_model_vis_pix` | The same two stages as two array jobs. Submit the second once the first has finished. |
| A cluster where JAX is unavailable or broken | **Numba only** | `batch_cpu/submit_initial_lens_model` | Both stages in one process with JAX disabled. The `vis_lp` stage is markedly slower this way. |

Both figures come from the acceptance runs recorded below, with the committed,
laptop-friendly `config/`. The "around 10 minutes per lens" quoted in the
repository's `README.md` and `start_here.py` is the DR1 science-run figure,
measured under that run's own configuration (`hpc.hpc_mode: true`, a much larger
`hpc.iterations_per_quick_update`, and its own sampler settings); it has not been
reproduced with the committed config, and the quick-update overhead alone does
not account for the gap. Treat the measured numbers above as the ones you will
see out of the box.

The GPU range above is node-to-node spread, not a setting you can choose. The two
runs used different A100 nodes, and the `vis_lp` stage — whose code and iteration
count were identical in both — was 25% slower on the second, which accounts for
most of the difference. The discrepancy against the science figure is still open
and tracked as
`PyAutoMind/draft/bug/euclid/gpu_per_lens_time_vs_documented_10_min.md`.

The GPU route is the simplest and, per lens, by far the fastest. The two-stage CPU route
exists because a cluster with hundreds of CPU cores and few GPUs can fit a large sample
faster in aggregate: the `vis_lp` MGE fit is a good fit for JAX on the CPU backend, while
the pixelized `vis_pix` fit, with JAX switched off, is fastest with the Numba sparse
operator and a forked process pool across many cores. That operator is the CPU route's
tool and is applied only under `--use_cpu`; the GPU route applies no sparse operator at
all and fits the plain dataset with JAX's own linear algebra. Under JAX the
`--number_of_cores` argument is ignored, because PyAutoFit routes a JAX likelihood
to its serial path and lets JAX vectorise instead.

Every script fits the committed example dataset by default (`sample=q1_walsmley`,
one entry in its `datasets` list, `--array=0-0`). To fit your own lenses, add their
dataset directory names to the list and widen `--array` to match.

## The `--stage` argument

`scripts/initial_lens_model.py` takes `--stage {all,vis_lp,vis_pix}` (default `all`):

- `--stage vis_lp` runs the MGE lens-light and source fit and stops.
- `--stage vis_pix` loads the completed `vis_lp` result and runs the pixelized source
  fit. If that result is not complete it stops immediately with an error naming the
  missing output directory and the `--stage vis_lp` command to run first, instead of
  silently starting `vis_lp` from scratch (which could also collide with a `vis_lp`
  job still running on the same lens).
- `--stage all` runs both, which is what the GPU route and the Numba-only route use.

`--skip_pix` still works as a deprecated alias of `--stage vis_lp`.

## Why the CPU route uses two processes

The two CPU stages want different environments. `vis_lp` under JAX wants the CPU
backend pinned (`JAX_PLATFORMS=cpu`, so it never grabs a GPU that was not allocated)
and every linear-algebra library given the full core count, because its parallelism
is threads. `vis_pix` wants every thread count pinned to `1`, because its parallelism
is `--number_of_cores` pool processes and giving each worker a full set of BLAS
threads oversubscribes the node. The scripts set these per stage; the two-job scripts
carry one block each, and the single-submission script applies each block with `env`
on its own command line so nothing leaks between the stages.

The stages also run in separate Python processes. This is a conservative default rather
than a measured necessity. PyAutoFit documents that a forked worker whose *likelihood*
touches JAX deadlocks in XLA compilation (`autofit/non_linear/search/nest/nautilus/search.py`),
and the DR1 science runs were submitted as two jobs on that basis. The `vis_pix`
likelihood under `--use_cpu` is Numba, not JAX, so the documented case does not apply
directly, and a control test did not reproduce a hang on either a laptop or a
cluster node:

| Leg (`hpc/diagnostics/jax_fork_control.py`) | Machine | Pool | Wall | Outcome |
|---|---|---|---|---|
| `control`: JAX likelihood evaluated in-process, then a `use_jax=False` pooled Nautilus | WSL2, 8 logical CPUs, CPU JAX | 4 | 265 s | PASS |
| `control_real`: the script's own `fit(stage="vis_lp")` under JAX, then `fit(stage="vis_pix", use_cpu=True)` in the same process | same | 4 | 494 s | PASS |
| `subprocess`: JAX stage in a child of a JAX-free parent, then the pooled fit | same | 4 | 233 s | PASS |
| `control` | RAL `euclid-ral-compute-1`, 16 cores | 16 | 216 s | PASS |
| `control_real` | same | 16 | 378 s | PASS |
| `subprocess` | same | 16 | 220 s | PASS |

The cluster rows add a 16-worker pool under real cgroup limits, which the laptop
run could not exercise. What remains untested is production sampler size (`n_live`
750 and 300, against 50 in the control) and multi-hour wall times: every leg above
ran a small fit. Until a production-size run passes, keep the process boundary. It
costs nothing with the single-submission script, which already gives you one
submission and one place in the queue. The single-process route is filed as a
follow-up rather than abandoned
(`PyAutoMind/draft/feature/euclid/single_process_cpu_route_jax_vis_lp_numba_vis_pix.md`).
`forkserver` and `spawn` are not an option: PyAutoFit pins the `fork` start method
because `forkserver` corrupts model instances (PyAutoFit#1437).

To repeat the measurement on your cluster:

```bash
python hpc/diagnostics/jax_fork_control.py --leg all --cores $SLURM_CPUS_PER_TASK \
    --output /path/to/scratch/jax_fork_control
```

It is a diagnostic, never part of CI, and it writes `results.json` with one entry per
leg (start method, whether an XLA backend was initialised at the fork point, wall
time, outcome).

`hpc/batch_cpu/submit_jax_fork_control` is that command as a SLURM job -- 16 cores,
all three legs, results under `output_diag/jax_fork_control` -- so the pooled legs are
measured at the size a production `vis_pix` fit actually forks. The RAL rows in
the table above came from that job (16 cores, all three legs, 13.6 min total).

## Acceptance on RAL

Both routes were run end to end from the scripts in this directory, on the
committed example lens (`q1_walsmley/102018665_NEG570040238507752998`), on the
RAL cluster on 2026-09-03:

| Route | Script | Allocation | `vis_lp` | `vis_pix` | Total | Outcome |
|---|---|---|---|---|---|---|
| GPU | `batch_gpu/submit_initial_lens_model` | A100 80GB PCIe (`euclid-ral-gpu-2`), 1 core | 30.2 min | 42.4 min | 1 h 14 min | COMPLETED |
| GPU, re-run after the sparse-operator fix | `batch_gpu/submit_initial_lens_model` | A100 80GB PCIe (`euclid-ral-gpu-1`), 1 core | 37.7 min | 63.8 min | 1 h 44 min | COMPLETED |
| Two-stage CPU | `batch_cpu/submit_initial_lens_model_two_stage` | 8 cores, 8 pool workers | 25.5 min | 2 h 51 min | 3 h 17 min | COMPLETED |

The second GPU row is the same lens re-run after `scripts/initial_lens_model.py`
stopped applying the Numba sparse operator on the JAX path (it is the CPU route's
tool; the science tree never applied it under JAX). It is recorded here because
the obvious expectation — that the GPU route would get faster — is wrong, and the
next reader should not have to re-run it to find that out:

- `vis_lp` does not touch the operator in either run, and both runs took exactly
  15 quick-update blocks, so it is a clean node-speed probe: 2.01 min per block
  against 2.51, a **25% slower node** on the re-run.
- `vis_pix` went from 3.26 min per block to 4.55, a factor of 1.40. Divide out the
  1.25 node factor and about 1.12 is left — one run against one run on a shared
  cluster, which is inside the noise.
- The quick updates themselves are unchanged (13.8 s against 13.2 s mean).

So the operator was neither the reason the GPU route is slow nor a meaningful
speed-up on it. The fix stands as a correctness fix — the JAX path now matches the
science tree and fits the plain dataset — and the gap to the documented ~10 min
per lens has to come from somewhere else.

The CPU chain's second process logged `Fit Already Completed: skipping
non-linear search` for `vis_lp` and went straight to `vis_pix`. That is the
`--stage vis_pix` guard doing its job: it found the cached `vis_lp` result and
did not re-fit it.

A first GPU attempt did not get that far. The allocated A100 was in MIG mode
with no instances configured, so `cuInit(0)` failed with
`CUDA_ERROR_NO_DEVICE`, JAX fell back to its CPU backend, and the job ran the
whole fit on its single allocated core -- `vis_lp` took 45 minutes and `vis_pix`
was about 5% done when the two-hour wall killed it. `nvidia-smi` reported a
healthy A100 throughout, and nothing in the job's own output said the GPU had
gone. Every script in `batch_gpu/` now checks `jax.default_backend()` before it
starts, so a GPU job whose JAX backend is CPU exits within seconds, naming the
node, instead of burning its entire walltime.

That guard also trips when the Python environment was never activated, because
then JAX is not importable at all and the check fails the same way. If it fires,
read the `.err` file before blaming the node: `No local .venv found (set
PYAUTO_HPC_BASE ...)` followed by `ModuleNotFoundError: No module named 'jax'`
means `activate.sh` found nothing, not that the GPU is missing. `sbatch` does not
carry your login shell's exports into the job unless you pass them, so submit with
`--export=ALL,PROJECT_PATH=...,PYAUTO_HPC_BASE=...` or set both in your cluster
shell profile.

## Setting up the cluster

1. **The environment.** Every submit script runs `source $PROJECT_PATH/activate.sh`.
   That file activates a local `.venv` if one exists, otherwise a shared PyAuto
   install named by `PYAUTO_HPC_BASE` (a directory holding a `PyAuto/` virtualenv
   beside editable checkouts of the PyAuto libraries). Export it in your cluster
   shell profile, or edit the default in `activate.sh`.
2. **The project path.** Export `PROJECT_PATH` to the project's root on the cluster
   before submitting; the scripts read it and it is the same location as
   `$HPC_BASE/$PROJECT_NAME` in `sync.conf`.
3. **The partition names.** The scripts use `--partition=cpu` and `--partition=gpu`.
   Rename them to your cluster's partitions, or override on the command line:
   `sbatch --partition=<name> hpc/batch_cpu/submit_initial_lens_model_two_stage`.
4. **Submit from the script's directory**, or via `hpc/sync submit`. The `-o` / `-e`
   directives are relative paths into `output/` and `error/` beside the script, and
   SLURM will not create them for you.
5. **Config for large runs.** The committed `config/` is laptop-friendly. Three
   `config/general.yaml` keys were set differently for the DR1 science runs and are
   worth changing in your cluster checkout (there is no override mechanism):
   `output.samples_to_csv: false` (one `samples.csv` per search adds up over
   thousands of lenses), `hpc.hpc_mode: true` (no GUI visualisation or screen
   logging; pair it with a larger `hpc.iterations_per_quick_update`), and
   `numba.cache: false` (Numba's on-disk cache is unreliable on NFS).

## Moving things with `sync`

`cp hpc/sync.conf.example hpc/sync.conf` and fill in your SSH alias, the storage
path and the project name (`sync.conf` is gitignored). Then, from the project root:

| Command | What it does |
|---|---|
| `hpc/sync push` / `push --no-data` | Upload code and config, with or without `dataset/`. |
| `hpc/sync submit cpu <script>` / `submit gpu <script>` | `sbatch` one of the scripts in `batch_cpu/` or `batch_gpu/`, from its own directory. |
| `hpc/sync push-submit cpu <script>` | Push, then submit. |
| `hpc/sync jobs` / `sacct` / `cancel <id>` | Queue, history, cancel. |
| `hpc/sync tail cpu` / `tail gpu` | Stream the live SLURM logs. |
| `hpc/sync logs` | Download only the SLURM logs (fast, use mid-run). |
| `hpc/sync pull` | Download the logs, then `output/` and the other result trees. |
| `hpc/sync wait-and-pull [secs]` | Poll until no jobs remain, then pull. |
| `hpc/sync status` / `check` / `du` | Dry-run transfer, connection test, remote disk usage. |

`hpc/sync help` lists everything.
