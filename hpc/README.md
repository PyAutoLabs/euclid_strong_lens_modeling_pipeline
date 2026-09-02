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
| A small subset of lenses (up to a few hundred) and a GPU node is available | **GPU** (recommended) | `batch_gpu/submit_initial_lens_model` | Both stages, `vis_lp` then `vis_pix`, in one process under JAX. About 10 minutes per lens. |
| A large sample and many CPU cores, or no GPU | **Two-stage CPU** | `batch_cpu/submit_initial_lens_model_two_stage` | One submission; `vis_lp` under JAX on the CPU backend, then `vis_pix` with the Numba sparse operator and a process pool, as two consecutive Python processes. |
| As above, but you want different walltime, memory or core counts per stage, or to re-run one stage alone | **Two-stage CPU, two jobs** | `batch_cpu/submit_initial_lens_model_vis_lp` then `batch_cpu/submit_initial_lens_model_vis_pix` | The same two stages as two array jobs. Submit the second once the first has finished. |
| A cluster where JAX is unavailable or broken | **Numba only** | `batch_cpu/submit_initial_lens_model` | Both stages in one process with JAX disabled. The `vis_lp` stage is markedly slower this way. |

The GPU route is the simplest and, per lens, by far the fastest. The two-stage CPU route
exists because a cluster with hundreds of CPU cores and few GPUs can fit a large sample
faster in aggregate: the `vis_lp` MGE fit is a good fit for JAX on the CPU backend, while
the pixelized `vis_pix` fit is fastest with the Numba sparse operator and a forked
process pool, which JAX's sparse linear algebra does not yet match. Under JAX the
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
directly, and a control test on a laptop did not reproduce a hang:

| Leg (`hpc/diagnostics/jax_fork_control.py`) | Machine | Pool | Wall | Outcome |
|---|---|---|---|---|
| `control`: JAX likelihood evaluated in-process, then a `use_jax=False` pooled Nautilus | WSL2, 8 logical CPUs, CPU JAX | 4 | 265 s | PASS |
| `control_real`: the script's own `fit(stage="vis_lp")` under JAX, then `fit(stage="vis_pix", use_cpu=True)` in the same process | same | 4 | 494 s | PASS |
| `subprocess`: JAX stage in a child of a JAX-free parent, then the pooled fit | same | 4 | 233 s | PASS |

Untested there: production sampler sizes (`n_live` 750 and 300 against 50 in the
test), multi-hour wall times, pools of 16 or more workers, and cluster cgroup limits.
Until the same run passes on a cluster node at production scale, keep the process
boundary. It costs nothing with the single-submission script, which already gives
you one submission and one place in the queue. `forkserver` and `spawn` are not an
option: PyAutoFit pins the `fork` start method because `forkserver` corrupts model
instances (PyAutoFit#1437).

To repeat the measurement on your cluster:

```bash
python hpc/diagnostics/jax_fork_control.py --leg all --cores $SLURM_CPUS_PER_TASK \
    --output /path/to/scratch/jax_fork_control
```

It is a diagnostic, never part of CI, and it writes `results.json` with one entry per
leg (start method, whether an XLA backend was initialised at the fork point, wall
time, outcome).

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
