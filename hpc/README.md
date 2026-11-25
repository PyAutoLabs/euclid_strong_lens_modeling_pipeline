# Running the pipeline on a cluster

This folder runs the pipeline under SLURM. For the DR1 sample it is one file,
`campaign/dr1.yaml`, which says what to fit, with what resources and in what order.
You drive it from your own machine with `hpc/sync`, which moves code and results and
runs the campaign commands on the cluster for you.

Two rules hold for every run:

- **Nothing is submitted without the human's yes.** `plan` and `pilot` only propose;
  `act submit` is the one command that starts jobs.
- **CPU work never goes on the `gpu` partition.** Not `gpu`, `ral,gpu` or `gpu,ral`,
  even when `ral` is busy. The campaign refuses such a request before it reaches
  SLURM.

## Getting started

Set up `hpc/sync` once (see [Setting up the cluster](#setting-up-the-cluster)), then,
from the project root:

1. **Build the manifest**, the list of lenses to fit, one `<sample> <dataset>` per line:

   ```bash
   python hpc/tools/build_campaign_manifest.py --sample dr1_sep1_rest
   ```

   It writes `hpc/campaign/dr1_manifest.txt` (not committed). A lens's line number is
   its index in every stage, so do not reorder the file once the campaign has started.
   `--exclude-file` leaves out lenses that are already done.

2. **Push** the code, the campaign and the manifest:

   ```bash
   hpc/sync push --no-data
   ```

3. **Plan.** This proposes resources, wall times and an ETA for each stage from earlier
   runs. It submits nothing:

   ```bash
   hpc/sync plan hpc/campaign/dr1.yaml
   ```

4. **Pick a pilot**, a small spread-out slice of the manifest to run first. It submits
   nothing either:

   ```bash
   hpc/sync pilot hpc/campaign/dr1.yaml
   ```

5. **Ask the human.** Show them the plan and the pilot. Go on only on their yes.

6. **Submit the pilot**, all three stages for the pilot's lenses:

   ```bash
   hpc/sync act submit hpc/campaign/dr1.yaml --indices <the pilot's indices>
   ```

   Add `--dry-run` first to print the batch scripts without submitting them.

7. **Watch it.** `status` collects the campaign on the cluster and prints a short
   digest: done, running, failed by cause, and anything that needs a decision.

   ```bash
   hpc/sync status hpc/campaign/dr1.yaml
   ```

   SED fits write to `output_sed/`, which `status` does not scan yet. For SED it
   sees only the stage's own record and the scheduler state, so an SED failure shows
   as a plain `failed` without the fit's own cause. Read the task's log for the cause.

8. **Resubmit failures.** `resubmit` writes a proposal sized from the failed tasks'
   own measurements (a longer wall time for timeouts, more memory for out-of-memory).
   It submits nothing. Once the human approves it, `act resubmit` submits it:

   ```bash
   hpc/sync resubmit hpc/campaign/dr1.yaml --stage vis_lp --class timeout
   hpc/sync act resubmit hpc/campaign/dr1.yaml resubmit_vis_lp_timeout.json
   ```

9. **Run the rest** with `act submit` over the remaining indices (or no `--indices`
   for the whole manifest), again only on the human's yes.

10. **Report.** At the end, `report` writes the compute used, energy and carbon, the
    failure census and a paragraph for a paper:

    ```bash
    hpc/sync report hpc/campaign/dr1.yaml
    ```

Every `act` (submit, resubmit, hold, release, throttle, update-pending, cancel) is
logged on the cluster in `campaign/euclid_dr1/actions.log`. The digest, lists and
proposals are pulled into `hpc/campaign/euclid_dr1/` (gitignored). Pull the fits
themselves with `hpc/sync pull`.

## The DR1 stages

| Stage | Runs | Starts when | Resources | Before the fit |
|---|---|---|---|---|
| `vis_lp` | `scripts/initial_lens_model.py --stage=vis_lp --setup=cpu`, JAX on the CPU | at once | 4 CPUs, 12 GB, 18 h, `ral`, up to 100 at a time | nothing |
| `vis_pix` | `scripts/initial_lens_model.py --stage=vis_pix --setup=cpu`, Numba sparse route, 8 pool processes (`--number_of_cores=auto`), BLAS pinned to 1 thread | that lens's `vis_lp` has succeeded | 8 CPUs, 16 GB, 24 h, `ral`, up to 60 | checks there is exactly one `vis_lp` zip |
| `sed` | `scripts/sersic_lens_model_waveband.py` into `output_sed/`, JAX on the CPU | that lens's `vis_lp` has succeeded | 8 CPUs, 16 GB, 3 h, `ral`, up to 40 | copies the `vis_lp` zip into `output_sed/` |

Each stage runs `hpc/batch_cpu/stage` once per lens. It reads the lens's manifest line,
writes a task record the campaign reads back, and runs the script. A failed check
before the fit marks the task `missing_seed`: resubmit `vis_lp` for that lens first.
SED only needs the `vis_lp` result, so it chains on `vis_lp` alone and runs alongside
`vis_pix`: a lens whose `vis_pix` fails still gets its SED fit. The SED wall time is short on purpose: its slow
tail is a hang in one band's search, and longer walls only burned node-hours.

Lens *i* of a stage waits only on lens *i* of the stage before (SLURM `aftercorr`),
so the stages overlap. The old chain scripts recorded that `aftercorr` never fired on
RAL's SLURM (22.05.9). A test on 2026-11-16 (SLURM 22.05.11) found that it fires within
seconds. It still fires after the upstream task's record has been purged (MinJobAge,
300 s), and a late fill submitted against a purged array runs too. A purged dependency
counts as met even if the upstream task failed, so the seed guards are what stop a lens
with no `vis_lp` result: `vis_pix` and `sed` refuse to start without exactly one
`vis_lp` zip and record `missing_seed`.

Each run also sets `PYAUTO_CONFIG_OVERLAY=hpc/config`, which layers `hpc/config/` over
`config/`. Today it only turns off Numba's disk cache, which is unreliable on the NFS
project directory with many jobs writing at once.

The campaign does not yet check each stage's result zip itself. A fit's zip is named
after its search identifier (`vis_lp/<hash>.zip`) and the campaign checks one fixed
path, so the checks are written out but commented in `campaign/dr1.yaml` until it can
match a pattern.

## Other routes and tools

The campaign is the route for the DR1 sample. The scripts in `batch_gpu/` and
`batch_cpu/` remain for smaller jobs and for the work around the fits.

| Your situation | Route | Script | What one job does |
|---|---|---|---|
| A small subset of lenses (up to a few hundred) and a GPU node is available | **GPU** | `batch_gpu/submit_initial_lens_model` | Both stages, `vis_lp` then `vis_pix`, in one process under JAX. Measured on one A100 80GB PCIe with the committed config: 1 h 14 min to 1 h 44 min per lens across two runs on two different nodes (`vis_lp` 30-38 min, `vis_pix` 42-64 min). |
| A handful of lenses on CPU nodes, one job each | **CPU, one job** | `batch_cpu/submit_initial_lens_model` | Both stages with `--setup=cpu`, as two Python processes in one job: `vis_lp` on JAX-CPU (NumPy where JAX is not installed), then `vis_pix` on the Numba sparse route. Both get the same allocation, so prefer the campaign beyond a few lenses. |
| One lens, by hand | **Template** | `batch_cpu/template` | The same two processes for one lens; copy it and edit the lens. |
| The SED chain on a free GPU node, small sample | **SED chain, GPU** | `batch_gpu/submit_sersic_waveband` | The same chain as the `sed` stage, on a GPU. Both SED analyses are `use_jax=True`, so only the backend differs. |
| Before building a manifest for new tiles | **Positions gate** | `batch_cpu/submit_positions_gate` | Runs `scripts/tools/positions_gate.py` per batch and writes `positions_meta.json` per tile. See [`../scripts/README.md`](../scripts/README.md). |
| The fits are finished and you want the catalogue | **Catalogue build** | `batch_cpu/submit_build_inspection_bundle`, or `submit_catalogue_chunked.sh` for DR1 | No fitting: builds `inspect/<sample>/` beside the results. `hpc/sync submit cpu submit_build_inspection_bundle --export=ALL,SAMPLE=dr1_sep1`. |
| Release work | **Release tools** | `submit_magnitudes`, `submit_structure_astrometry`, `submit_release_package`, `submit_v1_1_release_chain`, `submit_north_up_*`, `submit_render_north_up`, `submit_reload_wcs`, `submit_reload_coolest` | One-off producers and passes over finished results; each script's header says what it does. |

`hpc/sync submit cpu <script>` submits any of them; anything after the script name goes
to `sbatch` (for example `--export=ALL,SAMPLE=...`). `hpc/template.py` is a starting
point for a new pipeline script with the same command-line interface as the stage
scripts.

The GPU route is the fastest per lens. The CPU route exists because a cluster with
hundreds of CPU cores and few GPUs fits a large sample faster in aggregate: the
`vis_lp` MGE fit suits JAX on the CPU backend, while the pixelized `vis_pix` fit, with
JAX switched off, is fastest with the Numba sparse operator and a forked process pool
across many cores. Neither is chosen by hand any more: each stage passes
`--setup=cpu` (or the default `--setup=auto`, the same on a node with no GPU), and its
analysis resolves the designated setup for its own model when the fit starts —
`jax_cpu_dense_fp64` for `vis_lp`, `numba_cpu_sparse_fp64` (which attaches the sparse
operator) for `vis_pix` — and logs it as the fit's `Setup:` line. The GPU route
(`--setup=gpu` or `auto` on a GPU node) fits the plain dataset with JAX's own linear
algebra. `--number_of_cores=auto` (the default) resolves to one process for a JAX
likelihood, which never forks and lets JAX vectorise instead, and to the allocated
CPUs for a Numba one; each fit logs the outcome as its `Parallel:` line.
`--use_cpu` is a deprecated alias of `--setup=cpu`.

The GPU timings come from the acceptance runs below, with the committed,
laptop-friendly `config/`. The "around 10 minutes per lens" quoted in the
repository's `README.md` and `start_here.py` is the DR1 science-run figure, measured
under that run's own configuration; it has not been reproduced with the committed
config. The GPU range is node-to-node spread, not a setting you can choose, and the
gap to the science figure is tracked as
`PyAutoMind/draft/bug/euclid/gpu_per_lens_time_vs_documented_10_min.md`.

## The `--stage` argument

`scripts/initial_lens_model.py` takes `--stage {all,vis_lp,vis_pix}` (default `all`):

- `--stage vis_lp` runs the MGE lens-light and source fit and stops.
- `--stage vis_pix` loads the completed `vis_lp` result and runs the pixelized source
  fit. If that result is not complete it stops immediately with an error naming the
  missing output directory and the `--stage vis_lp` command to run first, instead of
  silently starting `vis_lp` from scratch (which could also collide with a `vis_lp`
  job still running on the same lens).
- `--stage all` runs both, which is what the GPU route uses. On CPU the campaign and
  the one-job scripts run `vis_lp` and `vis_pix` as separate processes.

`--skip_pix` still works as a deprecated alias of `--stage vis_lp`.

## Why the CPU route uses two processes

Each stage resolves its own setup, so the Numba stage forks its pool only in its own
process, never in one that has already initialised JAX. The two CPU stages also want
different environments. `vis_lp` under JAX wants the CPU
backend pinned (`JAX_PLATFORMS=cpu`, so it never grabs a GPU that was not allocated)
and every linear-algebra library given the full core count, because its parallelism
is threads. `vis_pix` wants every thread count pinned to `1`, because its parallelism
is `--number_of_cores` pool processes and giving each worker a full set of BLAS
threads oversubscribes the node. The campaign sets these per stage:
`hpc/batch_cpu/stage` pins every thread count to the allocated CPUs, and the `vis_pix`
stage sets `BLAS_THREADS=1` (which also caps Numba's own threads, and which the
setup's own thread pinning honours). These pins stay, set before Python starts, while
the RAL finding "fork() called from a process already using GNU OpenMP" (job 414150)
is open.

The SED chain has no such split, which is why the `sed` stage is a single process
with a single environment. Both of its stages —
`scripts/sersic_lens_model.py` and `scripts/lens_model_waveband.py` — build their
analysis with `use_jax=True`, so the whole chain is one JAX likelihood with the CPU
backend pinned, exactly like `vis_lp` above: no Numba stage, no pool, no fork boundary
to preserve. `scripts/sersic_lens_model_waveband.py` therefore passes no setup on: a
`--setup` (or the deprecated `--use_cpu`) given to the `sed` stage is ignored, and its
fits keep `use_jax=True`.

The stages also run in separate Python processes. This is a conservative default rather
than a measured necessity. PyAutoFit documents that a forked worker whose *likelihood*
touches JAX deadlocks in XLA compilation (`autofit/non_linear/search/nest/nautilus/search.py`),
and the DR1 science runs were submitted as two jobs on that basis. The `vis_pix`
likelihood under `--setup=cpu` is Numba, not JAX, so the documented case does not apply
directly, and a control test did not reproduce a hang on either a laptop or a
cluster node:

| Leg (`hpc/diagnostics/jax_fork_control.py`) | Machine | Pool | Wall | Outcome |
|---|---|---|---|---|
| `control`: JAX likelihood evaluated in-process, then a `use_jax=False` pooled Nautilus | WSL2, 8 logical CPUs, CPU JAX | 4 | 265 s | PASS |
| `control_real`: the script's own `fit(stage="vis_lp")` under JAX, then `fit(stage="vis_pix", use_cpu=True)` (now `setup="cpu"`) in the same process | same | 4 | 494 s | PASS |
| `subprocess`: JAX stage in a child of a JAX-free parent, then the pooled fit | same | 4 | 233 s | PASS |
| `control` | RAL `euclid-ral-compute-1`, 16 cores | 16 | 216 s | PASS |
| `control_real` | same | 16 | 378 s | PASS |
| `subprocess` | same | 16 | 220 s | PASS |

The cluster rows add a 16-worker pool under real cgroup limits, which the laptop
run could not exercise. What remains untested is production sampler size (`n_live`
750 and 300, against 50 in the control) and multi-hour wall times: every leg above
ran a small fit. Until a production-size run passes, keep the process boundary. It
costs little with the campaign, which submits both stages at once and chains them
lens by lens. The single-process route's draft was closed as superseded by
`--setup` (likelihood-lifecycle 5H); a production-size RAL witness of one process is
an optional follow-up that needs the human's go-ahead.
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
| Two-stage CPU | `batch_cpu/submit_initial_lens_model_two_stage` (retired; now the campaign's `vis_lp` and `vis_pix` stages) | 8 cores, 8 pool workers | 25.5 min | 2 h 51 min | 3 h 17 min | COMPLETED |

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

1. **`hpc/sync`.** `cp hpc/sync.conf.example hpc/sync.conf` and fill in your SSH alias,
   the storage path and the project name (`sync.conf` is gitignored). For the DR1
   campaign on RAL, `HPC_BASE=/mnt/ral/jnightin/complete` and `PROJECT_NAME=euclid_dr1`,
   so that `$HPC_BASE/$PROJECT_NAME` is the campaign's `project_root`. `hpc/sync check`
   tests the connection.
2. **The environment.** Every job runs `source activate.sh`. That file activates a local
   `.venv` if one exists, otherwise the PyAuto install named by `PYAUTO_HPC_BASE` (a
   directory holding a `PyAuto/` virtualenv beside editable checkouts of the PyAuto
   libraries). The campaign sets `PYAUTO_HPC_BASE` in each stage's `env`. Use your own
   install, never a shared one other people's runs depend on. The campaign commands
   run `python -m autofit.campaign` on the login node through the same `activate.sh`
   (override with `HPC_ENV` and `HPC_PYTHON` in `sync.conf`).
3. **Temp files and caches.** RAL compute nodes have a small local disk.
   `cache_root` in `campaign/dr1.yaml` sends `TMPDIR` and the tool caches to the
   project filesystem instead.
4. **The partition names.** The campaign and every `batch_cpu/` script use
   `--partition=ral`; the `batch_gpu/` scripts use `--partition=gpu`, because RAL names
   its CPU partition `ral` (it has no `cpu` partition). On another cluster, change the
   partitions in `campaign/dr1.yaml` and the scripts. **Never put CPU work on `gpu`**
   (human rule, 2026-09-30: `ral,gpu` CPU arrays took all 124 CPUs on both A100 nodes
   and idled all 8 GPUs for hours). The only exemption is a small CPU timing leg
   without `--gres` (at most 8 CPUs per task, throttle at most `%2`, no pending GPU jobs
   in `squeue -p gpu -t PD`). A partition name is not a device: check `gres/gpu` in
   `scontrol show job <id>` before calling a job a GPU run.
5. **Submit the hand-written scripts from their own directory**, or via
   `hpc/sync submit`. Their `-o` / `-e` directives are relative paths into `output/`
   and `error/` beside the script, and SLURM will not create them for you. The
   campaign writes its logs to `hpc/batch_cpu/output/` itself.
6. **Config for large runs.** The committed `config/` is laptop-friendly. Three
   `config/general.yaml` keys were set differently for the DR1 science runs:
   `output.samples_to_csv: false` (one `samples.csv` per search adds up over
   thousands of lenses), `hpc.hpc_mode: true` (no GUI visualisation or screen
   logging; pair it with a larger `hpc.iterations_per_quick_update`), and
   `numba.cache: false` (Numba's on-disk cache is unreliable on NFS). The campaign
   sets the last one through `hpc/config/`; put the others there too if you want
   them on the cluster only.

## Moving things with `sync`

From the project root:

| Command | What it does |
|---|---|
| `hpc/sync push` / `push --no-data` | Upload code and config, with or without `dataset/`. |
| `hpc/sync plan` / `pilot` / `status` / `resubmit` / `report` `hpc/campaign/dr1.yaml` | The campaign commands above. None of them submits anything. |
| `hpc/sync act submit hpc/campaign/dr1.yaml [--stage S] [--indices SPEC] [--dry-run]` | Submit campaign stages. The only command that starts campaign jobs. |
| `hpc/sync act <hold\|release\|throttle\|update-pending\|cancel> euclid_dr1 <job> ...` | Change queued campaign jobs, logged in `actions.log`. |
| `hpc/sync submit cpu <script>` / `submit gpu <script>` | `sbatch` one of the scripts in `batch_cpu/` or `batch_gpu/`, from its own directory. |
| `hpc/sync jobs` / `sacct` / `cancel <id>` | Queue, history, cancel. |
| `hpc/sync tail cpu` / `tail gpu` | Stream the live SLURM logs. |
| `hpc/sync logs` | Download only the SLURM logs (fast, use mid-run). |
| `hpc/sync pull [dir ...]` | Download the logs, then `output/`, `output_sed/` and `inspect/`; name roots (`pull inspect`) to fetch only those. |
| `hpc/sync wait-and-pull [secs]` | Poll until no jobs remain, then pull. |
| `hpc/sync status` / `check` / `du` | Dry-run transfer, connection test, remote disk usage. |

`hpc/sync help` lists everything.
