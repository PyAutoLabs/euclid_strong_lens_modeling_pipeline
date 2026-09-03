# Euclid Strong Lens Modeling Pipeline — Agent Guidance

Agent-agnostic instructions for this repository (Claude Code loads them via the
`@AGENTS.md` import in `CLAUDE.md`). It runs the Euclid strong lens modeling
pipeline built on **PyAutoLens**: automated lens models (SIE + shear mass, MGE
light) fitted to Euclid VIS imaging. `scripts/initial_lens_model.py` is the entry
point; the other `scripts/` pipelines chain off it for pixelized sources, Sersic
photometry and multi-band SED fits, and `catalogue/` turns finished fits into the
per-lens inspection bundle. `start_here.py` in the root is a **thin shim** over
`scripts/initial_lens_model.fit` — read and edit the script, not the shim.

Read before editing: [`README.md`](README.md) (install, script tables, the shared
CLI arguments, dataset requirements, testing and CI),
[`scripts/README.md`](scripts/README.md) (what each pipeline, diagnostic and tool
does) and [`docs/drift_report.md`](docs/drift_report.md) — the record of the DR1
pipeline-parity port: what was inherited from the science runs, what was
deliberately not, and why the non-obvious decisions (the Delaunay switch, the
latent API, the noise-mask fallback chain) were made.

## Repository Structure

```
util.py             # Shared: dataset loading, analysis, the latent catalogue, arg parsing
scripts/            # The fitting pipelines and the simulator  (scripts/README.md)
  tools/            #   Diagnostics and catalogue tooling, kept out of the pipelines
catalogue/          # Producers of the inspection bundle and master CSVs  (catalogue/README.md)
preprocess/         # Segmentation diagnostics, mask-tuning GUIs, FITS movers
workflow/           # Post-run analysis examples: csv_make.py, png_make.py, fits_make.py
tools/              # GUI utilities (extra-galaxies masking, PSF sizing)
hpc/                # SLURM submit scripts, the sync tooling, activate.sh's venv
config/             # PyAutoLens YAML; build/ holds the smoke profile + the no_run skip list
dataset/            # Input data: dataset/<sample>/<dataset_name>/  (dataset/README.md)
output/             # Results (generated at runtime, not committed)
tests/              # pytest suites, .github/ their CI workflows (see "Testing")
smoke_tests.txt     # Scripts the PyAutoHeart smoke runner executes, in order
```

## Running Scripts

Everything runs **from the repository root**. The shared argument parser, the two
environment variables and the per-script tables are in
[`README.md`](README.md#command-line-arguments) and
[`scripts/README.md`](scripts/README.md), not duplicated here. What is not there:

- `util.parse_fit_args()` returns a **6-tuple**: `(sample_name, dataset_name,
  iterations_per_quick_update, number_of_cores, use_cpu, stage)`.
- The diagnostics take their own arguments. `scripts/tools/diagnose_latent.py`
  accepts `--dataset` / `--sample` / `--output_path` / `--unique_tag` /
  `--search` / `--result_hash`; `scripts/tools/diagnose_latent_vis_pix.py` is a
  population sweep taking **no** `--dataset` (`--sample` / `--limit` /
  `--traceback` instead).
- `PYAUTO_TEST_MODE=1` makes every search finish almost instantly on trivial
  samples; `2` also skips the sampler (the `/smoke_test` skill's mode). Results
  land under `<output>/test_mode/`, never `<output>/`. Use it to check a script
  executes before a real fit or an HPC submission.
- From Codex or any restricted environment, point the caches somewhere writable:
  `NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib`.

## Testing

Commands and the CI job table are in README's "Testing and Continuous
Integration". `pytest.ini` sets `testpaths = tests` and registers one marker,
`slow`. The fast suite is deliberately **JAX-free** — no `use_jax=True`, no
non-linear search — and a few seconds end to end.

- `test_compute_latent_variable.py` — known-answer tests for all 12 latents
  against `dataset/simulated/euclid_dr1_like/truth.json`, asserted both as a
  bit-identical replay of `util.LatentEuclid` and against the *independent* truth
  blocks the simulator computed without it. Those two differ by documented
  physical offsets (the latents integrate the masked grid, the truth fluxes the
  whole frame) — read `truth["conventions"]` before changing a tolerance.
- `test_catalogue_parity.py` — each `catalogue/scripts/` producer's reconstructed
  CSV header versus the four DR1 reference header lines under
  `tests/data/dr1_headers/`; refresh those with `head -1` of the DR1 CSVs. Plus
  `test_util.py` (the pure helpers) and `test_repo_invariants.py` (the rules
  below).
- `test_latent_run_level.py` (`slow`, its own CI job) — one real-mode fit proving
  the pipeline still **writes** latents, which the fast suite cannot: it calls
  `LatentEuclid.variables` directly, an ungated path, while the write is gated by
  `autonerves.test_mode.skip_latents()` and nothing forces latents on under test
  mode. It draws with `af.Drawer(total_draws=10)` on a model anchored to the truth
  with one free parameter (over a free model a Drawer draws junk) and asserts
  `files/latent/latent_summary.json` holds exactly the 12 keys — a NaN latent is
  dropped from that file entirely, so the key count *is* the NaN check.

### Smoke tests

`smoke_tests.txt` lists what PyAutoHeart's smoke runner executes. Four things are
load-bearing when editing it:

- Entries run **in order, in one output directory** wiped once before the first.
  `scripts/tools/diagnose_latent.py` reads a finished result, so it must stay
  below the scripts that produce one.
- One `args_default` from `config/build/profile_smoke.yaml` is appended to
  **every** entry, so a script rejecting `--dataset` cannot be listed — hence
  `scripts/tools/diagnose_latent_vis_pix.py` living in `no_run.yaml`, and
  `scripts/simulator.py` accepting and ignoring those two arguments.
- Per-script environment goes in `profile_smoke.yaml`'s `overrides:`, matched by
  path substring. An in-file declaration is an `__Env__` docstring section opened
  with a bare `"""` and carrying an `ENV:` line with no leading `#`; the old
  column-0 `# ENV:` form was removed and now **raises**.
- `no_run.yaml` is **not** consulted when the runner is given the
  `smoke_tests.txt` allowlist — it is policy for the release mega-run, so a
  script legitimately appears in both. Every entry carries an inline reason.

## Continuous Integration

The workflows, jobs and what each red X means are tabulated in README's "Testing
and Continuous Integration". Both are thin callers into PyAutoHeart's reusable
`smoke-tests.yml@main`, which owns the dependency-chain checkout at the matching
branch and the five-library **source** install. `unit` and `slow` are two jobs on
purpose: "a latent value is wrong" and "the pipeline stopped writing latents at
all" should not arrive as the same red X. The smoke runner's `--report-dir` is
load-bearing — `build_util.execute_script` only records a failure and carries on
when a report was built, and `run_python.py` only propagates it when one exists;
without it the gate aborts at the first break *and* exits 0.

### Rules CI enforces (`tests/test_repo_invariants.py`)

1. **Nothing auto-simulates.** Only `scripts/simulator.py` may bring a dataset
   into being — no `SimulatorImaging`, `should_simulate` or `auto_simulate`
   anywhere else. Every fitting script reads a dataset off disk, the way a user
   with real Euclid data runs it; one that quietly simulates when it cannot find
   a dataset turns a broken data path into a green run.
2. **Every script is accounted for.** Each `*.py` under `scripts/` (recursively,
   so `scripts/tools/` included), `catalogue/scripts/`, `preprocess/`, `tools/`,
   `workflow/`, `.github/scripts/` and the repository root is either listed in
   `smoke_tests.txt` or excluded in `config/build/no_run.yaml` **with a written
   reason**. Doing neither fails the `unit` job.
3. **The allowlist is not stale.** Every `smoke_tests.txt` entry exists on disk.

## Dataset Layout

```
dataset/<sample_name>/<dataset_name>/
├── <dataset_name>.fits       # Multi-HDU FITS (VIS + NIR/EXT bands)   [required]
├── info.json                 # pixel_scale, mask_radius, mask_centre  [required]
├── segmentation/             # DR1 preprocessing: artefact_binary.fits (noise-scaling
│                             #   mask), lens_flux.fits (mask centre), source_flux.fits
│                             #   (positions, if positions.json is absent)  [optional]
├── mask_extra_galaxies.fits  # Fallback noise-scaling mask            [optional]
├── positions.json            # Multiple-image positions               [optional]
├── segmentation.png          # Collected into the inspection bundle   [optional]
└── rgb_0.png / rgb_1.png / rgb_0.jpg   # RGB thumbnails               [optional]
```

`util.load_vis_dataset()` reads all of it. Everything optional **degrades
gracefully** — the fit still runs, with reduced information:

| Input | Preferred | Falls back to | If neither |
|---|---|---|---|
| Noise-scaling mask | `segmentation/artefact_binary.fits` | `mask_extra_galaxies.fits` | No noise scaling. A mask whose shape does not match the cut-out is refused. |
| Mask centre | peak of `segmentation/lens_flux.fits` | `info["mask_centre"]` | Frame centre `(0, 0)`. |
| Positions | `positions.json` (2+ positions) | derived from `segmentation/source_flux.fits` + the VIS noise map | No positions likelihood. |

The FITS **primary** header must also carry the `WORST_BAND` / `WORST_PSF_*`
contract for the aperture-flux latents. It is stamped by the upstream Euclid
cut-out generator — neither this pipeline nor PyAutoReduce writes it — and is
documented where it is read, in `util.py` (`load_vis_dataset` for `WORST_BAND`
and its two degradation paths, `psf_fwhm_arcsec_from_primary_header` for the FWHM
key order and the `-99` sentinel).

[`dataset/README.md`](dataset/README.md) is the per-dataset table. Two ship:
`q1_walsmley/102018665_NEG570040238507752998/`, **real** Q1 imaging and every
script's default, which ships no `segmentation/` and no `positions.json` and so
exercises the *fallback* branches above; and `simulated/euclid_dr1_like/`, the
`scripts/simulator.py` mock, which ships everything the real one lacks and so
exercises the *preferred* branches, builds a complete 13-of-13 inspection bundle,
and whose `truth.json` is the known-answer source for the latent tests.

Two gotchas before adding a dataset:

- `.gitignore` carries `*.fits`, so **new committed FITS need `git add -f`** —
  without it they silently do not stage and the dataset lands half-committed.
- `preprocess/segmentation.py` **overwrites `positions.json`** from the local
  maxima of `segmentation/source_flux.fits`, discarding exact positions a
  simulator solved for. Back it up, or re-run the simulator (deterministic for a
  given `--seed`).

`scripts/simulator.py` is the only producer of simulated data (invariant 1);
[`scripts/README.md`](scripts/README.md) covers its two modes and the `truth.json`
it writes. Under `PYAUTO_TEST_MODE` it redirects to
`$PYAUTO_OUTPUT_DIR/simulator/<sample>/<dataset>/` so the smoke entry can never
overwrite the committed dataset (`--force-dataset-dir` overrides that).

## HPC

`hpc/` holds the `batch_gpu/` / `batch_cpu/` SLURM submit scripts (with their
`output/` and `error/` logs) and the `sync` script (`push | pull | sync | status`
plus `submit`, `push-submit`, `jobs`, `tail`, `logs`, `wait-and-pull` and more
(`hpc/sync help`)). Route choice, the `--stage` argument, the per-stage
environment blocks, the measured process-boundary test and the full `hpc/sync`
verb list are in `hpc/README.md`; the three cluster `config/general.yaml` keys
below are also repeated there.

The shipped `config/` is deliberately **laptop-friendly**. Three
`config/general.yaml` keys were different in the DR1 science runs — cluster
choices rather than sensible public defaults, so change them in your cluster
checkout (there is no override mechanism) and leave the repository defaults
alone: `output.samples_to_csv` → `false` (at 20,000 tiles a `samples.csv` per
search is a lot of disk; local users want them), `hpc.hpc_mode` → `true` (drops
GUI visualization and screen logging — pair with
`hpc.iterations_per_quick_update`, repo `10000`, science runs `100000`), and
`numba.cache` → `false` (Numba's on-disk cache is unreliable on NFS).
`docs/drift_report.md` §8 records why the rest of the science runs' config was
**not** adopted: it is stale relative to the installed library, not ahead of it.

## Not ported — available in `Science/euclid`

The DR1 science runs were driven from a private tree (`Science/euclid`). Phase 1
ported the analysis chain, `util.py`, the catalogue producers and the docs here;
some things were deliberately left behind. **If you need one, it is in that tree
— do not re-derive it from scratch without checking there first.**

`docs/drift_report.md` §10 is the full list with a reason for each. In summary:
the pixelized-source SED variants (`sersic_lens_model_pix*.py`,
`multi_lens_model_pix_waveband.py`), superseded by the Delaunay Source Pix stages
in `scripts/full_model.py`; `galaxy_sersic_model.py` (non-lens galaxy fitting,
out of scope); the one-offs against a private results tree
(`audit_sed_outputs.py`, `audit_unresolved_hpc.sh`, `reorganize_normies.py`);
`lens_model_waveband.py::fit_waveband_galaxy` / `::fit_waveband_pix` (no caller
in this chain); `preprocess/recenter_lens_centre.py` (the mask-centre fallback
chain handles its case); and 21 catalogue scripts, listed individually in
[`catalogue/README.md`](catalogue/README.md#not-ported--available-in-scienceeuclid).

## Scientific Context

For the science behind this pipeline — what Euclid is finding, why SIE + shear is
a reasonable default mass model, what MGE buys, how multipoles and external shear
affect substructure / cosmography downstream — see the lensing sub-wiki at
[`PyAutoLabs/PyAutoMemory`](https://github.com/PyAutoLabs/PyAutoMemory), locally
at `../PyAutoMemory/lensing_wiki/`: `entities/euclid-q1.md`,
`concepts/mass-models.md`, `concepts/multipoles.md`,
`concepts/external-convergence-shear.md`, `concepts/lens-finding.md`,
`entities/slam-pipeline.md`.

## Line Endings — Always Unix (LF)

All files must use Unix line endings (`\n`); CRLF breaks shell scripts on the
HPC. Verify with `file <path>` and convert with `dos2unix` if needed.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->

<!-- repos_sync:deliverable:begin -->
## Sessions end at their deliverable

A session ends when it reports its deliverable — never arm anything that
outlives the turn to wait for CI, a review or a merge: no `send_later`, no
`subscribe_pr_activity`, no `CronCreate`, no `ScheduleWakeup`, no `/loop`, no
`RemoteTrigger` create/update/run. Judge once, report, stop; the human re-runs
`/prm` (or the batch review) when it is green. Measured: five batch members
armed hourly check-ins on 2026-08-31, and a mobile `/prm` re-armed a 60-minute
`send_later` hourly all night on 2026-09-03 with no task active, draining usage.
<!-- repos_sync:deliverable:end -->
