# Euclid Strong Lens Modeling Pipeline — Agent Guidance

This file is for AI coding agents (Claude Code, Codex, Cursor, etc.) and humans
working in this repository. It is the agent-agnostic source of truth; Claude Code
loads it via the `@AGENTS.md` import in `CLAUDE.md`.

This repository provides the Euclid strong lens modeling pipeline, built on **PyAutoLens**. It fits automated lens models (SIE + shear mass, MGE light) to Euclid VIS imaging data. `scripts/initial_lens_model.py` is the entry point; the other `scripts/` pipelines chain off it for pixelized sources, Sersic photometry and multi-band SED fits, and `catalogue/` turns finished fits into the per-lens inspection bundle.

`start_here.py` in the repository root is a **thin shim** over `scripts/initial_lens_model.fit` — it exists only so older commands and bookmarks keep working. Read and edit `scripts/initial_lens_model.py`.

`docs/drift_report.md` is the durable record of the DR1 pipeline-parity port: what was inherited from the DR1 science runs, what was deliberately not, and why the non-obvious decisions (the Delaunay switch, the latent API, the noise-mask fallback chain) were made. Read it before changing any of them.

## Scientific Context

For the science behind this pipeline — what Euclid is finding, why SIE
+ shear is a reasonable default mass model, what MGE buys, how
multipoles and external shear affect substructure / cosmography
downstream — see the lensing sub-wiki at
[`PyAutoLabs/PyAutoMemory`](https://github.com/PyAutoLabs/PyAutoMemory),
locally at `../PyAutoMemory/lensing_wiki/`. Most directly relevant:
`entities/euclid-q1.md`, `concepts/mass-models.md`,
`concepts/multipoles.md`, `concepts/external-convergence-shear.md`,
`concepts/lens-finding.md`, `entities/slam-pipeline.md`.

---

## Repository Structure

```
start_here.py              # Thin shim over scripts/initial_lens_model.fit
util.py                    # Shared utilities: dataset loading, analysis, latents, arg parsing
activate.sh                # HPC venv activation (sets PYTHONPATH)
config/                    # PyAutoLens YAML configuration files
  latent.yaml              #   Which library latents this pipeline enables
  build/                   #   Smoke/CI profile (profile_smoke.yaml) and skip list (no_run.yaml)
dataset/                   # Input data: dataset/<sample>/<dataset_name>/
  README.md                #   Every shipped dataset, what reads it, how to regenerate it
output/                    # Results (generated at runtime, not committed)
docs/drift_report.md       # Record of the DR1 pipeline-parity port
.github/                   # CI: two workflows + their runners (see "Continuous Integration")
scripts/                   # The pipelines:
  initial_lens_model.py    #   ENTRY POINT — SIE + shear, MGE light, MGE source, Delaunay source pix
  sersic_lens_model.py     #   Sersic lens+source fits for SED photometry (chains off vis_lp)
  lens_model_waveband.py   #   Non-VIS bands with the VIS lens model fixed
  sersic_lens_model_waveband.py  # SED chain driver: vis_lp -> Sersic -> every band
  mge_lens_only.py         #   MGE lens-light-only subtraction
  full_model.py            #   Full SLaM pipeline (Delaunay source pix + power-law mass)
  diagnose_latent.py       #   Replay the latent catalogue on one converged result
  diagnose_latent_vis_pix.py  # The same replay swept over a sample's vis_pix results
  build_inspect.py         #   Collect the inspection bundle's PNGs from result zips
  build_inspection_bundle.sh  # Run all seven catalogue stages in order
  simulator.py             #   The ONLY producer of simulated data (--from-params/--from-result)
catalogue/                 # Producers of the per-lens inspection bundle and master CSVs
  README.md                #   13-file -> producer table, run order, upstream fits per stage
  scripts/                 #   deblending, lens_mass, lens_sersic, source_sersic,
                           #   multi_wavelength, magnitudes (+ shared catalogue_util.py)
preprocess/                # Preprocessing tools (all accept --sample):
  segmentation.py          #   Segmentation diagnostics + positions.json
  adjust_binary.py         #   GUI binary mask tuning (--object=artefact|source|lens)
  validation_GUI.py        #   Annotation GUI for segmentation QA
  move_segmentation_fits.py #  Move segmentation FITS into dataset folders
workflow/                  # Post-run analysis: csv_make.py, png_make.py, fits_make.py
tools/                     # GUI utilities (extra galaxies masking, PSF sizing)
hpc/                       # SLURM submit scripts and sync tooling
tests/                     # pytest suites (see "Testing")
  test_util.py             #   The shared helpers in util.py
  test_compute_latent_variable.py  # Known-answer latent values against truth.json
  test_catalogue_parity.py #   Catalogue columns vs the DR1 reference headers
  test_repo_invariants.py  #   The rules CI enforces on new scripts
  test_latent_run_level.py #   `slow`: a real fit still WRITES the latents
  data/dr1_headers/        #   The four DR1 CSV header lines, checked in as fixtures
pytest.ini                 # testpaths + the `slow` marker
smoke_tests.txt            # Scripts the PyAutoHeart smoke runner executes, in order
```

---

## Running Scripts

Scripts are run from the repository root:

```bash
python scripts/initial_lens_model.py --dataset=102018665_NEG570040238507752998 --sample=q1_walsmley

python scripts/full_model.py --dataset=102018665_NEG570040238507752998 --sample=q1_walsmley
```

Every fitting pipeline shares one parser, `util.parse_fit_args()`, which returns a
**6-tuple** `(sample_name, dataset_name, iterations_per_quick_update, number_of_cores, use_cpu, skip_pix)`:

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | *required* | Dataset subdirectory inside `dataset/<sample>/`. |
| `--sample` | none | Sample subdirectory inside `dataset/`. Omit for a flat layout. |
| `--iterations_per_quick_update` | `5000` | Sampler iterations between on-the-fly visualisation updates. |
| `--number_of_cores` | `1` | CPU cores for the non-JAX Nautilus searches. |
| `--use_cpu` | off | Disables JAX and applies the CPU sparse operator to the pixelized stage. |
| `--skip_pix` | off | Return after the MGE light-profile fit; skip the pixelized source stage. |

`mask_radius` is never an argument — it is always read from the dataset's `info.json`.

`PYAUTO_OUTPUT_DIR` (default `output`) relocates the results tree relative to the
repository root. `scripts/sersic_lens_model_waveband.py` exists to be run under
its own output dir so the many per-band results do not bloat the main tree, and
the catalogue's stages 6-7 read that separate tree by name.

The diagnostics take their own arguments. `scripts/diagnose_latent.py` accepts
`--dataset` / `--sample` / `--output_path` / `--unique_tag` / `--search` /
`--result_hash`; `scripts/diagnose_latent_vis_pix.py` is a population sweep and
takes **no** `--dataset` (`--sample` / `--limit` / `--traceback` instead).

---

## Test Mode Runs

Set `PYAUTO_TEST_MODE=1` to make all non-linear searches complete almost instantly with trivial samples. Use this to verify the full pipeline executes without errors before submitting to the HPC or running a real fit. Set `PYAUTO_TEST_MODE=2` to additionally skip the sampler step entirely — fastest mode, used by the `/smoke_test` skill.

Test-mode results are written under `<output>/test_mode/`, not `<output>/`.

```bash
D="--dataset=102018665_NEG570040238507752998 --sample=q1_walsmley"

PYAUTO_TEST_MODE=1 python scripts/initial_lens_model.py $D
PYAUTO_TEST_MODE=1 python scripts/full_model.py $D
PYAUTO_TEST_MODE=1 python scripts/sersic_lens_model.py $D
PYAUTO_TEST_MODE=1 python scripts/sersic_lens_model_waveband.py $D
PYAUTO_TEST_MODE=1 python scripts/mge_lens_only.py $D
PYAUTO_TEST_MODE=1 python scripts/lens_model_waveband.py $D
```

The example dataset lives at `dataset/q1_walsmley/102018665_NEG570040238507752998/`;
swap in `--dataset=euclid_dr1_like --sample=simulated` to run the same commands on
the committed mock.

---

## Testing

```bash
python -m pytest -q -m "not slow"      # 58 tests, ~4 s — the local default
python -m pytest -q -m slow            # 3 tests, 10-20 s — one real (non-test-mode) fit
python -m pytest -q                    # both
python3 .github/scripts/run_smoke.py   # the 9 scripts in smoke_tests.txt, in test mode
```

`pytest.ini` sets `testpaths = tests` and registers the one marker, `slow`.

### The fast suite (`-m "not slow"`)

Deliberately **JAX-free** — no `use_jax=True`, no non-linear search — and a few
seconds end to end. Four modules:

- `tests/test_util.py`: the pure helpers in `util.py` — the six-tuple CLI, the
  noise-scaling mask fallback chain, the `WORST_PSF_*` header contract, the
  `positions.json` fallback maths, and the latent **key set**.
- `tests/test_compute_latent_variable.py`: known-answer tests for all 12 latents
  against `dataset/simulated/euclid_dr1_like/truth.json`. Each latent is asserted
  **twice**: once that replaying `util.LatentEuclid` on the truth model reproduces
  `truth["latents"]` (a regression check — bit-identical in practice), and once
  against the *independent* truth blocks the simulator computed without touching
  `LatentEuclid`. The second set carries real tolerances, because the two sides
  differ by documented physical offsets: the latents integrate the **masked** grid
  (`mask_radius = 3"`) while the truth fluxes integrate the whole 100x100 frame.
  `truth["conventions"]` spells each offset out — read it before changing a
  tolerance.
- `tests/test_catalogue_parity.py`: the four DR1 reference CSV header lines are
  checked in verbatim under `tests/data/dr1_headers/`, and the header each
  `catalogue/scripts/` producer would write is reconstructed from its own
  `add_label_column` / `add_variable` calls (read with `ast`, with the suffix
  expansion delegated to autofit's own `Column.value`) and compared exactly, order
  included. To refresh the fixtures, re-copy `head -1` of the four DR1 reference
  CSVs.
- `tests/test_repo_invariants.py`: the rules below, enforced.

### The `slow` suite (`-m slow`)

`tests/test_latent_run_level.py` — one real-mode fit, 10-20 s depending on the
machine, in its own CI job. The fast suite proves the latent *values* are right
by calling `LatentEuclid.variables` directly, which is an **ungated** path: it
would keep passing even if the pipeline never wrote a latent again. The write is gated by
`autonerves.test_mode.skip_latents()` (`is_test_mode() or PYAUTO_SKIP_LATENTS`,
consumed at one line of PyAutoFit's `updater.py`) and **no environment variable
forces latents on under test mode** — so only a fit with `PYAUTO_TEST_MODE` unset
can see it.

Three things about that fit:

- It uses `af.Drawer(total_draws=10)`, not a sampler: the cheapest real search in
  PyAutoFit, while still running the full post-fit updater path. (The pipeline
  scripts hard-code `n_live=750` for Nautilus with no override; that is not a CI
  fit.) `latent_draw_via_pdf_size` is inert for `af.Drawer` — it falls back to all
  samples — so there is no knob to turn there.
- The model is **anchored on the truth** with one free parameter. A Drawer over a
  fully free model draws junk: the source's linear intensity solves to exactly
  zero and `magnification` becomes `0 / 0`.
- The assertion is that `files/latent/latent_summary.json` holds exactly the 12
  keys. **A NaN latent is dropped from that file entirely**, so the key count *is*
  the NaN check — a latent that failed to compute is a missing key, never a NaN
  value. (The file is `latent_summary.json` and not `files/latent.csv` because
  `config/output.yaml` sets `latent_draw_via_pdf: true`, on which branch the
  updater only calls `save_samples_summary`.)

### Smoke tests

`smoke_tests.txt` lists the scripts PyAutoHeart's smoke runner executes. Four
things about that runner are load-bearing when editing the file:

- Entries run **in order, in one output directory** that is wiped once before the
  first entry. `scripts/diagnose_latent.py` reads a finished result, so it must
  stay below the scripts that produce one.
- One `args_default` from `config/build/profile_smoke.yaml` is appended to
  **every** entry. A script that rejects `--dataset` cannot be listed —
  `scripts/diagnose_latent_vis_pix.py` is in `config/build/no_run.yaml` for
  exactly that reason. `scripts/simulator.py` accepts and ignores those two
  arguments so that it can be listed.
- Per-script environment goes in `profile_smoke.yaml`'s `overrides:`, matched by
  path substring. `scripts/diagnose_latent.py` uses one
  (`PYAUTO_OUTPUT_DIR: output/test_mode`) so it resolves the test-mode results
  the earlier entries wrote. An in-file declaration, if one is ever needed, is an
  `__Env__` docstring section opened with a bare `"""` and carrying an `ENV:`
  line with no leading `#` — the old column-0 `# ENV:` comment form was removed
  and now **raises**.
- `config/build/no_run.yaml` is **not** consulted when the runner is given the
  `smoke_tests.txt` allowlist; it is policy for the release mega-run. A script
  legitimately appears in both. It carries a reason inline for every entry.

---

## Continuous Integration

Two workflows, three jobs, each on the ubuntu x Python 3.12/3.13 matrix with
`fail-fast: false`. Both are thin callers into PyAutoHeart's reusable
`smoke-tests.yml@main`, which owns the ceremony — dependency-chain checkout at
the matching branch, the five-library **source** install, the cache dirs, the
env-profile validation.

| Workflow | Job | Runs | A red X means |
|---|---|---|---|
| `.github/workflows/smoke_tests.yml` | `smoke` | Every `smoke_tests.txt` entry under `PYAUTO_TEST_MODE`, via `.github/scripts/run_smoke.py` | **A script broke.** |
| `.github/workflows/tests.yml` | `unit` | `pytest -m "not slow"`, via `.github/scripts/run_tests.py` | **A latent value is wrong**, a catalogue column drifted from the DR1 reference, or a repository invariant broke. |
| `.github/workflows/tests.yml` | `slow` | `pytest -m slow`, via `.github/scripts/run_tests.py --slow` | **The pipeline stopped writing latents.** |

`unit` and `slow` are two jobs and not two legs of one on purpose: "a latent value
is wrong" and "the pipeline stopped writing latents at all" are different failures
and should not arrive as the same red X.

The smoke runner passes `--report-dir`, which is load-bearing rather than
cosmetic: `build_util.execute_script` only records a failure and carries on when a
report was built, and `run_python.py` only propagates the failure when a report
exists. Without it the gate aborts at the first break *and* exits 0. The report
directory is `test-results/` (gitignored), uploaded by the reusable workflow as
the `smoke-timings-<python-version>` artifact.

**PyAutoHeart picks this repository up automatically.** `heart/smoke.py`'s
`run_workspace` selects the modern runner on `.github/scripts/run_smoke.py`
existing, so adding that file moved euclid off the `_run_legacy_workspace`
fallback with no Heart-side change. Making the gate *release-blocking*
(`required_workflows.pipelines` in `PyAutoHeart/config/repos.yaml`) and adding it
to the weekly cloud sweep is a separate Heart-repo follow-up, filed at
`PyAutoMind/draft/test/pyautoheart/euclid_pipeline_release_blocking_gate.md`.

### Rules CI now enforces (`tests/test_repo_invariants.py`)

1. **Nothing auto-simulates.** Only `scripts/simulator.py` may bring a dataset
   into being — no `SimulatorImaging`, `should_simulate` or `auto_simulate`
   anywhere else. Every fitting script reads a dataset off disk, the way a user
   with real Euclid data runs it; a script that quietly simulates when it cannot
   find one turns a broken data path into a green run.
2. **Every script is accounted for.** Each `*.py` under `scripts/`,
   `catalogue/scripts/`, `preprocess/`, `tools/`, `workflow/`, `.github/scripts/`
   and the repository root is either listed in `smoke_tests.txt` or excluded in
   `config/build/no_run.yaml` **with a written reason**. Adding a script without
   doing one or the other fails the `unit` job.
3. **The allowlist is not stale.** Every `smoke_tests.txt` entry exists on disk.

---

## Codex / Sandboxed Runs

When running from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/initial_lens_model.py --dataset=102018665_NEG570040238507752998
```

---

## Dataset Layout

```
dataset/
└── <sample_name>/
    └── <dataset_name>/
        ├── <dataset_name>.fits    # Multi-HDU FITS (VIS + NIR/EXT bands)   [required]
        ├── info.json              # pixel_scale, mask_radius, mask_centre  [required]
        ├── segmentation/          # DR1 preprocessing outputs              [optional]
        │   ├── artefact_binary.fits  #   Preferred noise-scaling mask
        │   ├── lens_flux.fits        #   Peak gives the mask centre
        │   └── source_flux.fits      #   Used to derive positions if positions.json is absent
        ├── mask_extra_galaxies.fits  # Fallback noise-scaling mask         [optional]
        ├── positions.json         # Multiple-image positions               [optional]
        ├── segmentation.png       # Collected into the inspection bundle   [optional]
        └── rgb_0.png / rgb_1.png / rgb_0.jpg  # RGB thumbnails             [optional]
```

`util.load_vis_dataset()` reads all of the above. Everything marked optional
**degrades gracefully** — the fit still runs, with reduced information:

| Input | Preferred | Falls back to | If neither |
|---|---|---|---|
| Noise-scaling mask | `segmentation/artefact_binary.fits` | `mask_extra_galaxies.fits` | No noise scaling. A mask whose shape does not match the cut-out is refused. |
| Mask centre | peak of `segmentation/lens_flux.fits` | `info["mask_centre"]` | Frame centre `(0, 0)`. |
| Positions | `positions.json` (2+ positions) | derived from `segmentation/source_flux.fits` + the VIS noise map | No positions likelihood. |

The shipped example dataset has no `segmentation/` directory and no
`positions.json`, so it exercises the fallback branches — which is why they exist
as a chain and not a replacement.

The FITS **primary** header must additionally carry `WORST_BAND` and one of
`WORST_PSF_MER` / `WORST_PSF_HDR` / `WORST_PSF` for the aperture-flux latents.
These are stamped by the upstream Euclid cut-out generator; neither this pipeline
nor PyAutoReduce writes them. See the README's "Dataset Requirements" section for
the full contract and the two degradation paths.

### The two shipped datasets

[`dataset/README.md`](dataset/README.md) is the per-dataset table — one row each,
with what reads it and how to regenerate it. In short:

- `q1_walsmley/102018665_NEG570040238507752998/` — **real** Euclid Q1 MER
  imaging, 8 bands, 1.3 MB. Every fitting script's default, the smoke profile's
  `args_default`, and the catalogue's default sample. It ships no `segmentation/`
  and no `positions.json`, so it is the dataset that exercises the *fallback*
  branches above. **It stays the smoke default** — it is what users see, and CI
  fitting a mock everywhere would not.
- `simulated/euclid_dr1_like/` — the **simulated** DR1-layout mock written by
  `scripts/simulator.py`, 4 bands, 900 KB. It ships everything the real dataset
  lacks, so it exercises the *preferred* branch of every optional-input chain and
  is the one dataset that builds a complete 13-of-13 inspection bundle. Its
  `truth.json` is the known-answer source for the latent tests, and it is the
  dataset the `slow` run-level fit uses.

Two conventions to know before adding a dataset:

- `.gitignore` carries `*.fits`, so **new committed FITS need `git add -f`**.
  Without it they silently do not stage and the dataset lands half-committed.
- `preprocess/segmentation.py` is the producer of `segmentation.png`, and it
  **overwrites `positions.json`** from the local maxima of
  `segmentation/source_flux.fits` — discarding exact positions a simulator solved
  for. Back the file up and restore it, or re-run the simulator (deterministic for
  a given `--seed`).

`dataset/sample_group/` and `dataset/sample_point/` were removed in phase 2: 4.6 MB
with no consumer anywhere in the repository. Git history keeps them.

### `scripts/simulator.py`

The only producer of simulated data (invariant 1 above), and shared with phase 5 of
the DR1 prep epic. Two modes:

- `--from-params` — an analytic lens (`Isothermal` + `ExternalShear` mass, `Sersic`
  lens light, `Sersic` source) from the truth values at the top of the script. This
  wrote `dataset/simulated/euclid_dr1_like/` with `--from-params --seed 1`.
- `--from-result` — **phase 5's resimulation contract.** The tracer is rebuilt from
  a finished fit's `model.json` + maximum-log-likelihood sample (resolved with
  `diagnose_latent.py::resolve_files_path`, so it takes the same `--sample` /
  `--dataset` / `--unique_tag` / `--search` / `--result_hash`), and the bands, PSF
  stamps, zero-points, WCS and noise levels come from the dataset the fit was made
  on. Two rules are already implemented: a lens-light `sersic_index` at or above
  `--sersic-index-prior-edge` (5.0) is a pinned prior edge and is replaced by
  `--sersic-index-replacement` (3.0), with **both** values recorded in
  `truth.json`; and a single-band fit written to several bands applies the fitted
  intensities to every band — a **flat SED**, recorded as `sed: "flat"`. Per-band
  colour needs a per-band fit (`scripts/sersic_lens_model_waveband.py`) simulated
  band by band; that is phase 5's to change, not a bug here.

Every run writes `truth.json` beside the data: every model parameter, the per-band
lens / lensed-source / source fluxes in counts and microJansky, the four aperture
lens fluxes, the true magnification, the true Einstein radius, and the 12 pipeline
latents evaluated on the truth model.

Under `PYAUTO_TEST_MODE` the output is redirected to
`$PYAUTO_OUTPUT_DIR/simulator/<sample>/<dataset>/` instead of `dataset/`, so the
smoke entry for this script can never overwrite the committed dataset;
`--force-dataset-dir` overrides that.

---

## HPC

```
hpc/
├── batch_gpu/                      # GPU submit scripts + SLURM logs
│   ├── submit_initial_lens_model   #   scripts/initial_lens_model.py (array job)
│   ├── submit_full_model           #   scripts/full_model.py
│   ├── submit_sersic_waveband      #   scripts/sersic_lens_model_waveband.py (SED chain)
│   ├── output/                     #   SLURM stdout logs
│   └── error/                      #   SLURM stderr logs
├── batch_cpu/                      # CPU submit scripts + SLURM logs
│   ├── submit_initial_lens_model   #   Same, with --use_cpu --number_of_cores
│   ├── template                    #   One-off single run, not an array
│   ├── output/
│   └── error/
├── sync                            # Bidirectional sync script (local <-> HPC)
└── sync.conf.example               # Template config for sync
```

Setup: `cp hpc/sync.conf.example hpc/sync.conf` and edit with your HPC host and paths. `sync.conf` is gitignored.

Commands: `hpc/sync push`, `hpc/sync pull`, `hpc/sync sync` (push then pull), `hpc/sync status` (dry run).
`push` sends `catalogue/ config/ hpc/ preprocess/ scripts/ tests/ tools/ workflow/` plus the root
source files, and `dataset/` with `--ignore-existing`. `pull` retrieves `output/`,
`output_sed/` (the SED chain's tree) and `inspect/` (the built bundles); a
directory that does not exist on the cluster is skipped, not an error.

Submit the SLURM jobs from the cluster itself: `sbatch hpc/batch_gpu/submit_initial_lens_model`
with `PROJECT_PATH` exported to the remote project root.

### Config settings to change on the cluster

The shipped `config/` is deliberately **laptop-friendly**. Three settings were
different in the DR1 science runs, and they are cluster choices rather than
sensible public defaults, so they are documented here instead of being baked in.
Set them on the cluster; leave the repository defaults alone. There is no
override mechanism — edit the YAML in your cluster checkout.

All three live in `config/general.yaml`:

| Key | Repo default | Set on the cluster | Why |
|---|---|---|---|
| `output.samples_to_csv` | `true` | `false` | At 20,000 tiles a `samples.csv` per search is a lot of disk. Local users want them. (The science runs' config predates the rename and spells this `output.samples`.) |
| `hpc.hpc_mode` | `false` | `true` | Disables GUI visualization and screen logging, which are not suited to a batch node. Pair it with `hpc.iterations_per_quick_update` — repo `10000`, science runs `100000`. |
| `numba.cache` | `true` | `false` | Numba's on-disk cache is unreliable on NFS; disabling it trades a little start-up time for not failing. |

`docs/drift_report.md` §8 records why the rest of the science runs' config was
**not** adopted: it is stale relative to the installed library, not ahead of it.

---

## Not ported — available in `Science/euclid`

The DR1 science runs were driven from a private tree (`Science/euclid`). Phase 1
of the DR1 prep ported the analysis chain, `util.py`, the catalogue producers and
the docs into this repository. The following were deliberately left behind. If
you need one, it is in that tree — do not re-derive it from scratch without
checking there first. `docs/drift_report.md` §10 carries the same list with
fuller reasoning.

| Left in `Science/euclid` | Reason |
|---|---|
| `scripts/sersic_lens_model_pix.py`, `scripts/sersic_lens_model_pix_waveband.py`, `scripts/multi_lens_model_pix_waveband.py` | Pixelized-source SED variants, superseded by the Delaunay Source Pix stages in `scripts/full_model.py`. |
| `scripts/galaxy_sersic_model.py` | Non-lens galaxy fitting; out of scope for a lens-modelling pipeline. |
| `scripts/audit_sed_outputs.py`, `scripts/audit_unresolved_hpc.sh`, `scripts/reorganize_normies.py` | Operational one-offs against a private results tree. |
| `lens_model_waveband.py::fit_waveband_galaxy`, `::fit_waveband_pix` | Waveband variants with no caller in this repository's chain. |
| The 21 catalogue scripts (paper figures, QA copy tools, superseded precursors) | Listed individually with a reason each in [`catalogue/README.md`](catalogue/README.md#not-ported--available-in-scienceeuclid) — not duplicated here. |
| `preprocess/recenter_lens_centre.py` | Only needed on the `segmentation/lens_flux.fits` mask-centre path, which the fallback chain handles without it. |

---

## Line Endings — Always Unix (LF)

All files must use Unix line endings (`\n`). CRLF will break shell scripts on the HPC. After creating or editing files, verify with `file <path>` and convert with `dos2unix` if needed.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->
