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
output/                    # Results (generated at runtime, not committed)
docs/drift_report.md       # Record of the DR1 pipeline-parity port
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
tests/                     # pytest unit tests (JAX-free, no searches)
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

## Test Runs

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

The example dataset lives at `dataset/q1_walsmley/102018665_NEG570040238507752998/`.

### Unit tests

```bash
python -m pytest tests/ -q
```

`tests/` is deliberately fast and **JAX-free** — no `use_jax=True`, no non-linear
search. It covers the pure helpers in `util.py`, the six-tuple CLI, the
noise-scaling mask fallback chain, the `WORST_PSF_*` header contract and the
latent **key set**. It cannot cover latent *values*: `skip_latents()` is true in
every test mode, so no test-mode run computes one.

### Smoke tests

`smoke_tests.txt` lists the scripts PyAutoHeart's smoke runner executes. Three
things about that runner are load-bearing when editing the file:

- Entries run **in order, in one output directory** that is wiped once before the
  first entry. `scripts/diagnose_latent.py` reads a finished result, so it must
  stay below the scripts that produce one.
- One `args_default` from `config/build/profile_smoke.yaml` is appended to
  **every** entry. A script that rejects `--dataset` cannot be listed —
  `scripts/diagnose_latent_vis_pix.py` is in `config/build/no_run.yaml` for
  exactly that reason.
- Per-script environment goes in `profile_smoke.yaml`'s `overrides:`, matched by
  path substring. `scripts/diagnose_latent.py` uses one
  (`PYAUTO_OUTPUT_DIR: output/test_mode`) so it resolves the test-mode results
  the earlier entries wrote.

`config/build/no_run.yaml` carries a reason inline for every skipped script.

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
