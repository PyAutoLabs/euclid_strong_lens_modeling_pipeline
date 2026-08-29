# Euclid DR1 Pipeline Parity — Drift Report

This records phase 1 of the Euclid DR1 prep work, dated 2026-08-29: porting the
real DR1 analysis chain from the private science tree (referred to below as
`Science/euclid`) into this public repository, so a reader of this repository
alone can reproduce the DR1 results. It condenses the read-only survey that
preceded the port. The three commits that carried it out are the authoritative
record of what actually landed; this document explains what was ported, what
was deliberately left behind, and why the non-obvious decisions were made.

Sections 9 and 11 have since been updated by **phase 2** (TEST-mode CI on
committed simulated data, plus the latent tests), which discharged both of the
hand-offs phase 1 left open.

## 1. What was ported

### `util.py`

Four functions were missing entirely and are now present:

- `_find_local_maxima` — brute-force 4-neighbour interior scan over a flux
  array, sorted descending.
- `_pixel_to_arcsec` — converts a pixel `(row, col)` to an arcsec offset using
  AutoLens's half-pixel convention.
- `_compute_positions_from_source_flux` — builds a source-plane S/N map, walks
  the detection threshold down from S/N 3.0 in steps of 0.1 until at least one
  counter-image is found on the opposite side of the source, and returns the
  top `n_positions` local maxima. Mirrors the logic in
  `preprocess/segmentation.py`, the canonical writer of `positions.json`.
- `psf_fwhm_arcsec_from_primary_header` — reads a PSF FWHM in arcsec from a
  FITS primary header (see §6).

`load_vis_dataset` gained eight upgrades: an `image_tag` fallback between the
`_FLUX`/`_BGSUB` HDU names; a `segmentation/lens_flux.fits`-derived mask
centre with graceful fallback to `info["mask_centre"]`; a mask-centre-aware
brightest-pixel search region (previously the search always centred on the
cut-out frame, even for an offset lens); `header.get("MAGZERO", None)` instead
of a hard `KeyError` on bands lacking it; a noise-scaling mask fallback chain
(§4); a mask radius clamped to the cut-out frame so an offset lens cannot mask
off the edge; `WORST_BAND` warn-and-degrade hardening (§6); and a
`positions.json` fallback onto `_compute_positions_from_source_flux` when
fewer than two positions are found on disk. `VisualizerImaging` also picked up
`.jpg`/`.jpeg` support for the collected RGB thumbnail, matching how the DR1
tile dumps are actually stored on disk.

`parse_fit_args` now returns a 6-tuple, adding `--number_of_cores`,
`--use_cpu` and `--skip_pix`. This is a breaking change, and every call
site — all six scripts that unpack it — was updated in the same commit.

### The script chain

`scripts/initial_lens_model.py` carries the most consequential fix in the
whole port: the lens-light MGE's `ell_comps` priors are tightened from the
library default `[-1, 1]` to `[-0.5, 0.5]`. Beyond ±0.5 the MGE forms a
multi-blob shape that absorbs lensed-source flux into the lens light — a
known systematic; ±0.5 corresponds to an axis-ratio floor of roughly
q ≳ 0.17. The source MGE is now anchored with `centre=dataset_centre` rather
than left to a uniform prior over the whole cut-out. Search budgets were
raised (`vis_lp`: `n_live` 500→750, `n_like_max` 100000→200000; `vis_pix`:
`n_live` 150→300, `n_batch` 40→15, plus a quick-update interval and
`n_like_max` that were previously never passed at all). The `vis_pix` stage's
analysis object switched from the plain library `al.AnalysisImaging` to
`util.AnalysisImaging` — this is what had been silently dropping the Euclid
latents, the RGB visualizer and `wcs.json` from every pixelized-source fit;
all three are now restored.

The new `scripts/sersic_lens_model_waveband.py` (~56 lines, no new model
code) chains `initial_lens_model.fit(skip_pix=True)` →
`sersic_lens_model.fit_sersic` → `lens_model_waveband.fit_waveband`. It is
meant to run under a dedicated `PYAUTO_OUTPUT_DIR` so the per-band SED
outputs don't bloat the main results tree; each upstream stage
cache-short-circuits if its result zip already exists.

`scripts/diagnose_latent.py` and `scripts/diagnose_latent_vis_pix.py` are the
human-inspection route for the Euclid latent catalogue — the former replays
it on one converged result, the latter sweeps a population of `vis_pix`
results, reporting per-dataset OK/ERR so one bad tile cannot truncate the
picture. Both were rewritten, not copied verbatim (see §3).

The `catalogue/` producer tree (`catalogue/scripts/{deblending, lens_mass,
lens_sersic, source_sersic, multi_wavelength, magnitudes}.py`, plus the
shared `catalogue_util.py`) turns a directory of finished fits into the
inspection bundle described in `catalogue/README.md`.
`scripts/build_inspect.py` collects the bundle's image outputs, and
`scripts/build_inspection_bundle.sh` runs the full seven-stage order.

## 2. `start_here.py` collapsed to a shim

`start_here.py` had diverged from `scripts/initial_lens_model.py`: it was the
older copy, with no Source Pix stage at all, yet `README.md`, `AGENTS.md`,
the HPC submit scripts and `smoke_tests.txt` all pointed users at it. It is
now a thin shim over `scripts.initial_lens_model.fit`. Every existing
reference to `start_here.py` therefore still works unchanged, but
`scripts/initial_lens_model.py` is the file to read and edit going forward.

## 3. The latent-API inversion

The general drift rule for this port was "the science tree is ahead, port
from it". For latents that rule inverts. This repository's
`util.LatentEuclid(al.LatentLens)` plus `config/latent.yaml` is the *newer*
library API, and it is a strict superset of the science tree's legacy
185-line `AnalysisImaging.compute_latent_variables` with its 8 hardcoded
keys. This repository's set is 12 keys: the 8 config-enabled library latents
plus 4 Euclid-only FWHM aperture-flux µJy latents. The target's version was
kept as-is; `compute_latent_variables` was not ported at all; and the two
diagnostic scripts were rewritten onto `LatentEuclid.keys`/`.variables`
rather than copied, since their isolated Einstein-radius passes also needed
updating — the science tree's `Tracer.einstein_radius_from` no longer exists
in autolens 2026.8.17.1, and they now go through
`LensCalc.from_mass_obj(...).einstein_radius_from`/`_list_from` instead.

## 4. The `full_model.py` Delaunay switch

`al.mesh.RectangularAdaptImage` no longer exists in autolens 2026.8.17.1, so
`full_model.py` could not run at all before this port — the previous
`test_report.md` PASS for this script was stale evidence, predating the
library rename. Both Source Pix stages now use `al.mesh.Delaunay` with
`al.reg.AdaptSplit`.

This is a hard constraint, not a preference: `al.reg.Adapt` **cannot** JIT on
the Delaunay mesh family. Delaunay neighbours are computed by a direct
`scipy.spatial.Delaunay` call on the traced source-plane grid, and that call
cannot be traced under `jit`/`grad`. `AdaptSplit` is therefore mandatory for
any Delaunay-based stage under JAX, not a style choice.

Stage 1 builds its mesh grid from an `al.image_mesh.Overlay(shape=(26, 26))`
plus 30 circle-edge points, handed in via `AdaptImages`. Stage 2 uses
`al.image_mesh.Hilbert(pixels=500)`. The pixel count matches the Euclid
sibling `scripts/initial_lens_model.py` rather than the `autolens_workspace`
`delaunay.py` example's 1000, because Euclid VIS cut-outs are small and 1000
source pixels would inflate memory use for no benefit here.

## 5. The noise-mask fallback chain

The science tree reads its noise-scaling mask from
`dataset/<sample>/<name>/segmentation/artefact_binary.fits`. The shipped
example dataset has only `mask_extra_galaxies.fits` and no `segmentation/`
directory at all. Porting the science tree's path verbatim would have
silently disabled noise scaling on the example dataset, so it landed instead
as a fallback chain — try `segmentation/artefact_binary.fits`, then
`mask_extra_galaxies.fits` — with a shape guard so a mismatched mask is never
applied. The same graceful-degradation treatment was given to the
`segmentation/lens_flux.fits` mask-centre read and the `positions.json` →
`segmentation/source_flux.fits` positions fallback: both degrade to the prior
behaviour rather than raising when their preferred input is absent.

## 6. The `WORST_BAND` / `WORST_PSF_*` input contract

These FITS header keys are stamped by the upstream Euclid cut-out generator.
Neither this pipeline nor PyAutoReduce writes them; they are a hard input
contract on the dataset, not something either repository can regenerate.
`WORST_BAND` names the worst-seeing band and is used to index that band's PSF
HDU. `psf_fwhm_arcsec_from_primary_header` reads the FWHM in arcsec from
`WORST_PSF_MER`, then `WORST_PSF_HDR`, then `WORST_PSF`, skipping Euclid's
`-99` sentinel value, and **raises** rather than guessing if all three are
missing or sentinel — because the four aperture-flux latents are evaluated at
fixed multiples of this FWHM, and a wrong value would silently corrupt them.
When `WORST_BAND` itself is absent from the header, the aperture latents are
skipped with a warning instead of failing the fit.

## 7. The CRLF rule

The science tree's files are CRLF. This repository mandates LF line endings
(`.gitattributes` sets `* text=auto eol=lf`; `AGENTS.md` states "Line
Endings — Always Unix"). Every file ported from the science tree was
converted. CRLF matters here beyond style — it breaks shell scripts when they
run on the HPC.

## 8. "The science tree's config is stale, not ahead"

Configuration is the one place the drift rule fully reverses. The evidence:
retired class names in `priors/cosmology.yaml` (`model.LambdaCDMWrap` etc.
instead of the current `model.FlatLambdaCDM`) and in the NFW mass priors
(`scatter` instead of the current `scatter_sigma`); an `aadapt:` typo in
`general.yaml` that silently disables the whole adapt block; retired
`*Plotter` class names in the `visualize/plots.yaml` comments; and no
`latent.yaml` or `config/build/` at all, because the science tree predates
both. This repository's `config/` was kept wholesale rather than having
anything ported into it.

Three settings the science runs actually used were deliberately *not*
copied, because they are cluster choices rather than sensible public
defaults. In this repository's `config/general.yaml` they are
`output.samples_to_csv: true` (the science tree predates the rename and calls
it `output.samples`; it sets it `false` to save disk at 20,000-tile scale),
`hpc.hpc_mode: false` (the science runs set it `true`, alongside
`hpc.iterations_per_quick_update: 100000` rather than `10000`), and
`numba.cache: true` (the science runs set it `false`, an NFS workaround).
They are documented for cluster operators in `AGENTS.md`'s HPC section rather
than baked into the shipped config; there is no override mechanism.

### Config drift sweep

A follow-up read-only sweep of `config/` against `autolens_workspace/config/`
and the packaged library defaults confirmed the reversal above and found nine
euclid-local items, all fixed here. Two lookup rules drive every verdict: the
layered `conf` merges key by key, so an omitted key silently falls back to the
packaged default; and prior resolution is *path-sensitive* — a prior yaml only
fires when its directory path is a suffix of the class's real module path, so a
file at the wrong path is not an error, it is silently dead.

Deleted (the classes no longer exist in the installed stack, and nothing in the
repository referenced them):

- `priors/mesh/voronoi.yaml` — `Voronoi` removed from the library; the file was
  null-bodied and had zero references repo-wide.
- `priors/hyper_data.yaml` — `HyperBackgroundNoise` and `HyperImageSky` both
  removed; it declared real priors nothing could ever read.
- `priors/image_mesh/kmeans.yaml` — `KMeans` exists but is never used by this
  repository; `image_mesh/README.md` lost its `KMeans` example with it.

Split (the one finding with teeth):

- `priors/regularization/adaptive_brightness.yaml` → `adapt.yaml` +
  `adapt_split.yaml`, values byte-for-byte unchanged. The old path resolved to
  `regularization.adaptive_brightness.*`, but the classes live at
  `autoarray.inversion.regularization.adapt.Adapt` and
  `…regularization.adapt_split.AdaptSplit`, so the suffix match never fired and
  the whole file was dead — the packaged defaults were supplying the values
  instead. No number changes today, because the euclid blocks were identical to
  those defaults. It matters because `al.reg.AdaptSplit` *is* used by this
  pipeline (`scripts/full_model.py:503,571`, `scripts/initial_lens_model.py:343`),
  so any future repo-local tuning of those priors would have been silently
  discarded.

Stale keys and comments:

- `visualize/plots_search.yaml` — dropped the `pyswarms:` and `ultranest:`
  blocks; both searches have been removed from autofit.
- `visualize/plots.yaml` — dropped the `fit_quantity:` block; no reader in
  autolens or autogalaxy.
- `output.yaml` — added `model_graph: false`. It is the one omitted key with no
  packaged default beneath it (`should_output` on a wholly absent key).
- `latent.yaml` — the header claimed the library defaults all five keys to
  `false`. It does not: the three raw-flux keys default `true`, so the emitted
  catalogue was a superset of what the file described. Header corrected and the
  three raw-flux keys pinned explicitly, so the file now states its full output
  rather than half-inheriting it. No behaviour change.
- `config/README.md`, `config/non_linear/README.md` and `config/priors/README.md`
  refreshed from `autolens_workspace/config/` — all three listed folders and
  files that do not exist here. `config/visualize/README.md` was fixed in place
  instead, because its "Changing the colormap" section is euclid-specific and is
  linked from `README.md`.

#### Left alone on purpose

Eight further findings reproduce identically in `autolens_workspace/config/`
*and* in the packaged library config, so they are upstream state this
repository has faithfully inherited. Fixing them here would fork files that are
currently in parity; they belong in a PyAutoGalaxy issue instead:

1. `priors/light/operated/sersic.yaml` — key is `ersic`, missing its leading
   `S`, so the operated `Sersic` priors are dead.
2. `priors/mass/stellar/sersic_core.yaml` — `SersicCoreSph.mass_to_light_ratio`
   is not in the class `__init__`.
3. `priors/light/linear_operated/gaussian.yaml` — `Gaussian.intensity`, but
   linear profiles have no `intensity`.
4. `priors/light/linear/chameleon.yaml` — path mismatch, so `Chameleon` and
   `ChameleonSph` never resolve.
5. `priors/light/linear/eff.yaml` — path mismatch for `ElsonFreeFall` /
   `ElsonFreeFallSph`.
6. `priors/mass/dark/nfw_truncated_mcr.yaml` — path mismatch for
   `NFWTruncatedMCRScatterLudlowSph`.
7. `priors/point_sources.yaml` — `PointSourceChi` no longer exists.
8. `priors/cosmology.yaml` — dotted key `model.FlatLambdaCDM` matches no class
   path.

Also left as-is for the same parity reason, though not upstream *bugs*: the
per-search `dynesty:` / `emcee:` / `nautilus:` / `zeus:` sections of
`plots_search.yaml` (autofit now reads only the three family-level sections),
the readerless `fits.flip_for_ds9` key in `general.yaml`, the Voronoi sentence
in `priors/mesh/README.md`, and the `DelaunayNN:` line `priors/mesh/delaunay.yaml`
omits — inert, since both entries are null-bodied and the packaged file is the
fallback.

## 9. Coverage: 13 of 13 on the simulated dataset

The inspection bundle's 13 per-lens files, and which script produces each,
are catalogued in `catalogue/README.md` — see that file's producer table
rather than a duplicate here. Every one of the 13 now has a named producer in
this repository.

On the shipped **real** example dataset,
`dataset/q1_walsmley/102018665_NEG570040238507752998/`, 11 of the 13 can be
produced end to end. The two gaps are `segmentation.png` and
`vis_lp_image_with_positions.png`, both of which need inputs that dataset does
not ship — a `segmentation/` directory and a `positions.json` respectively.
This is a dataset-provenance gap, not a missing script: both files have a real
producer, they simply have nothing to read on that particular dataset.

Phase 2 closed the gap by *synthesising* the inputs rather than sourcing them.
`dataset/simulated/euclid_dr1_like/`, written by `scripts/simulator.py`, ships a
`segmentation/` directory and a `positions.json` (the true multiple-image
positions, solved with `al.PointSolver`), and
`scripts/build_inspection_bundle.sh` produces **13 of 13** on it — the first
dataset in this repository to do so. The 11-of-13 statement above still stands
for the real dataset and is expected to: it is a property of the upstream
cut-out, not of this pipeline.

## 10. Not ported — available in `Science/euclid`

The following remain in the science tree only, each with the reason it was
excluded:

- `scripts/sersic_lens_model_pix.py`, `sersic_lens_model_pix_waveband.py`,
  `multi_lens_model_pix_waveband.py` — pixelized-source SED variants,
  superseded by the Delaunay Source Pix stages already in `full_model.py`.
- `scripts/galaxy_sersic_model.py` — depends on the non-lens
  `fit_waveband_galaxy` variant, out of scope for a lens-modelling pipeline.
- `scripts/audit_sed_outputs.py`, `audit_unresolved_hpc.sh`,
  `reorganize_normies.py` — operational one-offs against a private results
  tree.
- `lens_model_waveband.py::fit_waveband_galaxy` and `::fit_waveband_pix` —
  unused waveband variants (non-lens galaxy fitting, and a pixelized-source
  multi-band fit) with no caller in this repository's chain.
- The 21 catalogue scripts left in `Science/euclid/catalogue/scripts/` —
  paper figures, QA copy tools and superseded precursors; see
  `catalogue/README.md`'s own "Not ported — available in `Science/euclid`"
  section for the full list and reasons.
- `preprocess/recenter_lens_centre.py` — only needed on the
  `segmentation/lens_flux.fits` mask-centre path, which the fallback chain in
  §5 does not require.

## 11. What phase 2 picked up — discharged

Phase 1 left two things to phase 2, and both are now done.

**Latent values under CI.** `tests/test_util.py` covered the *structure* of the
latent catalogue — that `LatentEuclid.keys()` returns exactly the 12 keys
`config/latent.yaml` enables, in order — but not the values, because
`skip_latents()` is true in every `PYAUTO_TEST_MODE` and only a real fit
computes one. `tests/test_compute_latent_variable.py` now asserts all 12
*values* against `dataset/simulated/euclid_dr1_like/truth.json`, and
`tests/test_latent_run_level.py` asserts a real fit still *writes* them. Both
run on every pull request. The two diagnostic scripts in §1 remain the human
route for inspecting one real result; they are no longer the only route.

**The two §9 coverage gaps.** Synthesised rather than sourced — see §9.

### Phase 2 (CI)

Two facts from building the run-level check that are not obvious from the code
and are worth carrying forward:

- **A NaN latent is dropped from `files/latent/latent_summary.json` entirely.**
  It is not written as a NaN value. So the run-level test's assertion that the
  summary holds exactly the 12 keys *is* the NaN check: a latent that failed to
  compute shows up as a missing key. (The file is `latent_summary.json` rather
  than `files/latent.csv` because `config/output.yaml` sets
  `latent_draw_via_pdf: true`, on which branch the updater calls only
  `save_samples_summary`.)
- **`af.Drawer` is the run-level fit, and it needs a truth-anchored model.**
  It draws uniformly from the priors and does no parameter search — the cheapest
  real search in PyAutoFit, while still running the full post-fit updater path,
  so `skip_latents()` and `latent_after_fit` behave exactly as in production.
  Over a *fully free* model it draws junk: the source's linear intensity solves
  to exactly zero and `magnification` becomes `0 / 0`, which then vanishes from
  the summary by the rule above. The test therefore anchors the model on the
  truth with one free parameter. `latent_draw_via_pdf_size` is inert for
  `af.Drawer` — it falls back to all samples — so it is not a knob for making
  that fit cheaper; `total_draws` is.
