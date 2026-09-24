The `scripts` folder contains the pipelines that fit Euclid data and the
simulator that makes data to fit. The diagnostics and the catalogue-bundle
tooling live one level down in `scripts/tools/`, keeping the model-running
scripts separate from the things that inspect their results. Everything here is
run from the **repository root**, not from this folder:

```bash
python scripts/initial_lens_model.py --sample=q1_walsmley --dataset=<name>
```

All fitting pipelines share one argument parser (`util.parse_fit_args`) —
`--dataset`, `--sample`, `--iterations_per_quick_update`, `--number_of_cores`,
`--use_cpu`, `--stage`. See the repository `README.md` for what each does.

# Fitting pipelines

- `initial_lens_model.py`: **The entry point.** MGE lens light + SIE + shear mass
  + MGE source (`vis_lp`), then a pixelized Delaunay source (`vis_pix`). The
  repository root's `start_here.py` is a thin shim over this script's `fit()`.
  `--stage=vis_lp` returns after `vis_lp`.
- `sersic_lens_model.py`: Sersic lens and source fits with the mass model fixed
  to the initial fit, giving more accurate photometry for SED fitting. Chains off
  `initial_lens_model.fit(..., stage="vis_lp")` — `vis_pix` replaces the source
  bulge with a pixelization, so its instance cannot seed a Sersic source prior.
- `lens_model_waveband.py`: After modeling the high resolution VIS imaging, model
  the lower resolution NIR / EXT imaging with the lens model held fixed.
- `sersic_lens_model_waveband.py`: The **SED chain** driver — runs
  `initial_lens_model --stage=vis_lp`, then `sersic_lens_model`, then
  `lens_model_waveband` over every band. Run it under its own
  `PYAUTO_OUTPUT_DIR` so the per-band results stay out of the main `output/`
  tree; both upstream stages cache-short-circuit if their result zips are
  already there.
- `mge_lens_only.py`: Multi-Gaussian Expansion subtraction of the lens emission
  only, so the source is revealed quickly for inspection.
- `full_model.py`: The full SLaM pipeline — MGE source, two Delaunay pixelized
  source stages, refined lens light, then a PowerLaw + shear mass model. Both
  pixelized stages use `al.mesh.Delaunay` with `al.reg.AdaptSplit`: `reg.Adapt`
  cannot JIT on the Delaunay family, because Delaunay neighbours come from a
  `scipy.spatial.Delaunay` call on the traced source-plane grid. The second stage
  uses `Hilbert(pixels=500)`, matching `initial_lens_model.py` rather than the
  `autolens_workspace` `delaunay.py` example's 1000 — Euclid VIS cut-outs are
  small.

# Simulating data

- `simulator.py`: The **only** producer of simulated data in this repository — no
  fitting script auto-simulates a missing dataset, and
  `tests/test_repo_invariants.py` keeps it that way. It writes an ordinary dataset
  of this pipeline (the multi-extension FITS contract `util.load_vis_dataset`
  reads, `info.json`, `positions.json`, `segmentation/`,
  `mask_extra_galaxies.fits`, RGB thumbnails) plus `truth.json`, which records
  every model parameter, the per-band lens / lensed-source / source fluxes in
  counts and microJansky, the four aperture lens fluxes, the true magnification
  and the true Einstein radius. Two modes:
  - `--from-params` (the default) builds an analytic lens — `Isothermal` +
    `ExternalShear` mass, `Sersic` lens light, `Sersic` source — from the truth
    values at the top of the script. This is what
    `dataset/simulated/euclid_dr1_like/` was made with
    (`--from-params --seed 1`), and `truth.json` is the known-answer source for
    `tests/test_compute_latent_variable.py`.
  - `--from-result` resimulates a fit you have already run: the tracer is loaded
    from that result's `tracer.json` — the maximum-log-likelihood lens with the
    linear light profiles already converted to the intensities the fit solved for,
    which `model.json` and the parameter vector do not carry (resolved with
    `tools/diagnose_latent.py::resolve_files_path`, so the arguments are the same
    `--sample` / `--dataset` / `--unique_tag` / `--search` / `--result_hash`) and
    the bands, PSF stamps, zero-points, WCS and noise levels come from the dataset
    it was fitted to. A single-band fit written to multiple bands applies the
    fitted intensities to every band — a *flat* SED, recorded as `sed: "flat"` in
    `truth.json`. A lens-light `sersic_index` at or above
    `--sersic-index-prior-edge` (5.0) is replaced by `--sersic-index-replacement`
    (3.0), with both values recorded.

  Under `PYAUTO_TEST_MODE` the dataset is written to
  `$PYAUTO_OUTPUT_DIR/simulator/` rather than `dataset/` unless
  `--force-dataset-dir` is passed, so the smoke run that executes this script
  cannot overwrite the committed dataset; `--dataset` / `--sample` are accepted and
  ignored in `--from-params` mode so it can carry the smoke runner's global
  `args_default`. See [`../dataset/README.md`](../dataset/README.md) for the
  datasets it has produced.

# Diagnostics

- `tools/diagnose_latent.py`: Replays the Euclid latent catalogue
  (`util.LatentEuclid`) on one converged result and prints every latent value,
  flagging NaN and zero sentinels, plus the Einstein radius in isolation. Runs no
  search. Test-mode results live under `<output>/test_mode/`, so pass
  `--output_path=output/test_mode` to inspect a smoke run.
- `tools/diagnose_latent_vis_pix.py`: The population version — the same replay over
  every `vis_pix` result in a sample, reporting per-dataset OK/ERR plus a
  summary. Takes `--sample`, not `--dataset`.

# Positions gate (before vis_lp)

- `tools/positions_gate.py`: Pre-submit gate for the `vis_lp` positions penalty.
  Per tile it reads the raw `positions.json` (never modified) and the light centre
  `vis_lp` fixes its mass centre to (`util.load_vis_dataset`'s `dataset_centre`),
  then: cuts positions within **0.15"** of the light centre (one PSF FWHM); fits a
  fixed-centre SIE + external shear in exactly `vis_lp`'s model space
  (`einstein_radius` [0, 8], `gamma_i` [-0.3, 0.3], `|ell_comps| < 0.95`) to get
  `s_min`, the smallest achievable max pairwise source-plane separation (what
  `PositionsLH` penalises), and a plausibility cost `J`; if `s_min > 0.2"` and
  there are >= 3 positions, drops the single position that makes the set trace
  (unique candidate; else lowest `J` by >= 4; else nearest if r < 0.3" or
  < 0.6x the others' median radius, outermost if > 1.5x; else review); flags,
  without dropping, a traceable set where one drop lowers `J` by >= 10; and sets
  **T = min(max(2 s_final, 0.3"), 0.5")** (2 s_final > 0.5" -> review). It writes
  `positions_meta.json` beside `positions.json` (positions used, drops and
  reasons, `s_min`, `J`, `T`, `status` keep | drop | review | flag | n_lt_2), which
  `util.load_vis_dataset` reads: gated positions + `T`, penalty off for `n_lt_2`,
  and the old behaviour (raw positions, T = 0.2) when there is no sidecar. A
  `--root` run also writes `positions_review.csv` and `positions_submit.txt`
  (review tiles held back unless `--include-review`). RAL:
  `hpc/batch_cpu/submit_positions_gate`. Research and census behind every
  parameter: `euclid_dr1` project, `inspect/positions_census/research/`
  (`SYNTHESIS.md`, `B_final_method.md`, `A_central_radius.md`).
  **Pair floor (phase 2):** when the tile ships `segmentation/source_flux.fits`
  and the VIS RMS map, a final set of exactly two positions needs both peak
  source-flux SNRs >= 3; otherwise the fainter is dropped and the tile goes to
  review (`pair_floor`) with no positions. Sidecar `version` 2.1; extra keys
  `method` (`gate`), `finder_version`, `s_final`, `snr`, `added` (always
  empty). **Production writer:** the same steps end
  `positions_finder.find_positions_gate` (package root, pure numpy), which
  writes new tiles' `positions.json` (`preprocess/segmentation.py`) and the
  `load_vis_dataset` fallback: SNR >= 3 source-flux peaks outside 0.15" of the
  light centre (merged within 0.15", brightest four, no SNR walk-down), then the
  gate steps and the pair floor, so a tile whose `positions.json` equals its
  SNR >= 3 peak set gets the same verdict seeded or unseeded. The
  model-guided reconcile loop (`positions_finder.find_positions`) and forward
  solver (`solve_images`) are non-production diagnostics (witness / tooling
  only).
- `tools/positions_finder_witness.py`: Runs the production path seeded (the gate
  on `positions.json`) and unseeded (the writer on the flux map) over every tile
  of a calibration sample (`--root`, any depth) without writing inside the
  tiles; prints both tables (with a `same` column) and writes
  `witness_table.md`, `witness.json` and `overlays/` under `<root>/witness`.
  `--reconcile` adds the diagnostic reconcile loop for comparison. On the
  10-lens `euclid_dr1` calibration sample (2026-09-24) the seeded path keeps the
  good five unchanged (s <= 0.025"), cuts the nucleus of Tile102014701, sends
  Tile102008208 and Tile102022005 to `pair_floor` review (counter-image SNR
  2.6 / 2.0 after the outlier drop) and Tile102012741 / Tile102023528 to
  review; unseeded, the good five trace at s <= 0.033" on different
  (newer-segmentation) peak sets.

# Catalogue orchestration

- `tools/build_inspect.py`: Collects the inspection bundle's PNGs out of the result
  zips PyAutoFit writes (falling back to an unzipped result directory).
- `build_inspection_bundle.sh`: Runs all ten catalogue stages in order for a
  sample. See [`../catalogue/README.md`](../catalogue/README.md) for the
  21-file to producer table, the run order and the upstream fit each stage needs.
