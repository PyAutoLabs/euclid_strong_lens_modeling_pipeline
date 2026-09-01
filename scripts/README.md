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
`--use_cpu`, `--skip_pix`. See the repository `README.md` for what each does.

# Fitting pipelines

- `initial_lens_model.py`: **The entry point.** MGE lens light + SIE + shear mass
  + MGE source (`vis_lp`), then a pixelized Delaunay source (`vis_pix`). The
  repository root's `start_here.py` is a thin shim over this script's `fit()`.
  `--skip_pix` returns after `vis_lp`.
- `sersic_lens_model.py`: Sersic lens and source fits with the mass model fixed
  to the initial fit, giving more accurate photometry for SED fitting. Chains off
  `initial_lens_model.fit(..., skip_pix=True)` — `vis_pix` replaces the source
  bulge with a pixelization, so its instance cannot seed a Sersic source prior.
- `lens_model_waveband.py`: After modeling the high resolution VIS imaging, model
  the lower resolution NIR / EXT imaging with the lens model held fixed.
- `sersic_lens_model_waveband.py`: The **SED chain** driver — runs
  `initial_lens_model --skip_pix`, then `sersic_lens_model`, then
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
  - `--from-result` resimulates a fit you have already run: the tracer is rebuilt
    from that result's `model.json` + maximum-log-likelihood sample (resolved with
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

# Catalogue orchestration

- `tools/build_inspect.py`: Collects the inspection bundle's PNGs out of the result
  zips PyAutoFit writes (falling back to an unzipped result directory).
- `build_inspection_bundle.sh`: Runs all seven catalogue stages in order for a
  sample. See [`../catalogue/README.md`](../catalogue/README.md) for the
  13-file to producer table, the run order and the upstream fit each stage needs.
