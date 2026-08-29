The `scripts` folder contains the pipelines that fit Euclid data, plus the
diagnostics and the two orchestrators that build the catalogue's inspection
bundle. Everything here is run from the **repository root**, not from this
folder:

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

# Diagnostics

- `diagnose_latent.py`: Replays the Euclid latent catalogue
  (`util.LatentEuclid`) on one converged result and prints every latent value,
  flagging NaN and zero sentinels, plus the Einstein radius in isolation. Runs no
  search. Test-mode results live under `<output>/test_mode/`, so pass
  `--output_path=output/test_mode` to inspect a smoke run.
- `diagnose_latent_vis_pix.py`: The population version — the same replay over
  every `vis_pix` result in a sample, reporting per-dataset OK/ERR plus a
  summary. Takes `--sample`, not `--dataset`.

# Catalogue orchestration

- `build_inspect.py`: Collects the inspection bundle's PNGs out of the result
  zips PyAutoFit writes (falling back to an unzipped result directory).
- `build_inspection_bundle.sh`: Runs all seven catalogue stages in order for a
  sample. See [`../catalogue/README.md`](../catalogue/README.md) for the
  13-file to producer table, the run order and the upstream fit each stage needs.
