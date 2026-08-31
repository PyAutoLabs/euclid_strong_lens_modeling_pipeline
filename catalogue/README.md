# Catalogue

The catalogue tree turns a directory of finished lens fits into the **inspection
bundle**: one folder per lens holding everything a scientist needs to judge that
lens, plus master CSVs spanning the whole sample. It is the direct ancestor of
the exported DR1 catalogue.

### Relation to `workflow/`

Both trees read finished fits out of `output/` through the PyAutoFit aggregator,
and they differ in purpose rather than in mechanism. `workflow/` holds *general
examples* of the aggregator export API — `csv_make.py`, `png_make.py` and
`fits_make.py` teach it step by step, and `workflow/example/` shows it applied to
real Euclid runs. `catalogue/` holds the *production* producers: the scripts the
bundle builder actually runs to write the 13 bundle files and the master CSVs.

The clearest way in is to read the two side by side.
`workflow/example/csv/lens_mass.py` builds the same table as
`catalogue/scripts/lens_mass.py` — the mass model parameters of every VIS fit,
scraped through the same `AggregateCSV` API — written as a flat top-to-bottom
tutorial that explains each call as it makes it, and saved under its own name
into `workflow/csv/`. The catalogue version is that same scrape wrapped in
argument parsing, sample-wide path resolution, the extra sigma columns and the
per-lens split, so it can be driven by `scripts/build_inspection_bundle.sh`.
Read the tutorial for the API; read the producer for what DR1 actually ships.

Producers live in `catalogue/scripts/`. The two orchestration scripts live in
`scripts/` beside the pipelines they read from:

```
scripts/build_inspection_bundle.sh   # runs the seven stages in order
scripts/tools/build_inspect.py             # stage 1 — collects the 6 bundle PNGs
catalogue/scripts/
  catalogue_util.py                  # shared path resolution + per-lens CSV split
  deblending.py                      # stage 2 — pre_psf.fits, model.fits
  lens_mass.py                       # stage 3 — lens_mass.csv
  lens_sersic.py                     # stage 4 — lens_sersic.csv
  source_sersic.py                   # stage 5 — source_sersic.csv
  multi_wavelength.py                # stage 6 — fit_multi_wavelength.png
  magnitudes.py                      # stage 7 — magnitudes.csv
```

---

## The 13 bundle files and what produces each

Every file of a complete per-lens bundle, the script that produces it, and
whether that script *generates* the file from the fit or *collects* an image the
fit already wrote.

| Output file | Producer | Generates / collects | Upstream fit it needs |
|---|---|---|---|
| `lens_mass.csv` | `catalogue/scripts/lens_mass.py` | generates | `initial_lens_model/vis_pix` |
| `lens_sersic.csv` | `catalogue/scripts/lens_sersic.py` | generates | `sersic_lens_model/vis` |
| `source_sersic.csv` | `catalogue/scripts/source_sersic.py` | generates | `sersic_lens_model/vis` |
| `magnitudes.csv` | `catalogue/scripts/magnitudes.py` | generates | multi-band `sersic_lens_model/<band>` in `output_sed/` |
| `pre_psf.fits` | `catalogue/scripts/deblending.py` | generates | `sersic_lens_model` (every waveband) |
| `model.fits` | `catalogue/scripts/deblending.py` | generates | `sersic_lens_model` (every waveband) |
| `fit_multi_wavelength.png` | `catalogue/scripts/multi_wavelength.py` | generates | multi-band `sersic_lens_model/<band>` in `output_sed/` |
| `vis_lp_fit.png` | `scripts/tools/build_inspect.py` | collects `image/fit.png` | `initial_lens_model/vis_lp` |
| `vis_pix_fit.png` | `scripts/tools/build_inspect.py` | collects `image/fit.png` | `initial_lens_model/vis_pix` |
| `vis_lp_image_with_positions.png` | `scripts/tools/build_inspect.py` | collects `image/image_with_positions.png` (best-effort) | `initial_lens_model/vis_lp` + the lens's `positions.json` |
| `rgb.png` | `scripts/tools/build_inspect.py` | collects `image/rgb.png`, else copies `dataset/<sample>/<lens>/rgb_0.jpg` | `initial_lens_model/vis_lp` |
| `segmentation.png` | `scripts/tools/build_inspect.py` | copies `dataset/<sample>/<lens>/segmentation.png` | `preprocess/segmentation.py` |
| `fit_sersic.png` | `scripts/tools/build_inspect.py` | collects `image/fit.png` | `sersic_lens_model/vis` |

`build_inspect.py` never re-renders: it streams the image out of the result zip
PyAutoFit writes when a search finishes, falling back to the unzipped result
directory's `image/` folder when a run was not zipped (an interrupted run, or a
`PYAUTO_TEST_MODE` run).

### Building a bundle off a test-mode run

Two things behave differently when the fits were made with `PYAUTO_TEST_MODE`,
and neither is a fault in the producers:

- **PNGs.** The collected PNGs only exist if the fit was run with visualization
  on. `PYAUTO_FAST_PLOTS=1` — set by the smoke profile in
  `config/build/profile_smoke.yaml` — suppresses PNG output entirely, and
  `PYAUTO_TEST_MODE=2` skips the sampler and so the post-fit visualization.
  A `PYAUTO_TEST_MODE=1` run *without* `PYAUTO_FAST_PLOTS` does produce them.
- **Latent columns are empty.** `autonerves.test_mode.skip_latents()` returns
  `True` for every test-mode level, so no `latent_summary.json` is written.
  `lens_mass.csv` therefore has its `effective_einstein_radius*` columns present
  but blank, and every value column of `magnitudes.csv` is blank. The column
  *structure* is still correct — only a real-mode fit fills them.

---

## Run order

`scripts/build_inspection_bundle.sh` runs the seven stages in dependency order.
It is idempotent — every stage skips work that is already done, so re-run it as
more fits land.

```bash
bash scripts/build_inspection_bundle.sh q1_walsmley
bash scripts/build_inspection_bundle.sh dr1_prelim_grade_ab run250
```

| Stage | Script | Reads | Writes |
|---|---|---|---|
| 1/7 | `scripts/tools/build_inspect.py` | `output/<sample>/` | the 6 collected PNGs |
| 2/7 | `catalogue/scripts/deblending.py` | `output/<sample>/` | `pre_psf.fits`, `model.fits` |
| 3/7 | `catalogue/scripts/lens_mass.py` | `output/<sample>/` | `lens_mass.csv` |
| 4/7 | `catalogue/scripts/lens_sersic.py` | `output/<sample>/` | `lens_sersic.csv` |
| 5/7 | `catalogue/scripts/source_sersic.py` | `output/<sample>/` | `source_sersic.csv` |
| 6/7 | `catalogue/scripts/multi_wavelength.py` | `output_sed/<sample>/` | `fit_multi_wavelength.png` |
| 7/7 | `catalogue/scripts/magnitudes.py` | `output_sed/<sample>/` | `magnitudes.csv` |

Everything lands in `inspect/<sample>[_<run_tag>]/`, with the master CSVs at the
root and one folder per lens holding that lens's own copy of every product.

Stages 6 and 7 read a **separate** results tree. The multi-band SED fits are run
with `PYAUTO_OUTPUT_DIR=output_sed` (see `scripts/sersic_lens_model_waveband.py`
and `scripts/lens_model_waveband.py`), so those two stages are skipped when
`output_sed/<sample>/` does not exist. Set `SKIP_SED=1` to skip them
deliberately while SED jobs are still running.

### Fits the bundle expects

Producing all 13 files needs three pipeline runs per lens:

```bash
python scripts/initial_lens_model.py --dataset=<lens> --sample=<sample>
python scripts/sersic_lens_model.py  --dataset=<lens> --sample=<sample>
PYAUTO_OUTPUT_DIR=output_sed python scripts/sersic_lens_model_waveband.py \
    --dataset=<lens> --sample=<sample>
```

plus `preprocess/segmentation.py` for `segmentation.png`.

### Environment

| Variable | Default | Effect |
|---|---|---|
| `OUTPUT_DIR` | `output` | results tree read by stages 1-5 |
| `SED_OUTPUT_DIR` | `output_sed` | results tree read by stages 6-7 |
| `SKIP_SED` | `0` | `1` skips stages 6-7 |
| `CREATE_ARCHIVE` | `1` | `0` skips the closing `tar czf` |
| `DATASET_PREFIX` | *(empty)* | restrict stage 1 to lens folders with this name prefix (`Tile` for DR1) |

Every producer can also be run on its own; each takes `--sample`,
`--output_path` and `--inspect_dir`, and `--help` documents the rest.

---

## Conventions

- **Only completed fits are catalogued.** Every aggregator query passes
  `completed_only=True`, so a lens without a `.completed` marker is absent
  rather than half-written. A sample still being fitted therefore yields a
  partial but never a corrupt catalogue.
- **Five value flavours per variable.** Each CSV column appears as median plus
  lower/upper 1σ and lower/upper 3σ (`magnitudes.csv` adds max-log-likelihood).
- **Intensity is never a column.** The Sersic profiles are `lp_linear.Sersic`,
  whose intensity is solved by linear algebra at each likelihood evaluation and
  so never enters the non-linear samples.
- **Fluxes are µJy**, from the `*_mujy` latents of `util.LatentEuclid` enabled in
  `config/latent.yaml`. They require `magzero` in the fit's info dict, which
  `util.load_vis_dataset` reads from the FITS header.
- **Master CSVs are split per lens.** `catalogue_util.write_per_tile_csv` drops
  each lens's rows into its own folder so a single lens folder is
  self-contained and can be shipped on its own.

---

## Not ported — available in `Science/euclid`

The DR1 science tree (`Science/euclid/catalogue/scripts/`) holds 21 further
files that are deliberately **not** in this repository. They are either
downstream consumers of a finished catalogue, superseded precursors, or
paper-figure code — none of them produce any of the 13 bundle files.

| Not ported | Count | Why |
|---|---|---|
| `plot_*.py` (`plot_lens_sersic_index`, `plot_mass_light_alignment_errors`, `plot_mass_light_position_angle`, `plot_perpendicular_population_test`, `plot_position_angle_errors`, `plot_shear_avoidance_error_control`, `plot_shear_vs_effective_radius`, `plot_shear_vs_light_mass_pa`, `plot_shear_zone_of_avoidance`, `plot_slack_summary_figures`, `plot_stars_vs_total_mass_alignment`) | 11 | paper and analysis figures drawn from a finished catalogue |
| `collect_*.py` (`collect_alignment_check_fits`, `collect_perpendicular_fits`, `collect_sersic_prior_check_fits`) | 3 | QA copy tools that read the exported release directory |
| `lens_mass_dr1_prelim_grade_ab.py`, `lens_mass_vis_pix_dr1_prelim_grade_ab.py` | 2 | superseded precursors of `lens_mass.py` (wrong stage / missing columns; they write a differently named `lens.csv`) |
| `deduplicate_deblending_fits.py`, `validate_magnitudes.py` | 2 | one-off QA/fixup passes, not on the bundle path |
| `lens_model.py` | 1 | ad-hoc `fit.png` compositor, superseded by `build_inspect.py` |
| `CLAUDE.md`, `AGENTS.md` | 2 | stale agent notes describing a directory layout this tree does not use |

Also left in the science tree: `preprocess/recenter_lens_centre.py`, and the
`scripts/` one-offs `audit_sed_outputs.py`, `audit_unresolved_hpc.sh` and
`reorganize_normies.py`.
