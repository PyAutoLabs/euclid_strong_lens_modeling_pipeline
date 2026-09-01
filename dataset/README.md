# Datasets

Every dataset this repository ships, what it is, and what reads it.

`.gitignore` has `*.fits`, so **every FITS here was force-added** (`git add -f`). A new
dataset's FITS will silently not be staged otherwise.

The layout and the FITS contract are documented in [`../AGENTS.md`](../AGENTS.md#dataset-layout)
and the README's "Dataset Requirements" section; the short version is

```
dataset/<sample>/<name>/
    <name>.fits    # PRIMARY + (<BAND>_BGSUB, <BAND>_PSF, <BAND>_RMS) x N, in that order
    info.json      # pixel_scale, mask_radius, mask_centre     [required]
    ...            # everything else is optional and degrades gracefully
```

---

## The datasets

| Path | Real / simulated | What it is | Read by | Regenerate |
|---|---|---|---|---|
| `q1_walsmley/102018665_NEG570040238507752998/` | **real** | A Euclid Q1 MER cut-out of a Walmsley et al. lens candidate: 8 bands (DES *griz* + VIS + NIR *YJH*), 100x100 at 0.1"/pixel, 1.3 MB. Ships no `segmentation/` and no `positions.json`, so it is the dataset that exercises the *fallback* branches of the optional-input chain. | Every fitting script's default; `config/build/profile_smoke.yaml`'s `args_default`; `scripts/tools/diagnose_latent.py` and `scripts/tools/diagnose_latent_vis_pix.py` defaults; `tests/test_util.py`; `tests/test_compute_latent_variable.py`; `catalogue/`'s default sample. | Not regenerable — real observed data, committed as-is. |
| `simulated/euclid_dr1_like/` | **simulated** | A DR1-layout mock: 4 bands (VIS + NIR *YJH*) at the real zero-points, PSF widths and noise levels, 100x100 at 0.1"/pixel, ~880 KB. An `Isothermal` + `ExternalShear` mass with `einstein_radius=1.2"`, a `Sersic` lens light and a `Sersic` source, producing four multiple images. Ships the inputs the real dataset lacks — `segmentation/`, `positions.json`, `mask_extra_galaxies.fits`, `rgb_0.png`/`rgb_1.png`, `segmentation.png` — so it exercises the *preferred* branch of every optional-input chain and is the one dataset that builds a complete 13-of-13 inspection bundle. `truth.json` holds every parameter, flux, aperture flux, magnification and Einstein radius that went in. | The latent known-answer unit tests; TEST-mode CI runs of the fitting chain. | `python scripts/simulator.py --from-params --seed 1`, then `python preprocess/segmentation.py --sample=simulated` for `segmentation.png` (restore `positions.json` afterwards — see below) |

---

## Adding a dataset

Simulated datasets come from `scripts/simulator.py` — it is the **only** producer of simulated
data in this repository, and no fitting script auto-simulates a missing dataset (most users fit
real Euclid imaging, and a silent auto-simulate would hide a missing-data mistake).

```bash
# an analytic lens from the truth values in the script
python scripts/simulator.py --from-params --output-dataset=my_lens

# resimulate a lens you have already fitted
python scripts/simulator.py --from-result \
    --sample=q1_walsmley --dataset=102018665_NEG570040238507752998 \
    --unique_tag=sersic_lens_model --search=vis \
    --output-dataset=my_lens_resimulated
```

Then commit the FITS with `git add -f`, and add a row to the table above.

`segmentation.png` is not written by the simulator — `preprocess/segmentation.py` is its producer,
for simulated and real datasets alike, and `scripts/tools/build_inspect.py` copies it into the
inspection bundle. Run it after the simulator:

```bash
python preprocess/segmentation.py --sample=simulated
```

**It rewrites `positions.json`** from the local maxima of `segmentation/source_flux.fits`, which
discards the exact multiple-image positions the simulator solved for. Back the file up first and
restore it afterwards, or simply re-run the simulator (it is deterministic for a given `--seed`).
It also leaves an overview copy at `dataset/<sample>/segmentation/<name>.png`, which is a
duplicate and is not committed.

Under `PYAUTO_TEST_MODE` the simulator writes to `$PYAUTO_OUTPUT_DIR/simulator/` instead of
`dataset/`, so a smoke run can never overwrite a committed dataset.
