# Sky orientation, phase 2d: north-up figures

Issue: PyAutoLabs/euclid_strong_lens_modeling_pipeline#115 (PR #120). The DR1 fits read every
cut-out without flipping its rows, so the fit frame has +y pointing south and +x west, and every
pipeline PNG showed the sky upside down. This phase redraws the per-lens release figures north
up, east left, without refitting any model. The user-facing summary is in `catalogue/CHANGELOG.md`
and `catalogue/KNOWN_ISSUES_DR1.md`.

## Method (`scripts/tools/render_north_up.py`)

1. It rebuilds each stored fit at its best-fit model, with `al.agg.FitImagingAgg`.
2. It mirrors the fit (y → −y):
   - the data, noise map, mask, PSF and adapt images are flipped top to bottom;
   - the adapt image-plane mesh grid and the dataset-model grid offset get y → −y;
   - the tracer goes through `sky_frame.plus_y_north_from`.

   The fit is then solved again, including the pixelized source.
3. It checks the mirrored fit against the original (the witness, one row per fit in
   `witness.csv`). The mirrored model image must equal the flipped original, with the
   chi-squared and the reconstruction unchanged:

   | fits | tolerance |
   |---|---|
   | VIS lens models (`vis_lp`, `vis_pix`) | 1e-8 image, 1e-9 chi-squared, 1e-6 reconstruction |
   | Sersic fits and the NISP/DECam bands | 1e-4 (they mirror only to about 1e-5) |
   | figure of merit (all fits) | 1e-3 |

   - The figure of merit tolerance allows for the Delaunay regularization log-determinant,
     which is conditioning limited.
   - A mirrored log-determinant that fails to factorize enters no drawn image. Such a fit is
     judged on the images alone and recorded as `pass_no_fom`.
   - Pixels that are NaN in both images are ignored.
4. It draws the figures with the library plotters. Only the RGB panel of `rgb.png` gets an N/E
   compass (white, top-right; human, 2026-11-09). A fit that fails the check is not drawn.

`scripts/tools/north_up_catalogue_tree.py` hard-links the built catalogue tree and swaps the
north-up PNGs in. Where none was drawn, it keeps the original PNG, as the human chose on
2026-11-05. Release packaging then runs on that tree (`hpc/batch_cpu/submit_north_up_release`).

## Evidence

- **Tests.** `tests/test_render_north_up.py` checks the mirror on a simulated lens with an
  off-centre mask, for a light-profile source and for a Delaunay pixelized source.
- **Pilot.** On 50 lenses, 431/431 fits passed the check, and the human reviewed the
  side-by-side sheets by eye. The 50 were:
  - 15 paper lenses;
  - 25 random lenses;
  - the 5 largest |γ₂| and the 5 largest |e₁|, the components that flip sign.
- **Full run, 15,007 lenses.** The first run (RAL 414229, fix-up 414321, rescue 414366) was
  repeated with the RGB-panel compass as RAL 414683; this table is from the repeat:

  | figure | north up | pipeline frame |
  |---|---:|---:|
  | `vis_lp_fit`, `rgb`, `segmentation` | 15,007 | 0 |
  | `vis_pix_fit` | 14,857 | 127 |
  | `fit_sersic` | 14,902 | 8 |
  | `fit_multi_wavelength` | 14,844 | 66 |

- **What the fallbacks are.**
  - Of the `vis_pix` fallbacks, 60 inversions do not solve after the mirror. For the others,
    the mirrored source solution differs: on co-circular mesh points the Delaunay
    triangulation is not mirror-symmetric.
  - Diagnostic job 414365 found that every original rebuild works, and that the failures sit
    only in the regularization log-determinant.
  - The band fallbacks differ by 1e-4 to 1e-2.
  - The `vis_pix` fallback set is not exactly repeatable: the same lenses gave 91 fallbacks on
    2026-11-05 and 127 on 2026-11-09, because the borderline triangulations differ between runs.
    `KNOWN_ISSUES_DR1.md` lists the run that was released.
- **Release check.** The release was compared with the sky-frame release by the
  sha256 of every member. Only PNGs changed or were added; the documents aside, every
  JSON, FITS and CSV table is byte-identical.
