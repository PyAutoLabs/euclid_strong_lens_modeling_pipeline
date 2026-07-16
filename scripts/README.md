The `scripts` folder contains example pipelines which perform lens modeling to fit different types of Euclid data and assuming different lens and source models.

# Files

- `initial_lens_model`: The same initial MGE + SIE lens model fit as the root `start_here.py` script, as an importable pipeline.
- `full_model`: Lens modeling pipeline which fits a full lens model, including a pixelized source, MGE lens light and custom complex mass models.
- `lens_model_waveband`: After modeling the high resolution VIS imaging, model lower resolution NIR / EXT imaging using a fixed lens model.
- `sersic_lens_model`: Fits Sersic lens and source models with the mass model fixed to the initial fit, giving more accurate photometry for SED fitting.
- `mge_lens_only`: Performs a foreground lens only MGE subtraction of the lens emission, such that the source can be quickly revealed for inspection.
