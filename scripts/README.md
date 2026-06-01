The `scripts` folder contains example pipelines which perform lens modeling to fit different types of Euclid data and assuming different lens and source models.

# Files

- `full_model`: Lens modeling pipeline which fits a full lens model, including a pixelized source, MGE lens light and custom complex mass models.
- `multi_wavelength_full`: Performs full lens modeling to VIS and then all multi-band data (NISP, EXT).
- `mge_lens_model_multi`: Lens modeling using a Multi Gaussian Expansion (MGE) lens light and source, which is fitted to VIS and then all multi-band data (NISP, EXT).
- `mge_lens_only`: Performs a foreground lens only MGE subtraction of the lens emission, such that the source can be quickly revealed for inspection.
