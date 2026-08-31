"""
GUI Preprocessing: Extra Galaxies Mask Dataset
==============================================

This tool allows one to mask a bespoke noise-map for a given image of a strong lens, using a GUI.

This noise-map is primarily used for increasing the variances of pixels that have non-modeled components in an image,
for example intervening line-of-sight galaxies that are near the lens, but not directly interfering with the
analysis of the lens and source galaxies.

This GUI is adapted from the following code: https://gist.github.com/brikeats/4f63f867fd8ea0f196c78e9b835150ab
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import json
import autolens as al
import autolens.plot as aplt
import numpy as np
from pathlib import Path

"""
__Dataset__

Setup the path the datasets we'll use to illustrate preprocessing, which is the 
folder `dataset/imaging/no_lens_light/mass_sie__source_sersic`.
"""

dataset_name = "102019586_NEG564463213499964061"
dataset_path = Path("dataset") / dataset_name

psf_full = al.Convolver.from_fits(
    file_path=Path(dataset_path) / "psf_full.fits", hdu=0, pixel_scales=0.1
)

print("PSF Shape Before Trimming:")
print(psf_full.shape_native)

new_shape = (11, 11)

print()
print(f"PSF being trimmed to {new_shape}")

psf = psf_full.resized_from(new_shape=new_shape)

psf.output_to_fits(file_path=Path(dataset_path) / "psf.fits", overwrite=True)


"""
__Png output__

Plot the full and trimmed PSFs side by side as log10 .png images in the dataset
folder, so the trim can be checked by eye: the trimmed kernel should retain the
PSF core and its first ring while the clipped wings are visibly negligible.
"""
aplt.plot_array(
    array=psf_full,
    output_path=dataset_path,
    output_filename="psf_full",
    output_format="png",
    use_log10=True,
)

aplt.plot_array(
    array=psf.kernel,
    output_path=dataset_path,
    output_filename="psf",
    output_format="png",
    use_log10=True,
)
