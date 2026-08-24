"""
GUI Preprocessing: Extra Galaxies Mask (Optional)
=================================================

There may be regions of an image that have signal near the lens and source that is from other galaxies not associated
with the strong lens we are studying. The emission from these images will impact our model fitting and needs to be
removed from the analysis.

This script creates a mask of these regions of the image, called the `mask_extra_galaxies`, which can be used to
prevent them from impacting a fit. This mask may also include emission from objects which are not technically galaxies,
but blend with the galaxy we are studying in a similar way. Common examples of such objects are foreground stars
or emission due to the data reduction process.

The mask can be applied in different ways. For example, it could be applied such that the image pixels are discarded
from the fit entirely, Alternatively the mask could be used to set the image values to (near) zero and increase their
corresponding noise-map to large values.

The exact method used depends on the nature of the model being fitted. For simple fits like a light profile a mask
is appropriate, as removing image pixels does not change how the model is fitted. However, for more complex models
fits, like those using a pixelization, masking regions of the image in a way that removes their image pixels entirely
from the fit can produce discontinuities in the pixelixation. In this case, scaling the data and noise-map values
may be a better approach.

This script outputs a `mask_extra_galaxies.fits` file, which can be loaded and used before a model fit, in whatever
way is appropriate for the model being fitted.

This script uses a GUI to mark the regions of the image where these extra galaxies are located, in contrast to the
example above which requires you to input these values manually.

__Links / Resources__

The script `data_preparation/gui/extra_galaxies_mask.ipynb` shows how to use a Graphical User Interface (GUI) to create
the extra galaxies mask.

__Start Here Notebook__

If any code in this script is unclear, refer to the `data_preparation/start_here.ipynb` notebook.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import argparse
import numpy as np
from pathlib import Path
import autolens as al

"""
__Dataset__

The path where the extra galaxy mask is output, which is `dataset/imaging/extra_galaxies`.
"""
parser = argparse.ArgumentParser(description="Lens Model Inputs")
parser.add_argument(
    "--dataset", metavar="path", required=True, help="the path to the dataset"
)
args = parser.parse_args()
dataset_name = args.dataset

dataset_main_path = Path("dataset") / dataset_name
dataset_fits_name = f"{dataset_name}.fits"

vis_index = 3

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the `Imaging` data, where the extra galaxies are visible in the data.
"""
data = al.Array2D.from_fits(
    file_path=dataset_main_path / dataset_fits_name,
    hdu=vis_index * 3 + 1,
    pixel_scales=pixel_scales,
)

data = al.Array2D(
    values=np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), mask=data.mask
)

"""
__Mask__

Create a 3.0" mask to plot over the image to guide where extra galaxy light needs its emission removed and noise scaled.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=data.shape_native, pixel_scales=data.pixel_scales, radius=mask_radius
)

"""
__Scribbler__

Load the Scribbler GUI for spray painting the scaled regions of the dataset. 

Push Esc when you are finished spray painting.
"""
scribbler = al.Scribbler(
    image=data.native,
    cmap="jet",
    norm="log",
    vmin=1.0e-3,
    vmax=np.max(data) / 3.0,
    brush_width=0.05,
    mask_overlay=mask,
)
mask = scribbler.show_mask()
mask = al.Mask2D(mask=mask, pixel_scales=pixel_scales)

"""
The GUI has now closed and the extra galaxies mask has been created.

Apply the extra galaxies mask to the image, which will remove them from visualization.
"""
data = data.apply_mask(mask=mask)

"""
__Output__

Output the extra galaxies mask, which will be load and used before a model fit.
"""
mask.output_to_fits(
    file_path=dataset_main_path / "mask_extra_galaxies.fits", overwrite=True
)
