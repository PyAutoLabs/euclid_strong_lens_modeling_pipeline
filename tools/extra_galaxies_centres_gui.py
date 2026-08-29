"""
GUI Preprocessing: Extra Galaxies Centres
=========================================

There may be extra galaxies nearby the lens and source galaxies, whose emission blends with the lens and source
and whose mass may contribute to the ray-tracing and lens model.

This script uses a GUI to mark the (y,x) arcsecond locations of these extra galaxies, each of which is then included
in the lens model as light profiles (a Multi Gasusian Expansion) and mass profiles (singular isothermal spheres) if
the `groups.py` pipeline is used (the `scripts/initial_lens_model.py` pipeline does not include extra galaxies for simplicity).

Extra galaxies require that the mask is expanded to include their light, which is done by computing the radial
distance of the furthest extra galaxy from the origin and adding a buffer. This mask is used in the `groups.py`
pipeline to mask the lens and source galaxies and the extra galaxies.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import argparse
import json
import numpy as np
import os
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import autolens as al
import autolens.plot as aplt

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from util import dataset_instrument_hdu_dict_via_fits_from

"""
__Dataset__

The path where the extra galaxy centres are output, which is `datasetgroup`.
"""
parser = argparse.ArgumentParser(description="Lens Model Inputs")
parser.add_argument(
    "--dataset", metavar="path", required=True, help="the path to the dataset"
)
parser.add_argument(
    "--sample",
    metavar="name",
    required=False,
    default=None,
    help="Sample subdirectory inside dataset/ containing the dataset.",
)
args = parser.parse_args()
dataset_name = args.dataset
sample_name = args.sample

if sample_name is not None:
    dataset_main_path = Path("dataset") / sample_name / dataset_name
else:
    dataset_main_path = Path("dataset") / dataset_name
dataset_fits_name = f"{dataset_name}.fits"

# Resolve the VIS HDU index from the FITS file rather than hardcoding it —
# different datasets have different numbers of ancillary bands preceding VIS.
vis_index = dataset_instrument_hdu_dict_via_fits_from(
    dataset_path=dataset_main_path,
    dataset_fits_name=dataset_fits_name,
    image_tag="_BGSUB",
)["vis"]

"""
The pixel scale of the imaging dataset.
"""
pixel_scales = 0.1

"""
Load the image which we will use to mark the lens light centre.
"""
data = al.Array2D.from_fits(
    file_path=dataset_main_path / dataset_fits_name,
    hdu=vis_index * 3 + 1,
    pixel_scales=pixel_scales,
)

"""
__Mask__

Create a 3.0" mask to plot over the image to guide where points should be marked.
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=data.shape_native, pixel_scales=data.pixel_scales, radius=mask_radius
)

grid = mask.derive_grid.edge

"""
__Search Box__

When you click on a pixel to mark a position, the search box looks around this click and finds the pixel with
the highest flux to mark the position.

The `search_box_size` is the number of pixels around your click this search takes place.
"""
search_box_size = 3

"""
__Clicker__

Set up the `Clicker` object from the `clicker.py` module, which monitors your mouse clicks in order to determine
the extra galaxy centres.
"""
clicker = al.Clicker(
    image=data, pixel_scales=pixel_scales, search_box_size=search_box_size
)

"""
Set up the clicker canvas and load the GUI which you can now click on to mark the extra galaxy centres.
"""
n_y, n_x = data.shape_native
hw = int(n_x / 2) * pixel_scales
ext = [-hw, hw, -hw, hw]
fig = plt.figure(figsize=(14, 14))
norm = LogNorm(vmin=1.0e-3, vmax=np.max(data) / 3.0)
plt.imshow(data.native, cmap="jet", norm=norm, extent=ext)
plt.scatter(y=grid[:, 0], x=grid[:, 1], c="k", marker="x", s=10)
plt.colorbar()
cid = fig.canvas.mpl_connect("button_press_event", clicker.onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)
plt.close(fig)

"""
Use the results of the Clicker GUI to create the list of extra galaxy centres.
"""
extra_galaxies_centres = al.Grid2DIrregular(values=clicker.click_list)

"""
__Mask Radius__

The circular mask radius is calculated as the radial distance of the furthest extra galaxy from the origin,
with a buffer of 0.2" added to this value.

If the lens has no extra galaxies, the default mask radius of 3.0" is used instead.

This will be used as the circular mask radius of any modeling pipeline if a user input value is not input
(recommended behaviour).
"""
if len(extra_galaxies_centres) > 1:
    extra_galaxies_radii = np.sqrt(
        extra_galaxies_centres[:, 0] ** 2 + extra_galaxies_centres[:, 1] ** 2
    )
    extra_galaxies_radius_buffer = 0.2
    extra_galaxies_max_radius = (
        np.max(extra_galaxies_radii) + extra_galaxies_radius_buffer
    )
    mask_radius = max(mask_radius, extra_galaxies_max_radius)

info = {}

if Path(dataset_main_path / "info.json").exists():
    try:
        with open(dataset_main_path / "info.json") as json_file:
            info = json.load(json_file)
            json_file.close()
    except FileNotFoundError:
        info = {}

    try:
        header = al.header_obj_from(
            file_path=dataset_main_path / dataset_fits_name,
            hdu=vis_index * 3 + 1,
        )
        magzero = header["MAGZERO"]
    except FileNotFoundError:
        magzero = None

info["mask_radius"] = mask_radius

with open(dataset_main_path / "info.json", "w") as json_file:
    json.dump(info, json_file)
    json_file.close()

"""
__Output__

Output this image of the extra galaxy centres to a .png file in the dataset folder for future reference.
"""
mask = al.Mask2D.circular(
    shape_native=data.shape_native, pixel_scales=data.pixel_scales, radius=mask_radius
)

aplt.plot_array(
    array=data,
    mask=mask,
    positions=extra_galaxies_centres,
    output_path=dataset_main_path,
    output_filename="extra_galaxies_centres",
    output_format="png",
    use_log10=True,
)

"""
Output the extra galaxy centres to the dataset folder of the lens, so that we can load them from a .json file 
when we model them.
"""
al.output_to_json(
    obj=extra_galaxies_centres,
    file_path=dataset_main_path / "extra_galaxies_centres.json",
)
