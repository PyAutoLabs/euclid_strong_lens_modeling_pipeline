"""
Results: MGE Inspect
====================

Scrape the results of lens modeling in the `output` folder and produces png  files suitable for inspecting if an
MGE lens + source model is a good fit.

The image contains the lens light subtracted image, source model image, zoomed in source plane image and full
source plane image.

The .png workflow API is described in `workflow/png_make.py` and you should read that tutorial first in full
before reading this example, as the same API is used here.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Path Prefix / Unique Tag__

In the modeling pipelines, the `path_prefix` `unique_tag` of the search is given the name and waveband of the dataset.

These name the fit in a descriptive and human-readable way, which we will exploit to make our .csv files
more descriptive and easier to interpret.

__Workflow Paths__

The workflow examples are designed to take large libraries of results and distill them down to the key information
required for your science, which are therefore placed in a single path for easy access.

The `workflow_path` specifies where these files are output, in this case the .csv files containing the key 
results we require.
"""
workflow_path = Path("workflow") / "csv"

"""
__Output Folder Structure__

The structure of the output folder depends on which Euclid lens modeling pipeline you have been running,
for example `mge_lens_only`, `mge_lens_model`, `slam` or another.

You also need to decide which waveband you want to inspect, with VIS the default.

You should choose the strings below corresponding to the results you want to load and analyse.
"""
# Example for start_here.py pipeline

pipeline_name = "initial_lens_model"
dataset_waveband = "vis"

"""
__Aggregator__

Set up the aggregator which will load results from the output folder of the model-fit.
"""
from autofit.aggregator.aggregator import Aggregator

agg = Aggregator.from_directory(directory=Path("output"), completed_only=True)

"""
Use a query on the aggregator to only get results for the `mass_total[1]` model-fit, which contains the final lens
model results.
"""
agg_query = agg.query(agg.unique_tag == pipeline_name)
agg_query = agg.query(agg_query.search.name == dataset_waveband)

"""
Extract the `AggregateImages` object, which has specific functions for loading image files (e.g. .png, .pdf) and
outputting results in an image format (e.g. .png, .pdf).
"""
agg_image = af.AggregateImages(aggregator=agg)

"""
__Output to Folder__

Output the images as single .png files for each model-fit in a single folder, using the `output_to_folder` method.

It can sometimes be easier and quicker to inspect the results of many model-fits when they are output to individual
files in a folder, as using an IDE you can click load and flick through the images. This contrasts a single .png
file you scroll through, which may be slower to load and inspect.
"""
subplots = [
    al.agg.subplot_fit.lens_light_subtracted_image,
    al.agg.subplot_fit.source_model_image,
    al.agg.subplot_fit.source_plane_image_zoom,
    al.agg.subplot_fit.source_plane_image,
]

image = agg_image.extract_image(subplots=subplots)

image.save(Path("workflow") / "png" / "0_master_image.png")

agg_image.output_to_folder(
    folder=Path("workflow") / "png",
    name=[search.path_prefix.parts[-1] for search in agg.values("search")],
    subplots=subplots,
)
