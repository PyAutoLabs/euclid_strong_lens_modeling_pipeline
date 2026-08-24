"""
Results: PNG Make
=================

This example is a results workflow example, which means it provides tool to set up an effective workflow inspecting
and interpreting the large libraries of modeling results.

In this tutorial, we use the aggregator to load .png files output by a model-fit, make them together to create
new .png images and then output them all to a single folder on your hard-disk.

For example, a common use case is extracting a subset of 3 or 4 images from `subplot_fit.png` which show the model-fit
quality, put them on a single line .png subplot and output them all to a single folder on your hard-disk. If you have
modeled 100+ datasets, you can then inspect all fits as .pngs in a single folder (or make a single. png file of all of
them which you scroll down), which is more efficient than clicking throughout the `output` folder to inspect
each lens result one-by-one.

Different .png images can be combined together, for example the goodness-of-fit images from `subplot.png`,
RGB images of each galaxy in the `dataset` folder and other images.

This enables the results of many model-fits to be concisely visualized and inspected, which can also be easily passed
on to other collaborators.

Internally, splicing uses the Python Imaging Library (PIL) to open, edit and save .png files. This is a Python library
that provides extensive file format support, an efficient internal representation and powerful image-processing
capabilities.

__CSV, Png and Fits__

Workflow functionality closely mirrors the `png_make.py` and `fits_make.py`  examples, which load results of
model-fits and output th em as .png files and .fits files to quickly summarise results.

The same initial fit creating results in a folder called `results_folder_csv_png_fits` is therefore used.

__Database File__

The aggregator can also load results from a `.sqlite` database file.

This is beneficial when loading results for large numbers of model-fits (e.g. more than hundreds)
because it is optimized for fast querying of results.

See the package `results/database` for a full description of how to set up the database and the benefits it provides,
especially if loading results from hard-disk is slow.
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

These name the fit in a descriptive and human-readable way, which we will exploit to make our .png files
more descriptive and easier to interpret.

__Workflow Paths__

The workflow examples are designed to take large libraries of results and distill them down to the key information
required for your science, which are therefore placed in a single path for easy access.

The `workflow_path` specifies where these files are output, in this case the .png files containing the key 
results we require.
"""
workflow_path = Path("workflow") / "png"

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
agg_image = af.AggregateImages(aggregator=agg_query)

"""
__Extract Images__

We now extract 3 images from the `subplot_fit.png` file and make them together into a single image.

We will extract the `data`, `model_image` and `normalized_residual_map` images, which are images you are used to
plotting and inspecting in the `output` folder of a model-fit.

We do this by simply passing the `agg_image.extract_image` method the `al.agg` attribute for each image we want to
extract.

This runs on all results the `Aggregator` object has loaded from the `output` folder, meaning that for this example
where two model-fits are loaded, the `image` object contains two images.

The `subplot_shape` input above determines the layout of the subplots in the final image, which for the example below
is a single row of 3 subplots.
"""
image = agg_image.extract_image(
    subplots=[
        al.agg.subplot_fit.data,
        al.agg.subplot_fit.model_data,
        al.agg.subplot_fit.normalized_residual_map,
    ],
)


"""
__Output Single Png__

The `image` object which has been extracted is a `Image` object from the Python package `PIL`, which we use
to save the image to the hard-disk as a .png file.

The .png is a single subplot of two rows, where each subplot is the data, model data and residual-map of a model-fit.
"""
image.save(workflow_path / "png_make_single_subplot.png")

"""
__Output to Folder__

An alternative way to output the image is to output them as single .png files for each model-fit in a single folder,
which is done using the `output_to_folder` method.

It can sometimes be easier and quicker to inspect the results of many model-fits when they are output to individual
files in a folder, as using an IDE you can click load and flick through the images. This contrasts a single .png
file you scroll through, which may be slower to load and inspect.

__Naming Convention__

We require a naming convention for the output files. In this example, we want each fit to be named after the dataset
name.

One way to name the .png files is to use the `path_prefix` of the search, which is unique to every model-fit. For
the Euclid pipeline, the `path_prefix` includes the dataset name, therefore this will informatively name the .png
files the names of the datasets.

We achieve this behaviour by inputting `name="path_prefix"` to the `output_to_folder` method. 
"""
agg_image.output_to_folder(
    folder=workflow_path,
    name="path_prefix",
    subplots=[
        al.agg.subplot_fit.data,
        al.agg.subplot_fit.model_data,
        al.agg.subplot_fit.normalized_residual_map,
    ],
)

"""
The `name` can be any search attribute, for example the `name` of the search, the `path_prefix` of the search, etc,
if they will give informative names to the .png files.

You can also manually input a list of names, one for each fit, if you want to name the .png files something else.
However, the list must be the same length as the number of fits in the aggregator, and you may not be certain of the
order of fits in the aggregator and therefore will need to extract this information, for example by printing the
`unique_tag` of each search (or another attribute containing the dataset name).
"""
print([search.path_prefix.parts[-1] for search in agg.values("search")])

agg_image.output_to_folder(
    folder=workflow_path,
    name=[search.path_prefix.parts[-1] for search in agg.values("search")],
    subplots=[
        al.agg.subplot_fit.data,
        al.agg.subplot_fit.model_data,
        al.agg.subplot_fit.normalized_residual_map,
    ],
)

"""
__Combine Images From Subplots__

We now combine images from two different subplots into a single image, which we will save to the hard-disk as a .png
file.

We will extract images from the `subplot_dataset.png` and `subplot_fit.png` images, which are images you are used to 
plotting and inspecting in the `output` folder of a model-fit.

We extract the `data` and `psf_log10` from the dataset and the `model_data` and `chi_squared_map` from the fit,
and combine them into a subplot with an overall shape of (2, 2).
"""
image = agg_image.extract_image(
    subplots=[
        al.agg.subplot_dataset.data,
        al.agg.subplot_dataset.psf_log_10,
        al.agg.subplot_fit.model_data,
        al.agg.subplot_fit.chi_squared_map,
    ]
    # subplot_shape=(2, 2),
)

image.save(workflow_path / "png_make_multi_subplot.png")


"""
__Add RGB Png__

We can also add the RGB image of the dataset, if it was output with the fit in `subplot_rgb.png`.

This requires us to define a custom ENum object, describing the RGB subplot. The subplot has 4 panels in a (2,2)
configuration, this ENum simply gives each a name that we can call.
"""
from enum import Enum


class SubplotRgb(Enum):
    """
    The subplots that can be extracted from the subplot_fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    rgb_0 = (0, 0)
    rgb_1 = (1, 0)
    rgb_0_zoom = (0, 1)
    rgb_1_zoom = (1, 1)


subplot_rgb = SubplotRgb

image = agg_image.extract_image(
    subplots=[
        subplot_rgb.rgb_0,
        al.agg.subplot_dataset.data,
        al.agg.subplot_fit.model_data,
        al.agg.subplot_fit.chi_squared_map,
    ]
    # subplot_shape=(2, 2),
)

image.save(workflow_path / "png_make_with_rgb_subplot.png")

"""
__Custom Subplots in Analysis__

Describe how a user can extend the `Analysis` class to compute custom images that are output to the .png files,
which they can then extract and make together.

__Path Navigation__

Example combinng `subplot_fit.png` from `source_lp[1]` and `mass_total[0]`.
"""
