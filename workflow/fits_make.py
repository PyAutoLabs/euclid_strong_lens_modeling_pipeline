"""
Results: Fits Make
==================

This example is a results workflow example, which means it provides tool to set up an effective workflow inspecting
and interpreting the large libraries of modeling results.

In this tutorial, we use the aggregator to load .fits files output by a model-fit, extract hdu images and create
new .fits files, for example all to a single folder on your hard-disk.

For example, a common use case is extracting an image from `model_galaxy_images.fits` of many fits and putting them
into a single .fits file on your hard-disk. If you have modeled 100+ datasets, you can then inspect all model images
in DS9 in .fits format n a single folder, which is more efficient than clicking throughout the `output` open each
.fits file one-by-one.

The most common use of .fits splciing is where multiple observations of the same galaxy are analysed, for example
at different wavelengths, where each fit outputs a different .fits files. The model images of each fit to each
wavelength can then be packaged up into a single .fits file.

This enables the results of many model-fits to be concisely visualized and inspected, which can also be easily passed
on to other collaborators.

Internally, splicing uses standard Astorpy functions to open, edit and save .fit files.

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

These name the fit in a descriptive and human-readable way, which we will exploit to make our .fits files
more descriptive and easier to interpret.

__Workflow Paths__

The workflow examples are designed to take large libraries of results and distill them down to the key information
required for your science, which are therefore placed in a single path for easy access.

The `workflow_path` specifies where these files are output, in this case the .fits files containing the key 
results we require.
"""
workflow_path = Path("workflow") / "fits"

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
Extract the `AggregateFITS` object, which has specific functions for loading .fits files and outputting results in 
.fits format.
"""
agg_fits = af.AggregateFITS(aggregator=agg_query)

"""
__Extract Images__

We now extract 2 images from the `fit.fits` file and combine them together into a single .fits file.

We will extract the `model_image` and `residual_map` images, which are images you are used to
plotting and inspecting in the `output` folder of a model-fit and can load and inspect in DS9 from the file
`fit.fits`.

By inspecting `fit.fits` you will see it contains four images which each have a an `ext_name`: `model_image`,
`residual_map`, `normalized_residual_map`, `chi_squared_map`.

We do this by simply passing the `agg_fits.extract_fits` method the name of the fits file we load from `fits.fit`
and the `ext_name` of what we extract.

This runs on all results the `Aggregator` object has loaded from the `output` folder, meaning that for this example
where two model-fits are loaded, the `image` object contains two images.
"""
hdu_list = agg_fits.extract_fits(
    hdus=[al.agg.fits_fit.model_data, al.agg.fits_fit.residual_map],
)

"""
__Output Single Fits__

The `image` object which has been extracted is an `astropy` `Fits` object, which we use to save the .fits to the 
hard-disk.

The .fits has 4 hdus, the `model_image` and `residual_map` for the two datasets fitted.
"""
hdu_list.writeto("fits_make_single.fits", overwrite=True)

"""
__Output to Folder__

An alternative way to output the .fits files is to output them as single .fits files for each model-fit in a single 
folder, which is done using the `output_to_folder` method.

It can sometimes be easier and quicker to inspect the results of many model-fits when they are output to individual
files in a folder, as using an IDE you can click load and flick through the images. This contrasts a single .png
file you scroll through, which may be slower to load and inspect.

__Naming Convention__

We require a naming convention for the output files. In this example, we have two model-fits, therefore two .fits
files are going to be output.

One way to name the .fits files is to use the `unique_tag` of the search, which is unique to every model-fit. For
the search above, the `unique_tag` was `simple_0` and `simple_1`, therefore this will informatively name the .fits
files the names of the datasets.

We achieve this behaviour by inputting `name="unique_tag"` to the `output_to_folder` method. 
"""
agg_fits.output_to_folder(
    folder=workflow_path,
    name="unique_tag",
    hdus=[al.agg.fits_fit.model_data, al.agg.fits_fit.residual_map],
)

"""
The `name` can be any search attribute, for example the `name` of the search, the `path_prefix` of the search, etc,
if they will give informative names to the .fits files.

You can also manually input a list of names, one for each fit, if you want to name the .fits files something else.
However, the list must be the same length as the number of fits in the aggregator, and you may not be certain of the
order of fits in the aggregator and therefore will need to extract this information, for example by printing the
`unique_tag` of each search (or another attribute containing the dataset name).
"""
print([search.unique_tag for search in agg.values("search")])

agg_fits.output_to_folder(
    folder=workflow_path,
    name=["hi_0.fits", "hi_1.fits"],
    hdus=[al.agg.fits_fit.model_data, al.agg.fits_fit.residual_map],
)


"""
__Add Extra Fits__

We can also add an extra .fits image to the extracted .fits file, for example an RGB image of the dataset.

We create an image of shape (1, 2) and add the RGB image to the left of the subplot, so that the new subplot has
shape (1, 3).

When we add a single .png, we cannot extract or make it, it simply gets added to the subplot.
"""
# image_rgb = Image.open(Path(dataset_path) / "rgb.png")
#
# image = agg_fits.extract_fits(
#     al.agg.subplot_dataset.data,
#     al.agg.subplot_dataset.psf_log_10,
#     subplot_shape=(1, 2),
# )

# image = al.add_image_to_left(image, additional_img)

# image.save("png_make_with_rgb.png")

"""
__Custom Fits Files in Analysis__

Describe how a user can extend the `Analysis` class to compute custom images that are output to the .png files,
which they can then extract and make together.
"""

"""
__Path Navigation__

Example combinig `fit.fits` from `source_lp[1]` and `mass_total[0]`.
"""
