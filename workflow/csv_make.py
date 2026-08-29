"""
Results: CSV
============

This example is a results workflow example, which means it provides tool to set up an effective workflow inspecting
and interpreting the large libraries of modeling results.

In this tutorial, we use the aggregator to load the results of model-fits and output them in a single .csv file.

This enables the results of many model-fits to be concisely summarised and inspected in a single table, which
can also be easily passed on to other collaborators.

__CSV, Png and Fits__

Workflow functionality closely mirrors the `png_make.py` and `fits_make.py`  examples, which load results of
model-fits and output th em as .png files and .fits files to quickly summarise results.

The same initial fit creating results in a folder called `results_folder_csv_png_fits` is therefore used.

__Real Examples__

This example illustrates the API for outputting results to a .csv file.

In the folder `workflow/examples`, we provide actual examples of how this APi is used to create .csv files
that output the results of real lens modeling using the Euclid pipeline, for example a .csv with all mass
model parameters for every lens model-fit to a "vis" dataset.

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
# Example for the scripts/initial_lens_model.py pipeline

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
Extract the `AggregateCSV` object, which has specific functions for outputting results in a CSV format.
"""
agg_csv = af.AggregateCSV(aggregator=agg_query)

"""
__Model Paths__

The paths are the tuples which define how model parameters are accessed from the model.
"""
model = [model for model in agg_query.values("model")][0]
print(model.paths)

"""
__Adding CSV Columns__

We first make a simple .csv which contains two columns, corresponding to the inferred median PDF values for
the y centre of the mass of the lens galaxy and its einstein radius.

To do this, we use the `add_variable` method, which adds a column to the .csv file we write at the end. Every time
we call `add_variable` we add a new column to the .csv file.

Note the API for the `centre`, which is a tuple parameter and therefore needs for `centre_0` to be specified.

The `results_folder_csv_png_fits` contained two model-fits to two different datasets, meaning that each `add_variable` 
call will add three rows, corresponding to the three model-fits.

This adds the median PDF value of the parameter to the .csv file, we show how to add other values later in this script.
"""
agg_csv.add_variable(argument="galaxies.lens.mass.centre.centre_0")
agg_csv.add_variable(argument="galaxies.lens.mass.einstein_radius")

"""
__Saving the CSV__

We can now output the results of all our model-fits to the .csv file, using the `save` method.

This will output in your current working directory (e.g. the `autolens_workspace/output.results_folder_csv_png_fits`) 
as a .csv file containing the median PDF values of the parameters, have a quick look now to see the format of 
the .csv file.
"""
agg_csv.save(path=workflow_path / "csv_simple.csv")

"""
__Customizing CSV Headers__

The headers of the .csv file are by default the argument input above based on the model. 

However, we can customize these headers using the `name` input of the `add_variable` method, for example making them
shorter or more readable.

We recreate the `agg_csv` first, so that we begin adding columns to a new .csv file.
"""
agg_csv = af.AggregateCSV(aggregator=agg_query)

agg_csv.add_variable(
    argument="galaxies.lens.mass.centre.centre_0",
    name="mass_centre_0",
)
agg_csv.add_variable(
    argument="galaxies.lens.mass.einstein_radius",
    name="mass_einstein_radius",
)

agg_csv.save(path=workflow_path / "csv_simple_custom_headers.csv")

"""
__Maximum Likelihood Values__

We can also output the maximum likelihood values of each parameter to the .csv file, using the `use_max_log_likelihood`
input.
"""
agg_csv = af.AggregateCSV(aggregator=agg_query)

agg_csv.add_variable(
    argument="galaxies.lens.mass.einstein_radius",
    name="mass_einstein_radius_max_lh",
    value_types=[af.ValueType.MaxLogLikelihood],
)

agg_csv.save(path=workflow_path / "csv_simple_max_likelihood.csv")

"""
__Errors__

We can also output PDF values at a given sigma confidence of each parameter to the .csv file, using 
the `af.ValueType.ValuesAt3Sigma` input and specifying the sigma confidence.

Below, we add the values at 3.0 sigma confidence to the .csv file, in order to compute the errors you would 
subtract the median value from these values. We add this after the median value, so that the overall inferred
uncertainty of the parameter is clear.

The method below adds three columns to the .csv file, corresponding to the values at the median, lower and upper sigma 
values.
"""
agg_csv = af.AggregateCSV(aggregator=agg_query)

agg_csv.add_variable(
    argument="galaxies.lens.mass.einstein_radius",
    name="mass_einstein_radius",
    value_types=[af.ValueType.Median, af.ValueType.ValuesAt3Sigma],
)

agg_csv.save(path=workflow_path / "csv_simple_errors.csv")

"""
__Column Label List__

We can add a list of values to the .csv file, provided the list is the same length as the number of model-fits
in the aggregator.

A useful example is adding the name and waveband of every dataset to the .csv file in a column on the left, indicating 
which dataset and waveband each row corresponds to.

To make this list, we use the `Aggregator` to loop over the `search` objects and extract their `path_prefix` 
and `unique_tag`'s, which  when we fitted the model above used the dataset name and waveband. This API can also be used 
to extract the `name` of the search and build an informative list for the names of the subplots.

The `path_prefix` contains the `output` folder prefix so a `parts` method is used to remove this, leaving just the
dataset name and waveband.

We then pass the column `name` and this list to the `add_label_column` method, which will add a column to the .csv file.
"""
agg_csv = af.AggregateCSV(aggregator=agg_query)

lens_name_list = [search.path_prefix.parts[-1] for search in agg.values("search")]

agg_csv.add_label_column(
    name="lens_name",
    values=lens_name_list,
)

waveband_list = [search.unique_tag for search in agg.values("search")]

agg_csv.add_label_column(
    name="waveband",
    values=waveband_list,
)

agg_csv.save(path=workflow_path / "csv_simple_dataset_name.csv")


"""
__Latent Variables__

Latent variables are not free model parameters but can be derived from the model, and they are described fully in
?.

This example was run with a latent variable called `example_latent`, and below we show that this latent variable
can be added to the .csv file using the same API as above.
"""
agg_csv = af.AggregateCSV(aggregator=agg_query)

agg_csv.add_variable(
    argument="galaxies.lens.shear.magnitude",
)

agg_csv.save(path=workflow_path / "csv_example_latent.csv")

"""
__Computed Columns__

We can also add columns to the .csv file that are computed from the median PDF instance values of the model.

To do this, we write a function which is input into the `add_computed_column` method, where this function takes the
median PDF instance as input and returns the computed value.

Below, we add a trivial example of a computed column, where the value is twice the sersic index of the bulge.
"""
agg_csv = af.AggregateCSV(aggregator=agg_query)


def einstein_radius_x2_from(samples):
    instance = samples.median_pdf()

    return 2.0 * instance.galaxies.lens.mass.einstein_radius


agg_csv.add_computed_column(
    name="bulge_einstein_radius_x2_computed",
    compute=einstein_radius_x2_from,
)

agg_csv.save(path=workflow_path / "csv_computed_columns.csv")

"""
__Lens Name and Wavelength__

We can add a list of values to the .csv file, provided the list is the same length as the number of model-fits
in the aggregator.

A useful example is adding the name and waveband of every dataset to the .csv file in a column on the left, indicating 
which dataset and waveband each row corresponds to.
"""
agg_csv = af.AggregateCSV(aggregator=agg_query)

lens_name_list = [search.path_prefix.parts[-1] for search in agg_query.values("search")]

agg_csv.add_label_column(
    name="lens_name",
    values=lens_name_list,
)

waveband_list = [search.unique_tag for search in agg_query.values("search")]

agg_csv.add_label_column(
    name="waveband",
    values=waveband_list,
)

"""
__Lens Model Parameters_

We now add the lens model parameters to the .csv file, including the maximum log likelihood value, median PDF value and
3.0 sigma confidence intervals.
"""
value_types = (
    af.ValueType.MaxLogLikelihood,
    af.ValueType.Median,
    af.ValueType.ValuesAt3Sigma,
)
value_types = (af.ValueType.MaxLogLikelihood,)


def add_var(arg_suffix: str, name: str):
    agg_csv.add_variable(
        argument=f"galaxies.lens.bulge.profile_list.{arg_suffix}",
        name=name,
        value_types=value_types,
    )


add_var("0.centre.centre_0", "bulge_centre_0")
add_var("0.centre.centre_1", "bulge_centre_1")
add_var("0.ell_comps.ell_comps_0", "bulge_ell_comps_0")
add_var("0.ell_comps.ell_comps_1", "bulge_ell_comps_1")

"""
__Saving the CSV__

We can now output the results of all our model-fits to the .csv file, using the `save` method.
"""
agg_csv.save(path=workflow_path / "result_lens_mge__mge_lens_only.csv")
