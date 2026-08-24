"""
Results: CSV Lens Magnitudes
============================

Scrape the results of lens modeling in the `output` folder and produces the  file `magnitudes.csv`
containing the lens, image-plane lensed source and delensed source-plane source magnitudes for all wavebands.

The .csv workflow API is described in `workflow/csv_make.py` and you should read that tutorial first in full
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

value_types = (af.ValueType.MaxLogLikelihood,)

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
__Lens / Source Fluxes__

Library latent keys (shipped in PyAutoLens) use bare, snake-case-lowercase
names with a ``_mujy`` suffix where applicable — no ``latent.`` prefix.
The aggregator reads them via single-string keys, not the old
``("latent", "X")`` tuple form.
"""
agg_csv.add_variable(
    argument="total_lens_flux_mujy",
)

agg_csv.add_variable(
    argument="total_lensed_source_flux_mujy",
)

agg_csv.add_variable(
    argument="total_source_flux_mujy",
)

"""
__Magnification__

``magnification`` is now a first-class library latent (shipped in
PyAutoLens), so we read it directly rather than computing it from the
two source fluxes by hand.
"""
agg_csv.add_variable(
    argument="magnification",
)


agg_csv.save(path=workflow_path / "magnitudes.csv")
