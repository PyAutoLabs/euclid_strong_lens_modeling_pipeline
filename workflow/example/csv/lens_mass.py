"""
Results: CSV Lens Magnitudes
============================

Scrape the results of lens modeling in the `output` folder and produces the  file `lens_mass.csv`
containing all lens mass model parameters for the fit to each "vis" dataset.

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
__Effective Einstein Radius__

Effective Einstein radius, R_Ein_eff

The effective Einstein radius of a lens system is defined as

    R_Ein_eff = sqrt(A / π)

where A is the area enclosed by the tangential critical curve of the lensing deflector.  
    
This quantity ensures consistency in comparing Einstein‐radii across different mass density profiles.  
In this dataset, it is the value given in the column “einstein_radius_effective” in the modelling_lens_mass.csv file.

Notes:
    - The standard Einstein radius of the mass model, θ_E^mass, is given (in this dataset) in a separate 
    column (“einstein_radius”) and is used internally by the modelling code.  
    - The effective Einstein radius is therefore a more literature‐conventional scalar lens radius derived from the 
    actual critical‐curve geometry, rather than just the parametrised lens mass model’s θ_E.
    - Units: The dataset uses angular units (arc-seconds) for all radii.  
"""


def einstein_radius_effective_from(result):

    samples = result.samples

    instance = samples.median_pdf()

    tracer = al.Tracer(galaxies=instance.galaxies)

    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

    return tracer.einstein_radius_from(grid=grid)


agg_csv.add_computed_column(
    name="einstein_radius_effective",
    compute=einstein_radius_effective_from,
)

"""
__SIE Mass Model__

Singular Isothermal Ellipsoid (SIE) mass model

The Singular Isothermal Ellipsoid (SIE) is a widely used analytic model
for describing the projected mass distribution of a gravitational lens.

Refer to the PyAutoLens API docs for a full description of the SIE mass model.
"""
agg_csv.add_variable(
    argument="galaxies.lens.mass.centre.centre_0",
    name="mass_centre_0",
)
agg_csv.add_variable(
    argument="galaxies.lens.mass.centre.centre_1",
    name="mass_centre_1",
)
agg_csv.add_variable(
    argument="galaxies.lens.mass.ell_comps.ell_comps_0",
    name="mass_ell_comps_0",
)
agg_csv.add_variable(
    argument="galaxies.lens.mass.ell_comps.ell_comps_1",
    name="mass_ell_comps_1",
)
agg_csv.add_variable(
    argument="galaxies.lens.mass.einstein_radius",
    name="mass_einstein_radius",
)

"""
__Shear__

The external shear field is parameterized by two components `gamma_1` and `gamma_2`.

See the **PyAutoLens** API docs for a full description of the External Shear model.
"""
agg_csv.add_variable(
    argument="galaxies.lens.shear.gamma_0",
    name="mass_ell_comps_0",
)
agg_csv.add_variable(
    argument="galaxies.lens.shear.gamma_1",
    name="mass_ell_comps_1",
)

agg_csv.save(path=workflow_path / "csv_q1_mass_model.csv")
