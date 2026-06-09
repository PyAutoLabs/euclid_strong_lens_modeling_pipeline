import util
import json
import numpy as np
from pathlib import Path

import autofit as af
import autolens as al

"""
__Dataset__
"""
dataset_name = "102018665_NEG570040238507752998"
sample_name = "q1_walsmley"

dataset_main_path = Path("dataset") / sample_name / dataset_name
dataset_fits_name = f"{dataset_name}.fits"

"""
__Dataset Wavebands__
"""
dataset_index_dict = util.dataset_instrument_hdu_dict_via_fits_from(
    dataset_path=dataset_main_path,
    dataset_fits_name=dataset_fits_name,
    image_tag="_BGSUB",  # Depends on how Euclid cutout was made
)

dataset_waveband = "vis"

vis_index = dataset_index_dict[dataset_waveband]

"""
__Dataset__
"""
dataset = al.Imaging.from_fits(
    data_path=dataset_main_path / dataset_fits_name,
    data_hdu=vis_index * 3 + 1,
    noise_map_path=dataset_main_path / dataset_fits_name,
    noise_map_hdu=vis_index * 3 + 3,
    psf_path=dataset_main_path / dataset_fits_name,
    psf_hdu=vis_index * 3 + 2,
    pixel_scales=0.1,
    check_noise_map=False,
)

dataset_centre = dataset.data.brightest_sub_pixel_coordinate_in_region_from(
    region=(-0.3, 0.3, -0.3, 0.3), box_size=2
)

"""
__Info__
"""
try:
    with open(dataset_main_path / "info.json") as json_file:
        info = json.load(json_file)
        json_file.close()
except FileNotFoundError:
    info = {}

"""
__Header__

"""
try:
    header = al.header_obj_from(
        file_path=dataset_main_path / dataset_fits_name,
        hdu=vis_index * 3 + 1,
    )
    magzero = header["MAGZERO"]
except FileNotFoundError:
    magzero = None

"""
__WCS__
"""
from astropy.wcs import WCS

pixel_wcs = WCS(header).celestial

"""
__Extra Galaxy Removal__
"""
try:
    mask_extra_galaxies = al.Mask2D.from_fits(
        file_path=dataset_main_path / "mask_extra_galaxies.fits",
        pixel_scales=0.1,
        invert=True,
    )

    dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)
except FileNotFoundError:
    pass

"""
__Mask__

"""
mask_radius = info.get("mask_radius") or 3.0
mask_centre = info.get("mask_centre") or (0.0, 0.0)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
    centre=mask_centre,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.1, 0.3],
    centre_list=[dataset_centre],
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)


"""
__Lowest Resolution PSF__
"""
header_primary = al.header_obj_from(
    file_path=dataset_main_path / dataset_fits_name,
    hdu=0,
)

lowest_resolution_waveband = header_primary.get("WORST_BAND", None).lower()

lowest_resolution_waveband_index = dataset_index_dict.get(
    lowest_resolution_waveband, None
)

psf_lowest_resolution = al.Convolver.from_fits(
    file_path=dataset_main_path / dataset_fits_name,
    hdu=lowest_resolution_waveband_index * 3 + 2,
    pixel_scales=0.1,
    normalize=True,
)

# Use OU-MER worst PSF FWHM if available, but if its -99 meaning the OU-MER pipeline failed used a
# fall back value computed during lens cutout creation.
psf_lowest_resolution_fwhm = float(header_primary.get("WORST_PSF_MER", None))

if psf_lowest_resolution_fwhm is None or psf_lowest_resolution_fwhm < -98:
    psf_lowest_resolution_fwhm = float(header_primary.get("WORST_PSF_HDR", None))

"""
__Redshifts__
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Model__

onal and well-conditioned, enabling efficient 
sampling with a high probability of finding the global maximum.      
"""
# Lens:

lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius,
    total_gaussians=20,
    gaussian_per_basis=2,
    centre_prior_is_uniform=True,
    centre=dataset_centre,
)

mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = dataset_centre[0]
mass.centre.centre_1 = dataset_centre[1]

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(
    al.Galaxy, redshift=redshift_lens, bulge=lens_bulge, mass=mass, shear=shear
)

# Source:

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

source = af.Model(al.Galaxy, redshift=redshift_source, bulge=source_bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Analysis__
"""
analysis = util.AnalysisImaging(
    dataset=dataset,
    use_jax=True,  # JAX will use GPUs for acceleration if available, else JAX will use multithreaded CPUs.
    dataset_main_path=dataset_main_path,
    title_prefix=dataset_waveband.upper(),
    plot_rgb=True,
    psf_lowest_resolution=psf_lowest_resolution,
    psf_lowest_resolution_fwhm=psf_lowest_resolution_fwhm,
    pixel_wcs=pixel_wcs,
    magzero=magzero,
)

"""
__Compute Latent Variable__
"""
batch_size = 2

param_vector = np.array(model.physical_values_from_prior_medians)

parameters = np.zeros(model.total_free_parameters)
parameters[:] = param_vector

latent_variables = util.LatentEuclid.variables(
    analysis,
    parameters=parameters,
    model=model,
)


def test_compute_latent_variables_smoke():
    assert len(latent_variables) == len(util.LatentEuclid.keys(analysis))
