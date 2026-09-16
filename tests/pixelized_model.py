"""
The ``vis_pix`` stage as a zero-free-parameter model, for the test modules
that need a pixelized-source fit without a non-linear search.

``scripts/initial_lens_model.py`` builds that stage from the preceding
``vis_lp`` result: the lens galaxy is carried over as an *instance* (light +
mass + shear), and the source's light profile is replaced by a
``Pixelization`` whose ``Delaunay`` mesh takes its vertices from an image-plane
grid built at run time — a ``Hilbert`` mesh drawn from the source's adapt
image, with a ring of circle-edge points appended and zeroed. That grid cannot
live in the model, so it travels to the analysis inside ``AdaptImages``.

Mirrored here with a truth model standing in for the ``vis_lp`` result: the
lens is the truth lens, and the adapt image is the truth source's own
image-plane (lensed) image with the script's floor applied, which is what the
``vis_lp`` source model image is an estimate of. The mesh is a quarter the size
the pipeline uses, purely so the inversion stays inside the fast suite. It is
the builder ``test_compute_latent_variable.py``'s ``pixelized_source_model``
fixture documents; ``test_wcs_dict.py`` and ``test_latent_run_level.py`` share
it from here.
"""

import numpy as np

# The `vis_pix` Delaunay mesh, mirrored from `scripts/initial_lens_model.py` at a
# size that keeps the inversion inside the fast suite: the script draws 500
# Hilbert points and appends a 30-point circle-edge ring, this draws 150 and
# appends the same ring. `HILBERT_WEIGHT_POWER` / `HILBERT_WEIGHT_FLOOR` and the
# adapt-image floor are the script's own values.
HILBERT_PIXELS = 150
EDGE_PIXELS_TOTAL = 30
HILBERT_WEIGHT_POWER = 3.5
HILBERT_WEIGHT_FLOOR = 0.01
ADAPT_IMAGE_FLOOR = 0.01

# The model path `AdaptImages` keys its per-galaxy entries by, which is why the
# pixelized model's galaxies are named `lens` / `source` (as the pipeline names
# them) rather than `galaxy_0` / `galaxy_1` (as `truth.json` does).
SOURCE_PATH = "('galaxies', 'source')"


def pixelized_model_and_adapt_images_from(lens, sersic_source, euclid_dataset):
    """
    The ``vis_pix`` model (zero free parameters) and the ``AdaptImages`` its
    ``Delaunay`` mesh needs, for a truth ``lens`` and ``sersic_source`` galaxy
    on the loaded ``euclid_dataset``.
    """
    import autofit as af
    import autolens as al

    dataset = euclid_dataset.dataset

    adapt_data = al.Tracer(
        galaxies=[lens, sersic_source]
    ).galaxy_image_2d_dict_from(grid=dataset.grids.lp)[sersic_source]
    adapt_data = adapt_data + np.max(adapt_data) * ADAPT_IMAGE_FLOOR

    image_plane_mesh_grid = al.image_mesh.Hilbert(
        pixels=HILBERT_PIXELS,
        weight_power=HILBERT_WEIGHT_POWER,
        weight_floor=HILBERT_WEIGHT_FLOOR,
    ).image_plane_mesh_grid_from(mask=dataset.mask, adapt_data=adapt_data)

    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=dataset.mask.mask_centre,
        radius=euclid_dataset.mask_radius + dataset.mask.pixel_scale / 2.0,
        n_points=EDGE_PIXELS_TOTAL,
    )

    adapt_images = al.AdaptImages(
        galaxy_name_image_dict={SOURCE_PATH: adapt_data},
        galaxy_name_image_plane_mesh_grid_dict={SOURCE_PATH: image_plane_mesh_grid},
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model.from_instance(lens),
            source=af.Model(
                al.Galaxy,
                redshift=sersic_source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=al.mesh.Delaunay(
                        pixels=image_plane_mesh_grid.shape[0],
                        zeroed_pixels=EDGE_PIXELS_TOTAL,
                    ),
                    regularization=al.reg.AdaptSplit(),
                ),
            ),
        )
    )

    assert model.prior_count == 0, (
        "the pixelized model must have zero free parameters, "
        f"got {model.prior_count}"
    )

    return model, adapt_images
