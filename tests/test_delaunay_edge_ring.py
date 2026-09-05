"""
The ``vis_pix`` Delaunay edge ring is really zeroed.

``scripts/initial_lens_model.py`` and ``scripts/full_model.py`` append a ring of
``edge_pixels_total`` points just outside the mask edge and hold their reconstructed
values at zero through ``zeroed_pixels`` on the ``Delaunay`` mesh. Until 2026-09 the
scripts passed the *appended* grid length as ``pixels`` while the mesh added
``zeroed_pixels`` on top, so ``mesh.pixels`` overstated the parameter count by the
ring size (PyAutoArray#526). This module mirrors the scripts' construction on a
small grid and asserts the zeroed indices are exactly the appended ring, inside the
grid, and that the counts the scripts pass add up to the grid they build.

Deliberately **JAX-free** and fit-free: it exercises the mesh-grid construction and
the mesh's index arithmetic only.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import autolens as al  # noqa: E402

EDGE_PIXELS_TOTAL = 30
MASK_RADIUS = 1.5
PIXEL_SCALE = 0.1


def _appended_grid():
    """
    The ``full_model.py`` SOURCE PIX 1 construction: a uniform ``Overlay`` grid clipped
    to a circular mask, then ``append_with_circle_edge_points`` at ``mask_radius`` plus
    half a pixel. Returns the interior grid and the appended grid.
    """
    mask = al.Mask2D.circular(
        shape_native=(40, 40), pixel_scales=PIXEL_SCALE, radius=MASK_RADIUS
    )

    image_plane_mesh_grid = al.image_mesh.Overlay(shape=(10, 10)).image_plane_mesh_grid_from(
        mask=mask
    )

    appended = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=MASK_RADIUS + mask.pixel_scale / 2.0,
        n_points=EDGE_PIXELS_TOTAL,
    )

    return image_plane_mesh_grid, appended


def test__appended_ring__is_the_last_edge_pixels_total_points_of_the_grid():
    interior, appended = _appended_grid()

    assert appended.shape[0] == interior.shape[0] + EDGE_PIXELS_TOTAL
    assert np.allclose(np.asarray(appended)[: interior.shape[0]], np.asarray(interior))

    ring = np.asarray(appended)[interior.shape[0] :]
    radii = np.hypot(ring[:, 0], ring[:, 1])

    assert np.allclose(radii, MASK_RADIUS + PIXEL_SCALE / 2.0)


def test__mesh_built_as_the_scripts_build_it__zeroes_exactly_the_ring():
    """
    ``pixels`` is the interior count (``hilbert_pixels`` in ``initial_lens_model.py``,
    ``image_plane_mesh_grid.shape[0] - edge_pixels_total`` in ``full_model.py``). The
    zeroed indices the inversion will use are the mesh's ``zeroed_pixels_from`` at the
    appended grid's length: the ring, and nothing else.
    """
    interior, appended = _appended_grid()

    mesh = al.mesh.Delaunay(
        pixels=appended.shape[0] - EDGE_PIXELS_TOTAL, zeroed_pixels=EDGE_PIXELS_TOTAL
    )

    assert mesh.pixels == interior.shape[0]
    assert mesh.zeroed_pixels == EDGE_PIXELS_TOTAL
    assert mesh.total_pixels == appended.shape[0]

    zeroed = mesh.zeroed_pixels_from(pixels=appended.shape[0])

    assert zeroed.shape == (EDGE_PIXELS_TOTAL,)
    assert zeroed.min() == interior.shape[0]
    assert zeroed.max() == appended.shape[0] - 1
    assert (zeroed == np.arange(interior.shape[0], appended.shape[0])).all()


def test__ring_is_zeroed_whatever_pixels_the_mesh_was_told():
    """
    The regression: the pre-fix form passed the appended length as ``pixels``. The
    ring is resolved against the grid the mapper receives, so both forms zero the same
    vertices — the archived ``vis_pix`` results (one mapper) were not affected.
    """
    interior, appended = _appended_grid()

    expected = np.arange(interior.shape[0], appended.shape[0])

    for pixels in (interior.shape[0], appended.shape[0]):
        mesh = al.mesh.Delaunay(pixels=pixels, zeroed_pixels=EDGE_PIXELS_TOTAL)

        assert (mesh.zeroed_pixels_from(pixels=appended.shape[0]) == expected).all()
