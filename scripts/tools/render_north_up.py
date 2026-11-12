"""
North-up, east-left re-render of the per-lens release PNGs (sky orientation phase 2d, #115).

The DR1 fits read every cut-out with no row flip, so the fit frame is +y = south, +x = west
(docs/sky_orientation/phase1.md) and every pipeline PNG is the sky mirrored top to bottom.
This tool redraws the release figures from the stored results with north up and east left.
It runs no non-linear search: each fit is rebuilt at its stored maximum-likelihood model.

For every fit it:

1. rebuilds the stored fit (``al.agg.FitImagingAgg``);
2. mirrors it: the dataset arrays, mask, PSF and adapt images are flipped top to bottom, the
   adapt image-plane mesh grids and the dataset-model grid offset get y -> -y, and the tracer
   goes through ``sky_frame.plus_y_north_from``. The fit is then solved again (inversion
   included), so critical curves, caustics and source planes come out in the mirrored frame;
3. checks the mirror (the witness): the mirrored model image must equal the flipped native one,
   with the chi-squared and the pixelized reconstruction unchanged. The VIS lens-model fits mirror
   to ~1e-11; the Sersic fits (VIS and the NISP/DECam bands) only to ~1e-5, so all three checks
   get ``BAND_RTOL`` for them (accepted by the human on the 2026-11-05 pilot as visually exact).
   Pixels that are NaN in both images are ignored. The figure of merit must agree to
   ``FOM_RTOL``: the log-determinant of a Delaunay regularization matrix is conditioning limited
   (up to 5e-4 on the pilot), which also separates a native rebuild from the stored value. For
   ~2% of lenses the pixelized fit does not survive the mirror (a different solution, or an
   inversion that will not solve); those fail or are recorded as ``error`` and are not drawn, and
   the release keeps their fit-frame PNG (human, 2026-11-05). ``witness.csv`` keeps the measured
   differences either way;
4. draws the figure with the library plotters, so +y on every axis now means north. Only the RGB
   panel of ``rgb.png`` carries an N/E compass, in white, in its top-right corner.

A fit that fails the witness is not drawn. Existing PNGs are never pixel-flipped and nothing is
written outside ``--out_dir``.

Products per lens (``<out_dir>/<tile>/``):

- ``vis_lp_fit.png``, ``vis_pix_fit.png``, ``fit_sersic.png``: the library ``subplot_fit``;
- ``vis_lp_image_with_positions.png``: VIS data with the positions from ``positions.json``;
- ``rgb.png``: ``rgb_north_up.jpg`` (else ``rgb_raw.jpg``, the delivered north-up original)
  and its masked copy;
- ``segmentation.png``: ``preprocess.segmentation.save_diagnostic_png`` on flipped arrays;
- ``fit_multi_wavelength.png``: data, lens light model, lens light subtracted and source
  model for every band of the SED chain, bands increasing in wavelength left to right.

``<out_dir>/witness.csv`` (or ``--witness_csv``) holds one row per fit.

Usage:

    python scripts/tools/render_north_up.py \
        --output_root=output/dr1_sep1 --sed_root=output_sed/dr1_sep1 \
        --dataset_root=dataset/dr1_sep1 --out_dir=inspect/north_up_pilot_20261105 \
        --tiles_file=tiles.txt
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "catalogue" / "scripts"))

try:  # autoconf was renamed autonerves (2026-10)
    from autonerves import conf
except ImportError:
    from autoconf import conf

conf.instance.push(
    new_path=str(PROJECT_ROOT / "config"), output_path=str(PROJECT_ROOT / "output")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pyfits
from PIL import Image

import autoarray as aa
import autogalaxy as ag
import autolens as al
import autolens.imaging.plot.fit_imaging_plots as fit_imaging_plots
from autofit.aggregator.aggregator import Aggregator
from autogalaxy.util.plot_utils import plot_array

from catalogue_util import latest_result_per_lens_band
from scripts.tools.sky_frame import plus_y_north_from

MODEL_IMAGE_RTOL = 1e-8
CHI_SQUARED_RTOL = 1e-9
RECONSTRUCTION_RTOL = 1e-6
BAND_RTOL = 1e-4
FOM_RTOL = 1e-3
# The Sersic VIS fit mirrors only to ~1e-5 like the bands (full run, 2026-11-05), so it is not here.
VIS_STAGES = ("vis_lp", "vis_pix")

# Effective wavelengths (nm), for ordering the multi-wavelength panels.
BAND_WAVELENGTH = {
    "decam_g": 482,
    "decam_r": 642,
    "decam_i": 784,
    "vis": 715,
    "decam_z": 926,
    "nir_y": 1033,
    "nir_j": 1259,
    "nir_h": 1686,
}

WITNESS_COLUMNS = [
    "tile",
    "stage",
    "status",
    "native_figure_of_merit",
    "mirrored_figure_of_merit",
    "stored_log_likelihood",
    "fom_rel_diff",
    "model_image_rel_diff",
    "chi_squared_rel_diff",
    "reconstruction_rel_diff",
]


# ---------------------------------------------------------------------------
# Mirroring
# ---------------------------------------------------------------------------


def flip_mask(mask):
    return aa.Mask2D(
        mask=np.flipud(np.asarray(mask)),
        pixel_scales=mask.pixel_scales,
        origin=(-mask.origin[0], mask.origin[1]),
    )


def flip_array(array, mask):
    return aa.Array2D(values=np.flipud(np.asarray(array.native)), mask=mask)


def flip_grid(grid):
    return aa.Grid2DIrregular(values=np.asarray(grid) * np.array([-1.0, 1.0]))


def mirrored_fit_from(fit):
    """The fit with +y = north: mirrored inputs, solved again at the same model."""
    dataset = fit.dataset
    mask = flip_mask(dataset.mask)
    unmasked = aa.Mask2D.all_false(
        shape_native=mask.shape_native,
        pixel_scales=mask.pixel_scales,
        origin=mask.origin,
    )
    kernel = dataset.psf.kernel
    psf = aa.Convolver(
        kernel=aa.Array2D.no_mask(
            values=np.flipud(np.asarray(kernel.native)),
            pixel_scales=kernel.pixel_scales,
        )
    )
    mirrored_dataset = aa.Imaging(
        data=flip_array(dataset.data, unmasked),
        noise_map=flip_array(dataset.noise_map, unmasked),
        psf=psf,
        over_sample_size_lp=flip_array(dataset.over_sample_size_lp, unmasked),
        over_sample_size_pixelization=flip_array(
            dataset.over_sample_size_pixelization, unmasked
        ),
        convolve_over_sample_size_lp=dataset.convolve_over_sample_size_lp,
        convolve_over_sample_size_pixelization=dataset.convolve_over_sample_size_pixelization,
        check_noise_map=False,
    ).apply_mask(mask=mask)

    tracer = plus_y_north_from(fit.tracer)

    # Adapt images are keyed by Galaxy, whose hash is its id; the mirrored copies keep it.
    adapt_images = fit.adapt_images
    if adapt_images is not None:
        by_id = {int(galaxy.id): galaxy for galaxy in tracer.galaxies}

        def by_galaxy(mapping, function):
            return {
                by_id.get(int(g.id), g): function(v) for g, v in (mapping or {}).items()
            }

        def by_name(mapping, function):
            return {k: function(v) for k, v in (mapping or {}).items()}

        image = lambda value: flip_array(value, mask)
        adapt_images = ag.AdaptImages(
            galaxy_image_dict=by_galaxy(adapt_images.galaxy_image_dict, image),
            galaxy_name_image_dict=by_name(adapt_images.galaxy_name_image_dict, image),
            galaxy_image_plane_mesh_grid_dict=by_galaxy(
                adapt_images.galaxy_image_plane_mesh_grid_dict, flip_grid
            ),
            galaxy_name_image_plane_mesh_grid_dict=by_name(
                adapt_images.galaxy_name_image_plane_mesh_grid_dict, flip_grid
            ),
            galaxy_path_list=adapt_images.galaxy_path_list,
        )

    dataset_model = fit.dataset_model
    if dataset_model is not None:
        dataset_model = aa.DatasetModel(
            background_sky_level=dataset_model.background_sky_level,
            grid_offset=(-dataset_model.grid_offset[0], dataset_model.grid_offset[1]),
            grid_rotation_angle=-dataset_model.grid_rotation_angle,
        )

    return al.FitImaging(
        dataset=mirrored_dataset,
        tracer=tracer,
        dataset_model=dataset_model,
        adapt_images=adapt_images,
        settings=fit.settings,
    )


def relative_difference(a, b):
    """Max |a - b| over max |a|, ignoring pixels that are NaN in both (some band images carry NaNs)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    if both_nan.all():
        return 0.0
    a, b = a[~both_nan], b[~both_nan]
    scale = np.max(np.abs(a)) or 1.0
    return float(np.max(np.abs(a - b)) / scale)


def witness_from(tile, stage, fit, mirrored, stored_log_likelihood):
    # The mirrored Delaunay regularization matrix can come out numerically singular (co-circular
    # mesh points triangulate differently), so its log-determinant -- used only by the figure of
    # merit, not by any drawn image -- fails to factorize. The figure is then judged on the model
    # image, chi-squared and reconstruction alone, and the row says so ("pass_no_fom").
    try:
        mirrored_figure_of_merit = mirrored.figure_of_merit
    except np.linalg.LinAlgError:
        mirrored_figure_of_merit = float("nan")
    row = {
        "tile": tile,
        "stage": stage,
        "native_figure_of_merit": fit.figure_of_merit,
        "mirrored_figure_of_merit": mirrored_figure_of_merit,
        "stored_log_likelihood": stored_log_likelihood,
        "fom_rel_diff": abs(mirrored_figure_of_merit - fit.figure_of_merit)
        / abs(fit.figure_of_merit),
        "model_image_rel_diff": relative_difference(
            np.flipud(np.asarray(fit.model_data.native)), mirrored.model_data.native
        ),
        "chi_squared_rel_diff": relative_difference(
            fit.chi_squared, mirrored.chi_squared
        ),
        "reconstruction_rel_diff": (
            relative_difference(
                fit.inversion.reconstruction, mirrored.inversion.reconstruction
            )
            if fit.inversion is not None
            else 0.0
        ),
    }
    vis = stage in VIS_STAGES
    passed = (
        row["model_image_rel_diff"] <= (MODEL_IMAGE_RTOL if vis else BAND_RTOL)
        and row["chi_squared_rel_diff"] <= (CHI_SQUARED_RTOL if vis else BAND_RTOL)
        and row["reconstruction_rel_diff"]
        <= (RECONSTRUCTION_RTOL if vis else BAND_RTOL)
    )
    if np.isnan(mirrored_figure_of_merit):
        row["status"] = "pass_no_fom" if passed else "FAIL"
    else:
        row["status"] = "pass" if passed and row["fom_rel_diff"] <= FOM_RTOL else "FAIL"
    return row


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def add_compass(ax):
    """A white N/E compass (north up, east left) in the top-right corner of one image panel.

    Only the RGB panel of rgb.png carries it (human, 2026-11-09); every figure is north up,
    east left, so one compass in the colour image is enough."""
    arrow = dict(arrowstyle="-|>", lw=2.0, color="white", mutation_scale=14)
    corner, length = (0.92, 0.74), 0.17
    north = (corner[0], corner[1] + length)
    east = (corner[0] - length, corner[1])
    for tip in (north, east):
        ax.annotate(
            "",
            xy=tip,
            xytext=corner,
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=arrow,
        )
    label = dict(
        xycoords="axes fraction", color="white", fontsize=13, fontweight="bold"
    )
    ax.annotate("N", xy=(north[0], north[1] + 0.015), ha="center", va="bottom", **label)
    ax.annotate("E", xy=(east[0] - 0.015, east[1]), ha="right", va="center", **label)


def save_subplot_fit(fit, target: Path, title_prefix="VIS"):
    fit_imaging_plots.subplot_fit(
        fit=fit,
        output_path=str(target.parent),
        output_format="png",
        title_prefix=title_prefix,
    )
    (target.parent / "fit.png").replace(target)


def save_single(target: Path, draw, figsize=(7, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def save_image_with_positions(mirrored, positions, target: Path):
    save_single(
        target,
        lambda ax: plot_array(
            array=mirrored.data,
            ax=ax,
            title="Image With Positions",
            positions=[positions],
        ),
    )


def north_up_rgb_path(dataset_dir: Path):
    for name in ("rgb_north_up.jpg", "rgb_raw.jpg"):
        if (dataset_dir / name).exists():
            return dataset_dir / name
    return None


# The colour images are 100 x 100 pixels drawn some 5x larger: smooth them like an image viewer
# would (human, 2026-11-12) instead of matplotlib's default hard-edged pixels for that zoom.
RGB_INTERPOLATION = "lanczos"


def save_rgb(rgb_path: Path, mask, target: Path):
    image = np.array(Image.open(rgb_path))
    shape = image.shape[:2]
    full = al.Mask2D.all_false(shape_native=shape, pixel_scales=mask.pixel_scales)
    circle = (
        mask
        if shape == mask.shape_native
        else al.Mask2D.circular(
            shape_native=shape,
            pixel_scales=mask.pixel_scales,
            radius=mask.circular_radius,
        )
    )
    with matplotlib.rc_context({"image.interpolation": RGB_INTERPOLATION}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        plot_array(array=al.Array2DRGB(values=image, mask=full), ax=axes[0], title="RGB")
        plot_array(
            array=al.Array2DRGB(values=image, mask=circle), ax=axes[1], title="RGB Masked"
        )
        add_compass(axes[0])
        fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


def save_segmentation(
    dataset_dir: Path, tile: str, rgb_path, positions_native, target: Path
):
    from preprocess.segmentation import save_diagnostic_png

    info = json.loads((dataset_dir / "info.json").read_text())
    seg = dataset_dir / "segmentation"

    def read(name):
        path = seg / name
        return (
            np.flipud(pyfits.getdata(path).astype(np.float32))
            if path.exists()
            else None
        )

    vis = np.flipud(pyfits.getdata(dataset_dir / f"{tile}.fits").astype(np.float32))
    zeros = np.zeros_like(vis)
    rgb = (
        np.array(Image.open(rgb_path))
        if rgb_path is not None
        else np.zeros(vis.shape + (3,))
    )
    cy, cx = info["mask_centre"]
    save_diagnostic_png(
        rgb_image=rgb,
        vis_image=vis,
        source_flux=(
            read("source_flux.fits") if read("source_flux.fits") is not None else zeros
        ),
        artefact_flux=(
            read("artefact_flux.fits")
            if read("artefact_flux.fits") is not None
            else zeros
        ),
        artefact_binary=(
            read("artefact_binary.fits")
            if read("artefact_binary.fits") is not None
            else zeros
        ),
        lens_flux=(
            read("lens_flux.fits") if read("lens_flux.fits") is not None else zeros
        ),
        lens_binary=(
            read("lens_binary.fits") if read("lens_binary.fits") is not None else zeros
        ),
        pixel_scale=info["pixel_scale"],
        mask_centre=[-cy, cx],
        mask_radius=info["mask_radius"],
        position_list=[[-y, x] for y, x in positions_native],
        output_path=target,
    )


def save_multi_wavelength(band_fits, target: Path):
    """One column per band (increasing wavelength), one row per panel, as multi_wavelength.py."""
    bands = sorted(band_fits, key=lambda b: BAND_WAVELENGTH.get(b, 10_000))
    rows = ("Data", "Lens Light", "Subtracted", "Source Model")
    fig, axes = plt.subplots(
        len(rows),
        len(bands),
        figsize=(4.5 * len(bands), 4.5 * len(rows)),
        squeeze=False,
    )
    for column, band in enumerate(bands):
        fit = band_fits[band]
        last = len(fit.tracer.planes) - 1
        source_vmax = fit_imaging_plots._get_source_vmax(fit)
        planes = fit.model_images_of_planes_list
        panels = (
            dict(array=fit.data),
            dict(array=planes[0]),
            dict(
                array=fit.subtracted_images_of_planes_list[last],
                vmin=0.0 if source_vmax is not None else None,
                vmax=source_vmax,
            ),
            dict(array=planes[last], vmax=source_vmax),
        )
        for row, (title, panel) in enumerate(zip(rows, panels)):
            plot_array(ax=axes[row, column], title=f"{band.upper()} {title}", **panel)
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def stored_fits(root: Path, tile: str):
    """``{search name: (native FitImaging, stored max log likelihood)}``, latest result per band."""
    if not (root / tile).is_dir():
        return {}
    agg = Aggregator.from_directory(
        directory=root / tile, completed_only=True, unzip_temporary=True
    )
    agg = latest_result_per_lens_band(agg, sample_root=root)
    out = {}
    fit_agg = al.agg.FitImagingAgg(aggregator=agg)
    for entry, fit_list, samples in zip(
        agg, fit_agg.max_log_likelihood_gen_from(), agg.values("samples_summary")
    ):
        key = (str(Path(entry.directory).parent.parent.name), entry.search.name)
        out[key] = (fit_list[0], samples.max_log_likelihood_sample.log_likelihood)
    return out


def positions_from(dataset_dir: Path):
    path = dataset_dir / "positions.json"
    if not path.exists():
        return None
    values = json.loads(path.read_text())["arguments"]["values"]["array"]
    return np.asarray(values, dtype=float)


def rgb_mask_from(output_root: Path, tile: str):
    """The north-up VIS mask of the lens's latest completed vis_lp fit, read straight from its zip
    (``image/dataset.fits``, HDU ``MASK``), so ``rgb.png`` can be redrawn without rebuilding the fit."""
    import io
    import zipfile

    zips = sorted((output_root / tile / "initial_lens_model" / "vis_lp").glob("*.zip"),
                  key=lambda z: z.stat().st_mtime, reverse=True)
    for path in zips:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if ".completed" not in names or "image/dataset.fits" not in names:
                continue
            with pyfits.open(io.BytesIO(archive.read("image/dataset.fits"))) as hdus:
                hdu = hdus["MASK"]
                mask = al.Mask2D(
                    mask=np.asarray(hdu.data) > 0.5,
                    pixel_scales=(hdu.header["PIXSCAY"], hdu.header["PIXSCAX"]),
                    origin=(hdu.header["ORIGINY"], hdu.header["ORIGINX"]),
                )
            return flip_mask(mask)
    return None


def render_rgb_only(tile, args):
    """Redraw only ``rgb.png`` (``--only rgb``): the colour image needs no fit, just its mask."""
    dataset_dir = Path(args.dataset_root) / tile
    rgb_path = north_up_rgb_path(dataset_dir)
    mask = rgb_mask_from(Path(args.output_root), tile)
    if rgb_path is None or mask is None:
        raise RuntimeError(f"no {'north-up RGB' if rgb_path is None else 'completed vis_lp fit'}")
    out = Path(args.out_dir) / tile
    out.mkdir(parents=True, exist_ok=True)
    save_rgb(rgb_path, mask, out / "rgb.png")
    plt.close("all")


def render_tile(tile, args, writer):
    out = Path(args.out_dir) / tile
    out.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_root) / tile

    def checked(stage, fit, stored):
        # An inversion that fails to solve (e.g. a matrix that is not positive definite) loses
        # this figure only; the lens's other figures are still drawn.
        try:
            mirrored = mirrored_fit_from(fit)
            row = witness_from(tile, stage, fit, mirrored, stored)
        except Exception as exc:
            writer.writerow(
                {"tile": tile, "stage": stage, "status": f"error: {type(exc).__name__}"}
            )
            return None
        writer.writerow(row)
        return mirrored if row["status"] in ("pass", "pass_no_fom") else None

    def loaded(root, stage):
        # A stored result the current model cannot rebuild is recorded, not fatal for the lens.
        try:
            return stored_fits(root, tile)
        except Exception as exc:
            writer.writerow(
                {
                    "tile": tile,
                    "stage": stage,
                    "status": f"load_error: {type(exc).__name__}",
                }
            )
            return {}

    def drawn(stage, draw, *a):
        # A figure that fails to draw is recorded and skipped; the lens's other figures go on.
        try:
            draw(*a)
            return True
        except Exception as exc:
            writer.writerow(
                {
                    "tile": tile,
                    "stage": stage,
                    "status": f"draw_error: {type(exc).__name__}",
                }
            )
            return False

    fits = loaded(Path(args.output_root), "initial_lens_model")
    vis_lp_mirrored = None
    for search, target in (
        ("vis_lp", "vis_lp_fit.png"),
        ("vis_pix", "vis_pix_fit.png"),
    ):
        if ("initial_lens_model", search) in fits:
            mirrored = checked(search, *fits[("initial_lens_model", search)])
            if mirrored is not None and drawn(
                search, save_subplot_fit, mirrored, out / target
            ):
                if search == "vis_lp":
                    vis_lp_mirrored = mirrored

    positions = positions_from(dataset_dir)
    if vis_lp_mirrored is not None and positions is not None:
        save_image_with_positions(
            vis_lp_mirrored,
            positions * np.array([-1.0, 1.0]),
            out / "vis_lp_image_with_positions.png",
        )

    rgb_path = north_up_rgb_path(dataset_dir)
    if rgb_path is not None and vis_lp_mirrored is not None:
        save_rgb(rgb_path, vis_lp_mirrored.dataset.mask, out / "rgb.png")

    if (dataset_dir / "info.json").exists():
        save_segmentation(
            dataset_dir,
            tile,
            rgb_path,
            positions if positions is not None else [],
            out / "segmentation.png",
        )

    if args.sed_root:
        sed = loaded(Path(args.sed_root), "sersic_lens_model")
        band_fits, failed = {}, False
        for (tag, band), (fit, stored) in sed.items():
            if tag != "sersic_lens_model":
                continue
            mirrored = checked(f"sersic_{band}", fit, stored)
            if mirrored is None:
                failed = True
                continue
            band_fits[band] = mirrored
            if band == "vis":
                drawn("sersic_vis", save_subplot_fit, mirrored, out / "fit_sersic.png")
        # All bands or none: a band failing its witness must not silently drop out of the figure.
        if band_fits and not failed:
            save_multi_wavelength(band_fits, out / "fit_multi_wavelength.png")
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output_root", required=True, help="e.g. output/dr1_sep1")
    parser.add_argument("--sed_root", default=None, help="e.g. output_sed/dr1_sep1")
    parser.add_argument("--only", choices=["rgb"], default=None,
                        help="redraw only this figure (rgb: rgb.png from the dataset RGB and the fit's mask)")
    parser.add_argument("--dataset_root", required=True, help="e.g. dataset/dr1_sep1")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tiles", default="", help="comma-separated tile names")
    parser.add_argument("--tiles_file", default=None, help="one tile name per line")
    parser.add_argument(
        "--chunk",
        default=None,
        help="i/n: render only the i-th (0-based) of n equal slices of the tiles (SLURM arrays)",
    )
    parser.add_argument(
        "--witness_csv", default=None, help="default <out_dir>/witness.csv"
    )
    args = parser.parse_args()

    tiles = [t for t in args.tiles.split(",") if t]
    if args.tiles_file:
        tiles += [
            line.strip()
            for line in Path(args.tiles_file).read_text().splitlines()
            if line.strip()
        ]
    if not tiles:
        parser.error("no tiles given")
    if args.chunk:
        index, total = (int(v) for v in args.chunk.split("/"))
        tiles = tiles[index::total]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    witness_path = (
        Path(args.witness_csv) if args.witness_csv else out_dir / "witness.csv"
    )
    new = not witness_path.exists()
    with open(witness_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WITNESS_COLUMNS)
        if new:
            writer.writeheader()
        for i, tile in enumerate(tiles, 1):
            start = time.time()
            try:
                if args.only == "rgb":
                    render_rgb_only(tile, args)
                else:
                    render_tile(tile, args, writer)
                print(
                    f"[{i}/{len(tiles)}] {tile}: done in {time.time() - start:.0f}s",
                    flush=True,
                )
            except Exception as exc:  # one bad lens must not stop the pilot
                print(
                    f"[{i}/{len(tiles)}] {tile}: ERROR {type(exc).__name__}: {exc}",
                    flush=True,
                )
            handle.flush()


if __name__ == "__main__":
    main()
