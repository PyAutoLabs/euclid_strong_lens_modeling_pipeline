"""
Binary Map Editor
=================

Simple GUI tool for tuning a binary mask for a given
object type (e.g. artefact, source, lens) for each lens in the dataset.

Set the object type via the OBJECT variable at the top of the script.
The tool will read ``segmentation/{OBJECT}_flux.fits`` and write
``segmentation/{OBJECT}_binary.fits``.

Displays two panels side by side:
  Left:  {OBJECT}_flux.fits the raw flux map
  Right: Image with the binary mask overlaid in colour

Controls:
  σ (sigma) text box      -- threshold for binarisation; pixels above
                             median + sigma * std are marked as artefact.
                             Set to 0 to use flux > 0 as threshold.
  d (dilation) text box   -- number of dilation iterations to expand
                             the binary mask (0 = no dilation)
  Image ID text box       -- jump directly to a lens by its leading number
  Left / Right arrow keys -- previous / next image

The resulting binary mask is saved automatically to
``segmentation/{OBJECT}_binary.fits`` on every update.

Run from the project root::

    python preprocess/adjust_binary.py --sample=dr1_top_500
    python preprocess/adjust_binary.py --sample=dr1_top_500 --object=source
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import project_paths  # noqa: E402
from astropy.io import fits
from scipy.ndimage import binary_dilation
from matplotlib.widgets import TextBox

OBJECT = "artefact"


def sort_key(p):
    return int(p.name.split("_")[0])


lens_dirs = []
idx = 0
dilation_size = 1
sigma = 0.0


def load_data(lens_dir):
    flux = fits.getdata(lens_dir / "segmentation" / f"{OBJECT}_flux.fits")

    png_path = lens_dir / f"{lens_dir.name}.png"

    if not png_path.exists():
        raise FileNotFoundError(f"Missing expected PNG: {png_path}")

    rgb = plt.imread(png_path)
    return flux, rgb


def compute_binary(flux, sigma, dilation):
    if sigma > 0:
        thresh = np.nanmedian(flux) + sigma * np.nanstd(flux)
        binary = flux > thresh
    else:
        binary = flux > 0

    if dilation > 0:
        binary = binary_dilation(binary, iterations=dilation)

    return binary.astype(np.float32)


def save_binary(lens_dir, binary):
    path = lens_dir / "segmentation" / f"{OBJECT}_binary.fits"
    fits.writeto(path, binary, overwrite=True)


def update():
    ax1.clear()
    ax2.clear()

    lens_dir = lens_dirs[idx]
    flux, rgb = load_data(lens_dir)

    binary = compute_binary(flux, sigma, dilation_size)
    save_binary(lens_dir, binary)
    ax1.imshow(flux, cmap="viridis")
    ax1.set_title("Flux")
    ax1.axis("off")

    if rgb.ndim == 3:
        rgb = rgb.mean(axis=-1)

    ax2.imshow(rgb, cmap="gray")

    mask = np.ma.masked_where(binary == 0, flux)
    ax2.imshow(mask, cmap="viridis", alpha=0.6)

    ax2.set_title(f"{lens_dir.name} | sigma={sigma} | dilation={dilation_size}")
    ax2.axis("off")
    fig.suptitle(f"{idx + 1}/{len(lens_dirs)} | {lens_dir.name}", fontsize=10)
    fig.canvas.draw_idle()


def on_key(event):
    global idx
    if event.key == "right":
        idx = (idx + 1) % len(lens_dirs)
        update()
    elif event.key == "left":
        idx = (idx - 1) % len(lens_dirs)
        update()


def submit_sigma(text):
    global sigma
    try:
        sigma = float(text)
    except:
        return
    update()


def submit_dilation(text):
    global dilation_size
    try:
        dilation_size = int(text)
    except:
        return
    update()


def submit_id(text):
    global idx
    try:
        num = int(text.strip())
        for i, d in enumerate(lens_dirs):
            if int(d.name.split("_")[0]) == num:
                idx = i
                update()
                break
    except ValueError:
        pass

    id_box.set_val("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        metavar="name",
        required=True,
        help="Sample subdirectory inside dataset/",
    )
    parser.add_argument(
        "--object",
        metavar="name",
        default="artefact",
        help="Object type to binarise (default: artefact)",
    )
    args = parser.parse_args()

    OBJECT = args.object
    DATASET_ROOT = project_paths.dataset_root() / args.sample

    lens_dirs = sorted(
        [
            d
            for d in DATASET_ROOT.iterdir()
            if d.is_dir() and "_" in d.name and d.name.split("_")[0].isdigit()
        ],
        key=sort_key,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.82, bottom=0.18, wspace=0.05)

    id_ax = plt.axes([0.12, 0.92, 0.1, 0.04])
    id_box = TextBox(id_ax, "Image #", initial="")
    id_box.on_submit(submit_id)

    sigma_ax = plt.axes([0.25, 0.05, 0.2, 0.05])
    sigma_box = TextBox(sigma_ax, "σ", initial="0")

    dilation_ax = plt.axes([0.55, 0.05, 0.2, 0.05])
    dilation_box = TextBox(dilation_ax, "d", initial="1")

    sigma_box.on_submit(submit_sigma)
    dilation_box.on_submit(submit_dilation)

    fig.canvas.mpl_connect("key_press_event", on_key)

    update()
    plt.show()
