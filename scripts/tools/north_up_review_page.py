"""
Review page for a north-up re-render (sky orientation phase 2d, #115).

For each lens rendered by ``render_north_up.py`` it writes one JPEG with a row per release
figure: the current release PNG (fit frame, north down) on the left and the north-up re-render
on the right. ``index.html`` lists the lenses with their selection reason and witness summary.
The page is small enough to copy off the cluster for a by-eye review.

Usage:

    python scripts/tools/north_up_review_page.py \
        --render_dir=inspect/north_up_pilot_20261105 \
        --current_dir=inspect/dr1_catalogue_v5 \
        --out_dir=inspect/north_up_pilot_20261105/review
"""

import argparse
import csv
import html
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

PRODUCTS = (
    "vis_lp_fit.png",
    "vis_pix_fit.png",
    "fit_sersic.png",
    "vis_lp_image_with_positions.png",
    "rgb.png",
    "segmentation.png",
    "fit_multi_wavelength.png",
)
COLUMN_WIDTH = 1100
LABEL_HEIGHT = 34


def scaled(path: Path):
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    return image.resize((COLUMN_WIDTH, round(image.height * COLUMN_WIDTH / image.width)))


def sheet_for(tile: str, render_dir: Path, current_dir: Path, target: Path):
    rows = []
    for product in PRODUCTS:
        old, new = scaled(current_dir / tile / product), scaled(render_dir / tile / product)
        if old is None and new is None:
            continue
        height = max(im.height for im in (old, new) if im is not None)
        row = Image.new("RGB", (2 * COLUMN_WIDTH, height + LABEL_HEIGHT), "white")
        draw = ImageDraw.Draw(row)
        for column, (image, label) in enumerate(
            ((old, f"{product}: current release (fit frame, north down)"), (new, f"{product}: north up, east left"))
        ):
            draw.text((column * COLUMN_WIDTH + 8, 8), label if image is not None else f"{label}: MISSING", fill="black")
            if image is not None:
                row.paste(image, (column * COLUMN_WIDTH, LABEL_HEIGHT))
        rows.append(row)
    if not rows:
        return False
    sheet = Image.new("RGB", (2 * COLUMN_WIDTH, sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(target, quality=80)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--current_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    render_dir, current_dir, out_dir = Path(args.render_dir), Path(args.current_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reasons = {}
    selection = render_dir / "selection.csv"
    if selection.exists():
        reasons = {r["tile_id"]: r["reason"] for r in csv.DictReader(open(selection))}
    witness = {}
    for path in sorted(render_dir.glob("witness*.csv")):
        for row in csv.DictReader(open(path)):
            witness.setdefault(row["tile"], Counter())[row["status"]] += 1

    tiles = sorted(p.name for p in render_dir.iterdir() if p.is_dir() and p.name != out_dir.name)
    items = []
    for tile in tiles:
        name = f"{tile}.jpg"
        if sheet_for(tile, render_dir, current_dir, out_dir / name):
            summary = ", ".join(f"{n} {s}" for s, n in sorted(witness.get(tile, {}).items()))
            items.append(
                f'<li><a href="{html.escape(name)}">{html.escape(tile)}</a> '
                f"&mdash; {html.escape(reasons.get(tile, ''))} &mdash; witness: {html.escape(summary)}</li>"
            )
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>North-up pilot review</title>"
        "<style>body{font-family:sans-serif;margin:16px}li{margin:4px 0}</style>"
        f"<h1>North-up pilot review ({len(items)} lenses)</h1>"
        "<p>Each sheet: current release figure (left) and the north-up re-render (right).</p>"
        f"<ol>{''.join(items)}</ol>\n"
    )
    print(f"review page: {len(items)} lenses -> {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
