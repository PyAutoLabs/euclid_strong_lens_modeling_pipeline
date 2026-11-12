"""
Build a north-up copy of a built catalogue tree (sky orientation phase 2d, #115).

Every file of ``--catalogue_dir`` is hard-linked into ``--out_dir``, so the copy costs no space,
and the source tree is left untouched. The exception is the release PNGs that
``render_north_up.py`` redraws: when ``--render_dir/<tile>/<png>`` exists, it replaces that PNG
in the copy. When it does not exist (a stored result that would not load, a lens with no SED
fit), the original fit-frame PNG is kept, as the human chose on 2026-11-05.

``<out_dir>_north_up_report.csv`` (a sibling, because the bundle and packaging stages treat
every sub-directory of the tree as a lens) records, for every lens and release PNG, whether it
is ``north_up``, ``fit_frame`` (original kept) or ``absent``. The packager then runs on
``--out_dir`` as on the original tree.

Usage:

    python scripts/tools/north_up_catalogue_tree.py \
        --catalogue_dir=inspect/dr1_catalogue_v5 \
        --render_dir=inspect/north_up_full_20261105 \
        --out_dir=inspect/dr1_catalogue_v5_northup
"""

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path

NORTH_UP_PNGS = (
    "vis_lp_fit.png",
    "vis_pix_fit.png",
    "fit_sersic.png",
    "vis_lp_image_with_positions.png",
    "rgb.png",
    "segmentation.png",
    "fit_multi_wavelength.png",
)


def link_tree(source: Path, target: Path):
    """Hard-link every file under ``source`` into ``target``, recreating the directories."""
    for root, _dirs, files in os.walk(source):
        out = target / Path(root).relative_to(source)
        out.mkdir(parents=True, exist_ok=True)
        for name in files:
            os.link(Path(root) / name, out / name)


def build(catalogue_dir: Path, render_dir: Path, out_dir: Path):
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: refusing to overwrite")
    report_path = out_dir.with_name(f"{out_dir.name}_north_up_report.csv")
    link_tree(catalogue_dir, out_dir)

    counts = Counter()
    with open(report_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tile", "product", "source"])
        for tile_dir in sorted(
            p for p in out_dir.iterdir() if p.is_dir() and not p.name.startswith("_")
        ):
            for product in NORTH_UP_PNGS:
                target = tile_dir / product
                north_up = render_dir / tile_dir.name / product
                if north_up.exists():
                    # Replace the link, never write through it: the source tree must not change.
                    if target.exists():
                        target.unlink()
                    shutil.copy2(north_up, target)
                    source = "north_up"
                elif target.exists():
                    source = "fit_frame"
                else:
                    source = "absent"
                writer.writerow([tile_dir.name, product, source])
                counts[(product, source)] += 1

    for (product, source), n in sorted(counts.items()):
        print(f"{product:34s} {source:10s} {n}")
    print(f"report -> {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--catalogue_dir", required=True)
    parser.add_argument("--render_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    build(Path(args.catalogue_dir), Path(args.render_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
