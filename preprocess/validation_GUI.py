"""
Annotation Tool
===============

Interactive matplotlib-based tool for manually annotating lens segmentation images.
Displays the segmentation.png for each lens and allows the user to assign a label.

Labels (keyboard shortcuts):
  1. OK                  -- segmentation looks good
  2. OK, no points       -- segmentation ok but no multiple-image positions
  3. Fix needed          -- segmentation has issues that need manual fixing
  4. Group / scale issue -- lens is part of a group or has a scale problem
  5. Recentering needed  -- lens needs recentering before analysis

Navigation:
  Left / Right arrow keys -- previous / next image
  Jump to # text box      -- jump directly to a lens by its leading number

Annotations are saved to a CSV file with columns ``name`` and ``label``.
Auto-next can be toggled to automatically advance after each annotation.

Run from the project root::

    python preprocess/validation_GUI.py --sample=dr1_top_500
    python preprocess/validation_GUI.py --sample=dr1_top_500 --csv=preprocess/annotations_dr1.csv
"""

import argparse
import os
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import project_paths  # noqa: E402
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from PIL import Image
import numpy as np


def sort_key(p):
    return int(p.name.split("_")[0])


CSV_PATH = None
lens_dirs = []
idx = 0
auto_next = [True]
annotations = {}


def save_annotation(name, label):
    annotations[name] = label
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "label"])
        for k, v in annotations.items():
            writer.writerow([k, v])


def load_image(path):
    img_path = path / "segmentation.png"
    img = Image.open(img_path).convert("RGB")
    return np.array(img)


def highlight(label):
    colors = {
        "1_ok": "#4CAF50",
        "2_ok_no_points": "#2196F3",
        "3_fix_needed": "#f44336",
        "4_group_scale": "#9C27B0",
        "5_recentering_needed": "#FF9800",
    }
    for l, btn in buttons.items():
        if l == label:
            btn.ax.set_facecolor(colors[l])
        else:
            btn.ax.set_facecolor("#dddddd")


def draw():
    ax.clear()
    lens_dir = lens_dirs[idx]
    img = load_image(lens_dir)
    ax.imshow(img)
    ax.axis("off")
    name = lens_dir.name
    label = annotations.get(name, None)
    if label:
        highlight(label)
    else:
        highlight(None)
    fig.suptitle(f"{idx + 1}/{len(lens_dirs)} | {name} | {label}")
    fig.canvas.draw_idle()


def next_img():
    global idx
    if idx < len(lens_dirs) - 1:
        idx += 1


def set_label(label):
    name = lens_dirs[idx].name
    save_annotation(name, label)
    if auto_next[0]:
        next_img()
    draw()


def on_key(event):
    global idx
    if event.key == "right":
        next_img()
        draw()
    elif event.key == "left":
        if idx > 0:
            idx -= 1
            draw()
    elif event.key == "1":
        set_label("1_ok")
    elif event.key == "2":
        set_label("2_ok_no_points")
    elif event.key == "3":
        set_label("3_fix_needed")
    elif event.key == "4":
        set_label("4_group_scale")
    elif event.key == "5":
        set_label("5_recentering_needed")


def on_jump(text):
    global idx
    try:
        num = int(text.strip())
        for i, d in enumerate(lens_dirs):
            if int(d.name.split("_")[0]) == num:
                idx = i
                draw()
                break
    except ValueError:
        pass
    jump_box.set_val("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        metavar="name",
        required=True,
        help="Sample subdirectory inside dataset/",
    )
    parser.add_argument(
        "--csv",
        metavar="path",
        default="preprocess/annotations.csv",
        help="Path to annotations CSV (default: preprocess/annotations.csv)",
    )
    args = parser.parse_args()

    ROOT_DIR = project_paths.dataset_root() / args.sample
    CSV_PATH = Path(args.csv)

    lens_dirs = sorted(
        [
            d
            for d in ROOT_DIR.iterdir()
            if d.is_dir() and "_" in d.name and d.name.split("_")[0].isdigit()
        ],
        key=sort_key,
    )

    if CSV_PATH.exists():
        with open(CSV_PATH, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) == 2:
                    annotations[row[0]] = row[1]

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.subplots_adjust(right=0.8, top=0.88)

    jump_ax = plt.axes([0.10, 0.92, 0.08, 0.05])
    jump_box = TextBox(jump_ax, "Jump to #", initial="")
    jump_box.on_submit(on_jump)

    btn1_ax = plt.axes([0.82, 0.72, 0.15, 0.06])
    btn2_ax = plt.axes([0.82, 0.62, 0.15, 0.06])
    btn3_ax = plt.axes([0.82, 0.52, 0.15, 0.06])
    btn4_ax = plt.axes([0.82, 0.42, 0.15, 0.06])
    btn5_ax = plt.axes([0.82, 0.32, 0.15, 0.06])
    check_ax = plt.axes([0.82, 0.18, 0.15, 0.06])

    btn1 = Button(btn1_ax, "[1] OK")
    btn2 = Button(btn2_ax, "[2] OK no pts")
    btn3 = Button(btn3_ax, "[3] Fix needed")
    btn4 = Button(btn4_ax, "[4] Group lens")
    btn5 = Button(btn5_ax, "[5] Recentering")
    toggle_btn = Button(check_ax, "Auto-next: ON", color="white")

    def toggle_auto(event):
        auto_next[0] = not auto_next[0]
        if auto_next[0]:
            toggle_btn.label.set_text("Auto-next: ON")
            toggle_btn.ax.set_facecolor("white")
        else:
            toggle_btn.label.set_text("Auto-next: OFF")
            toggle_btn.ax.set_facecolor("#aaaaaa")
        fig.canvas.draw_idle()

    buttons = {
        "1_ok": btn1,
        "2_ok_no_points": btn2,
        "3_fix_needed": btn3,
        "4_group_scale": btn4,
        "5_recentering_needed": btn5,
    }

    btn1.on_clicked(lambda event: set_label("1_ok"))
    btn2.on_clicked(lambda event: set_label("2_ok_no_points"))
    btn3.on_clicked(lambda event: set_label("3_fix_needed"))
    btn4.on_clicked(lambda event: set_label("4_group_scale"))
    btn5.on_clicked(lambda event: set_label("5_recentering_needed"))
    toggle_btn.on_clicked(toggle_auto)

    fig.canvas.mpl_connect("key_press_event", on_key)

    draw()
    plt.show()
