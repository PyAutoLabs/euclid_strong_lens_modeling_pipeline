"""
``north_up_catalogue_tree.build`` makes the tree the north-up release is packaged from, so the
properties the release relies on are tested: north-up PNGs replace their originals, a missing
render keeps the original, every other file is the same inode, and the source tree is untouched.
"""
import csv

import pytest

from scripts.tools import north_up_catalogue_tree


def test__build__swaps_rendered_pngs_and_links_the_rest(tmp_path):
    catalogue = tmp_path / "dr1_catalogue_v5"
    render = tmp_path / "render"
    out = tmp_path / "dr1_catalogue_v5_northup"
    for tile in ("TileA", "TileB"):
        (catalogue / tile).mkdir(parents=True)
        (catalogue / tile / "vis_lp_fit.png").write_bytes(b"old " + tile.encode())
        (catalogue / tile / "fit_sersic.png").write_bytes(b"old sersic")
        (catalogue / tile / "coolest.json").write_text("{}")
    (catalogue / "_build").mkdir()
    (catalogue / "_build" / "manifest.csv").write_text("tile\n")
    (render / "TileA").mkdir(parents=True)
    (render / "TileA" / "vis_lp_fit.png").write_bytes(b"north up A")

    north_up_catalogue_tree.build(catalogue, render, out)

    assert (out / "TileA" / "vis_lp_fit.png").read_bytes() == b"north up A"
    assert (out / "TileB" / "vis_lp_fit.png").read_bytes() == b"old TileB"
    assert (catalogue / "TileA" / "vis_lp_fit.png").read_bytes() == b"old TileA"
    assert (out / "TileA" / "coolest.json").stat().st_ino == (catalogue / "TileA" / "coolest.json").stat().st_ino
    assert (out / "_build" / "manifest.csv").exists()

    report = {(r["tile"], r["product"]): r["source"]
              for r in csv.DictReader(open(tmp_path / "dr1_catalogue_v5_northup_north_up_report.csv"))}
    assert report[("TileA", "vis_lp_fit.png")] == "north_up"
    assert report[("TileB", "vis_lp_fit.png")] == "fit_frame"
    assert report[("TileA", "fit_sersic.png")] == "fit_frame"
    assert report[("TileA", "rgb.png")] == "absent"


def test__build__refuses_an_existing_out_dir(tmp_path):
    (tmp_path / "c").mkdir()
    (tmp_path / "o").mkdir()
    with pytest.raises(SystemExit):
        north_up_catalogue_tree.build(tmp_path / "c", tmp_path / "r", tmp_path / "o")
