"""
Witness run of the model-guided positions finder over a calibration sample.

WHAT THIS DOES
    For every tile directory under ``--root`` (any depth; a tile is a directory
    holding ``positions.json`` and ``segmentation/source_flux.fits``) it runs
    ``positions_finder.find_positions`` exactly as the positions gate does --
    seeded with the tile's ``positions.json``, the gate's light centre and the
    segmentation writer's SNR map -- and, for comparison, unseeded (the path
    ``preprocess/segmentation.py`` now takes for a new tile). Nothing inside the
    tile directories is written.

    It prints one row per tile (old positions, new positions, seed and final
    ``s``, threshold, rounds, added / dropped, status and review reason, CPU
    seconds) and writes, under ``--out`` (default ``<root>/witness``):

    - ``witness_table.md`` / ``witness.json``: the table and the full results;
    - ``overlays/<tile>.png``: the VIS image and the source-flux map with the old
      positions (cyan circles), the new positions (red crosses), the model's
      predicted images (yellow x, labelled with ``mu``), added positions (green
      squares) and the light centre.

USAGE
    python scripts/tools/positions_finder_witness.py --root <sample_dir>
    python scripts/tools/positions_finder_witness.py --root <sample_dir> --no-overlays
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import positions_finder as pf  # noqa: E402
import positions_gate as gate  # noqa: E402


def find_tiles(root: Path):
    """Tile directories under ``root``, skipping the witness output folder."""
    out = []
    for sf in sorted(root.rglob("segmentation/source_flux.fits")):
        d = sf.parent.parent
        if "witness" in d.relative_to(root).parts:
            continue
        if (d / "positions.json").exists() and (d / f"{d.name}.fits").exists():
            out.append(d)
    return out


def run_tile(d: Path) -> dict:
    maps = gate.finder_maps_from_dataset(d)
    centre = gate.light_centre_from_dataset(d)
    seed = gate.read_positions(d)
    t0 = time.process_time()
    seeded = pf.find_positions(
        maps["source_flux"], maps["snr_map"], maps["lens_flux"], maps["pixel_scale"],
        seed=seed, light_centre=centre,
    )
    t_seeded = time.process_time() - t0
    t0 = time.process_time()
    unseeded = pf.find_positions(
        maps["source_flux"], maps["snr_map"], maps["lens_flux"], maps["pixel_scale"],
        light_centre=centre,
    )
    t_unseeded = time.process_time() - t0
    return dict(
        tile=d.name,
        group=d.parent.name,
        path=str(d),
        light_centre=[round(float(v), 4) for v in centre],
        old=[[round(float(v), 2) for v in p] for p in seed],
        seeded=seeded.to_dict(),
        unseeded=unseeded.to_dict(),
        cpu_s_seeded=round(t_seeded, 2),
        cpu_s_unseeded=round(t_unseeded, 2),
    )


def _fmt_pos(ps):
    return " ".join(f"[{p[0]:.2f},{p[1]:.2f}]" for p in ps) or "-"


def _fmt_s(v):
    return "-" if v is None else f"{v:.3f}"


def table_rows(results):
    rows = []
    for r in results:
        s = r["seeded"]
        rows.append(
            dict(
                group=r["group"],
                tile=r["tile"][:13],
                old=_fmt_pos(r["old"]),
                new=_fmt_pos(s["positions"]),
                s_seed=_fmt_s(s["s_seed"]),
                s_final=_fmt_s(s["s_final"]),
                T=_fmt_s(s["threshold"]),
                rounds=str(s["n_rounds"]),
                added=_fmt_pos([[a["y"], a["x"]] for a in s["added"]]),
                dropped=" ".join(
                    f"#{d['index'] + 1 if d['index'] is not None else '?'}:{d['reason']}" for d in s["dropped"]
                ) or "-",
                status=s["status"],
                review=s["review_reason"] or "-",
                cpu_s=f"{r['cpu_s_seeded']:.1f}",
                unseeded=f"{_fmt_pos(r['unseeded']['positions'])} ({r['unseeded']['status']}"
                + (f":{r['unseeded']['review_reason']}" if r["unseeded"]["review_reason"] else "")
                + ")",
            )
        )
    return rows


COLUMNS = ("group", "tile", "old", "new", "s_seed", "s_final", "T", "rounds", "added", "dropped", "status", "review", "cpu_s")


def markdown(rows, columns=COLUMNS):
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    return head + "".join("| " + " | ".join(r[c] for c in columns) + " |\n" for r in rows)


def overlay(r, out_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    d = Path(r["path"])
    with fits.open(d / f"{d.name}.fits") as hdul:
        vis = np.asarray(hdul["VIS_BGSUB"].data, float)
    sf = np.asarray(fits.getdata(d / "segmentation" / "source_flux.fits"), float)
    ps = 0.1
    ny, nx = vis.shape
    ext = [-nx / 2 * ps, nx / 2 * ps, -ny / 2 * ps, ny / 2 * ps]
    s = r["seeded"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, img, title in ((axes[0], vis, "VIS"), (axes[1], sf, "source_flux")):
        finite = img[np.isfinite(img)]
        lo, hi = np.percentile(finite, 1), np.percentile(np.abs(finite), 99.5)
        ax.imshow(np.arcsinh((img - lo) / max(hi / 10, 1e-12)), origin="upper", extent=ext, cmap="gray")
        for i, (y, x) in enumerate(r["old"]):
            ax.plot(x, y, "o", mfc="none", mec="cyan", ms=16, mew=1.5)
            ax.text(x + 0.15, y + 0.15, str(i + 1), color="cyan", fontsize=10)
        for y, x in s["positions"]:
            ax.plot(x, y, "+", color="red", ms=14, mew=2)
        for a in s["added"]:
            ax.plot(a["x"], a["y"], "s", mfc="none", mec="lime", ms=12, mew=1.5)
        for p in s["predicted"]:
            ax.plot(p["x"], p["y"], "x", color="yellow", ms=8, mew=1.5)
            ax.text(p["x"] - 0.1, p["y"] - 0.35, f"{p['mu']:.1f}", color="yellow", fontsize=8)
        cy, cx = r["light_centre"]
        ax.plot(cx, cy, "*", color="orange", ms=10)
        half = 3.0
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(title)
    fig.suptitle(
        f"{r['group']}/{r['tile'][:13]}: {s['status']}"
        + (f" ({s['review_reason']})" if s["review_reason"] else "")
        + f"  s {_fmt_s(s['s_seed'])} -> {_fmt_s(s['s_final'])}  T {_fmt_s(s['threshold'])}  rounds {s['n_rounds']}\n"
        "cyan o: positions.json   red +: finder   green []: added   yellow x: predicted (mu)   orange *: light centre",
        fontsize=10,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Witness run of positions_finder over a calibration sample.")
    parser.add_argument("--root", required=True, help="sample directory (tile directories at any depth)")
    parser.add_argument("--out", help="output directory (default <root>/witness)")
    parser.add_argument("--no-overlays", action="store_true", help="skip the overlay PNGs")
    args = parser.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "witness"
    tiles = find_tiles(root)
    if not tiles:
        print(f"no tiles under {root}")
        return 1
    results = [run_tile(d) for d in tiles]
    rows = table_rows(results)
    md = markdown(rows)
    print(md)
    print("Unseeded (segmentation-writer path):")
    for r in rows:
        print(f"  {r['group']}/{r['tile']}: {r['unseeded']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "witness_table.md").write_text(
        md + "\n\nUnseeded (segmentation-writer path):\n\n"
        + markdown(rows, ("group", "tile", "unseeded"))
    )
    (out / "witness.json").write_text(json.dumps(results, indent=1))
    if not args.no_overlays:
        for r in results:
            overlay(r, out / "overlays" / f"{r['group']}_{r['tile']}.png")
    total = sum(r["cpu_s_seeded"] for r in results)
    print(f"{len(results)} tiles, seeded finder {total:.1f} CPU-s total ({total / len(results):.1f} s/tile); outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
