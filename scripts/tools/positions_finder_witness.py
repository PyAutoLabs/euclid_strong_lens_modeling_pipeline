"""
Witness run of the production positions path over a calibration sample.

WHAT THIS DOES
    For every tile directory under ``--root`` (any depth; a tile is a directory
    holding ``positions.json`` and ``segmentation/source_flux.fits``) it runs
    the production path twice, writing nothing inside the tile directories:

    - **seeded** -- ``positions_gate.gate_tile(write=False)``: the phase 1 gate
      on the tile's ``positions.json`` plus the pair floor (what the pre-submit
      gate writes for an existing tile);
    - **unseeded** -- ``positions_finder.find_positions_gate``: SNR >= 3
      source-flux peaks outside 0.15" of the light centre, then the same gate
      steps and pair floor (what ``preprocess/segmentation.py`` and the
      ``util`` fallback write for a new tile).

    Both use the gate's light centre and the segmentation writer's SNR map, so
    they agree wherever the SNR >= 3 peak set equals ``positions.json``; the
    ``same`` column says whether the two final position sets match.

    ``--reconcile`` also runs the non-production diagnostic reconcile loop
    (``positions_finder.find_positions``, seeded and unseeded) for comparison.

    It prints the tables and writes, under ``--out`` (default ``<root>/witness``,
    overwritten):

    - ``witness_table.md`` / ``witness.json``: the tables and the full results;
    - ``overlays/<group>_<tile>.png``: the VIS image and the source-flux map
      with the old positions (cyan circles), the seeded result (red crosses),
      the unseeded result (magenta x) and the light centre (orange star).

USAGE
    python scripts/tools/positions_finder_witness.py --root <sample_dir>
    python scripts/tools/positions_finder_witness.py --root <sample_dir> --reconcile --no-overlays
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


def run_tile(d: Path, reconcile: bool = False) -> dict:
    maps = gate.finder_maps_from_dataset(d)
    centre = gate.light_centre_from_dataset(d)
    seed = gate.read_positions(d)
    t0 = time.process_time()
    seeded = gate.gate_tile(d, light_centre=centre, write=False)
    t_seeded = time.process_time() - t0
    t0 = time.process_time()
    writer = pf.find_positions_gate(
        maps["source_flux"], maps["snr_map"], light_centre=centre, pixel_scale=maps["pixel_scale"]
    )
    unseeded = pf.meta_from_result(writer)
    t_unseeded = time.process_time() - t0
    out = dict(
        tile=d.name,
        group=d.parent.name,
        path=str(d),
        light_centre=[round(float(v), 4) for v in centre],
        old=[[round(float(v), 2) for v in p] for p in seed],
        # The unseeded writer's candidate set (its raw positions).
        candidates=[[round(float(v), 2) for v in p] for p in writer.seed],
        seeded=seeded,
        unseeded=unseeded,
        cpu_s_seeded=round(t_seeded, 2),
        cpu_s_unseeded=round(t_unseeded, 2),
    )
    if reconcile:
        args = (maps["source_flux"], maps["snr_map"], maps["lens_flux"], maps["pixel_scale"])
        out["reconcile_seeded"] = pf.find_positions(*args, seed=seed, light_centre=centre).to_dict()
        out["reconcile_unseeded"] = pf.find_positions(*args, light_centre=centre).to_dict()
    return out


def _fmt_pos(ps):
    return " ".join(f"[{p[0]:.2f},{p[1]:.2f}]" for p in ps) or "-"


def _fmt_s(v):
    return "-" if v is None else f"{v:.3f}"


def _key(ps):
    return sorted((round(p[0], 2), round(p[1], 2)) for p in ps)


def _dropped(ds):
    return " ".join(f"#{d['index'] + 1 if d['index'] is not None else '?'}:{d['reason']}" for d in ds) or "-"


def gate_rows(results, which):
    rows = []
    for r in results:
        m = r[which]
        rows.append(
            dict(
                group=r["group"],
                tile=r["tile"][:13],
                old=_fmt_pos(r["old"]),
                input=_fmt_pos(r["candidates"]) if which == "unseeded" else _fmt_pos(r["old"]),
                snr=" ".join("-" if v is None else f"{v:.1f}" for v in (m.get("snr") or [])) or "-",
                new=_fmt_pos(m["positions_used"]),
                s_all=_fmt_s(m["s_min_all"]),
                s_final=_fmt_s(m["s_min"]),
                T=_fmt_s(m["threshold"]),
                dropped=_dropped(m["dropped"]),
                status=m["status"],
                review=m["review_reason"] or "-",
                same="yes" if _key(r["seeded"]["positions_used"]) == _key(r["unseeded"]["positions_used"]) else "no",
                cpu_s=f"{r['cpu_s_' + which]:.1f}",
            )
        )
    return rows


def reconcile_rows(results, which):
    rows = []
    for r in results:
        s = r[which]
        rows.append(
            dict(
                group=r["group"],
                tile=r["tile"][:13],
                new=_fmt_pos(s["positions"]),
                s_seed=_fmt_s(s["s_seed"]),
                s_final=_fmt_s(s["s_final"]),
                T=_fmt_s(s["threshold"]),
                rounds=str(s["n_rounds"]),
                added=_fmt_pos([[a["y"], a["x"]] for a in s["added"]]),
                dropped=_dropped(s["dropped"]),
                status=s["status"],
                review=s["review_reason"] or "-",
            )
        )
    return rows


SEEDED_COLUMNS = ("group", "tile", "old", "snr", "new", "s_all", "s_final", "T", "dropped", "status", "review", "cpu_s")
UNSEEDED_COLUMNS = ("group", "tile", "input", "snr", "new", "s_all", "s_final", "T", "dropped", "status", "review", "same")
RECONCILE_COLUMNS = ("group", "tile", "new", "s_seed", "s_final", "T", "rounds", "added", "dropped", "status", "review")


def markdown(rows, columns):
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
    s, u = r["seeded"], r["unseeded"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, img, title in ((axes[0], vis, "VIS"), (axes[1], sf, "source_flux")):
        finite = img[np.isfinite(img)]
        lo, hi = np.percentile(finite, 1), np.percentile(np.abs(finite), 99.5)
        ax.imshow(np.arcsinh((img - lo) / max(hi / 10, 1e-12)), origin="upper", extent=ext, cmap="gray")
        for i, (y, x) in enumerate(r["old"]):
            ax.plot(x, y, "o", mfc="none", mec="cyan", ms=16, mew=1.5)
            ax.text(x + 0.15, y + 0.15, str(i + 1), color="cyan", fontsize=10)
        for y, x in s["positions_used"]:
            ax.plot(x, y, "+", color="red", ms=14, mew=2)
        for y, x in u["positions_used"]:
            ax.plot(x, y, "x", color="magenta", ms=10, mew=1.5)
        cy, cx = r["light_centre"]
        ax.plot(cx, cy, "*", color="orange", ms=10)
        half = 3.0
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(title)

    def tag(m):
        return m["status"] + (f" ({m['review_reason']})" if m["review_reason"] else "") + f" s {_fmt_s(m['s_min'])} T {_fmt_s(m['threshold'])}"

    fig.suptitle(
        f"{r['group']}/{r['tile'][:13]}  seeded: {tag(s)}   unseeded: {tag(u)}\n"
        "cyan o: positions.json   red +: seeded gate   magenta x: unseeded writer   orange *: light centre",
        fontsize=10,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Witness run of the production positions path over a calibration sample.")
    parser.add_argument("--root", required=True, help="sample directory (tile directories at any depth)")
    parser.add_argument("--out", help="output directory (default <root>/witness)")
    parser.add_argument("--no-overlays", action="store_true", help="skip the overlay PNGs")
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="also run the non-production diagnostic reconcile loop (find_positions) for comparison",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "witness"
    tiles = find_tiles(root)
    if not tiles:
        print(f"no tiles under {root}")
        return 1
    results = [run_tile(d, reconcile=args.reconcile) for d in tiles]
    parts = [
        "Seeded (positions_gate: phase 1 gate on positions.json + pair floor):\n\n"
        + markdown(gate_rows(results, "seeded"), SEEDED_COLUMNS),
        "Unseeded (production writer: SNR >= 3 peaks + gate + pair floor):\n\n"
        + markdown(gate_rows(results, "unseeded"), UNSEEDED_COLUMNS),
    ]
    if args.reconcile:
        parts += [
            "Diagnostic reconcile loop, seeded (not production):\n\n"
            + markdown(reconcile_rows(results, "reconcile_seeded"), RECONCILE_COLUMNS),
            "Diagnostic reconcile loop, unseeded (not production):\n\n"
            + markdown(reconcile_rows(results, "reconcile_unseeded"), RECONCILE_COLUMNS),
        ]
    md = "\n\n".join(parts)
    print(md)
    out.mkdir(parents=True, exist_ok=True)
    (out / "witness_table.md").write_text(md)
    (out / "witness.json").write_text(json.dumps(results, indent=1))
    if not args.no_overlays:
        for r in results:
            overlay(r, out / "overlays" / f"{r['group']}_{r['tile']}.png")
    total = sum(r["cpu_s_seeded"] for r in results)
    print(f"{len(results)} tiles, seeded gate {total:.1f} CPU-s total ({total / len(results):.1f} s/tile); outputs in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
