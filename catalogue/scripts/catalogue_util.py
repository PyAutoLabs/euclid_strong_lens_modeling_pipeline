"""
Euclid Catalogue: Shared Helpers
=================================

The six catalogue producers under ``catalogue/scripts/`` all do the same three
things around their (different) aggregator queries:

1. resolve ``--output_path`` / ``--inspect_dir`` against the project root and
   push the pipeline's ``config/`` so notation, labels and plotting defaults
   match the fits being scraped;
2. write a master product to ``<inspect_dir>/``;
3. split that master into one self-contained copy per lens inside
   ``<inspect_dir>/<dataset_name>/``.

Steps 1 and 3 live here so the producers differ only where the science does.
Nothing in this module is Euclid-specific beyond the directory convention; it
deliberately holds no model, latent or aggregator logic.
"""

import argparse
import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_SAMPLE = "q1_walsmley"


def add_common_arguments(parser: argparse.ArgumentParser, default_output_path="output"):
    """
    Add the ``--sample`` / ``--output_path`` / ``--inspect_dir`` trio every
    catalogue producer takes.

    Parameters
    ----------
    parser
        The parser to extend.
    default_output_path
        Root results directory this producer reads by default. The multi-band
        producers (``multi_wavelength.py``, ``magnitudes.py``) default to
        ``output_sed`` because the SED chain is run with
        ``PYAUTO_OUTPUT_DIR=output_sed``; the rest read the main ``output``.
    """
    parser.add_argument(
        "--sample",
        metavar="name",
        default=DEFAULT_SAMPLE,
        help=(
            "Sample subdirectory inside the results directory. Default: "
            f"'{DEFAULT_SAMPLE}' (the shipped example dataset's sample)."
        ),
    )
    parser.add_argument(
        "--output_path",
        metavar="path",
        default=default_output_path,
        help=(
            f"Root results directory. Default: '{default_output_path}'. "
            "Test-mode runs land in '<output>/test_mode'."
        ),
    )
    parser.add_argument(
        "--inspect_dir",
        metavar="path",
        default=None,
        help=(
            "Directory the products are written to. "
            "Default: <project_root>/inspect/<sample>."
        ),
    )
    return parser


def resolve_paths(args):
    """
    Turn the parsed common arguments into absolute ``(output_path,
    inspect_path)`` and push the pipeline ``config/`` onto the autoconf stack.

    Relative paths are resolved against the project root, so a producer behaves
    the same whether it is run from the repo root or from ``catalogue/``. The
    inspect directory is created if absent.
    """
    from autolens import conf

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    if args.inspect_dir is not None:
        inspect_path = Path(args.inspect_dir)
        if not inspect_path.is_absolute():
            inspect_path = PROJECT_ROOT / inspect_path
    else:
        inspect_path = PROJECT_ROOT / "inspect" / args.sample

    conf.instance.push(new_path=PROJECT_ROOT / "config", output_path=output_path)

    inspect_path.mkdir(parents=True, exist_ok=True)

    return output_path, inspect_path


def sample_root_from(output_path: Path, sample: str) -> Path:
    """
    The directory holding one dataset folder per lens for ``sample``.
    """
    return output_path / sample if sample else output_path


def write_per_tile_csv(master_csv: Path, inspect_path: Path, filename: str) -> int:
    """
    Split a master CSV into one CSV per lens, dropped in that lens's own
    directory under ``inspect_path``, so every lens folder is self-contained.

    Rows are grouped by the ``lens_name`` column, so a producer emitting one row
    per lens (``lens_mass``, ``lens_sersic``, ``source_sersic``) and one
    emitting several rows per lens (``magnitudes``, one per waveband) both work.
    Rows without a ``lens_name`` are skipped.

    Returns the number of per-lens files written.
    """
    with open(master_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    rows_by_lens = {}
    for row in rows:
        lens_name = row.get("lens_name")
        if not lens_name:
            continue
        rows_by_lens.setdefault(lens_name, []).append(row)

    for lens_name, lens_rows in rows_by_lens.items():
        lens_dir = inspect_path / lens_name
        lens_dir.mkdir(parents=True, exist_ok=True)
        with open(lens_dir / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in lens_rows:
                writer.writerow(row)

    return len(rows_by_lens)


def dataset_names_from(sample_root: Path):
    """
    Sorted names of the per-lens result directories under ``sample_root``,
    or an empty list (with a message) when the sample has not been run.
    """
    if not sample_root.is_dir():
        print(f"no sample directory at {sample_root}; nothing to do")
        return []
    return sorted(
        name for name in os.listdir(sample_root) if (sample_root / name).is_dir()
    )
