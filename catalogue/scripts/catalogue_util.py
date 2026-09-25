"""
Euclid Catalogue: Shared Helpers
=================================

The eight catalogue producers under ``catalogue/scripts/`` all do the same three
things around their (different) aggregator queries:

1. resolve ``--output_path`` / ``--inspect_dir`` against the project root and
   push the pipeline's ``config/`` so notation, labels and plotting defaults
   match the fits being scraped;
2. write a master product to ``<inspect_dir>/``;
3. split that master into one self-contained copy per lens inside
   ``<inspect_dir>/<dataset_name>/``.

Steps 1 and 3 live here so the producers differ only where the science does.
The lens-mass producer also uses the explicit model-path fallback below while
old galaxy-attached-shear results and new ``MassField`` results coexist.
"""

import argparse
import csv
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_SAMPLE = "q1_walsmley"


def add_variable_with_fallback(
    agg_csv,
    argument: str,
    fallback_argument: str,
    name: str,
    value_types,
):
    """
    Add an aggregate variable with an explicit legacy model-path fallback.

    ``AggregateCSV`` normally leaves a cell blank when ``argument`` is absent.
    During the fields migration, new results store shear at
    ``fields.shear`` while legacy results store it at
    ``galaxies.lens.shear``. This column selects the new path when present and
    uses the legacy path otherwise; if neither exists,
    the normal warning and blank-cell behavior is retained.
    """
    from autofit.aggregator.summary.aggregate_csv.column import Column

    class FallbackColumn(Column):
        def __init__(self):
            super().__init__(
                argument=argument,
                name=name,
                value_types=value_types,
                strict=agg_csv._strict,
            )
            self.fallback_argument = fallback_argument

        def value(self, row):
            primary_argument = self.argument
            fallback_path = tuple(self.fallback_argument.split("."))
            if self.path not in row.known_paths and fallback_path in row.known_paths:
                self.argument = self.fallback_argument

            try:
                return super().value(row)
            finally:
                self.argument = primary_argument

    agg_csv._columns.append(FallbackColumn())


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

    The split reads the master back rather than being handed the rows, so the two
    can never disagree, and each per-lens file keeps the master's full header —
    the columns are identical, only the rows are fewer. That is what lets a
    single lens folder be shipped on its own and still be parsed by anything that
    reads the master.

    Lens directories are created as needed, so this works on a fresh inspect
    directory as well as alongside products earlier stages already wrote.

    Returns the number of per-lens files written (i.e. the number of distinct
    ``lens_name`` values), which for a multi-row producer is smaller than the
    number of rows in the master.
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

    # This master covers the sample. Remove only this producer's stale split
    # files, never a lens directory or products owned by another stage.
    for stale in inspect_path.glob(f"*/{filename}"):
        if stale.parent.name not in rows_by_lens:
            stale.unlink()

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


class BuildCounts:
    """Product counts; an incomplete fit is a skip, never a successful fit."""

    def __init__(self, stage, unit):
        self.stage = stage
        self.unit = unit
        self.built = 0
        self.already_present = 0
        self.skipped = 0
        self.errors = 0

    def skip(self, identity, reason):
        self.skipped += 1
        print(f"WARNING {self.stage}: skipping {identity}: {reason}")

    def report(self):
        print(
            f"{self.stage} ({self.unit}): built={self.built} "
            f"already_present={self.already_present} skipped={self.skipped} "
            f"errors={self.errors}"
        )


def reported(stage, unit="lenses"):
    """Report even on failure, but preserve every unexpected exception."""
    from functools import wraps

    def decorate(main):
        @wraps(main)
        def run():
            counts = BuildCounts(stage, unit)
            try:
                return main(counts)
            except Exception:
                counts.errors += 1
                raise
            finally:
                counts.report()

        return run

    return decorate


def missing_assets(result, *, fits=(), images=(), values=()):
    """Check named generated inputs, without catching deserialization errors.

    Use the aggregator's inventory so temporary ZIP extraction and nested JSON
    names follow the same rules as the actual readers. An existing malformed
    asset is not an absent asset; its reader must still raise.
    """
    missing = []
    fits_names = {item.name for item in result.fits}
    for name in fits:
        if name not in fits_names:
            missing.append(f"{name}.fits")
    for name in images:
        path = result.directory / "image" / f"{name}.png"
        try:
            path.stat()
        except FileNotFoundError:
            missing.append(f"{name}.png")
    value_names = {
        item.name
        for item in result.jsons + result.pickles + result.arrays + result.fits
    }
    for name in values:
        if name not in value_names:
            missing.append(name)
    return missing


def result_identity(result):
    search = result.search
    return (
        f"lens={search.path_prefix.parts[-1]} band={search.name} "
        f"result={result.directory.name}"
    )


def available_results(aggregator, counts, **requirements):
    """Filter only absent optional assets; labels must use these same survivors."""
    from autofit.aggregator.aggregator import Aggregator

    kept = []
    for result in aggregator:
        missing = missing_assets(result, **requirements)
        if missing:
            counts.skip(result_identity(result), f"missing {', '.join(missing)}")
        else:
            kept.append(result)
    return Aggregator(
        search_outputs=kept, grid_search_outputs=aggregator.grid_search_outputs
    )


def lens_assets_available(aggregator, counts, dataset_name, **requirements):
    """Keep a lens product complete across all selected results/bands."""
    missing = [
        f"{result_identity(result)}: {', '.join(assets)}"
        for result in aggregator
        if (assets := missing_assets(result, **requirements))
    ]
    if not len(aggregator):
        counts.skip(dataset_name, "no matching completed results")
        return False
    if missing:
        counts.skip(dataset_name, "missing assets; " + "; ".join(missing))
        return False
    return True


def write_fits_products(products, directory):
    """Stage all extracted products before replacing any published FITS file."""
    from tempfile import TemporaryDirectory

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fits-", dir=directory) as temporary:
        for name, hdus in products.items():
            hdus.writeto(Path(temporary) / name)
        for name in products:
            (Path(temporary) / name).replace(directory / name)
    return [directory / name for name in products]


def save_csv(aggregate, target):
    """Publish only after all rows have been read and written successfully."""
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix=".csv-", dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        aggregate.save(path=staged)
        staged.replace(target)


def clear_csv_rows(inspect_path, filename):
    """An empty refresh must not leave old rows masquerading as this build."""
    target = inspect_path / filename
    if target.exists():
        with target.open(newline="") as stream:
            header = next(csv.reader(stream))
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(prefix=".csv-", dir=inspect_path) as temporary:
            staged = Path(temporary) / filename
            with staged.open("w", newline="") as stream:
                csv.writer(stream).writerow(header)
            staged.replace(target)
    for stale in inspect_path.glob(f"*/{filename}"):
        stale.unlink()
