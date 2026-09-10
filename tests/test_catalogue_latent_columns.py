"""
The catalogue's latent columns must actually carry numbers.

``catalogue/scripts/lens_mass.py`` and ``catalogue/scripts/magnitudes.py`` asked
``af.AggregateCSV`` for their latents under a ``latent.`` prefix
(``latent.effective_einstein_radius``, ``latent.total_lens_flux_mujy``, ...).
That prefix is retired. ``Row.median_pdf_sample_kwargs`` merges the latent
summary's kwargs into the *same* dictionary as the model paths, under the bare
keys the fit wrote (``("effective_einstein_radius",)``), and
``Column.path`` is ``tuple(argument.split("."))`` — so the prefixed argument
looked up ``("latent", "effective_einstein_radius")``, found nothing, and
``Column.value`` turned that ``KeyError`` into ``None``.

Which is why nobody noticed. A missing argument is not raised, it is written
**blank**: the header stayed the full DR1 width and every latent cell under it
was empty — one blank column in ``lens_mass.csv`` and all eight variables, the
whole photometry table, in ``magnitudes.csv``.

``tests/test_catalogue_parity.py`` could not see it, because a blank cell and a
full one produce the same header, and that module deliberately runs nothing.
So this one runs the real thing:

* ``test_lens_mass_writes_no_blank_cells`` /
  ``test_magnitudes_writes_no_blank_cells`` build a completed result tree on
  disk with PyAutoFit's own serializers — ``DirectoryPaths.save_all`` for
  ``model.json`` / ``search.json``, ``save_samples_summary`` for
  ``samples_summary.json`` and for ``latent/latent_summary.json``, exactly as
  ``SearchUpdater._compute_latent_samples`` writes it — then run the producer's
  own ``main()`` over it and assert every cell of every row is non-empty, and
  that the header is still the DR1 header the parity test protects. A
  re-prefixed argument empties cells while leaving that header intact, so the
  two assertions together are the pin.
* ``test_every_non_model_argument_is_a_latent_key`` reads the ``argument``
  string of every ``add_variable`` call out of the producers' syntax trees and
  requires each one that is not a ``galaxies...`` model path to be, exactly, a
  key of ``util.LatentEuclid``. That catches a re-introduced prefix and a
  renamed latent without touching the disk.
* ``test_producers_survive_an_empty_query`` points a producer at a results tree
  holding only the *other* search and asserts it returns cleanly with no CSV,
  rather than letting ``AggregateCSV``'s ``ValueError("The aggregator is
  empty.")`` out. ``scripts/build_inspection_bundle.sh`` runs under ``set -e``,
  so that exception aborted every later stage of the bundle over a stage that
  simply had not run yet — the ordinary state of a ``vis_lp``-only tree scraped
  with the default ``--search_name=vis_pix``.

JAX-free and search-free, as the fast suite requires: no fit is run, the results
are written by hand from PyAutoFit's serializers and the latent values are
arbitrary constants — what is under test is the *lookup*, not the physics.
"""

import ast
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import util  # noqa: E402

# The ``ast`` reader below extends `test_catalogue_parity`'s, and shares its two
# node helpers rather than growing a second copy of them.
from test_catalogue_parity import _keyword, _literal  # noqa: E402


CATALOGUE_SCRIPTS = PROJECT_ROOT / "catalogue" / "scripts"
DR1_HEADERS = Path(__file__).parent / "data" / "dr1_headers"

PRODUCERS = ["lens_mass", "magnitudes"]

SAMPLE = "pytest_latent_columns"
LENS_NAMES = ("TileAAA", "TileBBB")
BANDS = ("vis", "nir_h")

# Enough samples for `SamplesPDF.summary()` to quantile a median and 1σ / 3σ
# bounds; the values themselves are arbitrary.
SAMPLE_COUNT = 5

# Written into `files/wcs.json`, which `magnitudes.py` reads for its third label
# column. `util.AnalysisImaging` writes this file beside a real result.
CRVAL_RA_DEG = 150.1234


def _latent_keys():
    """
    The authoritative Euclid latent names: the library latents
    ``config/latent.yaml`` enables plus ``LatentEuclid.APERTURE_LATENT_KEYS``.

    ``LatentEuclid.keys`` takes the analysis only to satisfy the ``Latent``
    extension-point signature and never reads it, so no fit is needed here.
    """
    return list(util.LatentEuclid.keys(None))


def _samples_for(model, offset):
    """
    A ``SamplesPDF`` over ``model`` with ``SAMPLE_COUNT`` samples, built the way
    a search builds one so that ``summary()`` produces a genuine
    ``SamplesSummary`` (median, max-likelihood and the 1σ / 3σ bounds).
    """
    from autofit.non_linear.samples.pdf import SamplesPDF
    from autofit.non_linear.samples.sample import Sample

    count = model.prior_count

    return SamplesPDF(
        model=model,
        sample_list=Sample.from_lists(
            model=model,
            parameter_lists=[
                [offset + 0.1 * index + 0.01 * parameter for parameter in range(count)]
                for index in range(SAMPLE_COUNT)
            ],
            log_likelihood_list=[float(index) for index in range(SAMPLE_COUNT)],
            log_prior_list=[0.0] * SAMPLE_COUNT,
            weight_list=[1.0] * SAMPLE_COUNT,
        ),
    )


def _latent_samples_for(keys, offset):
    """
    The latent samples a fit computes, mirrored from
    ``autofit.non_linear.analysis.latent.latent_samples_from``: one ``Sample``
    per posterior sample whose kwargs are the **bare** latent names, over a
    model built from those kwargs by ``simple_model_for_kwargs``.
    """
    from autofit.non_linear.samples.pdf import SamplesPDF
    from autofit.non_linear.samples.sample import Sample
    from autofit.non_linear.samples.util import simple_model_for_kwargs

    sample_list = [
        Sample(
            log_likelihood=float(index),
            log_prior=0.0,
            weight=1.0,
            kwargs={
                key: offset + 0.1 * index + 0.01 * position
                for position, key in enumerate(keys)
            },
        )
        for index in range(SAMPLE_COUNT)
    ]

    return SamplesPDF(
        model=simple_model_for_kwargs(sample_list[0].kwargs),
        sample_list=sample_list,
    )


def _write_result(model, *, path_prefix, name, unique_tag, offset, files=None):
    """
    Write one completed search output, using PyAutoFit's own writers.

    ``save_all`` writes ``files/model.json`` and ``files/search.json`` (the
    sentinel ``Aggregator.from_directory`` scans for, and the source of
    ``path_prefix`` / ``name`` / ``unique_tag``); the two
    ``save_samples_summary`` calls write ``files/samples_summary.json`` and
    ``files/latent/latent_summary.json`` under the names the search updater
    uses; ``completed()`` drops the ``.completed`` marker ``completed_only=True``
    filters on.
    """
    import autofit as af

    search = af.Drawer(
        name=name,
        path_prefix=path_prefix,
        unique_tag=unique_tag,
        total_draws=SAMPLE_COUNT,
    )

    paths = search.paths
    paths.model = model
    paths.search = search
    paths.save_all()
    paths.save_samples_summary(_samples_for(model, offset).summary())
    paths.save_samples_summary(
        _latent_samples_for(_latent_keys(), offset).summary(),
        "latent/latent_summary",
    )
    for file_name, value in (files or {}).items():
        paths.save_json(file_name, value)
    paths.completed()

    return paths.output_path


def _run_producer(producer, monkeypatch, output_path, inspect_path):
    """
    Load a producer by path (it is a script, not an importable package member)
    and run its real ``main()`` over ``output_path``.
    """
    spec = importlib.util.spec_from_file_location(
        f"_{producer}_under_test", CATALOGUE_SCRIPTS / f"{producer}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            f"{producer}.py",
            f"--sample={SAMPLE}",
            f"--output_path={output_path}",
            f"--inspect_dir={inspect_path}",
        ],
    )
    module.main()


def _assert_no_blank_cells(csv_path, producer):
    """
    Every cell of every row carries a value, and the header is still the DR1
    one. A missing ``add_variable`` argument is written blank rather than
    raised, so this is the only place the bug is visible.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    expected_header = (DR1_HEADERS / f"{producer}.txt").read_text().strip()

    assert ",".join(header) == expected_header, (
        f"{producer}.csv must carry the DR1 header; a producer whose columns "
        "have drifted is a different failure from a blank cell"
    )
    assert rows, f"{producer}.csv has a header but no rows; the fixture was not scraped"

    for number, row in enumerate(rows):
        for column, value in row.items():
            assert value is not None and value.strip() != "", (
                f"{producer}.csv row {number} ('{row.get('lens_name')}') has an "
                f"empty '{column}' cell: `add_variable` found nothing for that "
                "argument and wrote a blank rather than raising"
            )


@pytest.fixture
def pipeline_config(tmp_path):
    """
    Push the pipeline's own ``config/`` with the results root inside
    ``tmp_path``, so ``DirectoryPaths`` writes the fixture there, and restore
    the repository config afterwards (``conf.instance`` has no pop).
    """
    from autolens import conf

    output_path = tmp_path / "output"
    conf.instance.push(new_path=PROJECT_ROOT / "config", output_path=output_path)
    try:
        yield output_path
    finally:
        conf.instance.push(
            new_path=PROJECT_ROOT / "config", output_path=PROJECT_ROOT / "output"
        )


def test_lens_mass_writes_no_blank_cells(tmp_path, monkeypatch, pipeline_config):
    """
    The whole ``lens_mass.csv``, produced by the real ``main()`` over a real
    aggregator: seven mass parameters and the ``effective_einstein_radius``
    latent, five columns each, none of them empty.
    """
    import autofit as af
    import autolens as al

    # The `vis_pix` mass model: exactly the 7 free parameters `mass_args` names.
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                mass=al.mp.Isothermal,
                shear=al.mp.ExternalShear,
            )
        )
    )
    assert model.total_free_parameters == 7

    for offset, lens_name in enumerate(LENS_NAMES):
        _write_result(
            model,
            path_prefix=Path(SAMPLE) / lens_name,
            name="vis_pix",
            unique_tag="initial_lens_model",
            offset=1.0 + offset,
        )

    inspect_path = tmp_path / "inspect"
    _run_producer("lens_mass", monkeypatch, pipeline_config, inspect_path)

    _assert_no_blank_cells(inspect_path / "lens_mass.csv", "lens_mass")


def test_magnitudes_writes_no_blank_cells(tmp_path, monkeypatch, pipeline_config):
    """
    ``magnitudes.csv`` is the table the bug emptied completely: every one of its
    value columns is a latent. One row per ``(lens, waveband)``, and the three
    label columns (``lens_name``, ``waveband`` from ``search.name``,
    ``crval_ra_deg`` from the ``wcs.json`` the analysis writes) must be filled
    too, or a blank cell would prove nothing about the latents.
    """
    import autofit as af
    import autolens as al

    model = af.Collection(
        galaxies=af.Collection(lens=af.Model(al.Galaxy, redshift=0.5, bulge=al.lp.Sersic))
    )

    offset = 1.0
    for lens_name in LENS_NAMES:
        for band in BANDS:
            _write_result(
                model,
                path_prefix=Path(SAMPLE) / lens_name,
                name=band,
                unique_tag="sersic_lens_model",
                offset=offset,
                files={"wcs": {"crval_ra_deg": CRVAL_RA_DEG}},
            )
            offset += 1.0

    inspect_path = tmp_path / "inspect"
    _run_producer("magnitudes", monkeypatch, pipeline_config, inspect_path)

    csv_path = inspect_path / "magnitudes.csv"
    _assert_no_blank_cells(csv_path, "magnitudes")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == len(LENS_NAMES) * len(BANDS), (
        "magnitudes.csv carries one row per (lens, waveband); a different count "
        "means the de-duplication dropped a band"
    )


EMPTY_QUERY_PRODUCERS = ("lens_mass", "lens_sersic", "source_sersic")


@pytest.mark.parametrize("producer", EMPTY_QUERY_PRODUCERS)
def test_producers_survive_an_empty_query(producer, tmp_path, monkeypatch, pipeline_config):
    """
    A results tree that holds a completed ``vis_lp`` search and nothing else,
    scraped by a producer whose default ``--search_name`` is ``vis_pix``: the
    query matches nothing, and the producer must say so and return rather than
    raise.

    The tree is deliberately *not* empty — a missing sample directory is a
    different, already-handled path — so what is under test is an aggregator
    that found results and a query that excluded all of them, which is what
    ``af.AggregateCSV`` raises ``ValueError("The aggregator is empty.")`` on.

    The assertion is that no CSV is written, not that an empty one is: an absent
    file is the honest record of "nothing matched", and ``build_inspection_bundle.sh``
    and the per-lens split both already handle it.
    """
    import autofit as af
    import autolens as al

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=0.5,
                mass=al.mp.Isothermal,
                shear=al.mp.ExternalShear,
            )
        )
    )

    for offset, lens_name in enumerate(LENS_NAMES):
        _write_result(
            model,
            path_prefix=Path(SAMPLE) / lens_name,
            name="vis_lp",
            unique_tag="initial_lens_model",
            offset=1.0 + offset,
        )

    inspect_path = tmp_path / "inspect"
    _run_producer(producer, monkeypatch, pipeline_config, inspect_path)

    assert not (inspect_path / f"{producer}.csv").exists(), (
        f"{producer}.py wrote a CSV for a query that matched nothing; an empty "
        "query should leave no product behind"
    )


def arguments_from(path):
    """
    The ordered ``argument`` strings a producer hands to ``add_variable``, read
    out of the module's syntax tree.

    The same shapes ``test_catalogue_parity.column_specs_from`` recognises —
    a direct ``agg_csv.add_variable(argument="...", ...)`` call, and a
    ``for argument, name in <list of pairs>:`` loop unrolled over its literal —
    but keyed on the argument rather than the column name, because it is the
    argument that decides whether a value is found at all.
    """
    tree = ast.parse(path.read_text(), filename=str(path))

    environment = {}
    arguments = []

    def add_variable_calls(node):
        return [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_variable"
        ]

    def visit(statements):
        for statement in statements:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    literal = _literal(statement.value)
                    if literal is not None:
                        environment[target.id] = literal
                continue

            if isinstance(statement, ast.Expr) and isinstance(
                statement.value, ast.Call
            ):
                call = statement.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "add_variable"
                ):
                    argument = _literal(_keyword(call, "argument"))
                    if argument is not None:
                        arguments.append(argument)
                continue

            if isinstance(statement, ast.For):
                pairs = (
                    environment.get(statement.iter.id)
                    if isinstance(statement.iter, ast.Name)
                    else None
                )
                calls = add_variable_calls(statement)
                if pairs is not None and isinstance(statement.target, ast.Tuple) and calls:
                    for _call in calls:
                        for argument, _name in pairs:
                            arguments.append(argument)
                    continue

            for field in ("body", "orelse", "finalbody"):
                nested = getattr(statement, field, None)
                if nested:
                    visit(nested)

    visit(tree.body)

    return arguments


@pytest.mark.parametrize("producer", PRODUCERS)
def test_every_non_model_argument_is_a_latent_key(producer):
    """
    The hermetic half: a re-introduced ``latent.`` prefix, or a latent renamed
    on one side only, must fail here — naming the argument — rather than
    quietly emptying a column of the DR1 catalogue.
    """
    arguments = arguments_from(CATALOGUE_SCRIPTS / f"{producer}.py")

    assert arguments, (
        f"no add_variable arguments found in catalogue/scripts/{producer}.py; "
        "the syntax-tree reader has drifted from the producer"
    )

    latent_keys = _latent_keys()

    for argument in arguments:
        if argument.startswith("galaxies."):
            continue
        assert argument in latent_keys, (
            f"catalogue/scripts/{producer}.py asks add_variable for "
            f"'{argument}', which is neither a 'galaxies...' model path nor a "
            f"key of util.LatentEuclid ({sorted(latent_keys)}). "
            "`add_variable` looks its argument up in one merged dictionary of "
            "model paths and bare latent names, and writes a blank cell when it "
            "misses — so this would empty a DR1 column silently."
        )
