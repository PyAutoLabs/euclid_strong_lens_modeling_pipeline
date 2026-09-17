"""
The file-and-HDU contract of ``catalogue/scripts/lens_mass_maps.py``.

Stage 3 of the inspection bundle writes three FITS files per lens out of the
``image/tracer.fits`` a finished ``initial_lens_model/vis_pix`` search left in
its result zip. Three things about that are worth pinning, and none of them
needs a fit, a dataset or an aggregator:

1. **Which file gets which HDU.** ``mass_map_products()`` is the whole table —
   the filenames the bundle promises and the ``al.agg.fits_tracer`` members that
   go in each. A silent change here writes a differently-shaped bundle that
   still builds.
2. **The skip.** ``mass_maps_exist`` is what makes the stage cheap to re-run,
   and it is existence-only by design: a *partially* written lens must not be
   skipped, or it stays partial forever.
3. **The extraction.** ``af.AggregateFITS`` reaches ``tracer.fits`` only from
   the PyAutoFit fix this producer requires — before it, ``files/tracer.json``
   shadowed the FITS in ``SearchOutput.value`` (see the producer's module
   docstring). The control is here: a synthetic result directory holding a
   ``tracer.fits`` written with the library's own ``hdu_list_for_output_from``
   *and* a same-named JSON beside it, driven through the real
   ``af.AggregateFITS``, which pins the version floor as well as the contract.

JAX-free and fit-free: the only library code is ``autolens``'s enum,
``autofit``'s aggregator and ``autonerves``'s FITS writer, all imported inside
the tests.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "_lens_mass_maps_under_test",
    PROJECT_ROOT / "catalogue" / "scripts" / "lens_mass_maps.py",
)
lens_mass_maps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lens_mass_maps)

FILENAMES = ["convergence.fits", "potential.fits", "deflections.fits"]

# The extension names PyAutoLens's `fits_tracer` writes, in the order it writes
# them. `MASK` is the PrimaryHDU; the producer never asks for it.
TRACER_EXTNAMES = [
    "MASK",
    "CONVERGENCE",
    "POTENTIAL",
    "DEFLECTIONS_Y",
    "DEFLECTIONS_X",
]

SHAPE = (7, 7)


# ---------------------------------------------------------------------------
# Fixtures: a synthetic tracer.fits and the stubs that serve it
# ---------------------------------------------------------------------------


def _tracer_hdu_list():
    """
    A stand-in for one fit's ``image/tracer.fits``, written by the same
    ``autonerves`` helper ``tracer_plots.fits_tracer`` uses, so the extension
    names, the header keys and the PrimaryHDU-is-the-mask convention are the
    real ones rather than this module's idea of them.

    Each map is filled with a different constant, so an extraction that returns
    the right extension name but the wrong data is still caught.
    """
    from autonerves.fitsable import hdu_list_for_output_from

    values_list = [
        np.full(SHAPE, float(index), dtype="float64")
        for index in range(len(TRACER_EXTNAMES))
    ]

    return hdu_list_for_output_from(
        values_list=values_list,
        ext_name_list=[name.lower() for name in TRACER_EXTNAMES],
        header_dict={"PIXSCAY": 0.1, "PIXSCAX": 0.1},
    )


@pytest.fixture
def result_directory(tmp_path):
    """
    One search's result directory as a finished fit leaves it: ``image/tracer.fits``
    beside a same-named ``files/tracer.json``.

    The JSON is the collision — a real fit writes its max-log-likelihood
    ``Tracer`` there — so ``af.AggregateFITS`` is exercised against exactly the
    shape of result tree that defeated it before the PyAutoFit fix.
    """
    import json

    directory = tmp_path / "search"
    (directory / "image").mkdir(parents=True)
    (directory / "files").mkdir(parents=True)

    _tracer_hdu_list().writeto(directory / "image" / "tracer.fits", overwrite=True)
    (directory / "files" / "tracer.json").write_text(
        json.dumps({"not": "the fits file"})
    )

    return directory


def _aggregate_fits(directory):
    """
    The real ``af.AggregateFITS`` over one search output — the production object,
    built from a list of ``SearchOutput`` rather than an ``Aggregator`` so the
    control needs no scan, no ``search.json`` and no query.
    """
    import autofit as af
    from autofit import SearchOutput

    return af.AggregateFITS(aggregator=[SearchOutput(directory)])


class _StubAggregate:
    """
    The ``extract_fits``-exposing object ``write_mass_maps`` takes, standing in
    for ``af.AggregateFITS``. It records the HDU lists it was asked for so
    the writer can be checked independently of the extraction.
    """

    def __init__(self):
        self.calls = []

    def extract_fits(self, hdus, extname_prefix_list=None):
        from astropy.io import fits

        self.calls.append(list(hdus))

        hdu_list = [fits.PrimaryHDU()]
        for index, hdu in enumerate(hdus):
            image_hdu = fits.ImageHDU(
                data=np.full(SHAPE, float(index), dtype="float64")
            )
            image_hdu.header["EXTNAME"] = hdu.value
            hdu_list.append(image_hdu)

        return fits.HDUList(hdu_list)


# ---------------------------------------------------------------------------
# 1. The product table
# ---------------------------------------------------------------------------


def test_mass_map_products_names_the_three_bundle_files():
    """
    The filenames `catalogue/README.md` and the bundle builder promise, in the
    order they are written.
    """
    assert list(lens_mass_maps.mass_map_products()) == FILENAMES


def test_mass_map_products_selects_the_tracer_hdus():
    """
    Each file's HDUs are the ``FITSTracer`` members, not strings: it is the enum
    ``extract_fits`` keys off, and the deflections file holds both components,
    y before x.
    """
    import autolens as al

    products = lens_mass_maps.mass_map_products()

    assert products["convergence.fits"] == [al.agg.fits_tracer.convergence]
    assert products["potential.fits"] == [al.agg.fits_tracer.potential]
    assert products["deflections.fits"] == [
        al.agg.fits_tracer.deflections_y,
        al.agg.fits_tracer.deflections_x,
    ]


def test_mass_map_products_covers_every_tracer_map():
    """
    The control on the test above: every member of the enum is written to some
    file. A map added upstream (magnification, say) fails here rather than
    quietly not being catalogued.
    """
    import autolens as al

    written = [
        hdu for hdus in lens_mass_maps.mass_map_products().values() for hdu in hdus
    ]

    assert sorted(hdu.name for hdu in written) == sorted(
        member.name for member in al.agg.fits_tracer
    )


# ---------------------------------------------------------------------------
# 2. The skip
# ---------------------------------------------------------------------------


def test_a_lens_with_all_three_files_is_skipped(tmp_path):
    for filename in FILENAMES:
        (tmp_path / filename).write_bytes(b"")

    assert lens_mass_maps.mass_maps_exist(tmp_path, FILENAMES) is True


@pytest.mark.parametrize("missing", FILENAMES)
def test_a_partially_written_lens_is_not_skipped(tmp_path, missing):
    """
    The one that matters: a lens interrupted between files must be rebuilt, or
    the bundle keeps a lens that is permanently short of a map.
    """
    for filename in FILENAMES:
        if filename != missing:
            (tmp_path / filename).write_bytes(b"")

    assert lens_mass_maps.mass_maps_exist(tmp_path, FILENAMES) is False


def test_a_lens_with_no_folder_is_not_skipped(tmp_path):
    assert lens_mass_maps.mass_maps_exist(tmp_path / "absent", FILENAMES) is False


# ---------------------------------------------------------------------------
# 3. The extraction
# ---------------------------------------------------------------------------


def test_extraction_reads_the_fits_not_the_same_named_json(result_directory):
    """
    The regression the producer's PyAutoFit floor exists for.

    ``af.AggregateFITS`` resolves ``FITSTracer`` to the name ``"tracer"``. Before
    the fix it did so through ``SearchOutput.value``, which searches ``jsons``
    before ``fits`` and so returned the ``files/tracer.json`` object — measured
    against PyAutoFit/PyAutoLens 2026.8.17.1, where it fails with
    ``'Tracer' object has no attribute 'index_of'``. The result directory here
    carries both files, so a stack without the fix fails this test.
    """
    import autolens as al

    hdu_list = _aggregate_fits(result_directory).extract_fits(
        hdus=[al.agg.fits_tracer.convergence]
    )

    assert [hdu.header.get("EXTNAME") for hdu in hdu_list] == [None, "CONVERGENCE"]
    assert hdu_list[1].data.shape == SHAPE
    # CONVERGENCE is index 1 of TRACER_EXTNAMES, so the fixture filled it with 1.0.
    assert np.array_equal(hdu_list[1].data, np.full(SHAPE, 1.0))


def test_extraction_preserves_extnames_shapes_and_header(result_directory):
    """
    Every requested map arrives under its own name, with its own data and the
    zoomed mask's header keys — the pixel scale a reader needs to re-project the
    map is carried through the copy.
    """
    import autolens as al

    hdu_list = _aggregate_fits(result_directory).extract_fits(
        hdus=[al.agg.fits_tracer.deflections_y, al.agg.fits_tracer.deflections_x]
    )

    assert [hdu.header.get("EXTNAME") for hdu in hdu_list] == [
        None,
        "DEFLECTIONS_Y",
        "DEFLECTIONS_X",
    ]
    assert [hdu.data.shape for hdu in hdu_list[1:]] == [SHAPE, SHAPE]
    assert np.array_equal(hdu_list[1].data, np.full(SHAPE, 3.0))
    assert np.array_equal(hdu_list[2].data, np.full(SHAPE, 4.0))
    assert hdu_list[1].header["PIXSCAX"] == 0.1
    assert hdu_list[1].header["PIXSCAY"] == 0.1


def test_an_empty_aggregator_raises_the_value_error_main_catches():
    """
    ``main`` skips a lens on ``ValueError``; an empty aggregator is the "no
    completed search under this tag" case, and must raise rather than write an
    empty file.
    """
    import autofit as af

    with pytest.raises(ValueError):
        af.AggregateFITS(aggregator=[])


def test_a_result_without_tracer_fits_raises_the_file_not_found_main_catches(
    result_directory,
):
    """
    A test-mode result, or one fitted with ``fits_tracer`` off, has no
    ``tracer.fits`` — only the shadowing JSON. That is a skip, not a crash, so it
    must surface as the ``FileNotFoundError`` ``main`` catches beside
    ``ValueError``.
    """
    import autolens as al

    (result_directory / "image" / "tracer.fits").unlink()

    with pytest.raises(FileNotFoundError):
        _aggregate_fits(result_directory).extract_fits(
            hdus=[al.agg.fits_tracer.convergence]
        )


# ---------------------------------------------------------------------------
# 4. The writer
# ---------------------------------------------------------------------------


def test_write_mass_maps_writes_every_product_with_its_extnames(tmp_path):
    """
    The writer's own contract, exercised through a stub: one file per product,
    named as the table says, each holding an empty PrimaryHDU followed by its
    maps under the extension names the enum carries.
    """
    from astropy.io import fits

    products = lens_mass_maps.mass_map_products()
    aggregate = _StubAggregate()

    written = lens_mass_maps.write_mass_maps(aggregate, tmp_path / "TileXYZ", products)

    assert [path.name for path in written] == FILENAMES
    assert aggregate.calls == list(products.values())

    expected = {
        "convergence.fits": ["CONVERGENCE"],
        "potential.fits": ["POTENTIAL"],
        "deflections.fits": ["DEFLECTIONS_Y", "DEFLECTIONS_X"],
    }
    for filename, extnames in expected.items():
        with fits.open(tmp_path / "TileXYZ" / filename) as hdu_list:
            assert [hdu.header.get("EXTNAME") for hdu in hdu_list] == [None] + extnames
            assert all(hdu.data.shape == SHAPE for hdu in hdu_list[1:])


def test_write_mass_maps_replaces_a_surviving_file(tmp_path):
    """
    ``overwrite=True``: a lens left with some of its three files (the skip did
    not fire) is rebuilt rather than erroring on the file that is there.
    """
    lens_dir = tmp_path / "TileXYZ"
    lens_dir.mkdir()
    (lens_dir / "potential.fits").write_bytes(b"not a fits file")

    written = lens_mass_maps.write_mass_maps(_StubAggregate(), lens_dir)

    assert len(written) == len(FILENAMES)
    assert (lens_dir / "potential.fits").read_bytes()[:6] == b"SIMPLE"
