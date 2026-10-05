"""
``--n_parts=k`` must cut the distribution into EXACTLY k self-contained parts.

``scripts/tools/merge_catalogue.py`` packs whole lens folders into
``catalogue_part_NN.zip``. With ``--n_parts`` (driver ``PARTS=<k>``) the tile
list is cut into k contiguous groups of near-equal byte totals. These tests pin
the contract on ``split_parts`` (pure) and on ``plan_parts`` over real folders:
exactly k parts, every tile in exactly one part, order preserved (contiguous),
and each part within one largest folder of ``total / k``.
"""

import importlib.util
import random
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_merge():
    path = PROJECT_ROOT / "scripts" / "tools" / "merge_catalogue.py"
    spec = importlib.util.spec_from_file_location("_merge_catalogue_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


merge = _load_merge()


def _check(parts, tiles, sizes, k):
    assert len(parts) == k
    assert all(parts), "empty part"
    assert [t for part in parts for t in part] == list(tiles)  # every tile once, in order
    size_of = dict(zip(tiles, sizes))
    totals = [sum(size_of[t] for t in part) for part in parts]
    mean = sum(sizes) / k
    assert max(abs(total - mean) for total in totals) <= max(sizes)


@pytest.mark.parametrize("k", [1, 2, 3, 7, 10])
def test_split_parts_uniform(k):
    tiles = [f"Tile{i:05d}" for i in range(110)]
    sizes = [7_750_000] * 110
    parts = merge.split_parts(tiles, sizes, k)
    _check(parts, tiles, sizes, k)
    counts = [len(p) for p in parts]
    assert max(counts) - min(counts) <= 1


@pytest.mark.parametrize("seed", range(20))
def test_split_parts_random_sizes(seed):
    rng = random.Random(seed)
    n = rng.randint(10, 500)
    k = rng.randint(1, min(n, 15))
    tiles = [f"Tile{i:05d}" for i in range(n)]
    sizes = [rng.randint(1, 50_000_000) for _ in range(n)]
    _check(merge.split_parts(tiles, sizes, k), tiles, sizes, k)


def test_split_parts_skewed_still_exactly_k():
    # One huge folder at the front must not swallow the remaining parts.
    tiles = [f"Tile{i}" for i in range(6)]
    sizes = [10**9, 1, 1, 1, 1, 1]
    parts = merge.split_parts(tiles, sizes, 5)
    assert len(parts) == 5 and all(parts)
    assert [t for p in parts for t in p] == tiles


def test_split_parts_n_equals_k_and_too_few():
    tiles = ["A", "B", "C"]
    assert merge.split_parts(tiles, [5, 1, 9], 3) == [["A"], ["B"], ["C"]]
    with pytest.raises(ValueError):
        merge.split_parts(tiles, [1, 1, 1], 4)
    with pytest.raises(ValueError):
        merge.split_parts(tiles, [1, 1, 1], 0)


def test_plan_parts_n_parts_on_folders(tmp_path):
    rng = random.Random(0)
    tiles = [f"Tile{i:04d}" for i in range(40)]
    for tile in tiles:
        folder = tmp_path / tile
        folder.mkdir()
        (folder / "model.fits").write_bytes(b"x" * rng.randint(100, 5000))
        (folder / "lens_mass.csv").write_text("a,b\n1,2\n")
    (tmp_path / "_build").mkdir()  # never a lens folder
    (tmp_path / "lens_mass.csv").write_text("root file\n")
    sizes = [merge.folder_size(tmp_path / t) for t in tiles]

    parts = merge.plan_parts(tmp_path, cap_bytes=1, n_parts=10)
    _check(parts, tiles, sizes, 10)

    # Without n_parts the size cap still rules (cap of 1 byte: one folder per part).
    assert len(merge.plan_parts(tmp_path, cap_bytes=1)) == len(tiles)
