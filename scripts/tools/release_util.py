"""
Euclid Pipeline: Shared Distribution-Zip Helpers
================================================

The zip, hashing and part-cutting code shared by the two scripts that write
catalogue distributions:

- ``scripts/tools/merge_catalogue.py`` (``--archive=chunked``: the merged
  catalogue's ``catalogue_csvs.zip`` + ``catalogue_part_NN.zip``);
- ``scripts/tools/package_release.py`` (the csv / vis / full release tiers).

Conventions both follow:

- FITS / PNG / JPG (and already-compressed) members are ``ZIP_STORED``; they
  barely compress. Text members are ``ZIP_DEFLATED`` at level 6.
- Zips are written to a hidden ``.<name>.tmp`` sibling and renamed into place,
  so an interrupted write never leaves a truncated zip under its real name.
- Every zip is re-opened and CRC-tested (``ZipFile.testzip``) before the step
  that wrote it succeeds.
- ``split_parts`` cuts an ordered tile list into EXACTLY k contiguous parts of
  near-equal byte totals; ``plan_greedy`` packs up to a byte cap instead.

No CLI: import it from a sibling script after
``sys.path.insert(0, str(Path(__file__).resolve().parent))``.
"""

import hashlib
import zipfile
from pathlib import Path

STORED_SUFFIXES = {".fits", ".png", ".jpg", ".jpeg", ".zip", ".gz"}

HASH_BLOCK = 16 * 1024 * 1024


def compression_for(path):
    """``ZIP_STORED`` for image / FITS / compressed members, else ``ZIP_DEFLATED``."""
    suffix = Path(str(path)).suffix.lower()
    return zipfile.ZIP_STORED if suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED


def sha256(path: Path):
    """Hex SHA-256 of a file, read in 16 MB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes):
    return hashlib.sha256(data).hexdigest()


def split_parts(tiles, sizes, n_parts: int):
    """Cut ``tiles`` (in order) into EXACTLY ``n_parts`` contiguous, non-empty
    groups whose byte totals are near-equal.

    Boundary ``i`` falls where the running total crosses ``i * total / n_parts``
    (a tile joins the earlier group when its midpoint lies below the target),
    so each group is within one largest-folder of ``total / n_parts``.
    """
    n = len(tiles)
    if n_parts < 1:
        raise ValueError(f"n_parts must be >= 1, got {n_parts}")
    if n < n_parts:
        raise ValueError(f"cannot cut {n} lens folders into {n_parts} non-empty parts")
    total = sum(sizes)
    parts, start = [], 0
    for i in range(1, n_parts):
        target = total * i / n_parts
        cum = sum(sizes[:start]) + sizes[start]
        end = start + 1
        # Leave at least one tile for every later part.
        while end < n - (n_parts - i) and cum + sizes[end] / 2 <= target:
            cum += sizes[end]
            end += 1
        parts.append(list(tiles[start:end]))
        start = end
    parts.append(list(tiles[start:]))
    return parts


def plan_greedy(tiles, sizes, cap_bytes: int):
    """Pack ``tiles`` (in order) into parts of at most ``cap_bytes`` each; a
    single tile larger than the cap gets a part of its own."""
    parts, current, size = [], [], 0
    for tile, tile_size in zip(tiles, sizes):
        if current and size + tile_size > cap_bytes:
            parts.append(current)
            current, size = [], 0
        current.append(tile)
        size += tile_size
    if current:
        parts.append(current)
    return parts


def plan(tiles, sizes, cap_bytes: int, n_parts=None):
    """``split_parts`` with ``n_parts``, else ``plan_greedy`` at ``cap_bytes``."""
    if n_parts:
        return split_parts(tiles, sizes, n_parts)
    return plan_greedy(tiles, sizes, cap_bytes)


def part_name(index: int, n_parts: int):
    """``catalogue_part_NN.zip``, zero-padded to at least two digits."""
    width = max(2, len(str(n_parts)))
    return f"catalogue_part_{index:0{width}d}.zip"


def write_zip_files(zip_path: Path, members):
    """Write ``members`` (``(source_path, arcname)`` pairs) to ``zip_path``
    through a hidden temp file; compression per ``compression_for``."""
    zip_path = Path(zip_path)
    staged = zip_path.with_name(f".{zip_path.name}.tmp")
    with zipfile.ZipFile(staged, "w", allowZip64=True) as archive:
        for source, arcname in members:
            kind = compression_for(source)
            archive.write(
                source,
                arcname=arcname,
                compress_type=kind,
                compresslevel=6 if kind == zipfile.ZIP_DEFLATED else None,
            )
    staged.replace(zip_path)
    return zip_path.name


def add_file_hashed(archive: zipfile.ZipFile, source: Path, arcname: str):
    """Stream ``source`` into ``archive`` as ``arcname`` and return
    ``(bytes, sha256)`` of its contents, reading the file once."""
    info = zipfile.ZipInfo.from_file(source, arcname)
    info.compress_type = compression_for(source)
    digest = hashlib.sha256()
    size = 0
    with Path(source).open("rb") as src, archive.open(info, "w", force_zip64=True) as dest:
        for block in iter(lambda: src.read(HASH_BLOCK), b""):
            digest.update(block)
            dest.write(block)
            size += len(block)
    return size, digest.hexdigest()


def add_bytes_hashed(archive: zipfile.ZipFile, data: bytes, arcname: str, like: Path = None):
    """Write in-memory ``data`` as ``arcname`` (compression chosen from the
    arcname's suffix; timestamp from ``like`` when given); return
    ``(bytes, sha256)``."""
    if like is not None:
        info = zipfile.ZipInfo.from_file(like, arcname)
        info.file_size = len(data)
    else:
        info = zipfile.ZipInfo(arcname)
    info.compress_type = compression_for(arcname)
    with archive.open(info, "w", force_zip64=True) as dest:
        dest.write(data)
    return len(data), sha256_bytes(data)


def verify_zip(zip_path: Path):
    """Re-open and CRC-test a zip; return ``(name, sha256, member_names,
    {member: file_size})``. Raises on a CRC failure."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"{zip_path}: CRC failure on {bad}")
        infos = archive.infolist()
    names = [info.filename for info in infos]
    sizes = {info.filename: info.file_size for info in infos}
    return zip_path.name, sha256(zip_path), names, sizes


def write_sha256sums(path: Path, digests):
    """``SHA256SUMS`` in ``sha256sum -c`` format from ``{name: digest}``, sorted."""
    with Path(path).open("w") as stream:
        for name in sorted(digests):
            stream.write(f"{digests[name]}  {name}\n")
