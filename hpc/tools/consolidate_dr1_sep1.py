#!/usr/bin/env python3
"""
Consolidate every DR1 lens into the single sample ``dr1_sep1`` (2026-10-07).

Runs ON RAL from the project root (``PROJECT_PATH``, default
``/mnt/ral/jnightin/active/euclid_dr1``). Pure stdlib. All paths in the manifest are
relative to the project root.

Ops, in manifest order:

1. ``unlink``   every symlink in ``dataset/dr1_sep1`` (``extra`` = its target).
2. ``mv``       the real (stale 2026-09-11 ten-lens) dirs in ``dataset/dr1_sep1``
                -> ``archive_dr1_sep1_ten_20260911/dataset/<t>``.
3. ``mv``       every dir of ``dataset/dr1_sep1_rest`` and ``dataset/dr1_v2_new``
                -> ``dataset/dr1_sep1/<t>``; their non-dir files
                -> ``dataset/provenance_dr1_sep1/<old_sample>/<file>``.
4. ``copytree`` every dir of ``archive_top1000_20260925/dataset_dr1_sep1_top1000``
                -> ``dataset/dr1_sep1/<t>`` (cp -a semantics; the archive is
                never modified); its non-dir files are ``copyfile``d to
                ``dataset/provenance_dr1_sep1/dr1_sep1_top1000/``.
5. ``mv``       every dir of ``output{,_sed}/{dr1_sep1_rest,dr1_v2_new}``
                -> ``output{,_sed}/dr1_sep1/<t>``; non-dir entries
                -> ``output{,_sed}/provenance_dr1_sep1/<old_sample>/``.
6. ``rmdir``    the emptied ``dr1_sep1_rest`` / ``dr1_v2_new`` dirs.

Modes::

    --plan              scan the trees, write the manifest (moves.tsv)
    --dry-run           validate the manifest against the filesystem, simulating
                        the ops in order: per-op counts, collisions, missing
                        sources, post-state counts
    --execute --yes     apply; resumable (a finished op is skipped), never
                        overwrites, logs every op to moves.log
    --revert --yes      undo in reverse order (rmtree only of the top1000 copies)

The manifest is TSV: ``op<TAB>src<TAB>dst<TAB>extra``.
"""
import argparse
import collections
import os
import shutil
import sys
import time

ROOT = os.environ.get("PROJECT_PATH", "/mnt/ral/jnightin/active/euclid_dr1")
MANIFEST_DIR = "hpc/run_manifests/consolidate_dr1_sep1_20261007"
MANIFEST = f"{MANIFEST_DIR}/moves.tsv"
LOG = f"{MANIFEST_DIR}/moves.log"

TARGET = "dr1_sep1"
OLD_SAMPLES = ("dr1_sep1_rest", "dr1_v2_new")
TOP1000_ARCHIVE = "archive_top1000_20260925/dataset_dr1_sep1_top1000"
TEN_ARCHIVE = "archive_dr1_sep1_ten_20260911/dataset"
PROV = "provenance_dr1_sep1"
TILE_PREFIXES = ("Tile", "15arcsec_near_RADEC_Tile")


def scan(path):
    """One scandir pass: (dirs, symlinks{name: target}, files)."""
    dirs, links, files = [], {}, []
    with os.scandir(path) as it:
        for e in it:
            if e.is_symlink():
                links[e.name] = os.readlink(e.path)
            elif e.is_dir(follow_symlinks=False):
                dirs.append(e.name)
            else:
                files.append(e.name)
    return sorted(dirs), links, sorted(files)


def plan():
    ops, notes = [], []
    j = os.path.join

    # 1 + 2: dataset/dr1_sep1
    d, links, files = scan(f"dataset/{TARGET}")
    if files:
        sys.exit(f"unexpected non-dir files in dataset/{TARGET}: {files[:5]}")
    by_parent = collections.Counter(os.path.dirname(t.rstrip("/")) for t in links.values())
    notes.append(f"dataset/{TARGET}: {len(d)} real dirs, {len(links)} symlinks {dict(by_parent)}")
    for name in sorted(links):
        ops.append(("unlink", f"dataset/{TARGET}/{name}", "-", links[name]))
    for name in d:
        ops.append(("mv", f"dataset/{TARGET}/{name}", f"{TEN_ARCHIVE}/{name}", "stale_20260911"))

    # 3: dataset old samples
    tile_dirs = {}
    for s in OLD_SAMPLES:
        sd, sl, sf = scan(f"dataset/{s}")
        if sl:
            sys.exit(f"unexpected symlinks in dataset/{s}: {list(sl)[:5]}")
        # Only tile-named dirs are tiles; dr1_sep1_rest/positions_rewrite_priority250/
        # is a log directory and goes to provenance with the files.
        tiles = [n for n in sd if n.startswith(TILE_PREFIXES)]
        other = sorted(set(sd) - set(tiles))
        tile_dirs[s] = set(tiles)
        notes.append(f"dataset/{s}: {len(tiles)} tile dirs, other dirs {other}, {len(sf)} files")
        for name in tiles:
            ops.append(("mv", f"dataset/{s}/{name}", f"dataset/{TARGET}/{name}", s))
        for name in other + sf:
            ops.append(("mv", f"dataset/{s}/{name}", f"dataset/{PROV}/{s}/{name}", s))

    # 4: top1000 copies
    ad, al, af = scan(TOP1000_ARCHIVE)
    notes.append(f"{TOP1000_ARCHIVE}: {len(ad)} dirs, {len(al)} symlinks, {len(af)} files {af[:10]}")
    if al:
        sys.exit(f"unexpected symlinks in {TOP1000_ARCHIVE}: {list(al)[:5]}")
    for name in ad:
        ops.append(("copytree", f"{TOP1000_ARCHIVE}/{name}", f"dataset/{TARGET}/{name}", "dr1_sep1_top1000"))
    for name in af:
        ops.append(("copyfile", f"{TOP1000_ARCHIVE}/{name}", f"dataset/{PROV}/dr1_sep1_top1000/{name}", "dr1_sep1_top1000"))

    # 5: output trees
    for tree in ("output", "output_sed"):
        for s in OLD_SAMPLES:
            od, ol, of = scan(f"{tree}/{s}")
            notes.append(f"{tree}/{s}: {len(od)} dirs, {len(ol)} symlinks, {len(of)} files {of[:10]}")
            for name in od:
                ops.append(("mv", f"{tree}/{s}/{name}", f"{tree}/{TARGET}/{name}", s))
            for name in sorted(list(ol) + of):
                ops.append(("mv", f"{tree}/{s}/{name}", f"{tree}/{PROV}/{s}/{name}", s))

    # 6: rmdir emptied sample dirs
    for tree in ("dataset", "output", "output_sed"):
        for s in OLD_SAMPLES:
            ops.append(("rmdir", f"{tree}/{s}", "-", "-"))

    # cross-checks on names
    rest_links = {n for n, t in links.items() if t.rstrip("/").startswith("../dr1_sep1_rest/")}
    top_links = {n for n, t in links.items() if t.rstrip("/").startswith("../dr1_sep1_top1000/")}
    rest_dirs = tile_dirs["dr1_sep1_rest"]
    notes.append(f"check: rest links ⊆ rest dirs: {rest_links <= rest_dirs}; "
                 f"rest dirs - rest links = {len(rest_dirs - rest_links)} (all real dirs in {TARGET}: {rest_dirs - rest_links == set(d)})")
    notes.append(f"check: top1000 links == archive dirs: {top_links == set(ad)} ({len(top_links)} vs {len(ad)})")

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    with open(MANIFEST, "w") as f:
        f.write("op\tsrc\tdst\textra\n")
        for op in ops:
            f.write("\t".join(op) + "\n")
    for n in notes:
        print(n)
    c = collections.Counter(o[0] for o in ops)
    print(f"manifest {MANIFEST}: {len(ops)} ops {dict(c)}")


def read_manifest():
    with open(MANIFEST) as f:
        next(f)
        return [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]


class Sim:
    """Directory listings cached once per parent, mutated as ops are simulated."""

    def __init__(self):
        self.cache = {}

    def listing(self, parent):
        if parent not in self.cache:
            try:
                with os.scandir(parent) as it:
                    self.cache[parent] = {e.name: ("l" if e.is_symlink() else "d" if e.is_dir(follow_symlinks=False) else "f") for e in it}
            except FileNotFoundError:
                self.cache[parent] = None  # parent missing: created on execute
        return self.cache[parent]

    def kind(self, p):
        lst = self.listing(os.path.dirname(p))
        return None if lst is None else lst.get(os.path.basename(p))

    def add(self, p, k):
        par = os.path.dirname(p)
        if self.listing(par) is None:
            self.cache[par] = {}
        self.cache[par][os.path.basename(p)] = k

    def remove(self, p):
        self.cache[os.path.dirname(p)].pop(os.path.basename(p), None)


def dry_run():
    ops = read_manifest()
    sim, errors, counts = Sim(), [], collections.Counter()
    for op, src, dst, extra in ops:
        counts[op] += 1
        if op == "unlink":
            if sim.kind(src) != "l":
                errors.append(f"unlink: not a symlink: {src}")
            else:
                sim.remove(src)
        elif op in ("mv", "copytree", "copyfile"):
            k = sim.kind(src)
            want = {"copytree": ("d",), "copyfile": ("f",)}.get(op, ("d", "f", "l"))
            if k not in want:
                errors.append(f"{op}: source missing/wrong type ({k}): {src}")
                continue
            if sim.kind(dst) is not None:
                errors.append(f"{op}: COLLISION dst exists: {dst}")
                continue
            sim.add(dst, k)
            if op == "mv":
                sim.remove(src)
        elif op == "rmdir":
            lst = sim.listing(src)
            if lst:
                errors.append(f"rmdir: not empty after moves ({len(lst)} left, e.g. {sorted(lst)[:3]}): {src}")
    print(f"dry-run over {len(ops)} ops {dict(counts)}")
    for tree in ("dataset", "output", "output_sed"):
        lst = sim.listing(f"{tree}/{TARGET}") or {}
        print(f"post-state {tree}/{TARGET}: {sum(v == 'd' for v in lst.values())} dirs, "
              f"{sum(v == 'l' for v in lst.values())} symlinks, {sum(v == 'f' for v in lst.values())} files")
    print(f"errors: {len(errors)}")
    for e in errors[:30]:
        print("  " + e)
    return not errors


def log(fh, *fields):
    fh.write("\t".join([time.strftime("%Y-%m-%dT%H:%M:%S"), *fields]) + "\n")
    fh.flush()


def execute():
    ops = read_manifest()
    done = skipped = 0
    with open(LOG, "a") as fh:
        log(fh, "BEGIN", "execute", str(len(ops)))
        for i, (op, src, dst, extra) in enumerate(ops):
            if op == "unlink":
                if not os.path.lexists(src):
                    skipped += 1; continue
                if not os.path.islink(src):
                    sys.exit(f"refuse: {src} is no longer a symlink")
                os.unlink(src)
            elif op == "mv":
                if not os.path.lexists(src) and os.path.lexists(dst):
                    skipped += 1; continue
                if os.path.lexists(dst):
                    sys.exit(f"refuse to overwrite {dst}")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.rename(src, dst)
            elif op == "copytree":
                if os.path.lexists(dst):
                    skipped += 1; continue
                part = dst + ".partial"
                if os.path.lexists(part):
                    shutil.rmtree(part)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copytree(src, part, symlinks=True, copy_function=shutil.copy2)
                os.rename(part, dst)
            elif op == "copyfile":
                if os.path.lexists(dst):
                    skipped += 1; continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
            elif op == "rmdir":
                if not os.path.lexists(src):
                    skipped += 1; continue
                os.rmdir(src)  # raises if not empty
            log(fh, "OK", op, src, dst)
            done += 1
            if done % 1000 == 0:
                print(f"{i + 1}/{len(ops)} ops ({done} done, {skipped} skipped)", flush=True)
        log(fh, "END", "execute", f"done={done}", f"skipped={skipped}")
    print(f"execute: {done} done, {skipped} already done")


def revert():
    ops = read_manifest()
    n = 0
    with open(LOG, "a") as fh:
        log(fh, "BEGIN", "revert", str(len(ops)))
        for op, src, dst, extra in reversed(ops):
            if op == "rmdir" and not os.path.lexists(src):
                os.makedirs(src)
            elif op == "mv" and os.path.lexists(dst) and not os.path.lexists(src):
                os.makedirs(os.path.dirname(src), exist_ok=True)
                os.rename(dst, src)
            elif op == "copytree" and os.path.lexists(dst):
                if not os.path.isdir(src):
                    sys.exit(f"refuse: archive source {src} missing, not removing {dst}")
                shutil.rmtree(dst)
            elif op == "copyfile" and os.path.lexists(dst):
                os.unlink(dst)
            elif op == "unlink" and not os.path.lexists(src):
                os.symlink(extra, src)
            else:
                continue
            log(fh, "REVERTED", op, src, dst)
            n += 1
        log(fh, "END", "revert", str(n))
    print(f"revert: {n} ops undone")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    g.add_argument("--revert", action="store_true")
    ap.add_argument("--yes", action="store_true", help="required for --execute / --revert")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.plan:
        plan()
    elif a.dry_run:
        sys.exit(0 if dry_run() else 1)
    else:
        if not a.yes:
            sys.exit("--execute/--revert need --yes")
        # A fresh execute is gated on a clean dry run; a resumed one (moves.log
        # exists) is not, since its finished sources are already gone.
        if a.execute and not os.path.exists(LOG) and not dry_run():
            sys.exit("dry-run failed; not executing")
        execute() if a.execute else revert()


if __name__ == "__main__":
    main()
