"""
Write the campaign manifest: one "<sample> <dataset>" line per lens, sorted.

The DR1 campaign (hpc/campaign/dr1.yaml) runs every lens listed in its manifest,
hpc/campaign/dr1_manifest.txt. A line's 0-based number is the lens's campaign index,
so build the manifest once per campaign and do not reorder it while the campaign runs.

A lens is a directory dataset/<sample>/<dataset>/ that holds <dataset>.fits. The
dataset folder is the one project_paths.dataset_root() resolves (DATASET_ROOT, then
hpc/sync.conf, then <project>/dataset), or --dataset-root.

Usage, from the project root:

    python hpc/tools/build_campaign_manifest.py --sample dr1_sep1_rest
    python hpc/tools/build_campaign_manifest.py --sample dr1_sep1 --sample dr1_v2_new \\
        --exclude-file hpc/campaign/exclude.txt --output hpc/campaign/dr1_manifest.txt

An exclude file lists one lens per line, either "<sample> <dataset>" or just
"<dataset>" (that dataset in any sample); blank lines and # comments are ignored.

Stdlib only.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def read_excludes(path):
    """The excluded lenses: a set of (sample, dataset) pairs and a set of bare datasets."""
    pairs, datasets = set(), set()
    if path is None:
        return pairs, datasets
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) == 1:
            datasets.add(fields[0])
        else:
            pairs.add((fields[0], fields[1]))
    return pairs, datasets


def manifest_lines(dataset_root, samples, exclude_file=None):
    """
    The manifest's lines, sorted by sample then dataset.

    Raises
    ------
    FileNotFoundError
        If a sample has no folder under ``dataset_root``.
    """
    pairs, bare = read_excludes(exclude_file)
    lines = []
    for sample in samples:
        sample_dir = Path(dataset_root) / sample
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"No sample folder {sample_dir}")
        for directory in sample_dir.iterdir():
            name = directory.name
            if not (directory / f"{name}.fits").is_file():
                continue
            if (sample, name) in pairs or name in bare:
                continue
            lines.append((sample, name))
    return [f"{sample} {name}" for sample, name in sorted(lines)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sample", action="append", required=True,
        help="Sample folder under the dataset root; repeat for more than one.",
    )
    parser.add_argument(
        "--output", default="hpc/campaign/dr1_manifest.txt",
        help="Manifest to write, relative to the project root (default: %(default)s).",
    )
    parser.add_argument(
        "--exclude-file", default=None,
        help='Lenses to leave out: "<sample> <dataset>" or "<dataset>" per line.',
    )
    parser.add_argument(
        "--dataset-root", default=None,
        help="Folder holding <sample>/<dataset>/ (default: project_paths.dataset_root()).",
    )
    arguments = parser.parse_args(argv)

    if arguments.dataset_root is None:
        import project_paths

        dataset_root = project_paths.dataset_root()
    else:
        dataset_root = Path(arguments.dataset_root)

    lines = manifest_lines(dataset_root, arguments.sample, arguments.exclude_file)
    output = Path(arguments.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{line}\n" for line in lines))
    print(f"{len(lines)} lenses -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
