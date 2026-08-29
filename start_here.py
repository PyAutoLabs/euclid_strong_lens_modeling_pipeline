"""
Euclid Pipeline: Start Here
===========================

This is a **thin shim**. The entry point for fitting a Euclid strong lens
dataset is ``scripts/initial_lens_model.py`` — that is the file to run, read
and edit; the README, ``AGENTS.md`` and the HPC submit scripts all name it.

``start_here.py`` is kept only so that older commands, bookmarks and scripts
that call ``python start_here.py`` keep working unchanged. It was previously a
diverged copy of ``scripts/initial_lens_model.py`` that had fallen behind it
(no pixelized-source stage); collapsing it into a shim removed that drift.

**Installation:** see https://github.com/PyAutoLabs/euclid_strong_lens_modeling_pipeline

**Running as a black box:** pass it the dataset name and it fits automatically.
Results and visualisations are written to ``output/``. For small samples,
browsing ``output/`` directly is sufficient. For large samples use the
``workflow/`` scripts to export .csv, .fits, and .png summaries via the
database aggregator.

**Questions:** contact James Nightingale on the Euclid Consortium Slack.

Usage
-----
::

    python start_here.py --dataset=<name> --sample=<sample>

Every command-line argument accepted by ``scripts/initial_lens_model.py``
(``--dataset``, ``--sample``, ``--iterations_per_quick_update``,
``--number_of_cores``, ``--use_cpu``, ``--skip_pix``) is accepted here and
forwarded unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import util
from scripts.initial_lens_model import fit

__all__ = ["fit"]


if __name__ == "__main__":
    (
        sample_name,
        dataset_name,
        iterations_per_quick_update,
        number_of_cores,
        use_cpu,
        skip_pix,
    ) = util.parse_fit_args()
    fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
        number_of_cores=number_of_cores,
        use_cpu=use_cpu,
        skip_pix=skip_pix,
    )
