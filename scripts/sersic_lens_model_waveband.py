"""
Sersic Lens Model + Multi-Waveband Pipeline
============================================

End-to-end driver: ``vis_lp`` (cached short-circuit if available) →
``fit_sersic`` (cached short-circuit if available) → ``fit_waveband``
(multi-band Sersic fits on the non-VIS bands, seeded from the VIS Sersic
result).

This is the SED chain: it produces the matched multi-band photometry that the
photometric-redshift and SED fitting downstream of the lens model consume.

Designed to be run under a dedicated output directory (``PYAUTO_OUTPUT_DIR``)
so the many per-band outputs do not bloat the main ``initial_lens_model``
output tree::

    PYAUTO_OUTPUT_DIR=output_sed python scripts/sersic_lens_model_waveband.py \
        --dataset=<name> --sample=<sample>

Pre-conditions for the cached short-circuits:

- the ``vis_lp`` result for the tile must already exist under
  ``<output_dir>/<sample>/<tile>/initial_lens_model/vis_lp/<hash>.zip``
- the Sersic result must already exist under
  ``<output_dir>/<sample>/<tile>/sersic_lens_model/vis/<hash>.zip``

If they are not present those stages run from scratch (slow). Typically you
would copy the existing results into the alternate output directory before
submitting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import util
from scripts.initial_lens_model import fit
from scripts.sersic_lens_model import fit_sersic
from scripts.lens_model_waveband import fit_waveband


if __name__ == "__main__":
    (
        sample_name,
        dataset_name,
        iterations_per_quick_update,
        number_of_cores,
        use_cpu,
        skip_pix,
    ) = util.parse_fit_args()

    # `skip_pix=True` is forced: the Sersic source prior is seeded from
    # `galaxies.source.bulge`, which the `vis_pix` stage replaces with a
    # pixelization.
    vis_lp_result = fit(
        dataset_name=dataset_name,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
        number_of_cores=number_of_cores,
        use_cpu=use_cpu,
        skip_pix=True,
    )

    sersic_result = fit_sersic(
        dataset_name=dataset_name,
        vis_result=vis_lp_result,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
    )

    fit_waveband(
        dataset_name=dataset_name,
        unique_tag="sersic_lens_model",
        vis_result=sersic_result,
        use_sersic_over_sampling=True,
        sample_name=sample_name,
        iterations_per_quick_update=iterations_per_quick_update,
    )
