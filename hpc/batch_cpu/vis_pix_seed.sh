#!/bin/bash
#
# STAGE_SETUP for the campaign's `vis_pix` stage (hpc/campaign/dr1.yaml): check the
# lens has its finished vis_lp result before the pixelized fit starts.
#
# `scripts/initial_lens_model.py --stage=vis_pix` loads the vis_lp result from disk
# and fails if there is none. Checking first, here, records the task as failure class
# `missing_seed` (exit 3: resubmit vis_lp first) rather than as a Python exception. It
# needs exactly one zip in output/<sample>/<dataset>/initial_lens_model/vis_lp/, the
# guard the retired submit_initial_lens_model_vis_pix_* scripts ran inline.
#
# hpc/batch_cpu/stage runs this from the project root with the manifest fields exported
# as `sample` and `dataset`.

set -euo pipefail

sample="${sample:?sample is not set (run this through hpc/batch_cpu/stage)}"
dataset="${dataset:?dataset is not set (run this through hpc/batch_cpu/stage)}"

seed_source="output/${sample}/${dataset}/initial_lens_model/vis_lp"

shopt -s nullglob
seed_zips=("${seed_source}"/*.zip)
if [[ ${#seed_zips[@]} -ne 1 ]]; then
    echo "ERROR: expected one completed vis_lp zip in ${seed_source}, found ${#seed_zips[@]}" >&2
    exit 3
fi
echo "vis_lp seed: ${seed_zips[0]}"
