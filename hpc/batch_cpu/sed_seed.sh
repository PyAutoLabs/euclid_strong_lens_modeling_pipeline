#!/bin/bash
#
# STAGE_SETUP for the campaign's `sed` stage (hpc/campaign/dr1.yaml): copy the lens's
# finished vis_lp result into the SED output tree, so the SED chain
# (scripts/sersic_lens_model_waveband.py) loads it instead of fitting vis_lp again.
#
# hpc/batch_cpu/stage runs this from the project root, after it has read the manifest
# line, with the line's fields exported as `sample` and `dataset` (and passed as
# --sample=... --dataset=...). It needs exactly one zip in
#   output/<sample>/<dataset>/initial_lens_model/vis_lp/
# and copies it to
#   ${PYAUTO_OUTPUT_DIR}/<sample>/<dataset>/initial_lens_model/vis_lp/
# Zero zips (vis_lp not finished) or more than one (which one is the seed?) exits 3,
# which the stage records as failure class `missing_seed`: resubmit vis_lp first.
#
# This is the seed copy the retired submit_sersic_waveband_* scripts did inline.

set -euo pipefail

sample="${sample:?sample is not set (run this through hpc/batch_cpu/stage)}"
dataset="${dataset:?dataset is not set (run this through hpc/batch_cpu/stage)}"
output_dir="${PYAUTO_OUTPUT_DIR:?PYAUTO_OUTPUT_DIR is not set (the SED tree, e.g. output_sed)}"

seed_source="output/${sample}/${dataset}/initial_lens_model/vis_lp"
seed_target="${output_dir}/${sample}/${dataset}/initial_lens_model/vis_lp"

shopt -s nullglob
seed_zips=("${seed_source}"/*.zip)
if [[ ${#seed_zips[@]} -ne 1 ]]; then
    echo "ERROR: expected one completed vis_lp zip in ${seed_source}, found ${#seed_zips[@]}" >&2
    exit 3
fi

mkdir -p "${seed_target}"
cp -a "${seed_zips[0]}" "${seed_target}/"
echo "vis_lp seed: ${seed_zips[0]} -> ${seed_target}/"
