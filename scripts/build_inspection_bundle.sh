#!/usr/bin/env bash
#
# Euclid Pipeline: Inspection Bundle Builder
# ==========================================
#
# Runs the seven catalogue producers in dependency order and assembles a
# per-lens inspection bundle for one sample. Idempotent: safe to re-run as more
# results land — already-built lenses are skipped by each stage.
#
# Output layout (inspect/<sample>[_<run_tag>]/):
#
#   lens_mass.csv                        # master CSVs, one row per lens
#   lens_sersic.csv
#   source_sersic.csv
#   magnitudes.csv                       # one row per (lens, waveband)
#   <dataset_name>/
#       vis_lp_fit.png                   # collected by build_inspect.py
#       vis_pix_fit.png
#       vis_lp_image_with_positions.png
#       rgb.png
#       segmentation.png
#       fit_sersic.png
#       pre_psf.fits                     # lens light + lensed source, pre-PSF
#       model.fits                       # the same, post-PSF convolution
#       fit_multi_wavelength.png
#       lens_mass.csv                    # this lens's row of each master CSV
#       lens_sersic.csv
#       source_sersic.csv
#       magnitudes.csv
#
# Stages 6 and 7 read a separate results tree (default `output_sed`) holding the
# multi-band SED fits produced by running the waveband scripts with
# PYAUTO_OUTPUT_DIR=output_sed. They are skipped when that tree has no directory
# for the sample.
#
# Usage:
#   bash scripts/build_inspection_bundle.sh [sample] [run_tag]
#   bash scripts/build_inspection_bundle.sh q1_walsmley
#   bash scripts/build_inspection_bundle.sh dr1_prelim_grade_ab run250
#
# Environment:
#   OUTPUT_DIR        results tree for stages 1-5      (default: output)
#   SED_OUTPUT_DIR    results tree for stages 6-7      (default: output_sed)
#   SKIP_SED=1        skip stages 6-7 even if present  (default: 0)
#   CREATE_ARCHIVE=0  do not tar the bundle at the end (default: 1)
#   DATASET_PREFIX    only collect datasets with this name prefix (default: all)

set -euo pipefail

SAMPLE="${1:-q1_walsmley}"
RUN_TAG="${2:-}"
if [ -n "$RUN_TAG" ]; then
    INSPECT_DIR="inspect/${SAMPLE}_${RUN_TAG}"
else
    INSPECT_DIR="inspect/${SAMPLE}"
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd -- "$SCRIPT_DIR/.." && pwd )"

# On HPC, activate.sh activates the shared PyAuto venv under /mnt/ral and puts
# its checkouts on PYTHONPATH. It is HPC-only — sourcing it elsewhere fails — so
# it is used only when that venv is actually present and the caller has not
# already set up an environment (PYAUTO_ROOT is exported by a worktree
# activate.sh). Everywhere else the ambient install is used.
HPC_BASE="${PYAUTO_HPC_BASE:-/mnt/ral/jnightin/PyAuto}"
if [ -z "${PYAUTO_ROOT:-}" ] && [ -d "$HPC_BASE" ] && [ -f "$PROJECT_ROOT/activate.sh" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/activate.sh"
fi

# Writable caches for numba and matplotlib (see AGENTS.md).
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

OUTPUT_DIR="${OUTPUT_DIR:-output}"
SED_OUTPUT_DIR="${SED_OUTPUT_DIR:-output_sed}"
DATASET_PREFIX="${DATASET_PREFIX:-}"

cd "$PROJECT_ROOT"

mkdir -p "$INSPECT_DIR"

echo "==> [1/7] inspection PNGs (scripts/build_inspect.py)"
python "$PROJECT_ROOT/scripts/build_inspect.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR" \
    --dataset_prefix="$DATASET_PREFIX"

echo "==> [2/7] deblended FITS (catalogue/scripts/deblending.py)"
python "$PROJECT_ROOT/catalogue/scripts/deblending.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR"

echo "==> [3/7] lens mass CSV (catalogue/scripts/lens_mass.py)"
python "$PROJECT_ROOT/catalogue/scripts/lens_mass.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR"

echo "==> [4/7] lens Sersic CSV (catalogue/scripts/lens_sersic.py)"
python "$PROJECT_ROOT/catalogue/scripts/lens_sersic.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR"

echo "==> [5/7] source Sersic CSV (catalogue/scripts/source_sersic.py)"
python "$PROJECT_ROOT/catalogue/scripts/source_sersic.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR"

# Stages 6+7 read the multi-band SED tree. Set SKIP_SED=1 to refresh only the
# stable products while SED jobs are still running.
SED_OUTPUT_PATH="${PROJECT_ROOT}/${SED_OUTPUT_DIR}"
if [ "${SKIP_SED:-0}" = "1" ]; then
    echo "==> [6/7,7/7] skipped — SKIP_SED=1"
elif [ -d "$SED_OUTPUT_PATH/$SAMPLE" ]; then
    echo "==> [6/7] multi-wavelength PNG (catalogue/scripts/multi_wavelength.py)"
    python "$PROJECT_ROOT/catalogue/scripts/multi_wavelength.py" \
        --sample="$SAMPLE" \
        --output_path="$SED_OUTPUT_DIR" \
        --inspect_dir="$INSPECT_DIR"

    echo "==> [7/7] magnitudes CSV (catalogue/scripts/magnitudes.py)"
    python "$PROJECT_ROOT/catalogue/scripts/magnitudes.py" \
        --sample="$SAMPLE" \
        --output_path="$SED_OUTPUT_DIR" \
        --inspect_dir="$INSPECT_DIR"
else
    echo "==> [6/7,7/7] skipped — no $SED_OUTPUT_PATH/$SAMPLE/ (no SED runs yet)"
fi

echo ""
echo "Done. Bundle at: $INSPECT_DIR"
echo "  $(ls "$INSPECT_DIR" | wc -l) entries"

# Archive only when explicitly enabled — a full DR1 bundle is tens of GB, so an
# incremental refresh should not rebuild the tarball.
if [ "${CREATE_ARCHIVE:-1}" = "1" ]; then
    TAR_NAME="${INSPECT_DIR//\//_}.tar.gz"
    echo "==> Archiving -> $TAR_NAME"
    tar czf "$TAR_NAME" "$INSPECT_DIR/"
    echo "  $(ls -lh "$TAR_NAME" | awk '{print $5}')  $TAR_NAME"
else
    echo "==> Archive skipped — CREATE_ARCHIVE=0"
fi
