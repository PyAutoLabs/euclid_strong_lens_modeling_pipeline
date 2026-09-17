#!/usr/bin/env bash
#
# Euclid Pipeline: Inspection Bundle Builder
# ==========================================
#
# Runs the nine catalogue producers in dependency order and assembles a
# per-lens inspection bundle for one sample. Idempotent: safe to re-run as more
# results land — already-built lenses are skipped by each stage.
#
# Output layout (inspect/<sample>[_<run_tag>]/):
#
#   lens_mass.csv                        # master CSVs, one row per lens
#   lens_sersic.csv
#   source_sersic.csv
#   magnitudes.csv                       # one row per (lens, waveband)
#   astrometric_offsets.csv              # one row per (lens, non-VIS waveband)
#   <dataset_name>/
#       vis_lp_fit.png                   # collected by build_inspect.py
#       vis_pix_fit.png
#       vis_lp_image_with_positions.png
#       rgb.png
#       segmentation.png
#       fit_sersic.png
#       coolest.json                     # COOLEST template of the vis_pix fit
#       coolest_sersic.json              # COOLEST template of the sersic fit
#       pre_psf.fits                     # lens light + lensed source, pre-PSF
#       model.fits                       # the same, post-PSF convolution
#       convergence.fits                 # mass model maps, on the zoomed mask grid
#       potential.fits
#       deflections.fits                 # DEFLECTIONS_Y, DEFLECTIONS_X
#       fit_multi_wavelength.png
#       lens_mass.csv                    # this lens's row of each master CSV
#       lens_sersic.csv
#       source_sersic.csv
#       magnitudes.csv
#       astrometric_offsets.csv
#
# Stages 7-9 read a separate results tree (default `output_sed`) holding the
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
#   OUTPUT_DIR        results tree for stages 1-6      (default: output)
#   SED_OUTPUT_DIR    results tree for stages 7-9      (default: output_sed)
#   SKIP_SED=1        skip stages 7-9 even if present  (default: 0)
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

# On HPC, activate.sh activates the shared PyAuto venv under PYAUTO_HPC_BASE and
# puts its checkouts on PYTHONPATH. It is HPC-only — sourcing it elsewhere fails
# — so it is used only when that venv is actually present and the caller has not
# already set up an environment (PYAUTO_ROOT is exported by a worktree
# activate.sh). Everywhere else the ambient install is used. Export
# PYAUTO_HPC_BASE to point at your shared PyAuto checkout.
HPC_BASE="${PYAUTO_HPC_BASE:-/path/to/large/storage/your_username/PyAuto}"
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

echo "==> [1/9] inspection PNGs (scripts/tools/build_inspect.py)"
python "$PROJECT_ROOT/scripts/tools/build_inspect.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR" \
    --dataset_prefix="$DATASET_PREFIX"

echo "==> [2/9] deblended FITS (catalogue/scripts/deblending.py)"
python "$PROJECT_ROOT/catalogue/scripts/deblending.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR"

echo "==> [3/9] lens mass maps (catalogue/scripts/lens_mass_maps.py)"
python "$PROJECT_ROOT/catalogue/scripts/lens_mass_maps.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR"

echo "==> [4/9] lens mass CSV (catalogue/scripts/lens_mass.py)"
python "$PROJECT_ROOT/catalogue/scripts/lens_mass.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR"

echo "==> [5/9] lens Sersic CSV (catalogue/scripts/lens_sersic.py)"
python "$PROJECT_ROOT/catalogue/scripts/lens_sersic.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR"

echo "==> [6/9] source Sersic CSV (catalogue/scripts/source_sersic.py)"
python "$PROJECT_ROOT/catalogue/scripts/source_sersic.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR"

# Stages 7-9 read the multi-band SED tree. Set SKIP_SED=1 to refresh only the
# stable products while SED jobs are still running.
SED_OUTPUT_PATH="${PROJECT_ROOT}/${SED_OUTPUT_DIR}"
if [ "${SKIP_SED:-0}" = "1" ]; then
    echo "==> [7/9,8/9,9/9] skipped — SKIP_SED=1"
elif [ -d "$SED_OUTPUT_PATH/$SAMPLE" ]; then
    echo "==> [7/9] multi-wavelength PNG (catalogue/scripts/multi_wavelength.py)"
    python "$PROJECT_ROOT/catalogue/scripts/multi_wavelength.py" \
        --sample="$SAMPLE" \
        --output_path="$SED_OUTPUT_DIR" \
        --inspect_dir="$INSPECT_DIR"

    echo "==> [8/9] magnitudes CSV (catalogue/scripts/magnitudes.py)"
    python "$PROJECT_ROOT/catalogue/scripts/magnitudes.py" \
        --sample="$SAMPLE" \
        --output_path="$SED_OUTPUT_DIR" \
        --inspect_dir="$INSPECT_DIR"

    echo "==> [9/9] astrometric offsets CSV (catalogue/scripts/astrometric_offsets.py)"
    python "$PROJECT_ROOT/catalogue/scripts/astrometric_offsets.py" \
        --sample="$SAMPLE" \
        --output_path="$SED_OUTPUT_DIR" \
        --inspect_dir="$INSPECT_DIR"
else
    echo "==> [7/9,8/9,9/9] skipped — no $SED_OUTPUT_PATH/$SAMPLE/ (no SED runs yet)"
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
