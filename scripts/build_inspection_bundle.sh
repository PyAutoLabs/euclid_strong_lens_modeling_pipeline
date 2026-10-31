#!/usr/bin/env bash
#
# Euclid Pipeline: Inspection Bundle Builder
# ==========================================
#
# Runs the ten catalogue producers in dependency order and assembles a
# per-lens inspection bundle for one sample (or, with DATASETS_FILE, for a
# listed subset of its lenses — one chunk of the chunked DR1 build driven by
# hpc/batch_cpu/submit_catalogue_chunked.sh). Idempotent: safe to re-run as
# more results land — already-built lenses are skipped by each stage.
#
# Output layout (inspect/<sample>[_<run_tag>]/):
#
#   lens_mass.csv                        # master CSVs, one row per lens
#   lens_sersic.csv
#   source_sersic.csv
#   witt_wynne.csv                       # one row per lens: the SIEP projection
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
#       wcs_vis_lp.json                  # files/wcs.json of the vis_lp fit
#       wcs_vis_pix.json                 # files/wcs.json of the vis_pix fit
#       vis_pix_source_plane_reconstruction_0.csv  # pixelized source (CSV)
#       pre_psf.fits                     # lens light + lensed source, pre-PSF
#       model.fits                       # the same, post-PSF convolution
#       convergence.fits                 # mass model maps, on the zoomed mask grid
#       potential.fits
#       deflections.fits                 # DEFLECTIONS_Y, DEFLECTIONS_X
#       fit_multi_wavelength.png
#       lens_mass.csv                    # this lens's row of each master CSV
#       lens_sersic.csv
#       source_sersic.csv
#       witt_wynne.csv
#       witt_wynne.in                    # isit4or2or1 input, zero-centred
#       magnitudes.csv
#       astrometric_offsets.csv
#
# Two results trees are read. `OUTPUT_DIR` (default `output`) holds the VIS
# lens models (`initial_lens_model/vis_lp`, `vis_pix`): stages 1 (vis images),
# 3, 4 and 7. `SED_OUTPUT_DIR` (default `output_sed`) holds every Sersic fit —
# the VIS `sersic_lens_model/vis` fit and the per-band SED fits, written by the
# waveband scripts with PYAUTO_OUTPUT_DIR=output_sed: stages 1 (fit_sersic.png,
# coolest_sersic.json), 2, 5, 6 and 8-10. Stages 8-10 are skipped when that tree
# has no directory for the sample.
#
# Stage 2 keeps only the newest result per band (a retry sweep can leave two),
# and stage 2b (catalogue/scripts/deduplicate_deblending_fits.py) then guards
# pre_psf.fits / model.fits against any repeated EXTNAME, writing
# fits_deduplication_validation.csv + fits_deduplication_summary.json into the
# bundle root.
#
# Usage:
#   bash scripts/build_inspection_bundle.sh [sample] [run_tag]
#   bash scripts/build_inspection_bundle.sh q1_walsmley
#   bash scripts/build_inspection_bundle.sh dr1_prelim_grade_ab run250
#
# Environment:
#   OUTPUT_DIR        results tree for stages 1-7      (default: output)
#   SED_OUTPUT_DIR    results tree for stages 8-10     (default: output_sed)
#   SKIP_SED=1        skip stages 8-10 even if present (default: 0)
#   CREATE_ARCHIVE=1  tar the bundle at the end (default: 0 — a DR1 bundle is
#                     tens of GB; the chunked build zips in its merge step)
#   DATASET_PREFIX    only collect datasets with this name prefix (default: all)
#   DATASETS_FILE     only build the lenses listed in this file, one per line
#                     (default: every lens of the sample)
#   INSPECT_DIR       bundle directory (default: inspect/<sample>[_<run_tag>])

set -euo pipefail

SAMPLE="${1:-q1_walsmley}"
RUN_TAG="${2:-}"
if [ -n "${INSPECT_DIR:-}" ]; then
    :
elif [ -n "$RUN_TAG" ]; then
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
DATASETS_FILE="${DATASETS_FILE:-}"

cd "$PROJECT_ROOT"

# --datasets_file for every producer when DATASETS_FILE is set. The `+` form
# expands an empty array to nothing under `set -u` on any bash.
DS_ARGS=()
if [ -n "$DATASETS_FILE" ]; then
    DS_ARGS=(--datasets_file="$DATASETS_FILE")
    echo "==> scoped to $(grep -cv '^[[:space:]]*$' "$DATASETS_FILE") lenses listed in $DATASETS_FILE"
fi

mkdir -p "$INSPECT_DIR"

echo "==> [1/10] inspection PNGs (scripts/tools/build_inspect.py)"
python "$PROJECT_ROOT/scripts/tools/build_inspect.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --sersic_output_path="$SED_OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR" \
    --dataset_prefix="$DATASET_PREFIX" \
    ${DS_ARGS[@]+"${DS_ARGS[@]}"}

echo "==> [2/10] deblended FITS (catalogue/scripts/deblending.py)"
python "$PROJECT_ROOT/catalogue/scripts/deblending.py" \
    --sample="$SAMPLE" \
    --output_path="$SED_OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR" \
    ${DS_ARGS[@]+"${DS_ARGS[@]}"}

echo "==> [2b/10] de-duplicate deblended FITS HDUs (catalogue/scripts/deduplicate_deblending_fits.py)"
python "$PROJECT_ROOT/catalogue/scripts/deduplicate_deblending_fits.py" \
    --inspect_dir="$INSPECT_DIR" \
    ${DS_ARGS[@]+"${DS_ARGS[@]}"}

echo "==> [3/10] lens mass maps (catalogue/scripts/lens_mass_maps.py)"
python "$PROJECT_ROOT/catalogue/scripts/lens_mass_maps.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR" \
    ${DS_ARGS[@]+"${DS_ARGS[@]}"}

echo "==> [4/10] lens mass CSV (catalogue/scripts/lens_mass.py)"
python "$PROJECT_ROOT/catalogue/scripts/lens_mass.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR" \
    ${DS_ARGS[@]+"${DS_ARGS[@]}"}

echo "==> [5/10] lens Sersic CSV (catalogue/scripts/lens_sersic.py)"
python "$PROJECT_ROOT/catalogue/scripts/lens_sersic.py" \
    --sample="$SAMPLE" \
    --output_path="$SED_OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR" \
    ${DS_ARGS[@]+"${DS_ARGS[@]}"}

echo "==> [6/10] source Sersic CSV (catalogue/scripts/source_sersic.py)"
python "$PROJECT_ROOT/catalogue/scripts/source_sersic.py" \
    --sample="$SAMPLE" \
    --output_path="$SED_OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR" \
    ${DS_ARGS[@]+"${DS_ARGS[@]}"}

echo "==> [7/10] Witt-Wynne SIEP projection (catalogue/scripts/witt_wynne.py)"
python "$PROJECT_ROOT/catalogue/scripts/witt_wynne.py" \
    --sample="$SAMPLE" \
    --output_path="$OUTPUT_DIR" \
    --inspect_dir="$INSPECT_DIR" \
    ${DS_ARGS[@]+"${DS_ARGS[@]}"}

# Stages 8-10 read the multi-band SED tree. Set SKIP_SED=1 to refresh only the
# stable products while SED jobs are still running.
SED_OUTPUT_PATH="${PROJECT_ROOT}/${SED_OUTPUT_DIR}"
if [ "${SKIP_SED:-0}" = "1" ]; then
    echo "==> [8/10,9/10,10/10] skipped — SKIP_SED=1"
elif [ -d "$SED_OUTPUT_PATH/$SAMPLE" ]; then
    echo "==> [8/10] multi-wavelength PNG (catalogue/scripts/multi_wavelength.py)"
    python "$PROJECT_ROOT/catalogue/scripts/multi_wavelength.py" \
        --sample="$SAMPLE" \
        --output_path="$SED_OUTPUT_DIR" \
        --inspect_dir="$INSPECT_DIR" \
        ${DS_ARGS[@]+"${DS_ARGS[@]}"}

    echo "==> [9/10] magnitudes CSV (catalogue/scripts/magnitudes.py)"
    python "$PROJECT_ROOT/catalogue/scripts/magnitudes.py" \
        --sample="$SAMPLE" \
        --output_path="$SED_OUTPUT_DIR" \
        --inspect_dir="$INSPECT_DIR" \
        ${DS_ARGS[@]+"${DS_ARGS[@]}"}

    echo "==> [10/10] astrometric offsets CSV (catalogue/scripts/astrometric_offsets.py)"
    python "$PROJECT_ROOT/catalogue/scripts/astrometric_offsets.py" \
        --sample="$SAMPLE" \
        --output_path="$SED_OUTPUT_DIR" \
        --inspect_dir="$INSPECT_DIR" \
        ${DS_ARGS[@]+"${DS_ARGS[@]}"}
else
    echo "==> [8/10,9/10,10/10] skipped — no $SED_OUTPUT_PATH/$SAMPLE/ (no SED runs yet)"
fi

echo ""
echo "Done. Bundle at: $INSPECT_DIR"
echo "  $(ls "$INSPECT_DIR" | wc -l) entries"

# Archive only when explicitly enabled — a full DR1 bundle is tens of GB, so an
# incremental refresh should not rebuild the tarball.
if [ "${CREATE_ARCHIVE:-0}" = "1" ]; then
    TAR_NAME="${INSPECT_DIR//\//_}.tar.gz"
    echo "==> Archiving -> $TAR_NAME"
    tar czf "$TAR_NAME" "$INSPECT_DIR/"
    echo "  $(ls -lh "$TAR_NAME" | awk '{print $5}')  $TAR_NAME"
else
    echo "==> Archive skipped — CREATE_ARCHIVE=0"
fi
