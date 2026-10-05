#!/bin/bash
#
# Build the DR1 catalogue as N parallel chunks + one merge job, on RAL.
#
# WHY
#   scripts/build_inspection_bundle.sh walks a sample serially at ~11 min per
#   100 lenses on 4 CPUs (catalogue_test100_20261005), so the ~15,000-lens DR1
#   catalogue (dr1_sep1_rest + dr1_v2_new) is ~27 h in one job — over the 24 h
#   a single job should take. Split into N chunks (default 6) it is ~5-7 h of
#   wall per chunk, all chunks running at once, plus the merge.
#
# WHAT IT DOES (run ON RAL, on the login node; it only writes lists and sbatches)
#   1. tile list: every lens directory of output/<sample>/ for each SAMPLE
#      (or the "<sample> <tile>" lines of TILES_FILE), written to
#      inspect/<CATALOGUE>/_build/tiles.txt. Tile ids must be unique across
#      samples (checked).
#   2. split into N contiguous chunks _build/chunk_<k>.txt.
#   3. sbatch --array=1-N submit_catalogue_chunk: task k builds chunk k, per
#      sample, into _build/chunk_<k>/<sample>/ and touches _build/chunk_<k>.done.
#   4. sbatch --dependency=afterok:<array> submit_catalogue_merge: moves every
#      lens folder to inspect/<CATALOGUE>/<tile>/, concatenates the root CSVs
#      into ONE catalogue, writes the QA CSVs, grades.csv, wcs.csv /
#      source_clumps.csv, lens_index.csv, README.md and — with
#      ARCHIVE=chunked — the distribution zips in inspect/<CATALOGUE>_dist/.
#
#   Partition `ral` only (CPU work; never gpu). Walltime per chunk is computed
#   from the chunk's lens count at MIN_PER_100 (default 32 = 2x the ~16 min per
#   100 lenses measured in catalogue_test110_chunked_20261005, 37-lens chunks in
#   6.2-6.7 min) plus 30 min, capped at 23:59. N=6 over ~15,000 lenses: ~2,500
#   per chunk, ~6.7 h expected, 13:52 requested.
#
# USAGE
#   ssh euclid_jump
#   bash /mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu/submit_catalogue_chunked.sh \
#       CATALOGUE=dr1_catalogue_20261006 N=6 ARCHIVE=chunked [PARTS=10] [DRY=1]
#
#   Variables (as KEY=VALUE arguments or in the environment):
#     CATALOGUE     catalogue name -> inspect/<CATALOGUE>/   (required; must not exist)
#     SAMPLES       space-separated samples   (default "dr1_sep1_rest dr1_v2_new")
#     TILES_FILE    "<sample> <tile>" lines, instead of listing output/<sample>/
#     N             number of chunks          (default 6)
#     ARCHIVE       0 | chunked               (default 0)
#     PART_MAX_GB   cap per distribution part (default 10)
#     PARTS         write EXACTLY this many distribution parts (contiguous,
#                   near-equal bytes); overrides PART_MAX_GB   (default 0 = off)
#     EXCLUDED_FILE CSV (tile,sample,reason) of tiles deliberately left out;
#                   copied to inspect/<CATALOGUE>/excluded_tiles.csv (README lists it)
#     MIN_PER_100   walltime minutes per 100 lenses per chunk (default 32)
#     MERGE_TIME    merge walltime            (default 12:00:00)
#     DRY=1         write the lists, print the sbatch lines, submit nothing
#
#   A failed chunk k blocks the merge (afterok). Fix it, then re-run only it and
#   re-submit the merge against it — the driver prints both commands.

set -euo pipefail

for arg in "$@"; do
    case "$arg" in
        *=*) export "${arg?}" ;;
        *) echo "unknown argument: $arg (use KEY=VALUE)" >&2; exit 2 ;;
    esac
done

PROJECT_PATH=${PROJECT_PATH:-/mnt/ral/jnightin/euclid_dr1}
BATCH_DIR=$PROJECT_PATH/hpc/batch_cpu
CATALOGUE=${CATALOGUE:?CATALOGUE=<name> is required}
SAMPLES=${SAMPLES:-"dr1_sep1_rest dr1_v2_new"}
N=${N:-6}
ARCHIVE=${ARCHIVE:-0}
PART_MAX_GB=${PART_MAX_GB:-10}
PARTS=${PARTS:-0}
EXCLUDED_FILE=${EXCLUDED_FILE:-}
MIN_PER_100=${MIN_PER_100:-32}
MERGE_TIME=${MERGE_TIME:-12:00:00}
DRY=${DRY:-0}

CAT_DIR=$PROJECT_PATH/inspect/$CATALOGUE
BUILD=$CAT_DIR/_build
if [[ -e "$CAT_DIR" ]]; then
    echo "ERROR: $CAT_DIR already exists — pick a new CATALOGUE name (never overwrite a catalogue)" >&2
    exit 1
fi
if [[ -n "$EXCLUDED_FILE" && ! -f "$EXCLUDED_FILE" ]]; then
    echo "ERROR: EXCLUDED_FILE=$EXCLUDED_FILE not found" >&2
    exit 1
fi
mkdir -p "$BUILD/logs"
if [[ -n "$EXCLUDED_FILE" ]]; then
    cp "$EXCLUDED_FILE" "$CAT_DIR/excluded_tiles.csv"
    echo "excluded: $(( $(wc -l < "$CAT_DIR/excluded_tiles.csv") - 1 )) tiles -> $CAT_DIR/excluded_tiles.csv"
fi

# 1. tile list
if [[ -n "${TILES_FILE:-}" ]]; then
    # tr: a CRLF list would otherwise put a \r into every sample/tile path.
    tr -d '\r' < "$TILES_FILE" | grep -v '^[[:space:]]*\(#\|$\)' | awk '{print $1, $2}' > "$BUILD/tiles.txt"
else
    : > "$BUILD/tiles.txt"
    for sample in $SAMPLES; do
        ls "$PROJECT_PATH/output/$sample" | sort | awk -v s="$sample" '{print s, $1}' >> "$BUILD/tiles.txt"
    done
fi
total=$(wc -l < "$BUILD/tiles.txt")
dups=$(awk '{print $2}' "$BUILD/tiles.txt" | sort | uniq -d | wc -l)
if [[ "$dups" -ne 0 ]]; then
    echo "ERROR: $dups tile ids appear in more than one sample:" >&2
    awk '{print $2}' "$BUILD/tiles.txt" | sort | uniq -d | head >&2
    exit 1
fi
echo "tiles: $total over samples: $(awk '{print $1}' "$BUILD/tiles.txt" | sort | uniq -c | tr '\n' ' ')"

# 2. chunks (contiguous, near-equal)
per=$(( (total + N - 1) / N ))
split -l "$per" -d -a 2 --numeric-suffixes=1 --additional-suffix=.txt "$BUILD/tiles.txt" "$BUILD/chunk_"
n_made=$(ls "$BUILD"/chunk_[0-9]*.txt | wc -l)
minutes=$(( per * MIN_PER_100 / 100 + 30 ))
(( minutes > 1439 )) && minutes=1439
chunk_time=$(printf '%02d:%02d:00' $((minutes / 60)) $((minutes % 60)))
echo "chunks: $n_made x <=$per lenses, walltime $chunk_time each"

cat > "$BUILD/build.env" <<EOF
CATALOGUE=$CATALOGUE
SAMPLES="$SAMPLES"
N=$n_made
ARCHIVE=$ARCHIVE
PART_MAX_GB=$PART_MAX_GB
PARTS=$PARTS
CHUNK_TIME=$chunk_time
MERGE_TIME=$MERGE_TIME
GIT_REVISION=$(cd "$PROJECT_PATH" && git rev-parse --short HEAD 2>/dev/null || echo unknown)
EOF

cd "$BATCH_DIR"
chunk_cmd=(sbatch --parsable --partition=ral --array=1-"$n_made" -t "$chunk_time"
    -J "euclid_dr1_cat_chunk_$CATALOGUE"
    -o "$BUILD/logs/chunk.%A_%a.out" -e "$BUILD/logs/chunk.%A_%a.err"
    --export=ALL,PROJECT_PATH="$PROJECT_PATH",PYAUTO_HPC_BASE="${PYAUTO_HPC_BASE:-/mnt/ral/jnightin/PyAuto}",CATALOGUE="$CATALOGUE"
    submit_catalogue_chunk)
if [[ "$DRY" == 1 ]]; then
    echo "DRY: ${chunk_cmd[*]}"
    echo "DRY: sbatch --parsable --partition=ral --dependency=afterok:<array> -t $MERGE_TIME ... ARCHIVE=$ARCHIVE,PART_MAX_GB=$PART_MAX_GB,PARTS=$PARTS submit_catalogue_merge"
    exit 0
fi
array_id=$("${chunk_cmd[@]}")
merge_cmd=(sbatch --parsable --partition=ral --dependency=afterok:"$array_id" -t "$MERGE_TIME"
    -J "euclid_dr1_cat_merge_$CATALOGUE"
    -o "$BUILD/logs/merge.%j.out" -e "$BUILD/logs/merge.%j.err"
    --export=ALL,PROJECT_PATH="$PROJECT_PATH",PYAUTO_HPC_BASE="${PYAUTO_HPC_BASE:-/mnt/ral/jnightin/PyAuto}",CATALOGUE="$CATALOGUE",ARCHIVE="$ARCHIVE",PART_MAX_GB="$PART_MAX_GB",PARTS="$PARTS"
    submit_catalogue_merge)
merge_id=$("${merge_cmd[@]}")
echo "ARRAY_ID=$array_id" >> "$BUILD/build.env"
echo "MERGE_ID=$merge_id" >> "$BUILD/build.env"
echo "submitted chunk array $array_id (1-$n_made) and merge $merge_id (afterok)"
echo "logs: $BUILD/logs/"
echo "re-run one failed chunk k:"
echo "  cd $BATCH_DIR && sbatch --partition=ral --array=<k> -t $chunk_time -o $BUILD/logs/chunk.%A_%a.out -e $BUILD/logs/chunk.%A_%a.err --export=ALL,CATALOGUE=$CATALOGUE submit_catalogue_chunk"
echo "  then: sbatch --partition=ral --dependency=afterok:<new id> -t $MERGE_TIME -o $BUILD/logs/merge.%j.out -e $BUILD/logs/merge.%j.err --export=ALL,CATALOGUE=$CATALOGUE,ARCHIVE=$ARCHIVE,PART_MAX_GB=$PART_MAX_GB,PARTS=$PARTS submit_catalogue_merge"
