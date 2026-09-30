#!/bin/bash
#
# Chain the vis_pix follow-up onto the "rest" vis_lp arrays (375700-375703, parts 01-04)
# on RAL, one vis_pix task per tile. Idempotent: re-run it at any time to catch
# stragglers (race casualties, requeued vis_lp tasks, indices added later).
#
# Run ON RAL (ssh euclid_jump), from anywhere:
#   bash /mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu/submit_vis_pix_rest_chain.sh [--dry-run] [PART ...]
#   e.g.  ... --dry-run          (all parts, print only)
#         ... 01 02              (real run, parts 01 and 02 only)
#
# Per part, for every manifest index N (task N of vis_lp = line N of the manifest):
#   (a) skip N if a vis_pix_rest job for N is queued / running / completed (squeue +
#       sacct over jobs named euclid_dr1_vis_pix_rest_pNN), or if the tile already has
#       a vis_pix zip. A vis_pix task that ended FAILED / TIMEOUT / OUT_OF_MEMORY /
#       NODE_FAIL is listed and NOT resubmitted (look at it first). A CANCELLED one
#       (e.g. killed by --kill-on-invalid-dep) is treated as absent.
#   (b) vis_lp task N COMPLETED and the tile has exactly one vis_lp zip: goes into one
#       immediate array (no dependency, %60).
#   (c) vis_lp task N still queued (squeue: PENDING / RUNNING / requeued): one
#       single-task array `--array=N --dependency=afterok:<vis_lp>_N
#       --kill-on-invalid-dep=yes`. If sbatch rejects the dependency (the vis_lp task
#       finished and its record was purged between the snapshot and the sbatch, since
#       MinJobAge=300 s), N is submitted without it; the job's own vis_lp-zip guard
#       then decides.
#   (d) vis_lp task N FAILED / TIMEOUT / CANCELLED / OUT_OF_MEMORY / NODE_FAIL, or
#       COMPLETED without a zip (python crashed; the vis_lp script still exits 0):
#       listed and skipped.
# The filesystem is only touched for the indices that are candidates for (b) (vis_lp
# done, no vis_pix job yet), never for all tiles.
#
# Why not `--dependency=aftercorr:<vis_lp>`: it never fires on this slurmctld (22.05.9),
# see wiki/project/2026-09-29-vis-pix-v2grade-overnight.md.

set -uo pipefail

DRY=0
PARTS=()
for a in "$@"; do
    case "$a" in
        --dry-run) DRY=1 ;;
        0[1-4]) PARTS+=("$a") ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
[[ ${#PARTS[@]} -eq 0 ]] && PARTS=(01 02 03 04)

PROJECT_PATH=${PROJECT_PATH:-/mnt/ral/jnightin/euclid_dr1}
BATCH_DIR=$PROJECT_PATH/hpc/batch_cpu
SAMPLE=dr1_sep1_rest
SINCE=2026-09-30T00:00
declare -A VIS_LP=([01]=375700 [02]=375701 [03]=375702 [04]=375703)

cd "$BATCH_DIR" || exit 1

# Compress a sorted list of integers into a SLURM array spec: 0-3,7,9-12
to_ranges() {
    awk 'NR==1{s=$1;e=$1;next} $1==e+1{e=$1;next} {printf "%s%s,", s, (s==e?"":"-"e); s=$1;e=$1} END{if(NR) printf "%s%s", s, (s==e?"":"-"e)}'
}

# sbatch with retries on transient controller errors. Prints the job id.
# Returns 3 when the dependency itself is rejected.
sbatch_retry() {
    local out try
    for try in 1 2 3 4; do
        if out=$(sbatch --parsable "$@" 2>&1); then
            echo "${out%%;*}"
            return 0
        fi
        if grep -qi "dependency" <<<"$out"; then
            return 3
        fi
        echo "   sbatch error (try $try): $out" >&2
        sleep $((try * 5))
    done
    return 1
}

tot_imm=0; tot_dep=0
for part in "${PARTS[@]}"; do
    lp=${VIS_LP[$part]}
    script=submit_initial_lens_model_vis_pix_rest_p${part}
    name=euclid_dr1_vis_pix_rest_p${part}
    manifest=$PROJECT_PATH/hpc/run_manifests/vis_lp_rest_20260930_part${part}.txt
    mapfile -t tiles < "$manifest"
    n=${#tiles[@]}

    # --- vis_pix state per index -------------------------------------------------
    # Queued (pending singles / split or unsplit array tasks, all expanded by -r).
    pix_queued=$(squeue -h -r -u "$USER" -n "$name" -o "%K" | grep -E '^[0-9]+$' | sort -n -u)
    # Finished / started: latest job id per index wins.
    pix_state=$(sacct -u "$USER" -S "$SINCE" -X -n -P --name="$name" -o JobID,State 2>/dev/null |
        awk -F'|' '$1 ~ /^[0-9]+_\[?[0-9]+\]?$/ {split($1,a,"_"); gsub(/[\[\]]/,"",a[2]); split($2,s," ");
                   if (!(a[2] in id) || a[1]+0 > id[a[2]]) {id[a[2]]=a[1]+0; st[a[2]]=s[1]}}
                   END{for (i in st) print i, st[i]}' | sort -n -k1,1)

    # --- vis_lp state per index --------------------------------------------------
    lp_queued=$(squeue -h -r -j "$lp" -o "%K" 2>/dev/null | grep -E '^[0-9]+$' | sort -n -u)
    lp_state=$(sacct -j "$lp" -X -n -P -o JobID,State 2>/dev/null |
        awk -F'|' -v lp="$lp" '$1 ~ "^"lp"_[0-9]+$" {split($1,a,"_"); split($2,s," "); print a[2], s[1]}' | sort -n -u -k1,1)

    # --- classify ------------------------------------------------------------------
    # Output lines: "<index> <class> <detail>"
    classes=$(awk -v n="$n" '
        FILENAME==ARGV[1] {pq[$1]=1; next}
        FILENAME==ARGV[2] {ps[$1]=$2; next}
        FILENAME==ARGV[3] {lq[$1]=1; next}
        FILENAME==ARGV[4] {ls[$1]=$2; next}
        END {
            for (i=0; i<n; i++) {
                if (i in pq) {print i, "have", "vis_pix_queued"; continue}
                p = (i in ps) ? ps[i] : ""
                if (p=="COMPLETED" || p=="RUNNING" || p=="PENDING" || p=="REQUEUED" || p=="SUSPENDED" || p=="RESIZING") {print i, "have", "vis_pix_"p; continue}
                if (p!="" && p!="CANCELLED") {print i, "pixbad", "vis_pix_"p; continue}
                if (i in lq) {print i, "dep", "vis_lp_queued"; continue}
                l = (i in ls) ? ls[i] : "NONE"
                if (l=="COMPLETED") {print i, "done", "vis_lp_COMPLETED"; continue}
                print i, "lpbad", "vis_lp_"l
            }
        }' <(echo "$pix_queued") <(echo "$pix_state") <(echo "$lp_queued") <(echo "$lp_state"))

    # Filesystem check only for the (b) candidates.
    imm_idx=(); nozip=(); havezip=()
    shopt -s nullglob
    for i in $(awk '$2=="done"{print $1}' <<<"$classes"); do
        d=$PROJECT_PATH/output/$SAMPLE/${tiles[$i]}/initial_lens_model
        pz=("$d/vis_pix"/*.zip); lz=("$d/vis_lp"/*.zip)
        if [[ ${#pz[@]} -gt 0 ]]; then havezip+=("$i"); continue; fi
        if [[ ${#lz[@]} -ne 1 ]]; then nozip+=("$i:vis_lp_zips=${#lz[@]}"); continue; fi
        imm_idx+=("$i")
    done
    shopt -u nullglob
    dep_idx=($(awk '$2=="dep"{print $1}' <<<"$classes"))

    n_have=$(awk '$2=="have"' <<<"$classes" | grep -c . || true)
    echo "== part $part (vis_lp $lp, $n tiles): already_have_vis_pix=$n_have immediate=${#imm_idx[@]} dependent=${#dep_idx[@]}"
    echo "   vis_pix_zip_exists: ${#havezip[@]} [${havezip[*]:-}]"
    echo "   skipped vis_lp failed: [$(awk '$2=="lpbad"{printf "%s:%s ", $1, $3}' <<<"$classes")]"
    echo "   skipped vis_lp COMPLETED without one zip: [${nozip[*]:-}]"
    echo "   listed vis_pix failed (not resubmitted): [$(awk '$2=="pixbad"{printf "%s:%s ", $1, $3}' <<<"$classes")]"

    # --- (b) immediate array ---------------------------------------------------------
    if [[ ${#imm_idx[@]} -gt 0 ]]; then
        spec="$(printf '%s\n' "${imm_idx[@]}" | to_ranges)%60"
        if [[ $DRY -eq 1 ]]; then
            echo "   would: sbatch --array=$spec $script"
        else
            if jid=$(sbatch_retry --array="$spec" "$script"); then
                echo "   immediate array: $jid (${#imm_idx[@]} tasks: ${spec:0:120})"
                tot_imm=$((tot_imm + ${#imm_idx[@]}))
            else
                echo "   ERROR: immediate array submission failed" >&2
            fi
        fi
    fi

    # --- (c) dependent singles ------------------------------------------------------
    ids=(); nodep=(); failed=()
    for i in "${dep_idx[@]}"; do
        if [[ $DRY -eq 1 ]]; then
            continue
        fi
        jid=$(sbatch_retry --array="$i" --dependency="afterok:${lp}_${i}" --kill-on-invalid-dep=yes "$script")
        rc=$?
        if [[ $rc -eq 0 ]]; then
            ids+=("$jid")
        elif [[ $rc -eq 3 ]]; then
            if jid=$(sbatch_retry --array="$i" "$script"); then
                ids+=("$jid"); nodep+=("$i:$jid")
            else
                failed+=("$i")
            fi
        else
            failed+=("$i")
        fi
    done
    if [[ $DRY -eq 1 ]]; then
        [[ ${#dep_idx[@]} -gt 0 ]] && echo "   would: ${#dep_idx[@]} x sbatch --array=N --dependency=afterok:${lp}_N --kill-on-invalid-dep=yes $script  (N = ${dep_idx[0]} .. ${dep_idx[-1]})"
    else
        if [[ ${#ids[@]} -gt 0 ]]; then
            echo "   dependent single-task jobs: ${#ids[@]} (${ids[0]} .. ${ids[-1]})"
        fi
        [[ ${#nodep[@]} -gt 0 ]] && echo "   dependency rejected, submitted without it: ${nodep[*]}"
        [[ ${#failed[@]} -gt 0 ]] && echo "   ERROR: submission failed for indices: ${failed[*]} (re-run this driver)"
        tot_dep=$((tot_dep + ${#ids[@]}))
    fi
done
[[ $DRY -eq 0 ]] && echo "== total submitted: immediate=$tot_imm dependent=$tot_dep"
exit 0
