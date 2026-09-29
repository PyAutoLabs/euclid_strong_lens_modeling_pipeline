#!/bin/bash
#
# Submit the vis_pix follow-up of the v2-grade vis_lp arrays (parts 01-05) on RAL.
#
# Run ON RAL (ssh euclid_jump), from anywhere:
#   bash /mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu/submit_vis_pix_v2grade_overnight.sh [--dry-run]
#
# For each part it reads the state of every task of the matching vis_lp array
# (squeue for queued tasks, sacct for finished ones) and submits
#   - one array over the indices whose vis_lp task is COMPLETED (no dependency),
#     throttled at %60 per part;
#   - one single-task array (`--array=N`) per index whose vis_lp task is still
#     in the queue (squeue: RUNNING / PENDING, incl. requeued), with `--dependency=afterok:<vis_lp>_N
#     --kill-on-invalid-dep=yes`, so it starts when that vis_lp task succeeds.
# Indices whose vis_lp ended FAILED / TIMEOUT / CANCELLED / NODE_FAIL-without-
# requeue are listed and skipped.
#
# Why not one `--dependency=aftercorr:<vis_lp>` array: tried on 2026-09-29
# (372037/38/59/70/71) and it never fired, even for vis_lp tasks that completed
# after submission; completed task records are purged after MinJobAge=300 s.
#
# Re-running it later is safe for the indices it already covered only if their
# earlier vis_pix jobs are gone; it does not look for existing vis_pix jobs, so
# check `squeue -u jnightin -n euclid_dr1_vis_pix_v2g_pNN` first.

set -euo pipefail

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

BATCH_DIR=/mnt/ral/jnightin/euclid_dr1/hpc/batch_cpu
declare -A VIS_LP=([01]=366938 [02]=367039 [03]=367141 [04]=367243 [05]=367244)

cd "$BATCH_DIR"

# Compress a sorted list of integers into a SLURM array spec: 0-3,7,9-12
to_ranges() {
    awk 'NR==1{s=$1;e=$1;next} $1==e+1{e=$1;next} {printf "%s%s,", s, (s==e?"":"-"e); s=$1;e=$1} END{if(NR) printf "%s%s", s, (s==e?"":"-"e)}'
}

for part in 01 02 03 04 05; do
    lp=${VIS_LP[$part]}
    script=submit_initial_lens_model_vis_pix_v2grade_p${part}
    # Still in the queue (RUNNING / PENDING incl. requeued): one line per task.
    wait_idx=$(squeue -h -r -j "$lp" -o "%K" | sort -n -u)
    # Finished tasks: sacct lists each started task on its own line (latest record).
    states=$(sacct -j "$lp" -X -n -P -o JobID,State | awk -F'|' -v lp="$lp" '$1 ~ "^"lp"_[0-9]+$" {split($1,a,"_"); split($2,s," "); print a[2], s[1]}' | sort -n -u -k1,1)
    states=$(awk 'NR==FNR{w[$1]=1;next} !($1 in w)' <(echo "$wait_idx") <(echo "$states"))
    done_idx=$(awk '$2=="COMPLETED"{print $1}' <<<"$states")
    bad_idx=$(awk '$2!="COMPLETED" && $2!="RUNNING" && $2!="PENDING"{print $1":"$2}' <<<"$states")

    n_done=$(grep -c . <<<"$done_idx" || true)
    n_wait=$(grep -c . <<<"$wait_idx" || true)
    echo "== part $part (vis_lp $lp): completed=$n_done queued=$n_wait other=[$(tr '\n' ' ' <<<"$bad_idx")]"

    if [[ $n_done -gt 0 ]]; then
        spec="$(to_ranges <<<"$done_idx")%60"
        if [[ $DRY -eq 1 ]]; then
            echo "   would: sbatch --array=${spec:0:80}... $script"
        else
            jid=$(sbatch --parsable --array="$spec" "$script")
            echo "   immediate array: $jid ($n_done tasks)"
        fi
    fi

    ids=()
    for i in $wait_idx; do
        if [[ $DRY -eq 1 ]]; then
            continue
        fi
        if jid=$(sbatch --parsable --array="$i" --dependency="afterok:${lp}_${i}" --kill-on-invalid-dep=yes "$script" 2>/dev/null); then
            ids+=("$jid")
        else
            # vis_lp task finished and was purged between sacct and sbatch: submit
            # without a dependency; the vis_lp-zip guard covers a failed vis_lp.
            jid=$(sbatch --parsable --array="$i" "$script")
            ids+=("$jid")
            echo "   index $i: dependency rejected, submitted without it ($jid)"
        fi
    done
    if [[ ${#ids[@]} -gt 0 ]]; then
        echo "   dependent single-task jobs: ${#ids[@]} (${ids[0]} .. ${ids[-1]})"
    fi
done
