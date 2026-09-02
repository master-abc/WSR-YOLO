#!/usr/bin/env bash
set -u

repo=/data/zjq/paperb_upload/confirmatory_component/repo
python_bin=/data/zjq/miniconda3/envs/dwgsa-yolo/bin/python
state=/data/zjq/paperb_upload/confirmatory_component/state
logs=/data/zjq/paperb_upload/confirmatory_component/logs
locks=${state}/locks
mkdir -p "${state}" "${logs}" "${locks}"

variants=(
    confirm_yolo11s
    confirm_wsr_p3_r25
    confirm_wsr_p3_r25_no_hf
    confirm_wsr_p3_r25_no_ll
    confirm_wsr_p3_r25_fixed_haar
    confirm_wsr_p3_r25_equal_fusion
    confirm_wsr_p3_r25_random_route
    confirm_matched_conv_p3
    confirm_scale_only_p3
)
seeds=(13 42 3407)
# GPUs 0--6 are shared with unrelated long-running services. Keep this
# resumable queue on the three currently unallocated cards.
gpus=(7 8 9)
worker_count=6
tasks=()
for variant in "${variants[@]}"; do
    for seed in "${seeds[@]}"; do
        tasks+=("${variant}:${seed}")
    done
done

run_task() {
    local variant=$1
    local seed=$2
    local gpu=$3
    local log=${logs}/${variant}_seed_${seed}.log
    local marker=${state}/${variant}_seed_${seed}
    printf 'running gpu=%s\n' "${gpu}" >"${marker}.status"
    if cd "${repo}" && PYTHONPATH="${repo}" "${python_bin}" -m experiment.wsr.ablation \
        train --confirmatory --variant "${variant}" --seed "${seed}" --device "${gpu}" --resume \
        >"${log}" 2>&1; then
        printf 'complete\n' >"${marker}.status"
        return 0
    fi
    printf 'failed gpu=%s\n' "${gpu}" >"${marker}.status"
    return 1
}

worker() {
    local gpu=$1
    local offset=$2
    local index task variant seed
    local failed=0
    for ((index=offset; index<${#tasks[@]}; index+=worker_count)); do
        task=${tasks[index]}
        variant=${task%%:*}
        seed=${task##*:}
        if find "${repo}/experiment/wsr/generated/runs/confirmatory_component" \
            -path "*/${variant}/seed_${seed}/ablation_result.json" -type f -print -quit 2>/dev/null \
            | grep -q .; then
            printf 'complete\n' >"${state}/${variant}_seed_${seed}.status"
            continue
        fi
        if ! mkdir "${locks}/${variant}_seed_${seed}.lock" 2>/dev/null; then
            continue
        fi
        if ! run_task "${variant}" "${seed}" "${gpu}"; then
            failed=1
        fi
    done
    return "${failed}"
}

printf 'running_parallel\n' >"${state}/overall.status"
pids=()
# Two workers per card keep these small models near full utilization without
# launching all remaining jobs at once. Each worker advances through its own
# disjoint slice of the registered task list.
for ((index=0; index<worker_count; index++)); do
    gpu=${gpus[index % ${#gpus[@]}]}
    worker "${gpu}" "${index}" &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if [[ "${failed}" -eq 0 ]]; then
    printf 'complete\n' >"${state}/overall.status"
else
    printf 'failed\n' >"${state}/overall.status"
    exit 1
fi
