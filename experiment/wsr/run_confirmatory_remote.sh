#!/usr/bin/env bash
set -u

repo=/data/zjq/paperb_upload/confirmatory_component/repo
python_bin=/data/zjq/miniconda3/envs/dwgsa-yolo/bin/python
state=/data/zjq/paperb_upload/confirmatory_component/state
logs=/data/zjq/paperb_upload/confirmatory_component/logs
mkdir -p "${state}" "${logs}"

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
gpus=(1 2 3 4 7 8 9)
tasks=()
for variant in "${variants[@]}"; do
    for seed in "${seeds[@]}"; do
        tasks+=("${variant}:${seed}")
    done
done

worker() {
    local gpu=$1
    local offset=$2
    local failed=0
    local index task variant seed log marker
    for ((index=offset; index<${#tasks[@]}; index+=${#gpus[@]})); do
        task=${tasks[index]}
        variant=${task%%:*}
        seed=${task##*:}
        log=${logs}/${variant}_seed_${seed}.log
        marker=${state}/${variant}_seed_${seed}
        if find "${repo}/experiment/wsr/generated/runs/confirmatory_component" \
            -path "*/${variant}/seed_${seed}/ablation_result.json" -type f -print -quit 2>/dev/null \
            | grep -q .; then
            printf 'complete\n' >"${marker}.status"
            continue
        fi
        printf 'running gpu=%s\n' "${gpu}" >"${marker}.status"
        if cd "${repo}" && PYTHONPATH="${repo}" "${python_bin}" -m experiment.wsr.ablation \
            train --confirmatory --variant "${variant}" --seed "${seed}" --device "${gpu}" --resume \
            >"${log}" 2>&1; then
            printf 'complete\n' >"${marker}.status"
        else
            printf 'failed gpu=%s\n' "${gpu}" >"${marker}.status"
            failed=1
        fi
    done
    return "${failed}"
}

printf 'running\n' >"${state}/overall.status"
pids=()
for ((index=0; index<${#gpus[@]}; index++)); do
    worker "${gpus[index]}" "${index}" &
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
