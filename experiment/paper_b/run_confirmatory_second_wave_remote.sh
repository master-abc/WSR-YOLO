#!/usr/bin/env bash
set -u

repo=/data/zjq/paperb_upload/confirmatory_component/repo
python_bin=/data/zjq/miniconda3/envs/dwgsa-yolo/bin/python
state=/data/zjq/paperb_upload/confirmatory_component/state
logs=/data/zjq/paperb_upload/confirmatory_component/logs
locks=${state}/locks
mkdir -p "${state}" "${logs}" "${locks}"

variants=(
    confirm_wsr_p3_r25_no_hf
    confirm_wsr_p3_r25_no_hf
    confirm_wsr_p3_r25_no_ll
    confirm_wsr_p3_r25_no_ll
    confirm_wsr_p3_r25_no_ll
    confirm_wsr_p3_r25_fixed_haar
    confirm_wsr_p3_r25_fixed_haar
)
seeds=(42 3407 13 42 3407 13 42)
gpus=(1 2 3 4 7 8 9)
pids=()

printf 'running\n' >"${state}/second_wave.status"
for ((index=0; index<${#variants[@]}; index++)); do
    variant=${variants[index]}
    seed=${seeds[index]}
    gpu=${gpus[index]}
    marker=${state}/${variant}_seed_${seed}
    log=${logs}/${variant}_seed_${seed}.log
    if ! mkdir "${locks}/${variant}_seed_${seed}.lock" 2>/dev/null; then
        printf 'failed\n' >"${state}/second_wave.status"
        printf 'lock already exists for %s seed %s\n' "${variant}" "${seed}" >&2
        exit 1
    fi
    (
        printf 'running gpu=%s\n' "${gpu}" >"${marker}.status"
        if cd "${repo}" && PYTHONPATH="${repo}" "${python_bin}" -m experiment.paper_b.ablation \
            train --confirmatory --variant "${variant}" --seed "${seed}" --device "${gpu}" --resume \
            >"${log}" 2>&1; then
            printf 'complete\n' >"${marker}.status"
        else
            printf 'failed gpu=%s\n' "${gpu}" >"${marker}.status"
            exit 1
        fi
    ) &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if [[ "${failed}" -eq 0 ]]; then
    printf 'complete\n' >"${state}/second_wave.status"
else
    printf 'failed\n' >"${state}/second_wave.status"
    exit 1
fi
