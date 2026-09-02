#!/usr/bin/env bash
set -euo pipefail

project=/data/zjq/test/DWGSA-YOLO
python_bin=/data/zjq/miniconda3/envs/dwgsa-yolo/bin/python
bundle=/data/zjq/paperb_upload/revision_experiments
results=${bundle}/results
logs=${bundle}/logs
mkdir -p "${results}" "${logs}"

run_mechanism() {
    local dataset=$1
    local gpu=$2
    local data=${project}/experiment/paper_b/generated/datasets/${dataset}/dataset.yaml
    local weights=${project}/experiment/paper_b/generated/runs/controlled/${dataset}/wsr_yolo11s_p3_r25/seed_13/weights/best.pt
    if [[ -s "${results}/${dataset}_mechanism_controls.json" ]]; then
        return
    fi
    PYTHONPATH="${project}" "${python_bin}" "${bundle}/mechanism_diagnostics.py" \
        --weights "${weights}" \
        --data "${data}" \
        --output "${results}/${dataset}_mechanism_controls.json" \
        --split test \
        --control-split train \
        --controls \
        --shuffle-repeats 64 \
        --device "${gpu}" \
        >"${logs}/${dataset}_mechanism_controls.log" 2>&1
}

run_latency() {
    local dataset=$1
    local precision=$2
    local gpu=$3
    local data=${project}/experiment/paper_b/generated/datasets/${dataset}/dataset.yaml
    local baseline=${project}/experiment/paper_b/generated/runs/controlled/${dataset}/yolo11s/seed_13/weights/best.pt
    local candidate=${project}/experiment/paper_b/generated/runs/controlled/${dataset}/wsr_yolo11s_p3_r25/seed_13/weights/best.pt
    if [[ -s "${results}/${dataset}_${precision}_paired_latency.json" ]]; then
        return
    fi
    local half_arg=()
    if [[ "${precision}" == fp16 ]]; then
        half_arg=(--half)
    fi
    PYTHONPATH="${project}" "${python_bin}" "${bundle}/paired_benchmark.py" \
        --baseline-weights "${baseline}" \
        --candidate-weights "${candidate}" \
        --data "${data}" \
        --output "${results}/${dataset}_${precision}_paired_latency.json" \
        --split test \
        --device "${gpu}" \
        --warmup 50 \
        --cycles 12 \
        --repetitions-per-segment 25 \
        --maximum-images 50 \
        "${half_arg[@]}" \
        >"${logs}/${dataset}_${precision}_paired_latency.log" 2>&1
}

required=(
    "${project}/experiment/paper_b/generated/runs/controlled/deeppcb/yolo11s/seed_13/weights/best.pt"
    "${project}/experiment/paper_b/generated/runs/controlled/deeppcb/wsr_yolo11s_p3_r25/seed_13/weights/best.pt"
    "${project}/experiment/paper_b/generated/runs/controlled/dspcbsd_plus/yolo11s/seed_13/weights/best.pt"
    "${project}/experiment/paper_b/generated/runs/controlled/dspcbsd_plus/wsr_yolo11s_p3_r25/seed_13/weights/best.pt"
)
for path in "${required[@]}"; do
    test -f "${path}"
done

printf 'running\n' >"${bundle}/status.txt"
pids=()
run_mechanism deeppcb 1 & pids+=("$!")
run_mechanism dspcbsd_plus 2 & pids+=("$!")
run_latency deeppcb fp32 3 & pids+=("$!")
run_latency deeppcb fp16 4 & pids+=("$!")
run_latency dspcbsd_plus fp32 7 & pids+=("$!")
run_latency dspcbsd_plus fp16 8 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if [[ "${failed}" -eq 0 ]]; then
    printf 'complete\n' >"${bundle}/status.txt"
else
    printf 'failed\n' >"${bundle}/status.txt"
    exit 1
fi
