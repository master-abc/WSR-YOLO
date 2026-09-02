#!/usr/bin/env bash
set -u

repo=/data/zjq/paperb_upload/confirmatory_component/repo
python_bin=/data/zjq/miniconda3/envs/dwgsa-yolo/bin/python
state=/data/zjq/paperb_upload/confirmatory_component/state
logs=/data/zjq/paperb_upload/confirmatory_component/logs
runs=${repo}/experiment/wsr/generated/runs/confirmatory_component
variant=confirm_scale_only_p3
seed=3407
gpu=7
marker=${state}/${variant}_seed_${seed}
lock=${state}/locks/${variant}_seed_${seed}.lock
log=${logs}/${variant}_seed_${seed}.log

if find "${runs}" -path "*/${variant}/seed_${seed}/ablation_result.json" \
    -type f -print -quit 2>/dev/null | grep -q .; then
    printf 'complete\n' >"${marker}.status"
    exit 0
fi

if ! mkdir "${lock}" 2>/dev/null; then
    printf 'lock already exists for %s seed %s\n' "${variant}" "${seed}" >&2
    exit 1
fi

printf 'running gpu=%s\n' "${gpu}" >"${marker}.status"
if cd "${repo}" && PYTHONPATH="${repo}" "${python_bin}" -m experiment.wsr.ablation \
    train --confirmatory --variant "${variant}" --seed "${seed}" --device "${gpu}" --resume \
    >"${log}" 2>&1; then
    printf 'complete\n' >"${marker}.status"
    exit 0
fi

printf 'failed gpu=%s\n' "${gpu}" >"${marker}.status"
exit 1
