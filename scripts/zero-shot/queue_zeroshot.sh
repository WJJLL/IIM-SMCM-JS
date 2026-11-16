#!/bin/bash
# Queue zero-shot CLIP OOD detection on ImageNet-100 / 500-Easy / 500-Hard / 1K.
# Waits for BOTH IIM eval queues to finish first:
#   /tmp/run_eval_1k.sh     -> "EVAL_1K ALL DONE"     in /tmp/run_eval1k.out
#   /tmp/run_eval_full9_v2.sh -> "EVAL_FULL9 ALL DONE" in /tmp/run_full9v2.out
# Then, before each run, also require no train/eval process and a stable
# 2-minute idle GPU window, so we never collide with other queued jobs.
# Logs: output/zeroshot/logs/<tag>.log
set -u

cd /home/star/code/IIM-SMCM

PY=/home/star/miniconda3/envs/dassl/bin/python
ROOT=/home/star/code/IIM-SMCM/data/ood
BASE=output/zeroshot
LOG=${BASE}/logs
mkdir -p ${LOG}

BUSY_PAT="python.*eval_ood_detection\.py|python.*train\.py"
GPU_MIN_FREE=20000  # MiB

gpu_free () {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '
}

# Wait for a completion marker; if the driving script is gone, do not block forever.
wait_marker () {
  local f=$1 pat=$2 script=$3 i
  for i in $(seq 1 720); do
    grep -q "${pat}" "$f" 2>/dev/null && return 0
    if ! pgrep -f "bash ${script}" > /dev/null; then
      echo "--- ${script} is not running and '${pat}' absent, continue"
      return 0
    fi
    sleep 60
  done
  echo "--- timeout waiting for '${pat}' in ${f}"
  return 0
}

# Wait for no train/eval process and a stable 2-minute idle GPU window.
wait_free () {
  while true; do
    if ! pgrep -f "${BUSY_PAT}" > /dev/null; then
      free=$(gpu_free)
      if [ -n "${free}" ] && [ "${free}" -ge ${GPU_MIN_FREE} ]; then
        sleep 120
        if ! pgrep -f "${BUSY_PAT}" > /dev/null; then
          free=$(gpu_free)
          if [ -n "${free}" ] && [ "${free}" -ge ${GPU_MIN_FREE} ]; then
            return 0
          fi
        fi
        continue
      fi
    fi
    sleep 60
  done
}

run () {
  DS_CFG=$1   # dataset config
  PART=$2     # all / base / new
  IN_DS=$3    # imagenet100 / imagenet
  TAG=$4      # output tag
  OUT=${BASE}/${TAG}
  mkdir -p ${OUT}

  wait_free
  echo "=== ZERO-SHOT ${TAG} (${PART}) start $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES=0 ${PY} eval_ood_detection_zero_shot.py \
    --root ${ROOT} \
    --trainer ZeroShot \
    --dataset-config-file ${DS_CFG} \
    --in_dataset ${IN_DS} \
    --config-file configs/trainers/ZeroShot/null.yaml \
    --output-dir ${OUT} \
    --top_k 50 --T 1 --alpha 1.0 -b 256 \
    DATASET.NUM_SHOTS 0 \
    DATASET.SUBSAMPLE_CLASSES ${PART} \
    DATALOADER.TEST.BATCH_SIZE 256 \
    > ${LOG}/${TAG}.log 2>&1
  RC=$?
  echo "=== ZERO-SHOT ${TAG} done $(date '+%F %T') exit=${RC} ==="
}

echo "=== waiting for 1k eval queue $(date '+%F %T') ==="
wait_marker /tmp/run_eval1k.out "EVAL_1K ALL DONE" "/tmp/run_eval_1k.sh"
echo "=== waiting for full9 eval queue $(date '+%F %T') ==="
wait_marker /tmp/run_full9v2.out "EVAL_FULL9 ALL DONE" "/tmp/run_eval_full9_v2.sh"

run configs/datasets/imagenet100.yaml all  imagenet100 imagenet100
run configs/datasets/imagenet.yaml    base imagenet    imagenet500easy
run configs/datasets/imagenet.yaml    new  imagenet    imagenet500hard
run configs/datasets/imagenet.yaml    all  imagenet    imagenet1k

echo "=== ALL ZERO-SHOT DONE $(date '+%F %T') ==="
