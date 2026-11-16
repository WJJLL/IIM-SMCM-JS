#!/bin/bash
# Zero-shot CLIP OOD detection with 12 scores:
#   {MCM, GL-MCM, R-MCM, S-MCM} x {"", "-JS", "-JS-Spatial"}
# ID: ImageNet-100 / 500-Easy(base) / 500-Hard(new) / 1K(all)
# OOD: iNaturalist / SUN / Places365 / Texture (dtd)
set -u

cd /home/star/code/IIM-SMCM

PY=/home/star/miniconda3/envs/dassl/bin/python
ROOT=/home/star/code/IIM-SMCM/data/ood
BASE=output/zeroshot_js
LOG=${BASE}/logs
mkdir -p ${LOG}

run () {
  DS_CFG=$1   # dataset config
  PART=$2     # all / base / new
  IN_DS=$3    # imagenet100 / imagenet
  TAG=$4      # output tag
  OUT=${BASE}/${TAG}
  mkdir -p ${OUT}

  echo "=== ZERO-SHOT-JS ${TAG} (${PART}) start $(date '+%F %T') ==="
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
  echo "=== ZERO-SHOT-JS ${TAG} done $(date '+%F %T') exit=${RC} ==="
}

run configs/datasets/imagenet100.yaml all  imagenet100 imagenet100
run configs/datasets/imagenet.yaml    base imagenet    imagenet500easy
run configs/datasets/imagenet.yaml    new  imagenet    imagenet500hard
run configs/datasets/imagenet.yaml    all  imagenet    imagenet1k

echo "=== ALL ZERO-SHOT-JS DONE $(date '+%F %T') ==="
