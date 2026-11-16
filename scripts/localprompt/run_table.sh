#!/bin/bash
# Run LocalPrompt on ImageNet-100 / 500-Easy / 500-Hard, 4 & 16 shots, seeds 1-3,
# then evaluate OOD detection (FPR95 / AUROC).
set -u

cd /home/star/code/IIM-SMCM

PY=/home/star/miniconda3/envs/dassl/bin/python
ROOT=/home/star/code/IIM-SMCM/data/ood/
CFG=configs/trainers/LOCALPROMPT/vit_b16_ep30.yaml
BASE=output/localprompt_table
LOG=${BASE}/logs
mkdir -p ${LOG}

train_and_eval () {
  DS_CFG=$1      # dataset config file
  PART=$2        # all / base / new
  IN_DS=$3       # imagenet100 / imagenet
  TAG=$4         # output tag

  for SHOTS in 4 16; do
    for SEED in 1 2 3; do
      DIR=${BASE}/${TAG}/${SHOTS}shot/seed${SEED}
      mkdir -p ${DIR}

      echo "=== TRAIN ${TAG} ${SHOTS}shot seed${SEED} $(date '+%F %T') ==="
      CUDA_VISIBLE_DEVICES=0 ${PY} train.py \
        --root ${ROOT} --seed ${SEED} --trainer LOCALPROMPT \
        --dataset-config-file ${DS_CFG} \
        --config-file ${CFG} \
        --output-dir ${DIR} \
        --lambda_value 5 --div_value 0.5 --topk 50 --num_neg_prompts 300 \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${PART} \
        DATALOADER.TEST.BATCH_SIZE 256 \
        > ${LOG}/${TAG}_${SHOTS}shot_seed${SEED}_train.log 2>&1

      echo "=== EVAL  ${TAG} ${SHOTS}shot seed${SEED} $(date '+%F %T') ==="
      CUDA_VISIBLE_DEVICES=0 ${PY} eval_ood_detection.py \
        --root ${ROOT} --trainer LOCALPROMPT \
        --dataset-config-file ${DS_CFG} \
        --in_dataset ${IN_DS} \
        --config-file ${CFG} \
        --load-epoch 30 \
        --output-dir ${DIR}/eval --model-dir ${DIR} \
        --top_k 10 --num_neg_prompts 300 -b 256 \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${PART} \
        DATALOADER.TEST.BATCH_SIZE 256 \
        > ${LOG}/${TAG}_${SHOTS}shot_seed${SEED}_eval.log 2>&1
    done
  done
}

train_and_eval configs/datasets/imagenet100.yaml all  imagenet100 imagenet100
train_and_eval configs/datasets/imagenet.yaml    base imagenet    imagenet500easy
train_and_eval configs/datasets/imagenet.yaml    new  imagenet    imagenet500hard

echo "=== ALL DONE $(date '+%F %T') ==="
