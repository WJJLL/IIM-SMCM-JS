#!/bin/bash
# custom config
TRAINER=LOCALPROMPT

CSC=True
CTP=end

DATA=${1:-/home/star/code/IIM-SMCM/data/ood/}
DATASET=$2
CFG=$3
NCTX=${4:-16}
topk=${5:-10}
MODEL_dir=$6
Output_dir=${7:-${MODEL_dir}/eval}
EPOCH=${8:-30}
num_neg=${9:-300}

CUDA_VISIBLE_DEVICES=0 python eval_ood_detection.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--in_dataset ${DATASET} \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--load-epoch ${EPOCH} \
--output-dir ${Output_dir} \
--model-dir ${MODEL_dir} \
--top_k ${topk} \
--num_neg_prompts ${num_neg} \
TRAINER.LOCALPROMPT.N_CTX ${NCTX} \
TRAINER.LOCALPROMPT.CSC ${CSC} \
TRAINER.LOCALPROMPT.CLASS_TOKEN_POSITION ${CTP}
