#!/bin/bash
# ImageNet-1K (ID = all, 50k) 4-shot x 4 OOD sets, extra S-MCM-Spatial run.
# S-MCM / S-MCM-JS / S-MCM-JS-Spatial come from <ckpt>/eval_spatial/log.txt.
# JS_SPATIAL_W="0,1" -> pure spatial term (coef_js = 0, gamma = 0.2).
set -u

cd /home/star/zmluo/IIM-SMCM || exit 1

PY=/home/star/miniconda3/envs/dassl/bin/python
B=/mnt/mydrive/iim/output_1k/imagenet/IIM/common_4shots/numneg300_topk50_lamda5_div0.5

for S in 1 2 3; do
  D=${B}/seed${S}_all
  OUT=${D}/eval_smcm_spatial
  mkdir -p ${OUT}
  echo "=== im1k S-MCM-Spatial seed${S} start $(date '+%F %T') ==="
  JS_SPATIAL_W="0,1" PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 ${PY} eval_ood_detection.py \
    --root /home/star/code/IIM-SMCM/data/ood \
    --trainer IIM \
    --dataset-config-file configs/datasets/imagenet.yaml \
    --in_dataset imagenet \
    --config-file configs/trainers/IIM/common.yaml \
    --load-epoch 50 \
    --output-dir ${OUT} --model-dir ${D} \
    --top_k 50 --alpha 1.0 --num_neg_prompts 300 --batch-size 256 \
    DATASET.NUM_SHOTS 4 DATASET.SUBSAMPLE_CLASSES all \
    > /tmp/im1k_smcm_spatial_s${S}.log 2>&1
  echo "=== im1k S-MCM-Spatial seed${S} done $(date '+%F %T') exit=$? ==="
done

echo "=== IM1K S-MCM-SPATIAL ALL DONE $(date '+%F %T') ==="
