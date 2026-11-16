# Ablation using only the consensus terms (js_g / js_spatial)

IIM base, ImageNet-500-Hard as ID, 4-shot, seeds [1, 2, 3], 4 OOD sets; G=7 (2×2 patches per region), α=0.5 (share of js_g), γ=0.5.

## Main table

| score composition | pure-term FPR95 | pure-term AUROC | with base FPR95 | with base AUROC |
|---|---|---|---|---|
| js_g only | 65.30±0.96 | 84.55±0.88 | 29.19±1.29 | 93.78±0.30 |
| js_spatial only | 62.78±1.28 | 80.66±1.02 | 27.49±1.81 | 93.32±0.31 |
| both a=0.5 | 44.53±2.72 | 89.16±0.96 | 25.72±1.57 | 94.18±0.26 |
| S-MCM base | — | — | 29.28±1.48 | 93.33±0.20 |

## Per OOD dataset FPR95 (with base, a=0.5)

| set | FPR95 |
|---|---|
| iNaturalist | 9.11 |
| SUN | 26.85 |
| places365 | 34.09 |
| Texture | 32.81 |

## α sweep (γ=0.5, G=7)

| α | pure-term FPR95 | pure-term AUROC | with base FPR95 | with base AUROC |
|---|---|---|---|---|
| 0.0 | 62.78 | 80.66 | 27.49 | 93.32 |
| 0.1 | 57.03 | 83.03 | 26.77 | 93.60 |
| 0.2 | 51.56 | 85.23 | 26.35 | 93.82 |
| 0.3 | 47.12 | 87.08 | 25.88 | 94.00 |
| 0.4 | 44.59 | 88.43 | 25.71 | 94.12 |
| 0.5 | 44.53 | 89.16 | 25.72 | 94.18 |
| 0.6 | 46.34 | 89.24 | 25.98 | 94.20 |
| 0.7 | 50.11 | 88.71 | 26.44 | 94.17 |
| 0.8 | 55.15 | 87.65 | 27.35 | 94.09 |
| 0.9 | 60.49 | 86.21 | 28.19 | 93.96 |
| 1.0 | 65.30 | 84.55 | 29.19 | 93.78 |
