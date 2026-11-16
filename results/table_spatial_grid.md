# Spatial grid granularity G (S-MCM-JS-Spatial, γ=0.5, α_sp=0.5)

IIM base, ImageNet-500-Hard as ID, 4-shot, seeds [1, 2, 3], 4 OOD sets; FPR95/AUROC are the seed mean±std of the 4-OOD mean.

## Main table (only uniform partitions that divide 14 evenly)

| G | patches per region | Spatial single-feature AUROC | FPR95 | AUROC | ΔFPR95 |
|---|---|---|---|---|---|
| — (S-MCM base) | — | — | 29.28±1.48 | 93.33±0.20 | — |
| 2 | 49 | 61.55 | 27.89±1.38 | 93.74±0.28 | -1.39 |
| **7** | 4 | 80.66 | **25.72±1.57** | **94.18±0.26** | **-3.56** |
| 14 † | 1 | 50.00 | 27.88±1.15 | 93.88±0.24 | -1.39 |

† At $G=14$ each region has only 1 patch; here $\mathrm{JS}^{(m)}\equiv0$, so the Spatial term is identically 0, its single-feature AUROC is exactly 50.0, and the term effectively fails (only the JS with halved strength is left).

## Per OOD dataset FPR95 (γ=0.5)

| G | iNaturalist | SUN | places365 | Texture |
|---|---|---|---|---|
| 2 | 10.59 | 28.66 | 36.37 | 35.94 |
| 3 | 10.15 | 29.13 | 37.29 | 33.49 |
| 4 | 9.92 | 28.32 | 36.20 | 33.37 |
| 5 | 10.14 | 28.55 | 36.52 | 32.36 |
| 6 | 9.53 | 27.48 | 35.17 | 32.42 |
| 7 | 9.11 | 26.85 | 34.09 | 32.81 |
| 8 | 9.89 | 27.18 | 34.48 | 33.35 |
| 10 | 9.63 | 26.87 | 34.04 | 33.28 |
| 14 | 11.39 | 26.39 | 33.27 | 40.48 |

## Supplementary table (all G, including non-uniform partitions)

| G | patches per region | uniform | Spatial single-feature AUROC | FPR95 | AUROC | ΔFPR95 |
|---|---|---|---|---|---|---|
| 2 | 49 | yes | 61.55 | 27.89±1.38 | 93.74±0.28 | -1.39 |
| 3 | unequal | no | 68.63 | 27.52±1.61 | 93.82±0.28 | -1.76 |
| 4 | unequal | no | 72.52 | 26.95±1.70 | 94.00±0.26 | -2.32 |
| 5 | unequal | no | 76.55 | 26.90±1.42 | 94.05±0.25 | -2.38 |
| 6 | unequal | no | 77.21 | 26.15±1.54 | 94.14±0.26 | -3.12 |
| 7 | 4 | yes | 80.66 | 25.72±1.57 | 94.18±0.26 | -3.56 |
| 8 | unequal | no | 83.39 | 26.23±1.60 | 94.19±0.24 | -3.05 |
| 10 | unequal | no | 84.46 | 25.95±1.64 | 94.25±0.24 | -3.32 |
| 14 | 1 | yes | 50.00 | 27.88±1.15 | 93.88±0.24 | -1.39 |
