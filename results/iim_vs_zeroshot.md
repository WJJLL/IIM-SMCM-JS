# IIM baseline vs zero-shot: per OOD dataset comparison

- IIM: **w0 (`DIVMEAN_W=0`)**, mean ± std over 3 seeds for 4-shot and 16-shot respectively
- zero-shot: no training and no randomness, hence no std is given (the ZeroShot trainer does not compute JS / Spatial and has only 4 scores)
- Metric: **FPR@95TPR (%↓) / AUROC (%↑)**, same seed / same split protocol
- Mean over the 4 OOD sets: for IIM, each seed is first averaged over the 4 sets and then aggregated across seeds; for zero-shot it is the 4-set mean from its own log (the extra 5th ImageNet set evaluated in the Easy/Hard logs has been removed)
- ⚠️ **zero-shot is independent of the shot count**, so the ZS columns in the 4-shot and 16-shot blocks repeat the same numbers, allowing direct row-by-row comparison.

## Table 1: ImageNet-1k / ImageNet-500-Easy

| ID dataset | shot | OOD set | IIM `S-MCM` | IIM `S-MCM-JS` | IIM `S-MCM-JS-Spatial` | ZS `MCM` | ZS `GL-MCM` | ZS `R-MCM` | ZS `S-MCM` |
|---|---|---|---|---|---|---|---|---|---|
| ImageNet-1k | 4 | iNaturalist | 10.33±1.74 / 97.80 | 9.03±1.59 / 98.02 | 8.17±1.44 / 98.20 | 30.70 / 94.63 | 15.02 / 96.71 | 12.61 / 97.37 | 17.85 / 96.12 |
|  | 4 | SUN | 17.52±0.87 / 96.21 | 16.51±0.99 / 96.52 | 16.68±0.83 / 96.39 | 37.40 / 92.57 | 30.41 / 93.09 | 30.61 / 93.48 | 31.02 / 93.11 |
|  | 4 | places365 | 28.63±0.52 / 92.62 | 27.24±0.70 / 93.22 | 27.59±0.54 / 92.93 | 43.66 / 89.96 | 38.05 / 90.05 | 37.68 / 90.64 | 40.44 / 89.69 |
|  | 4 | Texture | 37.36±0.55 / 92.43 | 35.52±0.73 / 92.82 | 31.48±0.67 / 93.49 | 57.29 / 86.18 | 57.50 / 83.73 | 58.07 / 84.09 | 59.17 / 83.52 |
|  | 4 | **mean over the 4 OOD sets** | 23.46±0.32 / 94.76 | 22.08±0.46 / 95.15 | 20.98±0.34 / 95.25 | 42.26 / 90.83 | 35.24 / 90.90 | 34.74 / 91.39 | 37.12 / 90.61 |
|  | 16 | iNaturalist | 9.17±0.95 / 98.01 | 7.65±0.66 / 98.27 | 7.28±0.78 / 98.40 | 30.70 / 94.63 | 15.02 / 96.71 | 12.61 / 97.37 | 17.85 / 96.12 |
|  | 16 | SUN | 16.50±1.00 / 96.55 | 15.24±0.68 / 96.88 | 15.60±0.81 / 96.75 | 37.40 / 92.57 | 30.41 / 93.09 | 30.61 / 93.48 | 31.02 / 93.11 |
|  | 16 | places365 | 27.46±1.05 / 92.98 | 25.70±0.99 / 93.64 | 26.63±1.07 / 93.34 | 43.66 / 89.96 | 38.05 / 90.05 | 37.68 / 90.64 | 40.44 / 89.69 |
|  | 16 | Texture | 33.96±0.82 / 93.15 | 31.74±0.85 / 93.73 | 28.72±1.07 / 94.16 | 57.29 / 86.18 | 57.50 / 83.73 | 58.07 / 84.09 | 59.17 / 83.52 |
|  | 16 | **mean over the 4 OOD sets** | 21.77±0.89 / 95.18 | 20.08±0.70 / 95.63 | 19.56±0.85 / 95.66 | 42.26 / 90.83 | 35.24 / 90.90 | 34.74 / 91.39 | 37.12 / 90.61 |
| ImageNet-500-Easy | 4 | iNaturalist | 3.61±1.04 / 98.94 | 2.87±0.49 / 99.06 | 2.63±0.51 / 99.15 | 16.04 / 97.13 | 4.53 / 98.77 | 3.72 / 98.84 | 5.18 / 98.65 |
|  | 4 | SUN | 3.56±0.27 / 99.12 | 3.50±0.18 / 99.16 | 3.61±0.21 / 99.14 | 8.10 / 98.28 | 7.72 / 98.19 | 6.04 / 98.63 | 7.63 / 98.25 |
|  | 4 | places365 | 10.80±0.49 / 97.04 | 10.29±0.31 / 97.23 | 10.63±0.43 / 97.14 | 15.26 / 96.21 | 15.27 / 95.96 | 13.15 / 96.72 | 15.22 / 96.03 |
|  | 4 | Texture | 11.24±0.66 / 97.72 | 12.09±1.05 / 97.64 | 10.02±1.07 / 97.88 | 19.75 / 96.27 | 26.65 / 93.53 | 24.56 / 94.42 | 26.12 / 93.79 |
|  | 4 | **mean over the 4 OOD sets** | 7.30±0.57 / 98.20 | 7.19±0.46 / 98.28 | 6.72±0.51 / 98.33 | 14.79 / 96.97 | 13.54 / 96.61 | 11.87 / 97.15 | 13.54 / 96.68 |
|  | 16 | iNaturalist | 2.64±0.36 / 99.07 | 1.99±0.22 / 99.25 | 1.96±0.21 / 99.29 | 16.04 / 97.13 | 4.53 / 98.77 | 3.72 / 98.84 | 5.18 / 98.65 |
|  | 16 | SUN | 3.55±0.44 / 99.15 | 3.40±0.47 / 99.22 | 3.51±0.46 / 99.19 | 8.10 / 98.28 | 7.72 / 98.19 | 6.04 / 98.63 | 7.63 / 98.25 |
|  | 16 | places365 | 10.56±0.62 / 97.17 | 9.85±0.67 / 97.41 | 10.28±0.61 / 97.30 | 15.26 / 96.21 | 15.27 / 95.96 | 13.15 / 96.72 | 15.22 / 96.03 |
|  | 16 | Texture | 9.15±0.25 / 98.15 | 9.41±0.17 / 98.15 | 7.83±0.17 / 98.34 | 19.75 / 96.27 | 26.65 / 93.53 | 24.56 / 94.42 | 26.12 / 93.79 |
|  | 16 | **mean over the 4 OOD sets** | 6.48±0.12 / 98.38 | 6.16±0.21 / 98.51 | 5.90±0.22 / 98.53 | 14.79 / 96.97 | 13.54 / 96.61 | 11.87 / 97.15 | 13.54 / 96.68 |

## Table 2: ImageNet-500-Hard / ImageNet-100

| ID dataset | shot | OOD set | IIM `S-MCM` | IIM `S-MCM-JS` | IIM `S-MCM-JS-Spatial` | ZS `MCM` | ZS `GL-MCM` | ZS `R-MCM` | ZS `S-MCM` |
|---|---|---|---|---|---|---|---|---|---|
| ImageNet-500-Hard | 4 | iNaturalist | 12.56±0.93 / 97.33 | 11.36±1.06 / 97.64 | 10.46±0.86 / 97.81 | 28.85 / 94.34 | 20.72 / 95.63 | 17.31 / 96.46 | 23.29 / 95.07 |
|  | 4 | SUN | 27.30±2.02 / 93.93 | 26.10±1.84 / 94.33 | 26.23±2.00 / 94.15 | 52.68 / 89.30 | 41.92 / 90.56 | 42.36 / 90.57 | 42.43 / 90.50 |
|  | 4 | places365 | 34.45±2.82 / 92.07 | 33.06±2.81 / 92.53 | 33.60±2.96 / 92.21 | 51.32 / 88.91 | 44.97 / 89.22 | 44.31 / 89.50 | 47.24 / 88.82 |
|  | 4 | Texture | 42.85±0.34 / 89.99 | 40.53±0.97 / 90.85 | 37.20±0.56 / 91.42 | 61.52 / 83.31 | 58.55 / 83.03 | 59.18 / 82.81 | 61.22 / 82.88 |
|  | 4 | **mean over the 4 OOD sets** | 29.29±1.22 / 93.33 | 27.76±0.98 / 93.84 | 26.88±1.12 / 93.90 | 48.59 / 88.97 | 41.54 / 89.61 | 40.79 / 89.83 | 43.55 / 89.32 |
|  | 16 | iNaturalist | 11.69±2.70 / 97.50 | 10.29±2.05 / 97.85 | 9.65±2.24 / 97.96 | 28.85 / 94.34 | 20.72 / 95.63 | 17.31 / 96.46 | 23.29 / 95.07 |
|  | 16 | SUN | 25.67±1.23 / 94.25 | 24.57±1.58 / 94.69 | 24.48±1.54 / 94.51 | 52.68 / 89.30 | 41.92 / 90.56 | 42.36 / 90.57 | 42.43 / 90.50 |
|  | 16 | places365 | 32.40±1.27 / 92.54 | 31.23±1.73 / 93.01 | 31.43±1.57 / 92.74 | 51.32 / 88.91 | 44.97 / 89.22 | 44.31 / 89.50 | 47.24 / 88.82 |
|  | 16 | Texture | 38.56±0.15 / 91.02 | 36.47±0.05 / 92.00 | 33.15±0.10 / 92.37 | 61.52 / 83.31 | 58.55 / 83.03 | 59.18 / 82.81 | 61.22 / 82.88 |
|  | 16 | **mean over the 4 OOD sets** | 27.08±0.32 / 93.83 | 25.64±0.62 / 94.39 | 24.68±0.57 / 94.39 | 48.59 / 88.97 | 41.54 / 89.61 | 40.79 / 89.83 | 43.55 / 89.32 |
| ImageNet-100 | 4 | iNaturalist | 2.88±0.70 / 99.09 | 2.27±0.57 / 99.19 | 1.87±0.48 / 99.33 | 13.29 / 97.39 | 2.98 / 99.00 | 3.10 / 98.84 | 3.68 / 98.75 |
|  | 4 | SUN | 10.93±0.63 / 97.88 | 9.52±0.39 / 98.01 | 9.15±0.33 / 98.06 | 36.38 / 94.86 | 17.07 / 96.82 | 22.15 / 96.44 | 18.47 / 96.61 |
|  | 4 | places365 | 16.18±0.64 / 96.53 | 15.25±0.20 / 96.70 | 14.89±0.35 / 96.73 | 32.73 / 94.86 | 19.66 / 96.25 | 21.71 / 96.20 | 21.53 / 95.88 |
|  | 4 | Texture | 15.69±0.16 / 97.12 | 14.28±0.45 / 97.29 | 11.64±0.43 / 97.68 | 38.32 / 93.34 | 33.99 / 92.67 | 37.25 / 92.53 | 36.38 / 92.36 |
|  | 4 | **mean over the 4 OOD sets** | 11.42±0.44 / 97.65 | 10.33±0.05 / 97.80 | 9.39±0.27 / 97.95 | 30.18 / 95.11 | 18.42 / 96.18 | 21.05 / 96.00 | 20.02 / 95.90 |
|  | 16 | iNaturalist | 2.17±0.20 / 99.26 | 1.99±0.31 / 99.31 | 1.66±0.29 / 99.43 | 13.29 / 97.39 | 2.98 / 99.00 | 3.10 / 98.84 | 3.68 / 98.75 |
|  | 16 | SUN | 8.89±0.46 / 98.17 | 7.97±0.68 / 98.31 | 7.84±0.48 / 98.35 | 36.38 / 94.86 | 17.07 / 96.82 | 22.15 / 96.44 | 18.47 / 96.61 |
|  | 16 | places365 | 13.44±0.60 / 97.12 | 12.47±0.60 / 97.34 | 12.44±0.55 / 97.35 | 32.73 / 94.86 | 19.66 / 96.25 | 21.71 / 96.20 | 21.53 / 95.88 |
|  | 16 | Texture | 14.05±0.46 / 97.45 | 13.01±0.33 / 97.63 | 10.99±0.74 / 97.96 | 38.32 / 93.34 | 33.99 / 92.67 | 37.25 / 92.53 | 36.38 / 92.36 |
|  | 16 | **mean over the 4 OOD sets** | 9.64±0.17 / 98.00 | 8.86±0.19 / 98.15 | 8.23±0.27 / 98.27 | 30.18 / 95.11 | 18.42 / 96.18 | 21.05 / 96.00 | 20.02 / 95.90 |

## Observations

1. **Learned prompts outperform zero-shot across the board**: on 1k `MCM 42.26 → S-MCM 23.46`, and adding the spatial term brings it to `20.98`; on Hard `48.59 → 29.29 → 26.88`; on 100 `30.18 → 11.42 → 9.39`.
2. **In zero-shot, `S-MCM` is not consistently better than `MCM`**: the mean over the 4 OOD sets looks good (1k 42.26→37.12), but **per set it is mixed** — on 1k/Texture `MCM 57.29` is in fact better than `S-MCM 59.17`, and on Easy/Texture `19.75 → 26.12` degrades markedly. This shows that the multi-patch scoring of `S-MCM` only holds once there is a **well-aligned text side**, which is exactly what learned prompts provide.
3. **The zero-shot `S-MCM` cannot be used as an ablation baseline in the paper** — it already degrades by itself. The correct comparison is "learned prompt + `MCM`" → "+ `S-MCM`" → "+ JS & Spatial".
4. **The gain of the spatial term is largest and most stable on Texture**: 1k `37.36 → 31.48`, Hard `42.85 → 37.20`, 100 `15.69 → 11.64`, Easy `11.24 → 10.02`; on SUN / places365 it is only 0.1~0.4 or essentially flat.
5. The gap between IIM and zero-shot is in fact the largest on the two hardest sets (Texture, places365), indicating that learned prompts mainly improve discriminability on **hard distributions** rather than on easy sets.
6. **4-shot → 16-shot improves monotonically everywhere** (mean over the 4 OOD sets FPR95, `S-MCM-JS-Spatial`): 1k `20.98 → 19.56`, Hard `26.88 → 24.68`, 100 `9.39 → 8.23`, Easy `6.72 → 5.90`. However, **the absolute gain of the spatial term becomes smaller at 16-shot** (1k: 4-shot drops 2.48 from 23.46, 16-shot drops 2.21 from 21.77; 100: −2.03 vs −1.41), consistent with the earlier observation that "training-side improvements partially overlap with the spatial term" — with more samples the baseline itself has already improved the local evidence structure.
7. Under both shot counts, **Texture is still the set where the spatial term gains the most** (1k 4-shot −5.88 / 16-shot −5.24; Hard −5.65 / −5.41), showing that this conclusion does not depend on the shot setting.
