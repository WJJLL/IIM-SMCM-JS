# IIM base 4-shot: 4 bases + JS-Spatial (γ=0.5, α_spatial=0.5)

**Logged under**: `/mnt/mydrive/iim/output`, `base_4shots` / `new_4shots` / `all_4shots`  
**Hyperparameters**: `JS_GAMMA=0.5`, `JS_SPATIAL_A=0.5`, `JS_SPATIAL_GRID=7`  
**Seeds**: [1, 2, 3]  
**Metric**: FPR@95TPR (%↓) / AUROC (%↑), mean over the 4 OOD sets  
**Speed-up**: each image is forwarded once per seed; the components are dumped to disk and re-scored offline, without re-running the model

## ImageNet-500-Hard

| score | iNaturalist | SUN | places365 | Texture | **mean over the 4 OOD sets** |
|--------------------|--------------|--------------|--------------|--------------|--------------------|
| MCM | 30.22 | 50.47 | 50.46 | 63.16 | 48.58±0.00 / 88.72 |
| **MCM-JS-Spatial** | 12.38 | 32.12 | 36.66 | 39.44 | **30.15±0.88 / 93.22** |
| GL-MCM | 23.25 | 40.36 | 45.70 | 56.01 | 41.33±2.26 / 89.26 |
| **GL-MCM-JS-Spatial** | 14.11 | 34.08 | 40.79 | 40.19 | **32.29±2.00 / 91.91** |
| R-MCM | 16.68 | 37.09 | 40.39 | 57.83 | 38.00±2.09 / 90.98 |
| **R-MCM-JS-Spatial** | 8.80 | 28.66 | 34.31 | 39.68 | **27.86±1.68 / 93.56** |
| S-MCM | 12.55 | 27.30 | 34.43 | 42.84 | 29.28±1.48 / 93.33 |
| **S-MCM-JS-Spatial** | 9.11 | 26.85 | 34.08 | 32.81 | **25.71±1.57 / 94.18** |

## ImageNet-500-Easy

| score | iNaturalist | SUN | places365 | Texture | **mean over the 4 OOD sets** |
|--------------------|--------------|--------------|--------------|--------------|--------------------|
| MCM | 14.73 | 8.19 | 14.89 | 20.27 | 14.52±0.00 / 97.01 |
| **MCM-JS-Spatial** | 3.85 | 4.97 | 11.80 | 10.89 | **7.88±0.05 / 98.08** |
| GL-MCM | 6.62 | 7.08 | 14.07 | 20.89 | 12.16±1.17 / 96.94 |
| **GL-MCM-JS-Spatial** | 4.66 | 7.11 | 14.21 | 16.42 | **10.60±0.76 / 97.32** |
| R-MCM | 4.76 | 4.27 | 11.13 | 18.36 | 9.63±0.78 / 97.69 |
| **R-MCM-JS-Spatial** | 2.29 | 3.97 | 10.50 | 12.14 | **7.22±0.36 / 98.12** |
| S-MCM | 3.60 | 3.56 | 10.79 | 11.24 | 7.30±0.70 / 98.20 |
| **S-MCM-JS-Spatial** | 2.47 | 4.23 | 11.34 | 9.79 | **6.96±0.46 / 98.29** |

## ImageNet-100

| score | iNaturalist | SUN | places365 | Texture | **mean over the 4 OOD sets** |
|--------------------|--------------|--------------|--------------|--------------|--------------------|
| MCM | 11.86 | 29.43 | 28.80 | 37.54 | 26.91±0.00 / 95.18 |
| **MCM-JS-Spatial** | 1.91 | 13.08 | 16.47 | 16.06 | **11.88±0.42 / 97.72** |
| GL-MCM | 3.61 | 13.45 | 20.16 | 23.82 | 15.26±0.48 / 96.92 |
| **GL-MCM-JS-Spatial** | 2.16 | 11.86 | 18.72 | 15.53 | **12.07±0.21 / 97.62** |
| R-MCM | 4.50 | 19.99 | 21.75 | 31.24 | 19.37±0.66 / 96.45 |
| **R-MCM-JS-Spatial** | 1.27 | 10.55 | 14.76 | 14.40 | **10.24±0.11 / 97.98** |
| S-MCM | 2.64 | 10.16 | 15.41 | 14.86 | 10.77±0.67 / 97.72 |
| **S-MCM-JS-Spatial** | 1.61 | 9.42 | 15.73 | 9.80 | **9.14±0.22 / 98.05** |

## JS-Spatial gain (mean over the 4 OOD sets, 3-seed mean)

| ID dataset | base | original FPR95 | +JS-Spatial | ΔFPR | ΔAUROC |
|---|---|---|---|---|---|
| ImageNet-500-Hard | MCM | 48.58 | 30.15 | **-18.43** | +4.50 |
| ImageNet-500-Hard | GL-MCM | 41.33 | 32.29 | **-9.04** | +2.65 |
| ImageNet-500-Hard | R-MCM | 38.00 | 27.86 | **-10.13** | +2.58 |
| ImageNet-500-Hard | S-MCM | 29.28 | 25.71 | **-3.57** | +0.85 |
| ImageNet-500-Easy | MCM | 14.52 | 7.88 | **-6.64** | +1.07 |
| ImageNet-500-Easy | GL-MCM | 12.16 | 10.60 | **-1.57** | +0.37 |
| ImageNet-500-Easy | R-MCM | 9.63 | 7.22 | **-2.41** | +0.43 |
| ImageNet-500-Easy | S-MCM | 7.30 | 6.96 | **-0.34** | +0.09 |
| ImageNet-100 | MCM | 26.91 | 11.88 | **-15.03** | +2.54 |
| ImageNet-100 | GL-MCM | 15.26 | 12.07 | **-3.19** | +0.70 |
| ImageNet-100 | R-MCM | 19.37 | 10.24 | **-9.12** | +1.53 |
| ImageNet-100 | S-MCM | 10.77 | 9.14 | **-1.63** | +0.33 |

## Per-seed pairing (is JS-Spatial better on all 3/3 seeds?)

| ID dataset | base | seed1 | seed2 | seed3 | win rate |
|---|---|---|---|---|---|
| ImageNet-500-Hard | MCM | -17.82 | -19.43 | -18.03 | 3/3 |
| ImageNet-500-Hard | GL-MCM | -8.77 | -8.79 | -9.55 | 3/3 |
| ImageNet-500-Hard | R-MCM | -10.25 | -9.68 | -10.47 | 3/3 |
| ImageNet-500-Hard | S-MCM | -3.29 | -3.72 | -3.69 | 3/3 |
| ImageNet-500-Easy | MCM | -6.69 | -6.59 | -6.64 | 3/3 |
| ImageNet-500-Easy | GL-MCM | -1.56 | -2.08 | -1.06 | 3/3 |
| ImageNet-500-Easy | R-MCM | -2.90 | -2.21 | -2.12 | 3/3 |
| ImageNet-500-Easy | S-MCM | -0.61 | -0.31 | -0.11 | 3/3 |
| ImageNet-100 | MCM | -14.98 | -15.47 | -14.63 | 3/3 |
| ImageNet-100 | GL-MCM | -2.84 | -3.37 | -3.37 | 3/3 |
| ImageNet-100 | R-MCM | -8.34 | -9.79 | -9.25 | 3/3 |
| ImageNet-100 | S-MCM | -1.46 | -1.14 | -2.29 | 3/3 |

