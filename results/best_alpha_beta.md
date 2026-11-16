# Best configuration (ImageNet-500-Hard, IIM base 4-shot, 3 seeds)

| configuration | (α,β) | γ | FPR95 | AUROC | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|---|---|
| S-MCM base | (0,0) | 0 | **29.28±1.48** | 93.33±0.20 | 29.74 | 27.63 | 30.48 |
| Global JS (a=b=1) | (1,1) | 0.2 | **27.74±1.19** | 93.84±0.23 | 28.39 | 26.37 | 28.46 |
| best (a=.75,b=.5) | (0.75,0.5) | 0.2 | **27.38±0.94** | 93.99±0.27 | 28.34 | 26.47 | 27.33 |

## FPR95 per OOD dataset

| configuration | iNaturalist | SUN | places365 | Texture |
|---|---|---|---|---|
| S-MCM base | 12.55 | 27.30 | 34.43 | 42.84 |
| Global JS (a=b=1) | 11.34 | 26.09 | 33.05 | 40.48 |
| best (a=.75,b=.5) | 12.13 | 26.05 | 33.18 | 38.16 |
