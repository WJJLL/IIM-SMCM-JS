# Distribution of Global JS on ID / OOD (4-shot, seeds=[1, 2, 3])

Larger JS means more ID-like. Distribution statistics are **computed per seed and then reported as seed mean±sample std**; `n` is the number of samples in a **single ckpt** (identical to the boxes in the figure). FPR95/AUROC are the detection performance of **the JS feature alone** (computed for each OOD dataset separately).

## ImageNet-500-Hard

| set | n | mean | median | IQR | JS-only FPR95 | JS-only AUROC |
|---|---|---|---|---|---|---|
| ID | 25000 | 1.4185±0.0041 | 1.4257±0.0037 | 0.2235 | — | — |
| iNat | 10000 | 1.1702±0.0071 | 1.1695±0.0084 | 0.1380 | 62.07±5.68 | 89.45±1.06 |
| SUN | 10000 | 1.1800±0.0221 | 1.1859±0.0250 | 0.2025 | 62.41±4.42 | 85.21±1.64 |
| Places | 10000 | 1.2106±0.0220 | 1.2113±0.0238 | 0.1958 | 69.51±3.60 | 82.52±1.95 |
| Texture | 5640 | 1.2054±0.0237 | 1.2204±0.0271 | 0.2388 | 67.22±4.94 | 81.01±2.48 |

- ID median − OOD median: iNat -0.2561, SUN -0.2397, Places -0.2144, Texture -0.2053

## ImageNet-500-Easy

| set | n | mean | median | IQR | JS-only FPR95 | JS-only AUROC |
|---|---|---|---|---|---|---|
| ID | 25000 | 1.2649±0.0422 | 1.2656±0.0429 | 0.1859 | — | — |
| iNat | 10000 | 1.0985±0.0415 | 1.0974±0.0410 | 0.1294 | 74.78±4.35 | 83.63±1.83 |
| SUN | 10000 | 1.0755±0.0481 | 1.0808±0.0495 | 0.1772 | 64.04±3.17 | 83.61±0.98 |
| Places | 10000 | 1.1069±0.0475 | 1.1074±0.0481 | 0.1687 | 72.37±2.55 | 79.91±0.94 |
| Texture | 5640 | 1.1264±0.0269 | 1.1362±0.0276 | 0.2275 | 72.02±4.13 | 73.23±2.69 |

- ID median − OOD median: iNat -0.1682, SUN -0.1848, Places -0.1582, Texture -0.1294

## ImageNet-100

| set | n | mean | median | IQR | JS-only FPR95 | JS-only AUROC |
|---|---|---|---|---|---|---|
| ID | 5000 | 1.2765±0.0111 | 1.2821±0.0118 | 0.1909 | — | — |
| iNat | 10000 | 1.0607±0.0172 | 1.0586±0.0177 | 0.1359 | 57.19±3.37 | 89.00±1.35 |
| SUN | 10000 | 1.0891±0.0122 | 1.0940±0.0106 | 0.1729 | 66.25±4.20 | 83.48±1.48 |
| Places | 10000 | 1.1210±0.0083 | 1.1218±0.0068 | 0.1716 | 74.08±2.57 | 79.24±0.77 |
| Texture | 5640 | 1.1058±0.0073 | 1.1180±0.0052 | 0.2075 | 68.76±0.49 | 79.46±0.27 |

- ID median − OOD median: iNat -0.2235, SUN -0.1881, Places -0.1603, Texture -0.1641

## ImageNet-1k

| set | n | mean | median | IQR | JS-only FPR95 | JS-only AUROC |
|---|---|---|---|---|---|---|
| ID | 50000 | 1.4760±0.0055 | 1.4783±0.0056 | 0.2242 | — | — |
| iNat | 10000 | 1.2585±0.0023 | 1.2564±0.0020 | 0.1498 | 67.50±2.63 | 86.16±0.93 |
| SUN | 10000 | 1.2535±0.0091 | 1.2548±0.0096 | 0.2025 | 63.27±1.05 | 83.81±0.33 |
| Places | 10000 | 1.2924±0.0090 | 1.2894±0.0084 | 0.1957 | 72.15±0.96 | 79.77±0.39 |
| Texture | 5640 | 1.3190±0.0070 | 1.3313±0.0069 | 0.2485 | 74.15±0.64 | 73.50±0.64 |

- ID median − OOD median: iNat -0.2219, SUN -0.2235, Places -0.1889, Texture -0.1470

