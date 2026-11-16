# Distribution of Global JS on ID / OOD (4-shot, seeds=[1, 2, 3])

Larger JS means more ID-like. Distribution statistics are **computed per seed and then reported as seed mean±sample std**; `n` is the number of samples in a **single ckpt** (identical to the boxes in the figure). FPR95/AUROC are the detection performance of **the JS feature alone** (computed for each OOD dataset separately).

## ImageNet-500-Hard

| set | n | mean | median | IQR | JS-only FPR95 | JS-only AUROC |
|---|---|---|---|---|---|---|
| ID | 25000 | 0.1651±0.0021 | 0.1641±0.0023 | 0.0374 | — | — |
| iNat | 10000 | 0.1204±0.0020 | 0.1187±0.0021 | 0.0262 | 45.11±3.24 | 90.72±0.33 |
| SUN | 10000 | 0.1417±0.0024 | 0.1395±0.0021 | 0.0328 | 79.18±2.17 | 73.88±2.38 |
| Places | 10000 | 0.1451±0.0024 | 0.1429±0.0024 | 0.0343 | 82.19±2.08 | 70.54±2.11 |
| Texture | 5640 | 0.1224±0.0009 | 0.1182±0.0004 | 0.0310 | 44.65±1.33 | 87.51±0.97 |

- ID median − OOD median: iNat -0.0454, SUN -0.0246, Places -0.0212, Texture -0.0459

## ImageNet-500-Easy

| set | n | mean | median | IQR | JS-only FPR95 | JS-only AUROC |
|---|---|---|---|---|---|---|
| ID | 25000 | 0.1598±0.0012 | 0.1589±0.0011 | 0.0333 | — | — |
| iNat | 10000 | 0.1162±0.0010 | 0.1144±0.0013 | 0.0254 | 38.10±1.74 | 91.84±0.31 |
| SUN | 10000 | 0.1313±0.0019 | 0.1293±0.0020 | 0.0292 | 66.78±3.80 | 80.67±1.18 |
| Places | 10000 | 0.1351±0.0009 | 0.1331±0.0009 | 0.0307 | 72.25±2.02 | 76.93±0.65 |
| Texture | 5640 | 0.1185±0.0004 | 0.1144±0.0003 | 0.0292 | 39.47±0.99 | 88.55±0.33 |

- ID median − OOD median: iNat -0.0444, SUN -0.0295, Places -0.0258, Texture -0.0444

## ImageNet-100

| set | n | mean | median | IQR | JS-only FPR95 | JS-only AUROC |
|---|---|---|---|---|---|---|
| ID | 5000 | 0.1871±0.0021 | 0.1852±0.0022 | 0.0416 | — | — |
| iNat | 10000 | 0.1304±0.0038 | 0.1289±0.0037 | 0.0248 | 29.31±4.21 | 94.58±1.11 |
| SUN | 10000 | 0.1494±0.0035 | 0.1473±0.0035 | 0.0312 | 63.62±2.28 | 83.63±1.68 |
| Places | 10000 | 0.1537±0.0030 | 0.1517±0.0028 | 0.0332 | 69.64±1.90 | 80.23±1.62 |
| Texture | 5640 | 0.1360±0.0015 | 0.1328±0.0011 | 0.0293 | 38.22±2.02 | 90.91±0.91 |

- ID median − OOD median: iNat -0.0562, SUN -0.0379, Places -0.0335, Texture -0.0524

## ImageNet-1k

| set | n | mean | median | IQR | JS-only FPR95 | JS-only AUROC |
|---|---|---|---|---|---|---|
| ID | 50000 | 0.1666±0.0002 | 0.1660±0.0003 | 0.0369 | — | — |
| iNat | 10000 | 0.1196±0.0012 | 0.1175±0.0010 | 0.0277 | 41.11±2.51 | 91.24±0.78 |
| SUN | 10000 | 0.1406±0.0013 | 0.1387±0.0012 | 0.0329 | 76.43±1.71 | 76.20±0.99 |
| Places | 10000 | 0.1450±0.0008 | 0.1430±0.0010 | 0.0348 | 80.65±1.11 | 72.00±0.65 |
| Texture | 5640 | 0.1211±0.0008 | 0.1154±0.0006 | 0.0336 | 39.75±0.81 | 87.70±0.49 |

- ID median − OOD median: iNat -0.0484, SUN -0.0273, Places -0.0230, Texture -0.0505

