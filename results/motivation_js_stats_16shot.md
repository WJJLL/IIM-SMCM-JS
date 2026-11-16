# Distribution of Global JS on ID / OOD (16-shot, seeds=[1, 2, 3])

The larger the JS, the more ID-like. Distribution statistics are computed **per seed and then reported as seed mean ± sample std**; `n` is the number of samples of **a single ckpt** (consistent with the boxes in the figure). FPR95/AUROC are the detection performance of **the JS feature alone** (computed separately for each OOD set).

## ImageNet-500-Hard

| set | n | mean | median | IQR | JS-alone FPR95 | JS-alone AUROC |
|---|---|---|---|---|---|---|
| ID | 25000 | 1.4687±0.0059 | 1.4754±0.0063 | 0.2149 | — | — |
| iNat | 10000 | 1.2151±0.0059 | 1.2144±0.0057 | 0.1380 | 56.06±4.51 | 90.45±0.83 |
| SUN | 10000 | 1.2286±0.0099 | 1.2351±0.0131 | 0.2006 | 59.51±4.30 | 86.08±1.27 |
| Places | 10000 | 1.2627±0.0073 | 1.2626±0.0089 | 0.1932 | 67.28±3.31 | 83.03±1.45 |
| Texture | 5640 | 1.2486±0.0152 | 1.2623±0.0148 | 0.2383 | 63.64±1.99 | 82.29±0.71 |

- ID median − OOD median: iNat -0.2610, SUN -0.2403, Places -0.2128, Texture -0.2131

## ImageNet-500-Easy

| set | n | mean | median | IQR | JS-alone FPR95 | JS-alone AUROC |
|---|---|---|---|---|---|---|
| ID | 25000 | 1.2778±0.0223 | 1.2779±0.0229 | 0.1885 | — | — |
| iNat | 10000 | 1.0794±0.0170 | 1.0767±0.0172 | 0.1291 | 62.30±3.54 | 87.65±1.04 |
| SUN | 10000 | 1.0721±0.0243 | 1.0747±0.0279 | 0.1836 | 58.11±3.19 | 85.24±1.53 |
| Places | 10000 | 1.1087±0.0265 | 1.1069±0.0286 | 0.1759 | 67.79±3.64 | 81.14±2.00 |
| Texture | 5640 | 1.1228±0.0221 | 1.1307±0.0226 | 0.2232 | 68.79±2.12 | 76.06±1.47 |

- ID median − OOD median: iNat -0.2011, SUN -0.2032, Places -0.1709, Texture -0.1472

## ImageNet-100

| set | n | mean | median | IQR | JS-alone FPR95 | JS-alone AUROC |
|---|---|---|---|---|---|---|
| ID | 5000 | 1.3532±0.0121 | 1.3591±0.0112 | 0.2019 | — | — |
| iNat | 10000 | 1.1555±0.0319 | 1.1546±0.0332 | 0.1302 | 69.80±6.10 | 86.34±2.25 |
| SUN | 10000 | 1.1597±0.0274 | 1.1632±0.0300 | 0.1810 | 66.90±4.45 | 83.03±1.86 |
| Places | 10000 | 1.1893±0.0272 | 1.1873±0.0289 | 0.1742 | 74.27±4.15 | 79.59±2.06 |
| Texture | 5640 | 1.1840±0.0161 | 1.1989±0.0131 | 0.2121 | 71.06±2.02 | 78.27±1.19 |

- ID median − OOD median: iNat -0.2045, SUN -0.1960, Places -0.1719, Texture -0.1602

## ImageNet-1k

| set | n | mean | median | IQR | JS-alone FPR95 | JS-alone AUROC |
|---|---|---|---|---|---|---|
| ID | 50000 | 1.4492±0.0077 | 1.4520±0.0079 | 0.2115 | — | — |
| iNat | 10000 | 1.2354±0.0136 | 1.2328±0.0146 | 0.1431 | 64.74±3.49 | 86.90±0.97 |
| SUN | 10000 | 1.2351±0.0066 | 1.2373±0.0062 | 0.1961 | 62.11±1.39 | 83.87±0.61 |
| Places | 10000 | 1.2725±0.0057 | 1.2694±0.0052 | 0.1887 | 71.58±1.14 | 79.92±0.83 |
| Texture | 5640 | 1.2862±0.0091 | 1.2978±0.0075 | 0.2378 | 71.68±0.77 | 75.45±0.38 |

- ID median − OOD median: iNat -0.2192, SUN -0.2146, Places -0.1826, Texture -0.1542

