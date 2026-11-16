# Distribution of Global JS on ID / OOD (zero-shot, seeds=[1])

The larger the JS, the more ID-like. Distribution statistics are computed **per seed and then reported as seed mean ± sample std**; `n` is the number of samples of **a single ckpt** (consistent with the boxes in the figure). FPR95/AUROC are the detection performance of **the JS feature alone** (computed separately for each OOD set).

## ImageNet-500-Hard

| set | n | mean | median | IQR | JS-alone FPR95 | JS-alone AUROC |
|---|---|---|---|---|---|---|
| ID | 25000 | 0.0612±0.0000 | 0.0599±0.0000 | 0.0168 | — | — |
| iNat | 10000 | 0.0408±0.0000 | 0.0399±0.0000 | 0.0099 | 37.39±0.00 | 92.20±0.00 |
| SUN | 10000 | 0.0486±0.0000 | 0.0473±0.0000 | 0.0132 | 71.49±0.00 | 78.40±0.00 |
| Places | 10000 | 0.0506±0.0000 | 0.0490±0.0000 | 0.0143 | 76.42±0.00 | 74.13±0.00 |
| Texture | 5640 | 0.0436±0.0000 | 0.0422±0.0000 | 0.0127 | 49.77±0.00 | 86.79±0.00 |

- ID median − OOD median: iNat -0.0200, SUN -0.0126, Places -0.0109, Texture -0.0177

## ImageNet-500-Easy

| set | n | mean | median | IQR | JS-alone FPR95 | JS-alone AUROC |
|---|---|---|---|---|---|---|
| ID | 25000 | 0.0779±0.0000 | 0.0760±0.0000 | 0.0233 | — | — |
| iNat | 10000 | 0.0495±0.0000 | 0.0485±0.0000 | 0.0106 | 29.05±0.00 | 94.51±0.00 |
| SUN | 10000 | 0.0546±0.0000 | 0.0532±0.0000 | 0.0137 | 50.27±0.00 | 88.38±0.00 |
| Places | 10000 | 0.0571±0.0000 | 0.0554±0.0000 | 0.0148 | 58.60±0.00 | 84.89±0.00 |
| Texture | 5640 | 0.0520±0.0000 | 0.0491±0.0000 | 0.0161 | 37.34±0.00 | 89.13±0.00 |

- ID median − OOD median: iNat -0.0275, SUN -0.0228, Places -0.0206, Texture -0.0269

## ImageNet-100

| set | n | mean | median | IQR | JS-alone FPR95 | JS-alone AUROC |
|---|---|---|---|---|---|---|
| ID | 5000 | 0.0776±0.0000 | 0.0753±0.0000 | 0.0250 | — | — |
| iNat | 10000 | 0.0441±0.0000 | 0.0432±0.0000 | 0.0095 | 18.49±0.00 | 96.70±0.00 |
| SUN | 10000 | 0.0526±0.0000 | 0.0511±0.0000 | 0.0144 | 52.73±0.00 | 88.12±0.00 |
| Places | 10000 | 0.0547±0.0000 | 0.0528±0.0000 | 0.0154 | 59.07±0.00 | 85.41±0.00 |
| Texture | 5640 | 0.0509±0.0000 | 0.0491±0.0000 | 0.0144 | 45.66±0.00 | 89.74±0.00 |

- ID median − OOD median: iNat -0.0321, SUN -0.0242, Places -0.0225, Texture -0.0262

## ImageNet-1k

| set | n | mean | median | IQR | JS-alone FPR95 | JS-alone AUROC |
|---|---|---|---|---|---|---|
| ID | 50000 | 0.0635±0.0000 | 0.0621±0.0000 | 0.0170 | — | — |
| iNat | 10000 | 0.0438±0.0000 | 0.0429±0.0000 | 0.0099 | 41.77±0.00 | 91.31±0.00 |
| SUN | 10000 | 0.0508±0.0000 | 0.0496±0.0000 | 0.0133 | 71.47±0.00 | 78.42±0.00 |
| Places | 10000 | 0.0532±0.0000 | 0.0518±0.0000 | 0.0146 | 77.33±0.00 | 73.16±0.00 |
| Texture | 5640 | 0.0458±0.0000 | 0.0442±0.0000 | 0.0136 | 48.79±0.00 | 86.01±0.00 |

- ID median − OOD median: iNat -0.0192, SUN -0.0125, Places -0.0103, Texture -0.0179

