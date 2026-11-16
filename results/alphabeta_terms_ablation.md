# Two special cases of S_{α,β} and Global JS

IIM base, ImageNet-500-Hard as ID, 4-shot, seeds [1, 2, 3], 4 OOD sets; G=7, γ=0.5, a=0.5 (share of the global term).

Global term $x_g=\alpha H(\bar q)-\beta\frac1P\sum_p H(q_p)$; regional term $x_{sp}=\mathrm{std}_m[\alpha H(\bar q^{(m)})-\beta\frac{1}{|P_m|}\sum_{p\in P_m}H(q_p)]$.

## Single feature (the term used alone as a score, best direction)

| (α,β) | statistic | global-term FPR95 | global-term AUROC | regional-term FPR95 | regional-term AUROC |
|---|---|---|---|---|---|
| (1,0) | only H(q̄) (β=0) | 93.45 | 58.32 | 57.21 | 81.73 |
| (0,1) | only −mean patch entropy (α=0) | 61.37 | 80.53 | 53.26 | 83.36 |
| (1,1) | Global JS (α=β=1) | 65.30 | 84.55 | 62.78 | 80.66 |

## Adding on top of the base (γ=0.5, a=0.5, G=7)

Base S-MCM: FPR95 29.28±1.48, AUROC 93.33±0.20.

| (α,β) | statistic | global-only FPR95 | regional-only FPR95 | both FPR95 | global-only AUROC | regional-only AUROC | both AUROC |
|---|---|---|---|---|---|---|---|
| (1,0) | only H(q̄) (β=0) | 33.13±2.47 | 30.27±2.59 | 27.19±1.43 | 92.91 | 92.39 | 93.84 |
| (0,1) | only −mean patch entropy (α=0) | 32.43±3.10 | 30.14±2.55 | 30.84±2.71 | 91.94 | 92.47 | 92.40 |
| (1,1) | Global JS (α=β=1) | 29.19±1.29 | 27.49±1.81 | 25.72±1.57 | 93.78 | 93.32 | 94.18 |

## Sweep over the ratio r=β/α (γ=0.5, a=0.5, G=7)

Since each of the two terms is normalized by its ID std, (α,β)→(cα,cβ) are equivalent, and what actually matters is r=β/α.

| r | global-only FPR95 | regional-only FPR95 | both FPR95 | both AUROC |
|---|---|---|---|---|
| 0.00 | 33.13 | 30.27 | 27.19 | 93.84 |
| 0.25 | 32.25 | 30.21 | 26.89 | 93.93 |
| 0.50 | 30.75 | 29.92 | 26.73 | 94.01 |
| 0.75 | 29.65 | 28.85 | 26.40 | 94.05 |
| 1.00 | 29.19 | 27.49 | 25.72 | 94.18 |
| 1.25 | 29.42 | 28.18 | 27.37 | 93.74 |
| 1.50 | 29.99 | 29.08 | 28.38 | 93.37 |
| 2.00 | 30.63 | 29.74 | 29.24 | 93.01 |
| 4.00 | 31.68 | 30.03 | 30.22 | 92.64 |
