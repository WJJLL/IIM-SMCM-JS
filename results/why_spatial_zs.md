# 为什么需要 Spatial（zero-shot, ID=ImageNet-100）

| 分数 | AUROC | FPR95 | 检测率@ID-FPR5% |
|---|---|---|---|
| Global JS | 89.83 | 38.26 | 49.51 |
| Spatial JS | 90.03 | 43.52 | 56.23 |
| 融合 (w=1) | 91.92 | 34.32 | 61.08 |

## 固定 ID FPR=5% 时 OOD 的检出构成

| 类别 | 占全部 OOD |
|---|---|
| 两者都命中 | 37.10% |
| **仅 Global JS 命中** | 12.40% |
| **仅 Spatial 命中** | 19.13% |
| 都没命中 | 31.36% |

并集 = 68.64%，Global 漏掉的 50.49% 中有 19.13 个百分点由 Spatial 捞回。

## 控制 Global JS 后 Spatial 的箱内 AUROC = 68.86（min 62.80）
