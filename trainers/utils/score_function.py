import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip_w_local import clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import repeat

to_np = lambda x: x.data.cpu().numpy()


def MCM_Score(output, T):
    smax_global = to_np(F.softmax(output / T, dim=-1))
    mcm_global_score = -np.max(smax_global, axis=1)
    return mcm_global_score


def R_MCM_Score(output, output_local, T, topk=1, neg_output_local=None):

    mcm_score = MCM_Score(output, T)
    N, C = output_local.shape[1:]

    if neg_output_local is not None:
        # print(torch.mean(output_local), torch.mean(neg_output_local))
        smax_local = torch.topk((torch.exp(output_local / T) /
                                 torch.sum(torch.exp(torch.cat((output_local, neg_output_local), dim=-1) / T), dim=-1,
                                           keepdim=True)).reshape(-1, N * C), k=topk, dim=-1)[0]
    else:
        smax_local = torch.topk((torch.exp(output_local / T) /
                                 torch.sum(torch.exp(output_local / T), dim=-1,
                                           keepdim=True)).reshape(-1, N * C), k=topk, dim=-1)[0]
    r_mcm_score = -to_np(torch.mean(smax_local, dim=1))
    r_mcm_score = r_mcm_score + mcm_score
    return r_mcm_score



# def R_MCM_Score(
#     output,
#     output_local,
#     T,
#     topk=1,
#     neg_output_local=None
# ):
#     """
#     Multi-scale R-MCM score.

#     Args:
#         output: [B, C], global logits
#         output_local: [B, N, C] or list of [B, N_s, C]
#         T: temperature
#         topk: int or list of per-scale top-k values
#         neg_output_local: None, tensor, or list of tensors

#     Returns:
#         r_mcm_score: [B] numpy array
#     """

#     # 1. Global MCM score
#     mcm_score = MCM_Score(output, T)

#     # 2. Normalize local inputs to lists
#     if isinstance(output_local, (list, tuple)):
#         local_scales = list(output_local)
#     else:
#         local_scales = [output_local]

#     if neg_output_local is None:
#         neg_scales = [None] * len(local_scales)
#     elif isinstance(neg_output_local, (list, tuple)):
#         neg_scales = list(neg_output_local)
#     else:
#         neg_scales = [neg_output_local]

#     num_scales = len(local_scales)

#     if num_scales == 0:
#         raise ValueError("output_local must contain at least one scale")

#     if len(neg_scales) != num_scales:
#         raise ValueError(
#             "The number of negative-logit scales must match output_local"
#         )

#     # 3. Normalize topk
#     if isinstance(topk, int):
#         topk_scales = [topk] * num_scales
#     else:
#         topk_scales = list(topk)

#         if len(topk_scales) != num_scales:
#             raise ValueError(
#                 "The number of topk values must match the number of scales"
#             )

#     # 4. Compute local R-MCM score at each scale
#     scale_scores = []

#     for local_logits, neg_logits, k in zip(
#         local_scales, neg_scales, topk_scales
#     ):
#         # [B, N_s, C]
#         local_logits = local_logits.float()
#         B, N_s, C = local_logits.shape

#         # Positive logits: [B, N_s, C]
#         pos_exp = torch.exp(local_logits / T)

#         if neg_logits is not None:
#             # Denominator includes positive and negative classes
#             neg_logits = neg_logits.float()

#             denom = torch.sum(
#                 torch.exp(
#                     torch.cat([local_logits, neg_logits], dim=-1) / T
#                 ),
#                 dim=-1,
#                 keepdim=True
#             )
#         else:
#             denom = torch.sum(
#                 pos_exp,
#                 dim=-1,
#                 keepdim=True
#             )

#         # [B, N_s, C]
#         local_probs = pos_exp / denom

#         # Preserve original behavior: top-k over all patch-class pairs
#         local_probs = local_probs.reshape(B, N_s * C)

#         k = max(1, min(int(k), N_s * C))

#         topk_probs = torch.topk(
#             local_probs,
#             k=k,
#             dim=-1
#         ).values

#         # [B]
#         scale_score = -to_np(topk_probs.mean(dim=-1))
#         scale_scores.append(scale_score)

#     # 5. Average local scores across scales
#     fused_local_score = np.mean(
#         np.stack(scale_scores, axis=0),
#         axis=0
#     )

#     # 6. Add global MCM only once
#     r_mcm_score = fused_local_score + mcm_score

#     return r_mcm_score


def S_MCM_Score(output, output_local, T, lamda, topk=1, neg_output_local=None):
    # slcm_local_score_neg = SLCM_Score(output, output_local, T, lamda, topk=top_k,
    #                                   neg_output_local=neg_output_local)
    #
    r_mcm_score = R_MCM_Score(output, output_local, T, topk, neg_output_local)

    output_local_t = output_local.permute(0, 2, 1)
    N, C = output_local_t.shape[1:]
    S1 = torch.sum(torch.exp(output_local_t / T), dim=-1, keepdim=True)
    glcm_score = torch.topk((torch.exp(output_local_t / T) / S1).reshape(-1, N * C), k=topk, dim=-1)[0]
    glcm_score = -to_np(torch.mean(glcm_score, dim=1))
    glcm_score = lamda * glcm_score + r_mcm_score

    return glcm_score




