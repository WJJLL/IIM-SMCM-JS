"""JS / spatial-JS auxiliary scores for patch-consensus OOD detection.

Shared by the two score heads of this repo:

* ``trainers/iim.py``      -- IIM (learned local prompts + negative prompts)
* ``trainers/zeroshot.py`` -- frozen CLIP zero-shot baseline

Both report every base score (MCM / GL-MCM / R-MCM / S-MCM) three times:
``base``, ``base-JS`` (global patch-consensus JS) and ``base-JS-Spatial``
(global JS + spatial-organisation JS), giving 12 scores in total.

Conventions: scores are "smaller = more ID".  ``output_local`` must already be
in evaluation units (i.e. divided by 100) and shaped [B, P, C] with P a perfect
square (14x14 for ViT-B/16).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

to_np = lambda x: x.data.cpu().numpy()

# The JS budget gamma: the auxiliary term may shift the ID distribution of the
# base score by at most JS_GAMMA_TAU of its standard deviation (ID-only rule).
JS_SCALE = float(os.environ.get("JS_SCALE", "50"))
JS_GAMMA_TAU = 0.5
JS_GAMMA = float(os.environ.get("JS_GAMMA", str(JS_GAMMA_TAU)))

JS_SPATIAL_GRID = int(os.environ.get("JS_SPATIAL_GRID", "7"))
JS_SPATIAL_A = float(os.environ.get("JS_SPATIAL_A", "0.5"))
JS_SPATIAL_W = os.environ.get("JS_SPATIAL_W", "")


def configure(scale=None, gamma=None, grid=None, a=None, w=None):
    """Override the module-level knobs (used by the yacs config / CLI).

    Any argument left as ``None`` keeps its current value, so the environment
    variables above remain the default source of truth.  Must be called before
    the scoring helpers run (i.e. right after ``setup_cfg``).
    """
    global JS_SCALE, JS_GAMMA, JS_SPATIAL_GRID, JS_SPATIAL_A, JS_SPATIAL_W
    if scale is not None:
        JS_SCALE = float(scale)
    if gamma is not None:
        JS_GAMMA = float(gamma)
    if grid is not None:
        JS_SPATIAL_GRID = int(grid)
    if a is not None:
        JS_SPATIAL_A = float(a)
    if w is not None:
        JS_SPATIAL_W = str(w)
    print(f"[JS] scale={JS_SCALE:g} gamma={JS_GAMMA:g} "
          f"grid={JS_SPATIAL_GRID} a={JS_SPATIAL_A:g} "
          f"w={JS_SPATIAL_W or f'{JS_SPATIAL_A:g},{1 - JS_SPATIAL_A:g}'}")


def Patch_Agreement_Score(output_local, scale=None):
    """Per-image JS agreement across patches (larger = more ID-like)."""
    if scale is None:
        scale = JS_SCALE
    pi = torch.softmax(output_local * scale, dim=-1)            # B,P,C
    hbar = -(pi * pi.clamp_min(1e-12).log()).sum(-1).mean(-1)   # B
    q = pi.mean(dim=1)                                           # B,C
    hq = -(q * q.clamp_min(1e-12).log()).sum(-1)                 # B
    return to_np(hq - hbar)


def _spatial_weights(a=None):
    if JS_SPATIAL_W:
        g, s = [float(x) for x in JS_SPATIAL_W.split(",")]
        return g, s
    if a is None:
        a = JS_SPATIAL_A
    return a, 1.0 - a


def Patch_Spatial_Std_Score(output_local, scale=None, grid=None):
    """Std of the per-cell JS across a g x g grid (larger = more ID-like)."""
    if scale is None:
        scale = JS_SCALE
    if grid is None:
        grid = JS_SPATIAL_GRID
    B, P, C = output_local.shape
    side = int(round(P ** 0.5))
    assert side * side == P, (f"{P} patches is not a perfect square, "
                             f"cannot rebuild the spatial grid")
    pi = torch.softmax(output_local * scale, dim=-1)                # (B,P,C)
    h_p = -(pi * pi.clamp_min(1e-12).log()).sum(-1)                 # (B,P)
    pi_g = pi.reshape(B, side, side, C).permute(0, 3, 1, 2)         # (B,C,H,W)
    h_g = h_p.reshape(B, 1, side, side)                             # (B,1,H,W)
    cell = F.adaptive_avg_pool2d(pi_g, (grid, grid)).permute(0, 2, 3, 1)
    hq = -(cell * cell.clamp_min(1e-12).log()).sum(-1)              # (B,g,g)
    hb = F.adaptive_avg_pool2d(h_g, (grid, grid)).squeeze(1)        # (B,g,g)
    js = (hq - hb).reshape(B, grid * grid)
    return to_np(js.std(dim=-1))


def calibrate_js_spatial(std_base, js_all, js_std_all, a=None):
    """Return (coef_js, mu_js, coef_std, mu_std) using ID statistics only."""
    wg, ws = _spatial_weights(a)
    step = JS_GAMMA * float(np.std(std_base))
    sd_js = float(np.std(js_all)) + 1e-12
    sd_std = float(np.std(js_std_all)) + 1e-12
    return (wg * step / sd_js, float(np.mean(js_all)),
            ws * step / sd_std, float(np.mean(js_std_all)))


def combine_js_spatial(base, js, js_std, calib):
    """S = base - coef_js*(js - mu_js) - coef_std*(js_std - mu_std)."""
    coef_js, mu_js, coef_std, mu_std = calib
    return base - coef_js * (js - mu_js) - coef_std * (js_std - mu_std)
