import copy
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

import clip_w_local
from clip_w_local import clip
from clip_w_local.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np
from tqdm import tqdm
from PIL import Image
from einops import repeat
import math
from functools import reduce
from operator import mul
import os
from .utils.score_function import *
from .utils import js_spatial
from .utils.js_spatial import (
    Patch_Agreement_Score,
    Patch_Spatial_Std_Score,
    calibrate_js_spatial,
    combine_js_spatial,
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

_tokenizer = _Tokenizer()
softmax = nn.Softmax(dim=1).cuda()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model.cuda().eval()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.num_neg_prompts = cfg.num_neg_prompts
        self.num_local_prompts = n_cls
        n_ctx = cfg.TRAINER.IIM.N_CTX
        ctx_init = cfg.TRAINER.IIM.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 1. for global prompt initialization: frozen and hand-crafted 'a photo of {c}'
        classnames = [name.replace("_", " ") for name in classnames]
        global_prompts = ["a photo of a" + " " + name + "." for name in classnames]

        global_tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in global_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(global_tokenized_prompts).type(dtype)

        self.global_embedding = embedding
        self.global_tokenized_prompts = global_tokenized_prompts  # torch.Tensor  #1000,77
        self.class_token_position = cfg.TRAINER.IIM.CLASS_TOKEN_POSITION

        print(f'Global prompt context: a photo of a [CLASS].')

        # 2. for local prompt initialization: learnable
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            local_ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.IIM.CSC:
                print("Initializing class-specific contexts")
                local_ctx_vectors = torch.empty(self.num_local_prompts, n_ctx, ctx_dim, dtype=dtype)
            else:
                local_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(local_ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial local context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.local_ctx = nn.Parameter(local_ctx_vectors)  # to be optimized
        local_prompts = [prompt_prefix + " " + name + "." for name in classnames]
        local_tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in local_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(local_tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.local_tokenized_prompts = local_tokenized_prompts

        #  3. for negative prompt initialization: learnable
        print("Initializing negative local contexts")
        neg_ctx_vectors = torch.empty(self.num_neg_prompts, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(neg_ctx_vectors, std=0.02)
        neg_prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial local context: "{neg_prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.neg_ctx = nn.Parameter(neg_ctx_vectors)  # to be optimized

        neg_prompts = [neg_prompt_prefix + " " + "." for _ in range(self.num_neg_prompts)]
        neg_tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in neg_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(neg_tokenized_prompts).type(dtype)

        self.register_buffer("neg_token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("neg_token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.neg_tokenized_prompts = neg_tokenized_prompts

    def forward(self):
        assert self.class_token_position == 'end', 'not expected class token position.'

        local_ctx = self.local_ctx  # 100,16,512
        if local_ctx.dim() == 2:
            local_ctx = local_ctx.unsqueeze(0).expand(self.num_ood_prompts, -1, -1)

        prefix = self.token_prefix  # 100,1,512
        suffix = self.token_suffix  # 1000,60,512

        local_prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                local_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        if local_ctx.dim() == 2:
            local_ctx = local_ctx.unsqueeze(0).expand(self.num_neg_prompts, -1, -1)

        neg_prefix = self.neg_token_prefix  # 100,1,512
        neg_suffix = self.neg_token_suffix  # 1000,60,512

        neg_prompts = torch.cat(
            [
                neg_prefix,  # (n_cls, 1, dim)
                self.neg_ctx,  # (n_cls, n_ctx, dim)
                neg_suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return self.global_embedding, local_prompts, neg_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.global_tokenized_prompts = self.prompt_learner.global_tokenized_prompts
        self.local_tokenized_prompts = self.prompt_learner.local_tokenized_prompts
        self.neg_tokenized_prompts = self.prompt_learner.neg_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.zs_clip_model = copy.deepcopy(clip_model)
        self.zs_text_encoder = TextEncoder(self.zs_clip_model)

    def div_loss(self, text_features):
        num = text_features.shape[0]
        loss_div = torch.nn.functional.cosine_similarity(text_features[None, :, :],  # 1 300 512 ; 300, 1, 512
                                                         text_features[:, None, :], dim=-1)
        loss_div = torch.sum(loss_div, dim=-1) / num
        loss_div = torch.sum(loss_div, dim=-1) / (num - 1)
        return loss_div

    def forward(self, images):
        if self.training:

            _, local_prompts, neg_prompts = self.prompt_learner()
            local_tokenized_prompts = self.local_tokenized_prompts
            neg_tokenized_prompts = self.neg_tokenized_prompts
            local_text_features = self.text_encoder(local_prompts, local_tokenized_prompts)
            neg_text_features = self.text_encoder(neg_prompts, neg_tokenized_prompts)

            _, local_image_features = self.image_encoder(images.type(self.dtype))
            local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)

            local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True)
            neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_local = logit_scale * local_image_features @ local_text_features.t()
            neg_logits_local = logit_scale * local_image_features @ neg_text_features.t()

            # B, B, L
            loss_div = self.div_loss(neg_text_features)
            loss_div_local = self.div_loss(local_text_features)
            loss_div = loss_div_local + loss_div

            return logits_local, neg_logits_local, loss_div

        else:  # for inference
            global_prompts, local_prompts, neg_prompts = self.prompt_learner()

            global_tokenized_prompts = self.global_tokenized_prompts
            local_tokenized_prompts = self.local_tokenized_prompts
            neg_tokenized_prompts = self.neg_tokenized_prompts

            global_text_features = self.zs_text_encoder(global_prompts, global_tokenized_prompts)
            local_text_features = self.text_encoder(local_prompts, local_tokenized_prompts)
            neg_text_features = self.text_encoder(neg_prompts, neg_tokenized_prompts)

            image_features, _ = self.zs_clip_model.visual(images.type(self.dtype))
            # image_features, local_image_features = self.image_encoder(images.type(self.dtype))
            _, local_image_features = self.image_encoder(images.type(self.dtype))

            # local image features are much more robust with the effect of the visual prompts.

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)

            global_text_features = global_text_features / global_text_features.norm(dim=-1, keepdim=True)
            local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True)
            neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()

            logits = logit_scale * image_features @ global_text_features.t()
            logits_local = logit_scale * local_image_features @ local_text_features.t()
            neg_logits_local = logit_scale * local_image_features @ neg_text_features.t()

            return logits, logits_local, neg_logits_local


@TRAINER_REGISTRY.register()
class IIM(TrainerX):
    """
    IIM
    """

    # Names of the base scores; each is reported together with a -JS and a
    # -JS-Spatial variant (12 scores in total).
    BASE_NAMES = ["MCM", "GL-MCM", "R-MCM", "S-MCM"]

    # ID-only calibration constants for the JS / spatial-JS terms.
    # Set by the in-distribution pass (calibrate=True) and reused for every
    # OOD set, so no OOD information ever enters the coefficients.
    _js_calib = None

    def check_cfg(self, cfg):
        assert cfg.TRAINER.IIM.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.top_k = cfg.topk
        self.T = cfg.T
        self._js_calib = None

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.IIM.PREC == "fp32" or cfg.TRAINER.IIM.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        train_parameters = []
        train_parameters.append('text_encoder.text_projection')
        train_parameters.append('prompt_learner')
        if cfg.TRAINER.IIM.PROJ:
            train_parameters.append('image_encoder.proj')

        flag = 0
        for name, param in self.model.named_parameters():
            for str in train_parameters:
                flag = 0
                if name.startswith(str):
                    param.requires_grad_(True)
                    flag = 1
                    print(name)
                    break
            if flag == 0:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IIM.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def calculate_loss_local(self, output_local, neg_output_local, label):

        total_local_prompts = self.model.prompt_learner.num_local_prompts + self.model.prompt_learner.num_neg_prompts

        label_output_local = output_local.gather(2, repeat(label, 'b -> b n 1', n=output_local.shape[1]))  # batch,196,1
        label_topk_output_local = label_output_local.topk(k=self.top_k, dim=1)[0].squeeze()  # batch, k

        pos_topk_ouput_local = output_local.topk(k=self.top_k, dim=1)[0]  # batch, k, 500
        neg_topk_output_local = neg_output_local.topk(k=self.top_k, dim=1)[0]  # batch, k, 300

        if len(label_topk_output_local.shape) == 1:  # avoid the case k=1, if bs = 1, then unsqueeze(0) :)
            label_topk_output_local = label_topk_output_local.unsqueeze(1)

        # To prevent overflow. for simplicity, divide both the numerator and denominator by the first number
        common_factor = label_topk_output_local[:, 0]  # batch

        Pos = label_topk_output_local - repeat(common_factor, 'b-> b k', k=self.top_k)

        Neg = torch.cat((pos_topk_ouput_local, neg_topk_output_local), dim=-1)  # batch, k, 800
        Neg = Neg.reshape(-1, self.top_k * total_local_prompts)
        Neg = Neg - repeat(common_factor, 'b-> b k', k=self.top_k * total_local_prompts)

        local_contrastive = torch.sum(torch.exp(Pos / self.T), dim=-1) / torch.sum(torch.exp(Neg / self.T), dim=-1)
        loss_local = -torch.mean(torch.log(local_contrastive))

        return loss_local

    def calculate_loss_local_neg(self, neg_output_local, output_local, label):
        # bs, 196, 300; bs, 196, 500;
        total_local_prompts = self.model.prompt_learner.num_local_prompts + self.model.prompt_learner.num_neg_prompts
        num_neg_prompts = self.model.prompt_learner.num_neg_prompts

        label_output_local = output_local.gather(2, repeat(label, 'b -> b n 1', n=output_local.shape[1]))
        # bs, 196, 1
        label_topk_output_local_pos = label_output_local.topk(k=self.top_k * 2, dim=1)[1]
        label_topk_output_local_pos = label_topk_output_local_pos[:, self.top_k:2 * self.top_k]

        # bs, k, 1 # indices
        neg_topk_output_local = neg_output_local.gather(1, repeat(label_topk_output_local_pos, 'b k 1 -> b k n',
                                                                  n=self.model.prompt_learner.num_neg_prompts))
        # pos_topk_output_local = output_local.gather(1, repeat(label_topk_output_local_pos, 'b k 1 -> b k n',
        #                                                       n=self.model.prompt_learner.num_local_prompts))  # b, k, 500
        # as for the pos_topk_output_label, the above strictly follows our paper,
        # in practice, we find that the below one is better.
        pos_topk_output_local = output_local.topk(k=self.top_k * 2, dim=1)[0]
        pos_topk_output_local = pos_topk_output_local[:, self.top_k: self.top_k * 2]

        t = torch.cat((pos_topk_output_local, neg_topk_output_local), dim=-1)
        common_factor = torch.amax(t, dim=[1, 2], keepdim=True)  # b, k, 1

        Neg = neg_topk_output_local - repeat(common_factor, 'b 1 1 -> b k n', k=self.top_k, n=num_neg_prompts)
        Pos = torch.cat((neg_topk_output_local, pos_topk_output_local), dim=-1)
        Pos = Pos - repeat(common_factor, 'b 1 1 -> b k n ', k=self.top_k, n=total_local_prompts)

        loss_local_neg = torch.sum(torch.exp(Neg / self.T), dim=(1, 2)) / torch.sum(torch.exp(Pos / self.T), dim=(1, 2))
        loss_local_neg = -torch.mean(torch.log(loss_local_neg))

        return loss_local_neg        

    # def calculate_negative_entropy_loss(self, output_local, target_indices=None):
    #     """
    #     output_local:   [B, R, C]    logits over (region x ID class)
    #     target_indices: [B, k]      indices of the regions to constrain
    #     """
    #     # softmax over the response values to get a predictive distribution
    #     probs = F.softmax(output_local, dim=-1)  # [batch, k, num_classes]
        
    #     # entropy: -sum(p * log(p))
    #     epsilon = 1e-8
    #     entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)  # [batch, k]
        
    #     # average the entropy over all regions
    #     neg_entropy_loss = torch.mean(entropy)  # scalar
        
    #     return neg_entropy_loss



    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.IIM.PREC
        self.lambda_value = self.cfg.lambda_value
        self.div_value = self.cfg.div_value
        if prec == "amp":
            with autocast():
                output_local, neg_output_local, loss_div = self.model(image)
                loss_local = self.calculate_loss_local(output_local, neg_output_local, label)
                loss_local_negative  = self.calculate_loss_local_neg(neg_output_local, output_local, label)

                # pos_topk_ouput_local = output_local.topk(k=self.top_k, dim=1)[0]  # batch, k, 500
                # neg_entropy_loss =   self.calculate_negative_entropy_loss(
                #     pos_topk_ouput_local
                # )

                loss = loss_local + self.lambda_value * loss_local_negative + self.div_value * loss_div 


            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            raise NotImplementedError('fp32 easily falls into oom and fp16 suffers from nan loss. Should be amp')

        loss_summary = {
            "loss": loss.item(),
            "loss_local": loss_local.item(),
            "loss_local_negative": loss_local_negative.item() if loss_local_negative else 0,
            "loss_div": loss_div.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]  # get only one image is enough :) :) l
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # print(state_dict.keys())
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print('Successfully load global weights from pretrained model.')

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT  # test

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
        list_correct = []
        outputs = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)

            output_global, output_local, _ = self.model_inference(input)

            output_global /= 100.0
            output_local /= 100.0

            local_score = torch.topk(torch.exp(output_local / self.T), k=self.top_k, dim=1)[0]
            output = torch.exp(output_global) * torch.mean(local_score, dim=1)

            outputs.append(F.softmax(output, dim=-1).data.cpu().numpy())
            pred = output.max(dim=1)[1]
            for j in range(len(pred)):
                if pred[j] == label[j]:
                    cor = 1
                else:
                    cor = 0
                list_correct.append(cor)

            if len(output) == 2:
                output = output[0]
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0], np.concatenate(outputs, axis=0), list_correct

    def _base_and_aux_scores(self, output, output_local, neg_output_local, T, top_k):
        """Return the four base scores plus the two per-image JS auxiliaries.

        `*-JS`         : global patch-consensus JS (H(q) - mean_p H(pi_p))
        `*-JS-Spatial` : global JS + spatial-organisation JS (std of the JS of
                         g x g pooled cells), sharing one ID-only budget.
        """
        output = output / 100.0
        output_local = output_local / 100.0
        neg_output_local = neg_output_local / 100.0

        alpha = self.cfg.alpha
        lamda = alpha * 196 / self.model.prompt_learner.num_local_prompts

        bases = [
            MCM_Score(output, T),
            R_MCM_Score(output, output_local, T, topk=1),
            R_MCM_Score(output, output_local, T, topk=top_k,
                        neg_output_local=neg_output_local),
            S_MCM_Score(output, output_local, T, lamda, topk=top_k,
                        neg_output_local=neg_output_local),
        ]
        js = Patch_Agreement_Score(output_local)
        js_std = Patch_Spatial_Std_Score(output_local)
        return bases, js, js_std

    def _eval_ood_loader(self, data_loader, top_k, T, calibrate, parse_batch):
        """Run one loader and return 12 score arrays.

        Order per base score b in [MCM, GL-MCM, R-MCM, S-MCM]:
            b, b-JS, b-JS-Spatial

        With ``calibrate=True`` (the in-distribution pass) the JS coefficients
        are fitted from ID statistics only:
            coef   = JS_GAMMA * std_ID(base) / std_ID(js)
            mu_js  = mean_ID(js)
        and cached for the OOD passes, so no OOD data enters the calibration.
        """
        concat = lambda x: np.concatenate(x, axis=0)

        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        base_lists = [[] for _ in self.BASE_NAMES]
        js_list, js_std_list = [], []

        for batch in tqdm(data_loader):
            if parse_batch:
                images, labels = self.parse_batch_test(batch)
            else:
                images, labels, *id_flag = batch
                images = images.cuda()

            output, output_local, neg_output_local = self.model_inference(images)
            bases, js, js_std = self._base_and_aux_scores(
                output, output_local, neg_output_local, T, top_k)

            for i in range(len(self.BASE_NAMES)):
                base_lists[i].append(bases[i])
            js_list.append(js)
            js_std_list.append(js_std)

        n = len(data_loader.dataset)
        bases_all = [concat(b)[:n].copy() for b in base_lists]
        js_all = concat(js_list)[:n].copy()
        js_std_all = concat(js_std_list)[:n].copy()

        if calibrate:
            self._js_calib = []
            mu_js = float(js_all.mean())
            sd_js = float(np.std(js_all)) + 1e-12
            for name, base in zip(self.BASE_NAMES, bases_all):
                sd_base = float(np.std(base)) + 1e-12
                coef = js_spatial.JS_GAMMA * sd_base / sd_js
                sp = calibrate_js_spatial(base, js_all, js_std_all)
                self._js_calib.append((coef, mu_js, sp))
                print(f"[IIM {name}-JS] ID-only calibration: "
                      f"mu_js={mu_js:.6f} std_js={sd_js:.6f} "
                      f"std_base={sd_base:.6f} coef={coef:.6f} "
                      f"coef_js(spatial)={sp[0]:.6f} "
                      f"coef_std(spatial)={sp[2]:.6f}")

        if self._js_calib is None:
            print("[IIM JS] no ID calibration available - *-JS and "
                  "*-JS-Spatial fall back to the base scores")
            out = []
            for base in bases_all:
                out.extend([base, base, base])
            return out

        out = []
        for i, name in enumerate(self.BASE_NAMES):
            coef, mu_js, sp = self._js_calib[i]
            base = bases_all[i]
            base_js = base - coef * (js_all - mu_js)
            base_js_spatial = combine_js_spatial(base, js_all, js_std_all, sp)
            out.extend([base, base_js, base_js_spatial])

        return out

    @torch.no_grad()
    def test_ood(self, data_loader, top_k, T, calibrate=False):
        """Test-time OOD detection pipeline (tuple-style batches)."""
        return self._eval_ood_loader(data_loader, top_k, T, calibrate,
                                     parse_batch=False)

    @torch.no_grad()
    def test_ood_imagenet(self, data_loader, top_k, T, calibrate=False):
        """Test-time OOD detection pipeline (DASSL dict-style batches)."""
        return self._eval_ood_loader(data_loader, top_k, T, calibrate,
                                     parse_batch=True)

    @torch.no_grad()
    def test_visualize(self, image_path, label):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        device = "cuda"
        model, preprocess = clip_w_local.load("ViT-B/16", device=device)

        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        output, output_local, neg_output_local = self.model_inference(image)

        num_regions = output_local.shape[1]  # 1,

        label = torch.tensor(label).cuda()
        label = label.unsqueeze(0)

        label_output_local = output_local.gather(2, repeat(label, 'b -> b n 1', n=num_regions))
        # print(label_output_local.shape)
        # 4, 196, 1
        index = label_output_local.topk(k=100, dim=1)[1]
        # print(label_topk_output_local_pos.shape)  # indices
        neg_index = index[:, self.top_k:2 * self.top_k]
        pos_index = index[:, :self.top_k]
        # print(label_topk_output_local_pos.shape) # indices
        # 4, 50, 1

        return pos_index, neg_index

    def Info(self):
        names = []
        for name in self.BASE_NAMES:
            names += [f"{name} score", f"{name}-JS score",
                      f"{name}-JS-Spatial score"]
        return names