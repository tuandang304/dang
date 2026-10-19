"""DANG v5: DINOv3 dual-stream polyp segmenter with a prototype margin head and a dual foreground/background decoder.

Design, informed by 2025-2026 work:
  * Local stream: DINOv3-distilled ConvNeXt (timm `convnext_*.dinov3_lvd1689m`), fine-tuned, strides 4-32.
  * Semantic stream: frozen DINOv3 ViT (timm `vit_*_patch16_dinov3.lvd1689m`) at `v5_vit_size` (448 -> 28x28 grid),
    Conv-LoRA adapters on the last blocks, multi-depth readout (MedDINOv3, DINO-MVR, SegDINO).
  * Gated semantic fusion into the local pyramid at strides 8/16/32.
  * DANG dual-prototype margin head on the semantic tokens and the PASSE support envelope.
  * Light top-down decoder whose stages predict foreground and background logits (explicit background
    supervision, PraNet-V2) and gate their refinement with boundary uncertainty u = 4p(1-p) and a reverse map 1-p.
Setting `v5_vit=""` removes the semantic stream; the prototype head then reads the local stride-16/32 features.
"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dang.model import ConvBNAct, DualPrototypeMarginHead


class SepConvBNAct(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, BN, GELU."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class EvaConvLoRA(nn.Module):
    """Conv-LoRA on the output tokens of a frozen timm Eva/DINOv3 block (accepts rope/attn_mask kwargs)."""

    def __init__(self, block: nn.Module, dim: int, rank: int, num_prefix: int):
        super().__init__()
        self.block = block
        self.num_prefix = num_prefix
        self.down = nn.Linear(dim, rank, bias=False)
        self.dw = nn.Conv2d(rank, rank, 3, padding=1, groups=rank, bias=True)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x, **kwargs):
        out = self.block(x, **kwargs)
        B, N, _ = out.shape
        z = self.down(out)
        pre, sp = z[:, : self.num_prefix], z[:, self.num_prefix:]
        g = int(math.isqrt(sp.shape[1]))
        sp2 = sp.transpose(1, 2).reshape(B, -1, g, g)
        sp2 = F.gelu(sp2 + self.dw(sp2))
        z = torch.cat([pre, sp2.flatten(2).transpose(1, 2)], dim=1)
        return out + self.up(z)


class GatedSemanticFusion(nn.Module):
    """loc + gamma * sigmoid(gate([loc, sem])) * sem, gamma initialised small so training starts from the local stream."""

    def __init__(self, dim: int, gamma_init: float = 0.5):
        super().__init__()
        self.sem = ConvBNAct(dim, dim, kernel_size=3, padding=1)
        self.gate = nn.Conv2d(2 * dim, dim, 1)
        nn.init.constant_(self.gate.bias, 0.0)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), gamma_init))

    def forward(self, loc, sem):
        sem = self.sem(F.interpolate(sem, size=loc.shape[-2:], mode="bilinear", align_corners=False))
        g = torch.sigmoid(self.gate(torch.cat([loc, sem], dim=1)))
        return loc + self.gamma * g * sem


class DualDecoderStage(nn.Module):
    """One top-down stage: merge the coarser feature, refine with uncertainty and reverse attention, emit fg/bg logits."""

    def __init__(self, dim: int, has_top: bool, use_proto: bool):
        super().__init__()
        self.has_top = has_top
        self.use_proto = use_proto
        self.merge = SepConvBNAct(dim, dim)
        if use_proto:
            self.proto_embed = nn.Conv2d(1, dim, 1)
            nn.init.zeros_(self.proto_embed.weight)
            nn.init.zeros_(self.proto_embed.bias)
        if has_top:
            self.unc = nn.Conv2d(dim, dim, 1, bias=False)
            self.rev = nn.Conv2d(dim, dim, 1, bias=False)
            nn.init.zeros_(self.unc.weight)
            nn.init.zeros_(self.rev.weight)
            self.refine = SepConvBNAct(dim, dim)
        self.fg = nn.Conv2d(dim, 1, 1)
        self.bg = nn.Conv2d(dim, 1, 1)

    def forward(self, feat, top=None, top_fg=None, top_bg=None, proto=None):
        size = feat.shape[-2:]
        if top is not None:
            feat = feat + F.interpolate(top, size=size, mode="bilinear", align_corners=False)
        if self.use_proto and proto is not None:
            p = torch.sigmoid(F.interpolate(proto, size=size, mode="bilinear", align_corners=False))
            feat = feat + self.proto_embed(p.to(feat.dtype))
        x = self.merge(feat)
        if self.has_top:
            tf = F.interpolate(top_fg, size=size, mode="bilinear", align_corners=False)
            tb = F.interpolate(top_bg, size=size, mode="bilinear", align_corners=False)
            p = torch.sigmoid((tf - tb).float()).to(x.dtype)
            u = 4.0 * p * (1.0 - p)
            x = x + self.unc(x * u) + self.rev(x * (1.0 - p))
            x = self.refine(x)
            fg = self.fg(x) + tf
            bg = self.bg(x) + tb
        else:
            fg, bg = self.fg(x), self.bg(x)
        return x, fg, bg


class DANGv5(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim
        self.cnn = timm.create_model(cfg.v5_local, pretrained=True, features_only=True)
        self.loc_projs = nn.ModuleList([ConvBNAct(c, dim, kernel_size=1, padding=0) for c in self.cnn.feature_info.channels()])

        self.use_vit = bool(cfg.v5_vit)
        if self.use_vit:
            self.vit = timm.create_model(cfg.v5_vit, pretrained=True, dynamic_img_size=True)
            for p in self.vit.parameters():
                p.requires_grad = False
            vd = self.vit.embed_dim
            nb = len(self.vit.blocks)
            npre = getattr(self.vit, "num_prefix_tokens", 1)
            for i in range(nb - cfg.v5_lora_blocks, nb):
                self.vit.blocks[i] = EvaConvLoRA(self.vit.blocks[i], vd, cfg.v5_lora_rank, npre)
            self.vit_layers = list(cfg.v5_vit_layers)
            self.sem_projs = nn.ModuleList([ConvBNAct(vd, dim, kernel_size=1, padding=0) for _ in self.vit_layers])
            # learned depth mixing: one semantic map per fused scale (s8, s16, s32) from all readout depths
            self.depth_logits = nn.Parameter(torch.zeros(3, len(self.vit_layers)))
            self.fuse = nn.ModuleList([GatedSemanticFusion(dim) for _ in range(3)])
        # Privileged mode: the semantic stream is used only at training time (teacher path); inference runs the
        # local-only student path, so the deployed model is the local stream + local proto head + decoder.
        self.privileged = self.use_vit and bool(getattr(cfg, "v5_privileged", False))
        if not self.use_vit or self.privileged:
            self.sem16 = ConvBNAct(dim, dim, kernel_size=3, padding=1)

        self.proto_head = DualPrototypeMarginHead(dim=dim, proto_dim=cfg.proto_dim, num_prototypes=cfg.num_prototypes)

        self.dec4 = DualDecoderStage(dim, has_top=False, use_proto=True)
        self.dec3 = DualDecoderStage(dim, has_top=True, use_proto=True)
        self.dec2 = DualDecoderStage(dim, has_top=True, use_proto=False)
        self.dec1 = DualDecoderStage(dim, has_top=True, use_proto=False)
        self.head = nn.Sequential(ConvBNAct(dim * 4, dim, kernel_size=3, padding=1), nn.Dropout2d(0.1), nn.Conv2d(dim, 1, 1))

    def _semantic(self, x):
        v = self.cfg.v5_vit_size
        xv = F.interpolate(x, size=(v, v), mode="bilinear", align_corners=False)
        feats = self.vit.forward_intermediates(xv, indices=self.vit_layers, norm=True, output_fmt="NCHW",
                                               intermediates_only=True)
        return [proj(f) for proj, f in zip(self.sem_projs, feats)]

    @staticmethod
    def _passe(logit, coarse, proto):
        """PASSE: suppress far-field logits outside the dilated semantic support, reinforce its confident core."""
        h, w = logit.shape[-2:]
        p25 = proto.float()
        c25 = F.interpolate(coarse.float(), size=p25.shape[-2:], mode="bilinear", align_corners=False)
        p_sem = torch.sigmoid(0.5 * c25 + 0.5 * p25)
        env = F.avg_pool2d(F.max_pool2d(p_sem, 5, 1, 2), 3, 1, 1, count_include_pad=False)
        e_up = F.interpolate(env, size=(h, w), mode="bilinear", align_corners=False)
        p_up = F.interpolate(p_sem, size=(h, w), mode="bilinear", align_corners=False)
        return logit - 2.2 * F.relu(0.22 - e_up) / 0.22 + 1.2 * F.relu(p_up - 0.38)

    TRAIN_ONLY_PREFIXES = ("vit.", "sem_projs.", "depth_logits", "fuse.")

    def is_deploy_param(self, name: str) -> bool:
        """False for parameters only the training-time teacher path uses (privileged mode)."""
        return not (self.privileged and name.startswith(self.TRAIN_ONLY_PREFIXES))

    def _local_proto(self, loc, gt_mask):
        s16 = self.sem16(loc[2])
        s32 = F.interpolate(loc[3], size=s16.shape[-2:], mode="bilinear", align_corners=False)
        return self.proto_head([s16, s16, s32, s16], gt_mask=gt_mask)[0]

    def _fuse_semantic(self, x, loc, gt_mask):
        sem = self._semantic(x)
        proto_logits, _ = self.proto_head(sem, gt_mask=gt_mask)
        w = F.softmax(self.depth_logits, dim=-1)
        loc = list(loc)
        for k, li in enumerate((1, 2, 3)):
            mixed = sum(w[k, j] * s for j, s in enumerate(sem))
            loc[li] = self.fuse[k](loc[li], mixed)
        return loc, proto_logits

    def forward(self, x, gt_mask=None):
        H, W = x.shape[-2:]
        loc = [proj(f) for proj, f in zip(self.loc_projs, self.cnn(x))]  # s4, s8, s16, s32
        if self.privileged:
            out = self._decode(loc, self._local_proto(loc, gt_mask), H, W)
            if self.training:
                loc_t, proto_t = self._fuse_semantic(x, loc, gt_mask)
                out["teacher"] = self._decode(loc_t, proto_t, H, W)
                out["feat_pairs"] = [(loc[li], loc_t[li]) for li in (1, 2, 3)]
            return out
        if self.use_vit:
            loc, proto_logits = self._fuse_semantic(x, loc, gt_mask)
        else:
            proto_logits = self._local_proto(loc, gt_mask)
        return self._decode(loc, proto_logits, H, W)

    def _decode(self, loc, proto_logits, H, W):
        p4, f4, b4 = self.dec4(loc[3], proto=proto_logits)
        p3, f3, b3 = self.dec3(loc[2], p4, f4, b4, proto=proto_logits)
        p2, f2, b2 = self.dec2(loc[1], p3, f3, b3)
        p1, f1, b1 = self.dec1(loc[0], p2, f2, b2)

        size1 = p1.shape[-2:]
        cat = torch.cat([p1] + [F.interpolate(p, size=size1, mode="bilinear", align_corners=False) for p in (p2, p3, p4)], dim=1)
        stage = (f1 - b1).float() * 0.5
        z = 0.5 * self.head(cat).float() + 0.5 * stage
        z = 0.8 * z + 0.2 * F.interpolate(proto_logits.float(), size=size1, mode="bilinear", align_corners=False)
        z = self._passe(z, f4 - b4, proto_logits)

        up = lambda t: F.interpolate(t.float(), size=(H, W), mode="bilinear", align_corners=False)
        return {
            "logits": up(z),
            "aux_logits": [up(0.5 * (f2 - b2)), up(0.5 * (f3 - b3)), up(0.5 * (f4 - b4))],
            "bg_logits": [up(b1), up(b2), up(b3), up(b4)],
            "boundary_logits": None,
            "proto_logits": proto_logits,
        }
