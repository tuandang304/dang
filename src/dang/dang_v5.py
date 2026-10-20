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

import json
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


class HalfInstanceNorm(nn.Module):
    """IBN-Net style: instance-normalise the first half of the channels (removes per-image style/colour statistics),
    keep the rest; applied as a forward hook after an early backbone stage."""

    def __init__(self, ch: int):
        super().__init__()
        self.half = ch // 2
        self.inorm = nn.InstanceNorm2d(self.half, affine=True)

    def forward(self, x):
        a, b = x[:, : self.half], x[:, self.half:]
        return torch.cat([self.inorm(a.float()).to(x.dtype), b], dim=1)


class PhaseStructureBranch(nn.Module):
    """Phase-only reconstruction of the stride-4 features (unit amplitude spectrum keeps edges/structure and drops
    contrast/illumination energy), projected and added through a zero-initialised gate."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = ConvBNAct(dim, dim, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            f = torch.fft.fft2(x.float(), norm="ortho")
            ph = torch.fft.ifft2(f / (f.abs() + 1e-6), norm="ortho").real
            ph = (ph - ph.mean(dim=(2, 3), keepdim=True)) / (ph.std(dim=(2, 3), keepdim=True) + 1e-6)
        return x + self.gamma * self.proj(ph.to(x.dtype))


class DWConvAttention(nn.Module):
    """Drop-in depthwise-convolution replacement of a pretrained ViT attention block (Scribano et al., 2605.22132,
    blockwise DW formulation, re-implemented): out = W_O . DW3x3(W_V x) on the patch grid. W_V and W_O are the block's
    frozen pretrained value and output projections; the 3x3 kernels (Gaussian init) and the prefix-token mixing weights
    are trained on the task loss. Prefix tokens (cls + registers) exchange information with the grid via its mean."""

    def __init__(self, attn: nn.Module, num_prefix: int):
        super().__init__()
        dim = attn.proj.in_features
        heads = attn.num_heads
        self.num_prefix = num_prefix
        self.v = nn.Linear(dim, dim, bias=attn.qkv.bias is not None or getattr(attn, "v_bias", None) is not None)
        self.v.weight.data.copy_(attn.qkv.weight.data[2 * dim:])
        if self.v.bias is not None:
            vb = attn.qkv.bias[2 * dim:] if attn.qkv.bias is not None else attn.v_bias
            self.v.bias.data.copy_(vb.data)
        self.proj = attn.proj
        for p in list(self.v.parameters()) + list(self.proj.parameters()):
            p.requires_grad = False
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        c = torch.arange(3) - 1
        g = torch.exp(-(c[:, None] ** 2 + c[None, :] ** 2) / 2.0)
        self.dw.weight.data.copy_((g / g.sum()).expand_as(self.dw.weight))
        hd = dim // heads
        self.pre_from_grid = nn.Parameter(torch.full((heads,), 0.5).repeat_interleave(hd))
        self.grid_from_pre = nn.Parameter(torch.full((heads,), 0.5).repeat_interleave(hd))

    def forward(self, x, rope=None, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        v = self.v(x)
        vp, vg = v[:, : self.num_prefix], v[:, self.num_prefix:]
        g = int(math.isqrt(vg.shape[1]))
        grid = vg.transpose(1, 2).reshape(B, C, g, g) + (self.grid_from_pre * vp.mean(1))[:, :, None, None]
        out_g = self.dw(grid).flatten(2).transpose(1, 2)
        out_p = vp + (self.pre_from_grid * vg.mean(1))[:, None, :]
        return self.proj(torch.cat([out_p, out_g], dim=1))


@torch.no_grad()
def attention_variability(vit, chunks, size: int):
    """Per-block mean over heads of sum_ij std_x(E(x)_h[i, j]) (the Sigma_b criterion of 2605.22132): low = the
    block's attention is nearly input-independent, i.e. convolution-like. Welford over all images in `chunks`, fp32."""
    blocks = [getattr(b, "block", b) for b in vit.blocks]
    stats, hooks, saved = {}, [], []

    def hook(idx):
        def fn(mod, inp, out):
            for a in out.float():  # one image at a time: (heads, N, N)
                n, mean, m2 = stats.get(idx, (0, torch.zeros_like(a), torch.zeros_like(a)))
                n += 1
                d = a - mean
                mean = mean + d / n
                stats[idx] = (n, mean, m2 + d * (a - mean))
        return fn

    for i, b in enumerate(blocks):
        saved.append(b.attn.fused_attn)
        b.attn.fused_attn = False
        hooks.append(b.attn.attn_drop.register_forward_hook(hook(i)))
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            for images in chunks:
                xv = F.interpolate(images.float(), size=(size, size), mode="bilinear", align_corners=False)
                vit.forward_features(xv)
    finally:
        for h in hooks:
            h.remove()
        for b, f in zip(blocks, saved):
            b.attn.fused_attn = f
    return [float((stats[i][2] / max(stats[i][0] - 1, 1)).sqrt().sum(dim=(-2, -1)).mean()) for i in range(len(blocks))]

class LiteConvBlock(nn.Module):
    """ConvNeXt-style residual block: depthwise 7x7, BN, pointwise 2x expansion, GELU, pointwise back."""

    def __init__(self, ch: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.pw1 = nn.Conv2d(ch, 2 * ch, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(2 * ch, ch, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.bn(self.dw(x)))))


class DetailStem(nn.Module):
    """Shallow convolutional spatial-prior stem (cf. ViT-Adapter SPM, SegDINO): stride-4 and stride-8 detail
    features straight from the image, used when the frozen ViT is the only deep trunk (v12_trunk="vit")."""

    def __init__(self, c4: int = 64, c8: int = 128):
        super().__init__()
        self.s4 = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
                                nn.Conv2d(32, c4, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c4), nn.GELU(),
                                LiteConvBlock(c4), LiteConvBlock(c4))
        self.s8 = nn.Sequential(nn.Conv2d(c4, c8, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c8), nn.GELU(),
                                LiteConvBlock(c8), LiteConvBlock(c8))
        self.channels = [c4, c8]

    def forward(self, x):
        f4 = self.s4(x)
        return [f4, self.s8(f4)]


class TokenPyramid(nn.Module):
    """Token Pyramid Adaptation (SegDINO, 2606.17972, re-implemented): the k-th ViT readout depth becomes the k-th
    pyramid level (shallow -> stride 4, deep -> stride 32) via a per-level learned resize; no convolutional trunk."""

    def __init__(self, dim: int):
        super().__init__()
        self.up4 = nn.Sequential(nn.ConvTranspose2d(dim, dim, 2, stride=2, groups=dim, bias=False), nn.BatchNorm2d(dim),
                                 nn.GELU(), SepConvBNAct(dim, dim))
        self.l8 = SepConvBNAct(dim, dim)
        self.l16 = ConvBNAct(dim, dim, kernel_size=1, padding=0)
        self.down32 = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim, bias=False),
                                    nn.BatchNorm2d(dim), nn.GELU(), ConvBNAct(dim, dim, kernel_size=1, padding=0))

    def forward(self, sem, H, W):
        rs = lambda t, d: F.interpolate(t, size=(H // d, W // d), mode="bilinear", align_corners=False)
        return [self.up4(rs(sem[0], 8)), self.l8(rs(sem[1], 8)), self.l16(rs(sem[2], 16)), self.down32(rs(sem[3], 16))]


class ResidualDepthwiseBlock(nn.Module):
    """SegDINO-v2 (2606.17972, github.com/script-Yang/segdino_v2 @0965c799 dpt.py), verbatim: x + gamma * GELU(GN(PW(DW(x))))
    with gamma initialised to zero."""

    def __init__(self, channels, use_group_norm=True):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = (
            nn.GroupNorm(min(32, channels), channels)
            if use_group_norm
            else nn.BatchNorm2d(channels)
        )
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = self.act(self.norm(self.pointwise(self.depthwise(x))))
        return x + self.gamma * residual


class ScaleAwareDecoder(nn.Module):
    """SegDINO-v2 SAD (TPASADDecoder minus its own token resampling, which V12-b's TokenPyramid already provides):
    intra-level residual refinement, then top-down x_k = R(up(x_{k+1}) + level_k); 1x1 output at stride 4.
    Extra (DANG): 1x1 deep-supervision heads on the s8/s16/s32 merged levels."""

    def __init__(self, dim: int):
        super().__init__()
        self.intra = nn.ModuleList([ResidualDepthwiseBlock(dim) for _ in range(4)])
        self.inter = nn.ModuleList([ResidualDepthwiseBlock(dim) for _ in range(4)])
        self.out_conv = nn.Conv2d(dim, 1, 1)
        self.aux = nn.ModuleList([nn.Conv2d(dim, 1, 1) for _ in range(3)])

    def forward(self, pyr):
        lv = [blk(p) for blk, p in zip(self.intra, pyr)]          # s4, s8, s16, s32
        x = self.inter[3](lv[3])
        merged = [x]
        for k in (2, 1, 0):
            x = self.inter[k](F.interpolate(x, size=lv[k].shape[-2:], mode="bilinear", align_corners=False) + lv[k])
            merged.append(x)
        x1 = merged[-1]
        aux = [head(m) for head, m in zip(self.aux, (merged[2], merged[1], merged[0]))]   # s8, s16, s32
        return self.out_conv(x1), aux


class QueryDecoderLayer(nn.Module):
    """Mask2Former decoder layer (masked cross-attention -> self-attention -> FFN, post-norm)."""

    def __init__(self, dim: int, heads: int = 8, ffn: int = 768):
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, q, qpos, mem, mpos, attn_mask=None):
        q = self.n1(q + self.cross(q + qpos, mem + mpos, mem, attn_mask=attn_mask, need_weights=False)[0])
        q = self.n2(q + self.self_attn(q + qpos, q + qpos, q, need_weights=False)[0])
        return self.n3(q + self.ffn(q))


class DualQueryMaskDecoder(nn.Module):
    """Query-based mask decoding on the token pyramid (Mask2Former 2112.01527; encoder-only variants EoMT 2503.19108,
    EoSeg 2607.00223). Two learnable queries (foreground, background); a light FPN pixel decoder gives stride-4 pixel
    embeddings; 3 decoder layers attend to s32, s16, s8 tokens in turn. The foreground query's cross-attention is masked
    to where the previous prediction is foreground (fallback: all tokens). Logit = (e_fg - e_bg) . pixel embedding."""

    def __init__(self, dim: int, num_layers: int = 3):
        super().__init__()
        self.pix_lat = nn.ModuleList([ConvBNAct(dim, dim, kernel_size=1, padding=0) for _ in range(4)])
        self.pix_out = nn.Sequential(SepConvBNAct(dim, dim), nn.Conv2d(dim, dim, 1))
        self.query_feat = nn.Embedding(2, dim)
        self.query_pos = nn.Embedding(2, dim)
        self.level_embed = nn.Embedding(3, dim)
        self.mem_proj = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(3)])
        self.layers = nn.ModuleList([QueryDecoderLayer(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.mask_embed = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.scale = dim ** -0.5

    @staticmethod
    def _pos2d(h, w, dim, device, dtype):
        """Fixed 2D sine position encoding (DETR), flattened to (h*w, dim)."""
        npf = dim // 4
        y, x = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing="ij")
        omega = 1.0 / (10000 ** (torch.arange(npf, device=device, dtype=torch.float32) / npf))
        out = [torch.sin(2 * math.pi * x.flatten()[:, None] * omega), torch.cos(2 * math.pi * x.flatten()[:, None] * omega),
               torch.sin(2 * math.pi * y.flatten()[:, None] * omega), torch.cos(2 * math.pi * y.flatten()[:, None] * omega)]
        return torch.cat(out, dim=1).to(dtype)

    def _predict(self, q, pix):
        e = self.mask_embed(self.norm(q))                                      # B, 2, C
        maps = torch.einsum("bqc,bchw->bqhw", e, pix) * self.scale
        return maps[:, 0:1], maps[:, 1:2]                                     # fg, bg logits at stride 4

    def forward(self, pyr):
        size4 = pyr[0].shape[-2:]
        pix = sum(F.interpolate(l(p), size=size4, mode="bilinear", align_corners=False) for l, p in zip(self.pix_lat, pyr))
        pix = self.pix_out(pix)
        B = pix.shape[0]
        q = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)
        qpos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        fg, bg = self._predict(q, pix)
        outs = [(fg, bg)]
        heads = self.layers[0].cross.num_heads
        for i, layer in enumerate(self.layers):
            lvl = 2 - (i % 3)                                                    # s32 (pyr[3]) -> s16 -> s8
            m = self.mem_proj[lvl](pyr[lvl + 1])
            h, w = m.shape[-2:]
            mem = m.flatten(2).transpose(1, 2)
            mpos = (self._pos2d(h, w, m.shape[1], m.device, mem.dtype) + self.level_embed.weight[lvl]).unsqueeze(0)
            # masked attention for the foreground query (Mask2Former): attend where p(fg) >= 0.5 at this level
            p = torch.sigmoid(F.interpolate((fg - bg).float(), size=(h, w), mode="bilinear", align_corners=False))
            block = (p.flatten(1) < 0.5)                                         # B, HW (True = not allowed)
            block = block & ~block.all(dim=1, keepdim=True)                     # fallback: all tokens if empty
            am = torch.zeros(B, 2, h * w, dtype=torch.bool, device=m.device)
            am[:, 0] = block
            am = am.repeat_interleave(heads, dim=0)                              # (B*heads, 2, HW)
            q = layer(q, qpos, mem, mpos, attn_mask=am.detach())
            fg, bg = self._predict(q, pix)
            outs.append((fg, bg))
        return outs


class DANGv5(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim
        # v12 trunk: "dual" = V8-b (fine-tuned ConvNeXt local stream + frozen ViT); "vit" = the frozen ViT is the only deep
        # trunk, a shallow conv stem supplies s4/s8 detail and s16/s32 come from the semantic tokens
        self.trunk = getattr(cfg, "v12_trunk", "dual")
        if self.trunk == "vit":
            assert cfg.v5_vit, "v12_trunk='vit' needs a semantic stream"
            self.stem = DetailStem()
            loc_ch = self.stem.channels
        elif self.trunk == "tpa":
            assert cfg.v5_vit and len(cfg.v5_vit_layers) == 4, "v12_trunk='tpa' needs 4 ViT readout depths"
            loc_ch = []
        else:
            self.cnn = timm.create_model(cfg.v5_local, pretrained=True, features_only=True)
            loc_ch = self.cnn.feature_info.channels()
        self.loc_projs = nn.ModuleList([ConvBNAct(c, dim, kernel_size=1, padding=0) for c in loc_ch])

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
            if self.trunk == "tpa":
                self.depth_logits = None                                       # fixed depth -> scale mapping
                self.pyramid = TokenPyramid(dim)
            elif self.trunk == "vit":
                self.fuse = nn.ModuleList([GatedSemanticFusion(dim)])          # s8 only: stem detail + semantics
                self.sem_s16 = ConvBNAct(dim, dim, kernel_size=3, padding=1)
                self.sem_s32 = ConvBNAct(dim, dim, kernel_size=3, padding=1)
            else:
                self.fuse = nn.ModuleList([GatedSemanticFusion(dim) for _ in range(3)])
        else:
            self.sem16 = ConvBNAct(dim, dim, kernel_size=3, padding=1)

        # round-1 architecture switches (all off = V8-b)
        self.fuse_s4 = bool(getattr(cfg, "v5_fuse_s4", False)) and self.use_vit
        if self.fuse_s4:
            self.depth_logits_s4 = nn.Parameter(torch.zeros(1, len(self.vit_layers)))
            self.fuse4 = GatedSemanticFusion(dim)
        self.ibn = None
        if getattr(cfg, "v5_ibn", False):
            chs = self.cnn.feature_info.channels()
            self.ibn = nn.ModuleList([HalfInstanceNorm(chs[0]), HalfInstanceNorm(chs[1])])
            self.cnn.stages_0.register_forward_hook(lambda m, i, o: self.ibn[0](o))
            self.cnn.stages_1.register_forward_hook(lambda m, i, o: self.ibn[1](o))
        self.phase = PhaseStructureBranch(dim) if getattr(cfg, "v5_phase", False) else None
        self.detail = None
        if getattr(cfg, "v5_detail_s2", False):
            self.detail = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
                                        ConvBNAct(32, 32, kernel_size=3, padding=1))
            self.detail_ctx = ConvBNAct(dim, 32, kernel_size=1, padding=0)
            self.detail_head = nn.Sequential(ConvBNAct(65, 32, kernel_size=3, padding=1), nn.Conv2d(32, 1, 1))
            nn.init.zeros_(self.detail_head[1].weight)
            nn.init.zeros_(self.detail_head[1].bias)

        if tuple(getattr(cfg, "v12_dw_fixed", ())):   # latency benchmark only: fixed blocks, no data needed
            self.apply_dw(tuple(cfg.v12_dw_fixed))

        # ablation switch: v12_proto=False removes the prototype head, its logit term and its loss (sad decoder only)
        self.use_proto = bool(getattr(cfg, "v12_proto", True))
        self.proto_head = (DualPrototypeMarginHead(dim=dim, proto_dim=cfg.proto_dim, num_prototypes=cfg.num_prototypes)
                           if self.use_proto else None)

        # round 2: decoding stack on the token pyramid ("dual" = V12-b) and the PASSE logit envelope
        self.decoder = getattr(cfg, "v12_decoder", "dual")
        self.use_passe = bool(getattr(cfg, "v12_passe", True))
        if self.decoder == "sad":
            self.sad = ScaleAwareDecoder(dim)
        elif self.decoder == "query":
            self.qdec = DualQueryMaskDecoder(dim)
        if self.decoder != "dual":
            return                                                         # no dual decoder stages / 768-ch head

        self.dec4 = DualDecoderStage(dim, has_top=False, use_proto=True)
        self.dec3 = DualDecoderStage(dim, has_top=True, use_proto=True)
        self.dec2 = DualDecoderStage(dim, has_top=True, use_proto=False)
        self.dec1 = DualDecoderStage(dim, has_top=True, use_proto=False)
        if getattr(cfg, "v12_head", "v8b") == "slim":   # 4 x 32-ch bottleneck + separable 3x3 (~0.6 vs 10.3 GMACs)
            self.head_red = nn.ModuleList([ConvBNAct(dim, 32, kernel_size=1, padding=0) for _ in range(4)])
            self.head = nn.Sequential(SepConvBNAct(128, 64), nn.Dropout2d(0.1), nn.Conv2d(64, 1, 1))
        else:
            self.head_red = None
            self.head = nn.Sequential(ConvBNAct(dim * 4, dim, kernel_size=3, padding=1), nn.Dropout2d(0.1), nn.Conv2d(dim, 1, 1))

    @torch.no_grad()
    def prepare(self, loader, device):
        """Data-dependent set-up before training (training images only): pick the v12_dw_blocks ViT blocks with the
        most input-invariant attention and replace their attention with DWConvAttention."""
        k = int(getattr(self.cfg, "v12_dw_blocks", 0))
        if k <= 0 or not self.use_vit:
            return
        imgs, n_s = [], int(getattr(self.cfg, "v12_dw_samples", 64))
        for batch in loader:
            imgs.append(batch["image"])
            if sum(t.shape[0] for t in imgs) >= n_s:
                break
        imgs = torch.cat(imgs)[:n_s].to(device)
        sigma_b = attention_variability(self.vit, imgs.split(4), self.cfg.v5_vit_size)
        chosen = sorted(sorted(range(len(sigma_b)), key=lambda i: sigma_b[i])[:k])
        print("V12_DW_SELECTION_JSON: " + json.dumps({"sigma_b": [round(v, 2) for v in sigma_b], "chosen": chosen,
                                                      "n_samples": int(imgs.shape[0])}), flush=True)
        self.apply_dw(chosen)

    def apply_dw(self, indices):
        npre = getattr(self.vit, "num_prefix_tokens", 1)
        dev = next(self.vit.parameters()).device
        for i in indices:
            inner = getattr(self.vit.blocks[i], "block", self.vit.blocks[i])
            inner.attn = DWConvAttention(inner.attn, npre).to(dev)
        self.dw_blocks = list(indices)

    def _semantic(self, x):
        v = self.cfg.v5_vit_size
        xv = F.interpolate(x, size=(v, v), mode="bilinear", align_corners=False)
        if getattr(self.cfg, "v5_vit_fp32", False) and xv.is_cuda:
            # DINOv3 ViT-L overflows in fp16 on T4 (no bf16); run the semantic stream in fp32
            with torch.autocast(device_type="cuda", enabled=False):
                feats = self.vit.forward_intermediates(xv.float(), indices=self.vit_layers, norm=True,
                                                       output_fmt="NCHW", intermediates_only=True)
        else:
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

    def forward(self, x, gt_mask=None):
        H, W = x.shape[-2:]
        if self.trunk == "tpa":
            sem = self._semantic(x)
            proto_logits = self.proto_head(sem, gt_mask=gt_mask)[0] if self.use_proto else None
            loc = self.pyramid(sem, H, W)                                     # s4, s8, s16, s32 from tokens only
        elif self.trunk == "vit":
            loc = [proj(f) for proj, f in zip(self.loc_projs, self.stem(x))]  # s4, s8
            sem = self._semantic(x)
            proto_logits, _ = self.proto_head(sem, gt_mask=gt_mask)
            w = F.softmax(self.depth_logits, dim=-1)
            mixed = [sum(w[k, j] * s for j, s in enumerate(sem)) for k in range(3)]
            loc[1] = self.fuse[0](loc[1], mixed[0])
            rs = lambda t, d: F.interpolate(t, size=(H // d, W // d), mode="bilinear", align_corners=False)
            loc += [self.sem_s16(rs(mixed[1], 16)), self.sem_s32(rs(mixed[2], 32))]
        else:
            loc = [proj(f) for proj, f in zip(self.loc_projs, self.cnn(x))]  # s4, s8, s16, s32

        if self.use_vit and self.trunk == "dual":
            sem = self._semantic(x)
            proto_logits, _ = self.proto_head(sem, gt_mask=gt_mask)
            w = F.softmax(self.depth_logits, dim=-1)
            for k, li in enumerate((1, 2, 3)):
                mixed = sum(w[k, j] * s for j, s in enumerate(sem))
                loc[li] = self.fuse[k](loc[li], mixed)
            if self.fuse_s4:
                w4 = F.softmax(self.depth_logits_s4, dim=-1)
                loc[0] = self.fuse4(loc[0], sum(w4[0, j] * s for j, s in enumerate(sem)))
        elif not self.use_vit:
            s16 = self.sem16(loc[2])
            s32 = F.interpolate(loc[3], size=s16.shape[-2:], mode="bilinear", align_corners=False)
            proto_logits, _ = self.proto_head([s16, s16, s32, s16], gt_mask=gt_mask)

        if self.phase is not None:
            loc[0] = self.phase(loc[0])

        up = lambda t: F.interpolate(t.float(), size=(H, W), mode="bilinear", align_corners=False)
        if self.decoder == "sad":
            y, aux = self.sad(loc)
            z = y.float()
            if proto_logits is not None:
                z = z + 0.2 * F.interpolate(proto_logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
            return {"logits": up(z), "aux_logits": [up(a) for a in aux], "bg_logits": None, "boundary_logits": None,
                    "proto_logits": proto_logits}
        if self.decoder == "query":
            outs = self.qdec(loc)
            fg, bg = outs[-1]
            z = (fg - bg).float() + 0.2 * F.interpolate(proto_logits.float(), size=fg.shape[-2:], mode="bilinear",
                                                        align_corners=False)
            return {"logits": up(z), "aux_logits": [up(f - b) for f, b in outs[:-1]], "bg_logits": [up(b) for _, b in outs],
                    "boundary_logits": None, "proto_logits": proto_logits}

        p4, f4, b4 = self.dec4(loc[3], proto=proto_logits)
        p3, f3, b3 = self.dec3(loc[2], p4, f4, b4, proto=proto_logits)
        p2, f2, b2 = self.dec2(loc[1], p3, f3, b3)
        p1, f1, b1 = self.dec1(loc[0], p2, f2, b2)

        size1 = p1.shape[-2:]
        ps = [p1, p2, p3, p4] if self.head_red is None else [r(p) for r, p in zip(self.head_red, (p1, p2, p3, p4))]
        cat = torch.cat([ps[0]] + [F.interpolate(p, size=size1, mode="bilinear", align_corners=False) for p in ps[1:]], dim=1)
        stage = (f1 - b1).float() * 0.5
        z = 0.5 * self.head(cat).float() + 0.5 * stage
        z = 0.8 * z + 0.2 * F.interpolate(proto_logits.float(), size=size1, mode="bilinear", align_corners=False)
        if self.use_passe:
            z = self._passe(z, f4 - b4, proto_logits)

        if self.detail is not None:
            d = self.detail(x)
            size2 = d.shape[-2:]
            ctx = F.interpolate(self.detail_ctx(p1), size=size2, mode="bilinear", align_corners=False)
            z2 = F.interpolate(z, size=size2, mode="bilinear", align_corners=False)
            z = z2 + self.detail_head(torch.cat([d, ctx.to(d.dtype), z2.to(d.dtype)], dim=1)).float()
        return {
            "logits": up(z),
            "aux_logits": [up(0.5 * (f2 - b2)), up(0.5 * (f3 - b3)), up(0.5 * (f4 - b4))],
            "bg_logits": [up(b1), up(b2), up(b3), up(b4)],
            "boundary_logits": None,
            "proto_logits": proto_logits,
        }
