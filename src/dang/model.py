import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def parallel_selective_scan_1d(u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Exact parallel associative prefix scan (Hillis-Steele) for first-order input-gated recurrence:
        h_t = a_t * h_{t-1} + (1 - a_t) * u_t
    along the last dimension L in O(log2 L) parallel steps.
    """
    L = u.shape[-1]
    b = (1.0 - a) * u
    a_cur = a
    n_steps = int(math.ceil(math.log2(max(L, 2))))
    for step in range(n_steps):
        d = 1 << step
        if d >= L:
            break
        a_prev = a_cur[..., :-d]
        b_prev = b[..., :-d]
        a_hi = a_cur[..., d:]
        b_hi = b[..., d:]
        b = torch.cat([b[..., :d], a_hi * b_prev + b_hi], dim=-1)
        a_cur = torch.cat([a_cur[..., :d], a_hi * a_prev], dim=-1)
    return b


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1, dilation: int = 1, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ViTBlockAdapterWrapper(nn.Module):
    """Wraps a frozen DINOv2 ViT block with a Parameter-Efficient Bottleneck Adapter (Linear, Conv-LoRA, or Freq-MoE Conv-LoRA)."""

    def __init__(self, block: nn.Module, dim: int = 768, rank: int = 4, mode: str = "freq_moe_conv_lora", num_prefix: int = 1):
        super().__init__()
        self.block = block
        self.dim = dim
        self.rank = rank
        self.mode = mode
        self.num_prefix = num_prefix

        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = 1.0

        if mode == "conv_lora":
            self.dw_conv = nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=True)
            self.act = nn.GELU()
        elif mode == "freq_moe_conv_lora":
            # 3 parallel multi-scale dilated depthwise experts (d=1, 2, 3) + Joint Content & HF-Energy Router + High-Frequency Gate
            self.experts = nn.ModuleList([
                nn.Conv2d(rank, rank, kernel_size=3, padding=d, dilation=d, groups=rank, bias=True)
                for d in (1, 2, 3)
            ])
            self.router = nn.Linear(rank * 2, 3, bias=True)
            self.freq_gate = nn.Sequential(
                nn.Conv2d(rank, rank, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.mode == "none":
            return out

        B, N, C = out.shape
        z = self.down(out)  # [B, N, r]

        if self.mode == "linear_lora":
            delta = self.up(F.gelu(z))
            return out + self.scale * delta

        prefix = z[:, :self.num_prefix, :]
        spatial = z[:, self.num_prefix:, :]
        S = spatial.shape[1]
        H = W = int(math.isqrt(S))
        sp_2d = spatial.transpose(1, 2).reshape(B, self.rank, H, W)

        if self.mode == "conv_lora":
            sp_2d = self.act(sp_2d + self.dw_conv(sp_2d))
        elif self.mode == "freq_moe_conv_lora":
            low_freq = F.avg_pool2d(sp_2d, kernel_size=3, stride=1, padding=1)
            high_freq = sp_2d - low_freq
            gap = sp_2d.mean(dim=(-2, -1))              # [B, r] global content descriptor
            hf_energy = high_freq.abs().mean(dim=(-2, -1))  # [B, r] high-frequency energy descriptor
            weights = F.softmax(self.router(torch.cat([gap, hf_energy], dim=-1)), dim=-1)  # [B, 3]
            exp_out = sum(
                weights[:, k].view(B, 1, 1, 1) * exp(sp_2d)
                for k, exp in enumerate(self.experts)
            )
            fg = self.freq_gate(high_freq)
            sp_2d = self.act(sp_2d + exp_out * (1.0 + fg))

        spatial_out = sp_2d.flatten(2).transpose(1, 2)
        z_out = torch.cat([prefix, spatial_out], dim=1)
        return out + self.scale * self.up(z_out)


class SS2DLite(nn.Module):
    """4-Directional 2D Input-Gated State-Space Recurrence solved via parallel associative scan."""

    def __init__(self, dim: int, cond_dim: int = 0):
        super().__init__()
        self.in_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        gate_in = dim + cond_dim
        self.decay_proj = nn.Sequential(
            nn.Conv2d(gate_in, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(dim)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def _scan_2d(self, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B, C, H, W = u.shape
        u_h = u.permute(0, 2, 1, 3).reshape(B * H, C, W)
        a_h = a.permute(0, 2, 1, 3).reshape(B * H, C, W)
        h_lr = parallel_selective_scan_1d(u_h, a_h).reshape(B, H, C, W).permute(0, 2, 1, 3)
        h_rl = torch.flip(
            parallel_selective_scan_1d(torch.flip(u_h, dims=[-1]), torch.flip(a_h, dims=[-1])),
            dims=[-1],
        ).reshape(B, H, C, W).permute(0, 2, 1, 3)

        u_v = u.permute(0, 3, 1, 2).reshape(B * W, C, H)
        a_v = a.permute(0, 3, 1, 2).reshape(B * W, C, H)
        h_tb = parallel_selective_scan_1d(u_v, a_v).reshape(B, W, C, H).permute(0, 2, 3, 1)
        h_bt = torch.flip(
            parallel_selective_scan_1d(torch.flip(u_v, dims=[-1]), torch.flip(a_v, dims=[-1])),
            dims=[-1],
        ).reshape(B, W, C, H).permute(0, 2, 3, 1)
        return 0.25 * (h_lr + h_rl + h_tb + h_bt)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        B, C, H, W = x.shape
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=1)
        u = F.silu(self.dw_conv(u))

        gate_in = torch.cat([u, cond], dim=1) if cond is not None else u
        a = 0.10 + 0.85 * self.decay_proj(gate_in)

        if self.training and (H * W > 484):
            ssm_state = torch.utils.checkpoint.checkpoint(self._scan_2d, u, a, use_reentrant=False)
        else:
            ssm_state = self._scan_2d(u, a)
        out = self.norm(ssm_state) * F.silu(z)
        return x + self.out_proj(out)


class GDSSAFusion(nn.Module):
    """Gated Dual-Stream State-Space Alignment (GD-SSA): Learned Sigmoid Affinity Channel Swapping (SSCS) + Cross-Conditioned 2D-SSM."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale_loc = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.scale_sem = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.sscs_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.ssm_loc = SS2DLite(dim, cond_dim=dim)
        self.ssm_sem = SS2DLite(dim, cond_dim=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )
        self.out_conv = ConvBNAct(dim, dim, kernel_size=3, padding=1)

    def forward(self, f_loc: torch.Tensor, f_sem: torch.Tensor) -> torch.Tensor:
        f_l = f_loc * self.scale_loc
        f_s = f_sem * self.scale_sem
        # Learned Sigmoid Affinity Channel Swapping (SSCS)
        alpha = self.sscs_gate(torch.cat([f_l, f_s], dim=1))
        swap_l = alpha * f_l + (1.0 - alpha) * f_s
        swap_s = alpha * f_s + (1.0 - alpha) * f_l

        # Cross-stream-conditioned 2D selective recurrence
        s_l = self.ssm_loc(swap_l, cond=f_s)
        s_s = self.ssm_sem(swap_s, cond=f_l)
        g = self.gate(torch.cat([s_l, s_s], dim=1))
        fused = g * s_s + (1.0 - g) * s_l
        return self.out_conv(fused + f_loc + f_sem)


class A2BDBlock(nn.Module):
    """Attention-Augmented Boundary Decoder (A2BD) block with Local/Dilated Spatial Branches, 2D-SSM, and Sobel+Laplacian Edge Gate."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.sla_conv = nn.Sequential(
            nn.Conv2d(half, half, kernel_size=3, padding=1, groups=half, bias=False),
            nn.BatchNorm2d(half),
            nn.GELU(),
            nn.Conv2d(half, half, kernel_size=1, bias=False),
        )
        self.dla_d2 = nn.Conv2d(half, half, kernel_size=3, padding=2, dilation=2, groups=half, bias=False)
        self.dla_d3 = nn.Conv2d(half, half, kernel_size=3, padding=3, dilation=3, groups=half, bias=False)
        self.dla_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(half, half, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.dla_proj = nn.Sequential(
            nn.BatchNorm2d(half),
            nn.GELU(),
            nn.Conv2d(half, half, kernel_size=1, bias=False),
        )
        self.ssm = SS2DLite(dim, cond_dim=dim)
        self.attn_mod = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Fixed 3x3 Sobel kernels for explicit gradient magnitude extraction
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        self.register_buffer("sobel_kx", kx.repeat(dim, 1, 1, 1))
        self.register_buffer("sobel_ky", ky.repeat(dim, 1, 1, 1))

        self.edge_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )
        self.out_proj = ConvBNAct(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        a_sla = self.sla_conv(x1)
        g_dla = self.dla_gate(x2)
        a_dla = self.dla_proj(g_dla * self.dla_d2(x2) + (1.0 - g_dla) * self.dla_d3(x2))
        attn_feat = torch.cat([a_sla, a_dla], dim=1)

        # Condition 2D-SSM state recurrence directly on multi-scale spatial features
        ssm_feat = self.ssm(x, cond=attn_feat)
        mod_feat = ssm_feat * (1.0 + self.attn_mod(attn_feat)) + attn_feat

        gx = F.conv2d(x, self.sobel_kx.to(dtype=x.dtype), padding=1, groups=self.dim)
        gy = F.conv2d(x, self.sobel_ky.to(dtype=x.dtype), padding=1, groups=self.dim)
        sobel_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        lap = (x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)).abs()
        edge_gate = self.edge_enhance(0.5 * (sobel_mag + lap))
        return self.out_proj(mod_feat + x * edge_gate)


class DANGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim

        self.cnn = timm.create_model(cfg.cnn_backbone, pretrained=True, features_only=True)
        cnn_channels = self.cnn.feature_info.channels()
        self.loc_projs = nn.ModuleList([
            ConvBNAct(c, dim, kernel_size=1, padding=0) for c in cnn_channels
        ])

        self.use_dinov2 = cfg.use_dinov2
        if self.use_dinov2:
            self.vit = timm.create_model(cfg.vit_backbone, pretrained=True, dynamic_img_size=True)
            for p in self.vit.parameters():
                p.requires_grad = False

            num_prefix = getattr(self.vit, "num_prefix_tokens", 1)
            vit_dim = self.vit.embed_dim
            if cfg.peft_mode != "none":
                for i in range(len(self.vit.blocks)):
                    self.vit.blocks[i] = ViTBlockAdapterWrapper(
                        self.vit.blocks[i],
                        dim=vit_dim,
                        rank=cfg.lora_rank,
                        mode=cfg.peft_mode,
                        num_prefix=num_prefix,
                    )

            self.extract_layers = [2, 5, 8, 11]
            self.sem_projs = nn.ModuleList([
                ConvBNAct(vit_dim, dim, kernel_size=1, padding=0) for _ in range(4)
            ])

            if cfg.fusion_mode == "concat":
                self.fusers = nn.ModuleList([
                    ConvBNAct(dim * 2, dim, kernel_size=3, padding=1) for _ in range(4)
                ])
            elif cfg.fusion_mode == "gd_ssa":
                self.fusers = nn.ModuleList([
                    GDSSAFusion(dim) for _ in range(4)
                ])
            elif cfg.fusion_mode == "coarse_gd_ssa":
                self.fusers = nn.ModuleList([
                    ConvBNAct(dim * 2, dim, kernel_size=3, padding=1) if i < 2 else GDSSAFusion(dim)
                    for i in range(4)
                ])
                self.fine_gates = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
                        nn.BatchNorm2d(dim),
                        nn.Sigmoid(),
                    )
                    for _ in range(2)
                ])

        if cfg.decoder_mode == "a2bd":
            self.dec_blocks = nn.ModuleList([A2BDBlock(dim) for _ in range(4)])
        else:
            self.dec_blocks = nn.ModuleList([ConvBNAct(dim, dim, kernel_size=3, padding=1) for _ in range(4)])

        self.boundary_head = nn.Sequential(
            ConvBNAct(dim, dim // 2, kernel_size=3, padding=1),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
        )
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.boundary_alpha = nn.Parameter(torch.tensor(0.2))

        self.head = nn.Sequential(
            ConvBNAct(dim * 4, dim, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(dim, 1, kernel_size=1),
        )
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(dim, 1, kernel_size=1) for _ in range(3)
        ])

    def _extract_vit_pyramid(self, x: torch.Tensor, target_shapes: list[tuple[int, int]]) -> list[torch.Tensor]:
        B = x.shape[0]
        v_size = self.cfg.vit_img_size
        x_vit = F.interpolate(x, size=(v_size, v_size), mode="bilinear", align_corners=False)

        h = self.vit.patch_embed(x_vit)
        h = self.vit._pos_embed(h)
        h = self.vit.patch_drop(h)
        h = self.vit.norm_pre(h)

        num_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        H_p = W_p = v_size // 14

        sem_feats = []
        stage_idx = 0
        for idx, blk in enumerate(self.vit.blocks):
            h = blk(h)
            if idx in self.extract_layers:
                h_norm = self.vit.norm(h)
                sp = h_norm[:, num_prefix:, :].transpose(1, 2).reshape(B, -1, H_p, W_p)
                sp_proj = self.sem_projs[stage_idx](sp)
                th, tw = target_shapes[stage_idx]
                sp_resized = F.interpolate(sp_proj, size=(th, tw), mode="bilinear", align_corners=False)
                sem_feats.append(sp_resized)
                stage_idx += 1
        return sem_feats

    def forward(self, x: torch.Tensor) -> dict:
        H, W = x.shape[-2:]
        raw_loc = self.cnn(x)
        loc_feats = [proj(f) for proj, f in zip(self.loc_projs, raw_loc)]

        if self.use_dinov2:
            target_shapes = [(f.shape[-2], f.shape[-1]) for f in loc_feats]
            sem_feats = self._extract_vit_pyramid(x, target_shapes)
            fused_feats = []
            for i in range(4):
                if self.cfg.fusion_mode == "concat":
                    fused_feats.append(self.fusers[i](torch.cat([loc_feats[i], sem_feats[i]], dim=1)))
                elif self.cfg.fusion_mode == "gd_ssa":
                    fused_feats.append(self.fusers[i](loc_feats[i], sem_feats[i]))
                elif self.cfg.fusion_mode == "coarse_gd_ssa":
                    if i < 2:
                        g = self.fine_gates[i](torch.cat([loc_feats[i], sem_feats[i]], dim=1))
                        fused_feats.append(loc_feats[i] + g * self.fusers[i](torch.cat([loc_feats[i], sem_feats[i]], dim=1)))
                    else:
                        fused_feats.append(self.fusers[i](loc_feats[i], sem_feats[i]))
                else:
                    fused_feats.append(loc_feats[i] + sem_feats[i])
        else:
            fused_feats = loc_feats

        p4 = self.dec_blocks[3](fused_feats[3])
        p3 = self.dec_blocks[2](fused_feats[2] + F.interpolate(p4, size=fused_feats[2].shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.dec_blocks[1](fused_feats[1] + F.interpolate(p3, size=fused_feats[1].shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.dec_blocks[0](fused_feats[0] + F.interpolate(p2, size=fused_feats[0].shape[-2:], mode="bilinear", align_corners=False))

        b_logits_low = self.boundary_head(p1)
        if self.cfg.use_boundary_loss:
            p1 = p1 + self.boundary_alpha * torch.sigmoid(b_logits_low) * self.boundary_refine(p1)

        h1, w1 = p1.shape[-2:]
        cat_feat = torch.cat([
            p1,
            F.interpolate(p2, size=(h1, w1), mode="bilinear", align_corners=False),
            F.interpolate(p3, size=(h1, w1), mode="bilinear", align_corners=False),
            F.interpolate(p4, size=(h1, w1), mode="bilinear", align_corners=False),
        ], dim=1)

        logits = F.interpolate(self.head(cat_feat), size=(H, W), mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(b_logits_low, size=(H, W), mode="bilinear", align_corners=False)
        aux_logits = [
            F.interpolate(self.aux_heads[0](p2), size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(self.aux_heads[1](p3), size=(H, W), mode="bilinear", align_corners=False),
            F.interpolate(self.aux_heads[2](p4), size=(H, W), mode="bilinear", align_corners=False),
        ]
        return {
            "logits": logits,
            "boundary_logits": boundary_logits,
            "aux_logits": aux_logits,
        }
