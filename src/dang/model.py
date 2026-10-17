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


class LayerScale2d(nn.Module):
    """Per-channel learnable residual scaling initialized small so new branches start near identity."""

    def __init__(self, dim: int, init_val: float = 0.05):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


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
            gap = sp_2d.mean(dim=(-2, -1))
            hf_energy = high_freq.abs().mean(dim=(-2, -1))
            weights = F.softmax(self.router(torch.cat([gap, hf_energy], dim=-1)), dim=-1)
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
    """4-Directional 2D Input-Gated State-Space Recurrence with short local-memory initialization."""

    def __init__(self, dim: int, cond_dim: int = 0):
        super().__init__()
        self.in_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        gate_in = dim + cond_dim
        self.decay_proj = nn.Sequential(
            nn.Conv2d(gate_in, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # Initialize decay bias to -2.0 so initial retention a ≈ 0.20 (local memory, preventing cross-shaped blur)
        nn.init.zeros_(self.decay_proj[0].weight)
        nn.init.constant_(self.decay_proj[0].bias, -2.0)

        self.norm = nn.BatchNorm2d(dim)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.ls = LayerScale2d(dim, init_val=0.05)

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
        return x + self.ls(self.out_proj(out))


class EdgeSnappedSemUp(nn.Module):
    """
    Content-Aware Edge-Snapped Semantic Upsampling:
    Uses sharp local CNN features to predict a per-pixel k*k reassembly kernel over upsampled DINOv2
    features, snapping coarse 14x14 patch semantics onto fine CNN lesion boundaries.
    """

    def __init__(self, dim: int, k: int = 5, dilation: int = 2):
        super().__init__()
        self.k = k
        self.d = dilation
        self.kernel_pred = nn.Sequential(
            ConvBNAct(dim, dim // 4, kernel_size=1, padding=0),
            nn.Conv2d(dim // 4, k * k, kernel_size=3, padding=1, bias=True),
        )
        # Initialize to uniform box kernel at step 0
        nn.init.zeros_(self.kernel_pred[1].weight)
        nn.init.zeros_(self.kernel_pred[1].bias)

    def forward(self, sem_lr: torch.Tensor, loc_feat: torch.Tensor) -> torch.Tensor:
        B, C = sem_lr.shape[:2]
        H, W = loc_feat.shape[-2:]
        sem_up = F.interpolate(sem_lr, size=(H, W), mode="bilinear", align_corners=False)
        w = F.softmax(self.kernel_pred(loc_feat), dim=1).unsqueeze(1)  # [B, 1, k*k, H, W]
        pad = self.d * (self.k // 2)
        patches = F.unfold(sem_up, kernel_size=self.k, dilation=self.d, padding=pad)
        patches = patches.view(B, C, self.k * self.k, H, W)
        return (patches * w).sum(dim=2)


class GDSSAFusion(nn.Module):
    """
    Identity-Preserving Gated Dual-Stream State-Space Alignment (GD-SSA v2):
    Asymmetrically initialized SSCS channel mixing + Cross-Conditioned 2D-SSM + Spatial Reliability Gate.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale_loc = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.scale_sem = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.sscs_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # Initialize SSCS gate asymmetrically (alpha ≈ 0.90) so local and semantic streams start distinct
        nn.init.zeros_(self.sscs_gate[1].weight)
        nn.init.constant_(self.sscs_gate[1].bias, math.log(0.9 / 0.1))

        self.ssm_loc = SS2DLite(dim, cond_dim=dim)
        self.ssm_sem = SS2DLite(dim, cond_dim=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )
        self.out_conv = ConvBNAct(dim, dim, kernel_size=3, padding=1)
        self.reliability = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.reliability[-2].weight)
        nn.init.constant_(self.reliability[-2].bias, -1.5)
        self.ls = LayerScale2d(dim, init_val=0.05)

    def forward(self, f_loc: torch.Tensor, f_sem: torch.Tensor) -> torch.Tensor:
        f_l = f_loc * self.scale_loc
        f_s = f_sem * self.scale_sem
        alpha = self.sscs_gate(torch.cat([f_l, f_s], dim=1))
        swap_l = alpha * f_l + (1.0 - alpha) * f_s
        swap_s = alpha * f_s + (1.0 - alpha) * f_l

        s_l = self.ssm_loc(swap_l, cond=f_s)
        s_s = self.ssm_sem(swap_s, cond=f_l)
        g = self.gate(torch.cat([s_l, s_s], dim=1))
        fused = g * s_s + (1.0 - g) * s_l
        delta = self.out_conv(fused)
        conf = self.reliability(torch.cat([f_loc, f_sem], dim=1))
        # Preserve pretrained local pathway as identity at initialization
        return f_loc + conf * self.ls(delta)


class SafeConcatFusion(nn.Module):
    """Identity-preserving gated concat fusion for Dual-Stream Control arm (R5b)."""

    def __init__(self, dim: int):
        super().__init__()
        self.fuser = ConvBNAct(dim * 2, dim, kernel_size=3, padding=1)
        self.reliability = nn.Conv2d(dim * 2, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.reliability.weight)
        nn.init.constant_(self.reliability.bias, -1.5)
        self.ls = LayerScale2d(dim, init_val=0.05)

    def forward(self, f_loc: torch.Tensor, f_sem: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([f_loc, f_sem], dim=1)
        conf = torch.sigmoid(self.reliability(cat))
        return f_loc + conf * self.ls(self.fuser(cat))


class A2BDBlock(nn.Module):
    """
    Attention-Augmented Boundary Decoder (A2BD v2) block:
    Baseline FPN conv + Gated Residual (Multi-Scale Local/Dilated Spatial Branches + 2D-SSM + Bounded L1-Sobel/HighPass Gate).
    """

    def __init__(self, dim: int, use_ssm: bool = True):
        super().__init__()
        self.dim = dim
        self.use_ssm = use_ssm
        self.base = ConvBNAct(dim, dim, kernel_size=3, padding=1)

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
        if use_ssm:
            self.ssm = SS2DLite(dim, cond_dim=dim)
            self.attn_mod = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

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
        self.ls = LayerScale2d(dim, init_val=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_y = self.base(x)
        x1, x2 = base_y.chunk(2, dim=1)
        a_sla = self.sla_conv(x1)
        g_dla = self.dla_gate(x2)
        a_dla = self.dla_proj(g_dla * self.dla_d2(x2) + (1.0 - g_dla) * self.dla_d3(x2))
        attn_feat = torch.cat([a_sla, a_dla], dim=1)

        if self.use_ssm:
            ssm_feat = self.ssm(base_y, cond=attn_feat)
            mod_feat = ssm_feat * (1.0 + self.attn_mod(attn_feat)) + attn_feat
        else:
            mod_feat = base_y + attn_feat

        # Bounded fp32 L1-Sobel gradient + box high-pass filter (no bf16 sqrt gradient spikes)
        xf = base_y.float()
        gx = F.conv2d(xf, self.sobel_kx, padding=1, groups=self.dim)
        gy = F.conv2d(xf, self.sobel_ky, padding=1, groups=self.dim)
        sobel_l1 = (gx.abs() + gy.abs()).to(dtype=base_y.dtype)
        box_hp = (base_y - F.avg_pool2d(base_y, kernel_size=3, stride=1, padding=1)).abs()
        edge_gate = self.edge_enhance(0.5 * (sobel_l1 + box_hp))
        return base_y + self.ls(self.out_proj(mod_feat + base_y * edge_gate))


class DetailFirstDecoderStage(nn.Module):
    """Protects fine lateral CNN detail with a spatial top-down reliability gate before A2BD v2 refinement."""

    def __init__(self, dim: int, has_top: bool = True, use_ssm: bool = True, top_bias: float = 0.0):
        super().__init__()
        self.has_top = has_top
        if has_top:
            self.top_gate = nn.Conv2d(dim * 2, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.top_gate.weight)
            nn.init.constant_(self.top_gate.bias, top_bias)
        self.block = A2BDBlock(dim, use_ssm=use_ssm)

    def forward(self, lateral: torch.Tensor, top: torch.Tensor | None = None) -> torch.Tensor:
        x = lateral
        if self.has_top and top is not None:
            top_up = F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            g = 2.0 * torch.sigmoid(self.top_gate(torch.cat([lateral, top_up], dim=1)))
            x = lateral + g * top_up
        return self.block(x)


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
            self.adapter_start_block = getattr(cfg, "adapter_start_block", 6)
            if cfg.peft_mode != "none":
                for i in range(self.adapter_start_block, len(self.vit.blocks)):
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

            if cfg.fusion_mode == "safe_concat":
                self.fusers = nn.ModuleList([
                    nn.Identity(),
                    SafeConcatFusion(dim),
                    SafeConcatFusion(dim),
                    SafeConcatFusion(dim),
                ])
            elif cfg.fusion_mode == "safe_gd_ssa_v2":
                self.edge_snap_s8 = EdgeSnappedSemUp(dim, k=5, dilation=2)
                self.edge_snap_s16 = EdgeSnappedSemUp(dim, k=3, dilation=1)
                self.fuser_s8 = SafeConcatFusion(dim)
                self.fuser_s16 = GDSSAFusion(dim)
                self.fuser_s32 = GDSSAFusion(dim)
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

        if cfg.decoder_mode == "detail_a2bd_v2":
            # p4(11x11, SSM), p3(22x22, SSM), p2(44x44, SSM), p1(88x88, local edge-gated without 88x88 scan)
            self.dec_p4 = DetailFirstDecoderStage(dim, has_top=False, use_ssm=True)
            self.dec_p3 = DetailFirstDecoderStage(dim, has_top=True, use_ssm=True, top_bias=0.0)
            self.dec_p2 = DetailFirstDecoderStage(dim, has_top=True, use_ssm=True, top_bias=-0.2)
            self.dec_p1 = DetailFirstDecoderStage(dim, has_top=True, use_ssm=False, top_bias=-0.4)
        elif cfg.decoder_mode == "a2bd":
            self.dec_blocks = nn.ModuleList([A2BDBlock(dim, use_ssm=True) for _ in range(4)])
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
        nn.init.zeros_(self.boundary_refine[-1].weight)
        self.boundary_alpha = nn.Parameter(torch.tensor(0.2))

        self.head = nn.Sequential(
            ConvBNAct(dim * 4, dim, kernel_size=3, padding=1),
            nn.Dropout2d(0.1),
            nn.Conv2d(dim, 1, kernel_size=1),
        )
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(dim, 1, kernel_size=1) for _ in range(3)
        ])

    def _extract_vit_pyramid(self, x: torch.Tensor) -> list[torch.Tensor]:
        B = x.shape[0]
        v_size = self.cfg.vit_img_size
        x_vit = F.interpolate(x, size=(v_size, v_size), mode="bilinear", align_corners=False)

        num_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        H_p = W_p = v_size // 14

        sem_feats_25 = []
        stage_idx = 0

        with torch.no_grad():
            h = self.vit.patch_embed(x_vit)
            h = self.vit._pos_embed(h)
            h = self.vit.patch_drop(h)
            h = self.vit.norm_pre(h)
            for idx in range(self.adapter_start_block):
                h = self.vit.blocks[idx](h)
                if idx in self.extract_layers:
                    h_norm = self.vit.norm(h)
                    sp = h_norm[:, num_prefix:, :].transpose(1, 2).reshape(B, -1, H_p, W_p)
                    sem_feats_25.append((stage_idx, sp.detach()))
                    stage_idx += 1

        h = h.detach()
        for idx in range(self.adapter_start_block, len(self.vit.blocks)):
            h = self.vit.blocks[idx](h)
            if idx in self.extract_layers:
                h_norm = self.vit.norm(h)
                sp = h_norm[:, num_prefix:, :].transpose(1, 2).reshape(B, -1, H_p, W_p)
                sem_feats_25.append((stage_idx, sp))
                stage_idx += 1

        proj_25 = [self.sem_projs[s_idx](sp) for s_idx, sp in sem_feats_25]
        return proj_25

    def forward(self, x: torch.Tensor) -> dict:
        H, W = x.shape[-2:]
        raw_loc = self.cnn(x)
        loc_feats = [proj(f) for proj, f in zip(self.loc_projs, raw_loc)]

        if self.use_dinov2:
            sem_25 = self._extract_vit_pyramid(x)
            if self.cfg.fusion_mode == "safe_gd_ssa_v2":
                # Stage 0 (88x88): Pure high-resolution CNN features (protected from 25x25 patch blur)
                f0 = loc_feats[0]
                # Stage 1 (44x44): Edge-snapped semantic reassembly + reliability-gated residual
                s1 = self.edge_snap_s8(sem_25[1], loc_feats[1])
                f1 = self.fuser_s8(loc_feats[1], s1)
                # Stage 2 (22x22): Edge-snapped semantic reassembly + Safe GD-SSA v2
                s2 = self.edge_snap_s16(sem_25[2], loc_feats[2])
                f2 = self.fuser_s16(loc_feats[2], s2)
                # Stage 3 (11x11): Safe GD-SSA v2 at coarse semantic resolution
                s3 = F.interpolate(sem_25[3], size=loc_feats[3].shape[-2:], mode="bilinear", align_corners=False)
                f3 = self.fuser_s32(loc_feats[3], s3)
                fused_feats = [f0, f1, f2, f3]
            elif self.cfg.fusion_mode == "safe_concat":
                f0 = loc_feats[0]
                fused_feats = [f0]
                for i in range(1, 4):
                    si = F.interpolate(sem_25[i], size=loc_feats[i].shape[-2:], mode="bilinear", align_corners=False)
                    fused_feats.append(self.fusers[i](loc_feats[i], si))
            else:
                fused_feats = []
                for i in range(4):
                    si = F.interpolate(sem_25[i], size=loc_feats[i].shape[-2:], mode="bilinear", align_corners=False)
                    if i < 2:
                        g = self.fine_gates[i](torch.cat([loc_feats[i], si], dim=1))
                        fused_feats.append(loc_feats[i] + g * self.fusers[i](torch.cat([loc_feats[i], si], dim=1)))
                    else:
                        fused_feats.append(self.fusers[i](loc_feats[i], si))
        else:
            fused_feats = loc_feats

        if self.cfg.decoder_mode == "detail_a2bd_v2":
            p4 = self.dec_p4(fused_feats[3])
            p3 = self.dec_p3(fused_feats[2], p4)
            p2 = self.dec_p2(fused_feats[1], p3)
            p1 = self.dec_p1(fused_feats[0], p2)
        else:
            p4 = self.dec_blocks[3](fused_feats[3])
            p3 = self.dec_blocks[2](fused_feats[2] + F.interpolate(p4, size=fused_feats[2].shape[-2:], mode="bilinear", align_corners=False))
            p2 = self.dec_blocks[1](fused_feats[1] + F.interpolate(p3, size=fused_feats[1].shape[-2:], mode="bilinear", align_corners=False))
            p1 = self.dec_blocks[0](fused_feats[0] + F.interpolate(p2, size=fused_feats[0].shape[-2:], mode="bilinear", align_corners=False))

        b_logits_low = self.boundary_head(p1)
        if getattr(self.cfg, "use_boundary_feedback", self.cfg.use_boundary_loss):
            edge_prob = torch.sigmoid(b_logits_low).detach()
            p1 = p1 + self.boundary_alpha * edge_prob * self.boundary_refine(p1)

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
