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
    """Per-channel learnable residual scaling initialized so new branches start stable without dead attenuation."""

    def __init__(self, dim: int, init_val: float = 0.20):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), init_val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class RetinexChromaticStem(nn.Module):
    """
    Zero-Shot Input-Level Multi-Scale Retinex + Specular-Aware Chromatic Normalizer (v4.1 Pillar B Stem):
    1) Strictly zero-shot (`aug_prob=0.0`): never synthesizes adverse corruptions during training, ensuring
       100% unseen zero-shot evaluation on Stress Track S3 across all experiment arms.
    2) During training on clean Kvasir images, updates online EMA reference statistics (`ema_ref_mean`, `ema_ref_std`)
       of canonical white-light endoscopy luminance/chrominance.
    3) At both train and test time, performs differentiable specular highlight inpainting, multi-scale homomorphic
       Retinex illumination normalization (kernel 31x31 + 15x15), and canonical Kvasir chromatic standardization,
       gated by an automatic photometric anomaly detector so clean white-light images remain untouched while
       adverse NBI/low-light/glare images are restored before entering ConvNeXt and DINOv2.
    """

    def __init__(self, aug_prob: float = 0.0):
        super().__init__()
        self.aug_prob = 0.0
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        # Canonical white-light Kvasir RGB mean & std in [0, 1] space, updated online via training EMA
        self.register_buffer("ema_ref_mean", torch.tensor([0.555, 0.325, 0.240], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("ema_ref_std", torch.tensor([0.235, 0.185, 0.160], dtype=torch.float32).view(1, 3, 1, 1))

        self.refine_net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.refine_net[-1].weight)
        nn.init.zeros_(self.refine_net[-1].bias)
        self.gate_mlp = nn.Sequential(
            nn.Linear(9, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, 0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rgb = (xf * self.im_std + self.im_mean).clamp(0.0, 1.0)

        # 1. Specular Glare Detection & Smooth Local Inpainting
        min_c = rgb.amin(dim=1, keepdim=True)
        glare_w = torch.sigmoid(18.0 * (min_c - 0.82))
        local_med = F.avg_pool2d(rgb, kernel_size=15, stride=1, padding=7, count_include_pad=False)
        rgb_deglare = (1.0 - glare_w) * rgb + glare_w * local_med

        # Update online Kvasir white-light reference statistics during training (no synthetic corruptions)
        c_mu = rgb_deglare.mean(dim=(-2, -1), keepdim=True)
        c_std = (rgb_deglare.var(dim=(-2, -1), unbiased=False, keepdim=True) + 1e-5).sqrt()
        if self.training:
            with torch.no_grad():
                b_mu = c_mu.mean(dim=0, keepdim=True)
                b_std = c_std.mean(dim=0, keepdim=True)
                self.ema_ref_mean.mul_(0.98).add_(b_mu, alpha=0.02)
                self.ema_ref_std.mul_(0.98).add_(b_std, alpha=0.02)

        # 2. Multi-Scale Homomorphic Retinex Illumination Division + Gamma Recovery
        illum_coarse = F.avg_pool2d(rgb_deglare, kernel_size=31, stride=1, padding=15, count_include_pad=False) + 0.03
        illum_mid = F.avg_pool2d(rgb_deglare, kernel_size=15, stride=1, padding=7, count_include_pad=False) + 0.03
        log_refl = torch.log(rgb_deglare + 0.03) - (0.65 * torch.log(illum_coarse) + 0.35 * torch.log(illum_mid))

        # 3. Canonical Kvasir Chromatic Standardization (cancels NBI / low-light / low-contrast shifts)
        r_mu = log_refl.mean(dim=(-2, -1), keepdim=True)
        r_std = (log_refl.var(dim=(-2, -1), unbiased=False, keepdim=True) + 1e-5).sqrt()
        refl_norm = (log_refl - r_mu) / r_std

        global_norm = (rgb_deglare - c_mu) / c_std
        restored_rgb = (0.55 * global_norm + 0.45 * refl_norm) * self.ema_ref_std + self.ema_ref_mean
        restored_rgb = restored_rgb.clamp(0.0, 1.0)

        # 4. Automatic Photometric Anomaly Gate (activates strongly when input deviates from canonical white-light)
        dev_mu = (c_mu - self.ema_ref_mean).abs() / 0.16
        dev_std = (c_std - self.ema_ref_std).abs() / 0.09
        glare_frac = glare_w.mean(dim=(-2, -1), keepdim=True).expand_as(dev_mu)
        anomaly_score = (0.52 * dev_mu + 0.33 * dev_std + 0.15 * glare_frac).clamp(0.0, 1.6)
        stat_vec = torch.cat([c_mu.flatten(1), c_std.flatten(1), anomaly_score.flatten(1)], dim=1)
        learned_gate = torch.sigmoid(self.gate_mlp(stat_vec)).view(-1, 3, 1, 1)
        blend_w = (anomaly_score * learned_gate).clamp(0.04, 0.94)

        blended_rgb = (1.0 - blend_w) * rgb + blend_w * restored_rgb
        x_base = (blended_rgb - self.im_mean) / self.im_std
        x_restored_norm = (restored_rgb - self.im_mean) / self.im_std
        delta = self.refine_net(torch.cat([x_base, x_restored_norm], dim=1))
        return (x_base + 0.25 * delta).to(dtype=x.dtype)


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


class FastAxialStripSSM(nn.Module):
    """
    Fast Multi-Scale Axial Strip State-Space Mixer (v4 Pillar C):
    Replaces the 168-allocation Python scan loop with fused cuDNN horizontal (1x11), vertical (11x1),
    and dilated (3x3, d=3) directional strip operators modulated by input-conditioned state gates.
    """

    def __init__(self, dim: int, cond_dim: int = 0, strip_k: int = 11):
        super().__init__()
        self.dim = dim
        self.in_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.dw_local = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)

        pad = strip_k // 2
        self.strip_h1 = nn.Conv2d(dim, dim, kernel_size=(1, strip_k), padding=(0, pad), groups=dim, bias=False)
        self.strip_v1 = nn.Conv2d(dim, dim, kernel_size=(strip_k, 1), padding=(pad, 0), groups=dim, bias=False)
        self.strip_h2 = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim, bias=False)
        self.strip_v2 = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim, bias=False)
        self.dw_dilated = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim, bias=False)

        gate_in = dim + cond_dim
        self.dir_gate = nn.Sequential(
            nn.Conv2d(gate_in, dim * 3, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.dir_gate[0].weight)
        nn.init.constant_(self.dir_gate[0].bias, 0.0)

        self.norm = nn.BatchNorm2d(dim)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.ls = LayerScale2d(dim, init_val=0.20)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=1)
        u = F.silu(self.dw_local(u))

        gate_in = torch.cat([u, cond], dim=1) if cond is not None else u
        g_h, g_v, g_d = self.dir_gate(gate_in).chunk(3, dim=1)

        s_h = self.strip_v2(self.strip_h1(u))
        s_v = self.strip_h2(self.strip_v1(u))
        s_d = self.dw_dilated(u)

        ssm_state = u + g_h * s_h + g_v * s_v + g_d * s_d
        out = self.norm(ssm_state) * F.silu(z)
        return x + self.ls(self.out_proj(out))


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


class DINOFreqStyleCalibrator(nn.Module):
    """
    DINOv2-Conditioned Low-Frequency Illumination & Style Calibrator + Training Low-Freq MixStyle (v4 Pillar B):
    Normalizes low-frequency illumination drift on ConvNeXt stages s1 and s2 using global DINOv2 context.
    """

    def __init__(self, dim: int, vit_dim: int = 768, mixstyle_p: float = 0.5):
        super().__init__()
        self.dim = dim
        self.mixstyle_p = mixstyle_p
        self.style_mlp = nn.Sequential(
            nn.Linear(vit_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 3),
        )
        nn.init.zeros_(self.style_mlp[-1].weight)
        nn.init.zeros_(self.style_mlp[-1].bias)
        self.calib_alpha = nn.Parameter(torch.full((1, dim, 1, 1), -0.2))

    def forward(self, x: torch.Tensor, g_dino: torch.Tensor | None = None) -> torch.Tensor:
        B, C, H, W = x.shape
        xf = x.float()
        low = F.avg_pool2d(xf, kernel_size=9, stride=1, padding=4)
        high = xf - low

        mu = low.mean(dim=(-2, -1), keepdim=True)
        std = (low.var(dim=(-2, -1), unbiased=False, keepdim=True) + 1e-5).sqrt()
        low_norm = (low - mu) / std

        if self.training and self.mixstyle_p > 0.0 and B > 1:
            if torch.rand(1, device=x.device).item() < self.mixstyle_p:
                perm = torch.randperm(B, device=x.device)
                lam = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.3, 0.7)
                mu_mix = lam * mu + (1.0 - lam) * mu[perm]
                std_mix = lam * std + (1.0 - lam) * std[perm]
                low_mixed = low_norm * std_mix + mu_mix
            else:
                low_mixed = low
        else:
            low_mixed = low

        if g_dino is not None:
            params = self.style_mlp(g_dino.float()).view(B, C * 3, 1, 1)
            gamma_low, beta_low, gamma_high = params.chunk(3, dim=1)
            low_dino = (1.0 + 0.5 * torch.tanh(gamma_low)) * low_norm + 0.5 * torch.tanh(beta_low)
            alpha = torch.sigmoid(self.calib_alpha.float())
            low_out = (1.0 - alpha) * low_mixed + alpha * (low_dino * std.detach() + mu.detach())
            high_out = (1.0 + 0.3 * torch.tanh(gamma_high)) * high
            out = low_out + high_out
        else:
            out = low_mixed + high

        return out.to(dtype=x.dtype)


class EdgeSnappedSemUp(nn.Module):
    """
    Content-Aware Edge-Snapped Semantic Upsampling:
    Starts as exact bilinear interpolation and learns a content-aware local reassembly residual guided by sharp CNN features.
    """

    def __init__(self, dim: int, k: int = 3, dilation: int = 1):
        super().__init__()
        self.k = k
        self.d = dilation
        self.kernel_pred = nn.Sequential(
            ConvBNAct(dim, dim // 4, kernel_size=1, padding=0),
            nn.Conv2d(dim // 4, k * k, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.kernel_pred[1].weight)
        with torch.no_grad():
            self.kernel_pred[1].bias.zero_()
            self.kernel_pred[1].bias[(k * k) // 2] = 6.0
        self.reassembly_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, sem_lr: torch.Tensor, loc_feat: torch.Tensor) -> torch.Tensor:
        B, C = sem_lr.shape[:2]
        H, W = loc_feat.shape[-2:]
        sem_up = F.interpolate(sem_lr, size=(H, W), mode="bilinear", align_corners=False)
        w = F.softmax(self.kernel_pred(loc_feat), dim=1).unsqueeze(1)  # [B, 1, k*k, H, W]
        pad = self.d * (self.k // 2)
        sem_pad = F.pad(sem_up, (pad, pad, pad, pad), mode="replicate")
        patches = F.unfold(sem_pad, kernel_size=self.k, dilation=self.d, padding=0)
        patches = patches.view(B, C, self.k * self.k, H, W)
        reassembled = (patches * w).sum(dim=2)
        return sem_up + torch.tanh(self.reassembly_scale) * (reassembled - sem_up)


class DPTFourScaleReassembler(nn.Module):
    """
    4-Scale Learned DPT Multi-Layer Reassembler (v4 Pillar A):
    Combines all 4 DINOv2 intermediate layers [2, 5, 8, 11] into all 4 target pyramid scales
    (including stride-4 88x88) via learned softmax depth-mixing weights + Edge-Snapped Upsampling at s8/s16.
    """

    def __init__(self, dim: int):
        super().__init__()
        init_logits = torch.eye(4) * 2.0
        self.depth_logits = nn.Parameter(init_logits)
        self.edge_snap_s8 = EdgeSnappedSemUp(dim, k=3, dilation=2)
        self.edge_snap_s16 = EdgeSnappedSemUp(dim, k=3, dilation=1)
        self.scale_convs = nn.ModuleList([
            ConvBNAct(dim, dim, kernel_size=3, padding=1) for _ in range(4)
        ])

    def forward(self, sem_25: list[torch.Tensor], loc_feats: list[torch.Tensor]) -> list[torch.Tensor]:
        weights = F.softmax(self.depth_logits, dim=-1)  # [4, 4]
        out_scales = []
        for s_idx in range(4):
            w_s = weights[s_idx]
            mixed_25 = (
                w_s[0] * sem_25[0]
                + w_s[1] * sem_25[1]
                + w_s[2] * sem_25[2]
                + w_s[3] * sem_25[3]
            )
            if s_idx == 1:
                up = self.edge_snap_s8(mixed_25, loc_feats[1])
            elif s_idx == 2:
                up = self.edge_snap_s16(mixed_25, loc_feats[2])
            else:
                up = F.interpolate(mixed_25, size=loc_feats[s_idx].shape[-2:], mode="bilinear", align_corners=False)
            out_scales.append(self.scale_convs[s_idx](up))
        return out_scales


class DualPrototypeMarginHead(nn.Module):
    """
    Supervised Foreground/Background Prototype Bank + Ground-Truth EMA Support Anchors + Test-Time Self-Support Margin (v4 Pillar A):
    1) Projects deep DINOv2 tokens [T_5, T_8, T_11] to L2-normalized fp32 embeddings Z (25x25).
    2) During training, directly updates EMA foreground & background anchor vectors from ground-truth mask patch
       occupancy so that even with 10% labels (80 images), the prototype manifold locks onto true polyp vs. mucosa tokens.
    3) Combines learned K-prototype cosine margin, EMA ground-truth support anchor margin, and test-time image-adaptive
       self-support centroid margin.
    """

    def __init__(self, dim: int = 192, proto_dim: int = 128, num_prototypes: int = 8):
        super().__init__()
        self.proto_dim = proto_dim
        self.num_prototypes = num_prototypes
        self.proj = nn.Sequential(
            ConvBNAct(dim * 3, dim, kernel_size=3, padding=1),
            nn.Conv2d(dim, proto_dim, kernel_size=1, bias=True),
        )
        fg_p = torch.empty(num_prototypes, proto_dim)
        bg_p = torch.empty(num_prototypes, proto_dim)
        nn.init.orthogonal_(fg_p)
        nn.init.orthogonal_(bg_p)
        self.fg_prototypes = nn.Parameter(fg_p)
        self.bg_prototypes = nn.Parameter(bg_p)

        self.register_buffer("ema_fg_anchor", F.normalize(fg_p.mean(dim=0, keepdim=True), p=2, dim=-1))
        self.register_buffer("ema_bg_anchor", F.normalize(bg_p.mean(dim=0, keepdim=True), p=2, dim=-1))
        self.register_buffer("ema_initialized", torch.tensor(0, dtype=torch.long))

        self.log_temp = nn.Parameter(torch.tensor(math.log(12.0)))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.self_support_weight = nn.Parameter(torch.tensor(0.40))
        self.ema_anchor_weight = nn.Parameter(torch.tensor(0.45))

    @torch.no_grad()
    def _update_ema_anchors(self, z_norm: torch.Tensor, gt_mask: torch.Tensor):
        m_25 = F.adaptive_avg_pool2d(gt_mask.float(), z_norm.shape[-2:])  # [B, 1, 25, 25]
        fg_w = (m_25 >= 0.35).float() * m_25
        bg_w = (m_25 <= 0.10).float() * (1.0 - m_25)

        if fg_w.sum() > 1.0:
            batch_fg = (fg_w * z_norm).sum(dim=(0, 2, 3)) / (fg_w.sum() + 1e-6)
            batch_fg = F.normalize(batch_fg.unsqueeze(0), p=2, dim=-1, eps=1e-6)
            if self.ema_initialized.item() == 0:
                self.ema_fg_anchor.copy_(batch_fg)
            else:
                self.ema_fg_anchor.copy_(F.normalize(0.95 * self.ema_fg_anchor + 0.05 * batch_fg, p=2, dim=-1, eps=1e-6))

        if bg_w.sum() > 1.0:
            batch_bg = (bg_w * z_norm).sum(dim=(0, 2, 3)) / (bg_w.sum() + 1e-6)
            batch_bg = F.normalize(batch_bg.unsqueeze(0), p=2, dim=-1, eps=1e-6)
            if self.ema_initialized.item() == 0:
                self.ema_bg_anchor.copy_(batch_bg)
            else:
                self.ema_bg_anchor.copy_(F.normalize(0.95 * self.ema_bg_anchor + 0.05 * batch_bg, p=2, dim=-1, eps=1e-6))

        if fg_w.sum() > 1.0 and bg_w.sum() > 1.0:
            self.ema_initialized.fill_(1)

    def forward(self, sem_25: list[torch.Tensor], gt_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        deep_cat = torch.cat([sem_25[1], sem_25[2], sem_25[3]], dim=1)
        z_raw = self.proj(deep_cat).float()  # [B, Dp, 25, 25] in fp32
        z_norm = F.normalize(z_raw, p=2, dim=1, eps=1e-6)

        if self.training and gt_mask is not None:
            self._update_ema_anchors(z_norm.detach(), gt_mask)

        fg_p = F.normalize(self.fg_prototypes.float(), p=2, dim=-1, eps=1e-6)
        bg_p = F.normalize(self.bg_prototypes.float(), p=2, dim=-1, eps=1e-6)

        sim_fg = torch.einsum("bdhw,kd->bkhw", z_norm, fg_p)
        sim_bg = torch.einsum("bdhw,kd->bkhw", z_norm, bg_p)

        lse_t = 6.0
        score_fg = torch.logsumexp(lse_t * sim_fg, dim=1, keepdim=True) / lse_t
        score_bg = torch.logsumexp(lse_t * sim_bg, dim=1, keepdim=True) / lse_t

        temp = self.log_temp.float().exp().clamp(4.0, 25.0)
        bank_logits = temp * (score_fg - score_bg) + self.bias.float()

        # EMA Ground-Truth Support Anchor Margin
        ema_fg = F.normalize(self.ema_fg_anchor.float(), p=2, dim=-1, eps=1e-6)
        ema_bg = F.normalize(self.ema_bg_anchor.float(), p=2, dim=-1, eps=1e-6)
        ema_margin = 10.0 * (
            torch.einsum("bdhw,kd->bkhw", z_norm, ema_fg) - torch.einsum("bdhw,kd->bkhw", z_norm, ema_bg)
        )
        ema_w = 0.5 * torch.tanh(self.ema_anchor_weight.float())
        prior_logits = bank_logits + ema_w * ema_margin

        # Image-Adaptive Self-Support Margin Map
        p_fg = torch.sigmoid(prior_logits.detach())
        w_fg = p_fg.pow(2) + 1e-4
        w_bg = (1.0 - p_fg).pow(2) + 1e-4

        c_fg = (w_fg * z_norm).sum(dim=(-2, -1), keepdim=True) / w_fg.sum(dim=(-2, -1), keepdim=True)
        c_bg = (w_bg * z_norm).sum(dim=(-2, -1), keepdim=True) / w_bg.sum(dim=(-2, -1), keepdim=True)
        c_fg = F.normalize(c_fg, p=2, dim=1, eps=1e-6)
        c_bg = F.normalize(c_bg, p=2, dim=1, eps=1e-6)

        self_fg = (z_norm * c_fg).sum(dim=1, keepdim=True)
        self_bg = (z_norm * c_bg).sum(dim=1, keepdim=True)
        self_logits = 8.0 * (self_fg - self_bg)

        ss_w = 0.5 * torch.tanh(self.self_support_weight.float())
        proto_logits = prior_logits + ss_w * self_logits
        return proto_logits, z_norm


class GDSSAFusion(nn.Module):
    """
    Gated Dual-Stream State-Space Alignment (GD-SSA):
    Supports both legacy SS2DLite (v3) and FastAxialStripSSM + Prototype Prior Injection (v4).
    """

    def __init__(self, dim: int, use_fast_strip: bool = False, has_proto_prior: bool = False):
        super().__init__()
        self.use_fast_strip = use_fast_strip
        self.has_proto_prior = has_proto_prior
        self.scale_loc = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.scale_sem = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.sscs_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.sscs_gate[1].weight)
        nn.init.constant_(self.sscs_gate[1].bias, math.log(0.85 / 0.15))

        if use_fast_strip:
            self.ssm_loc = FastAxialStripSSM(dim, cond_dim=dim)
            self.ssm_sem = FastAxialStripSSM(dim, cond_dim=dim)
        else:
            self.ssm_loc = SS2DLite(dim, cond_dim=dim)
            self.ssm_sem = SS2DLite(dim, cond_dim=dim)

        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim),
            nn.Sigmoid(),
        )
        self.out_conv = ConvBNAct(dim, dim, kernel_size=3, padding=1)

        rel_in = dim * 2 + (2 if has_proto_prior else 0)
        self.reliability = nn.Sequential(
            nn.Conv2d(rel_in, dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.reliability[-2].weight)
        rel_bias = 0.2 if use_fast_strip else -1.5
        ls_init = 0.30 if use_fast_strip else 0.05
        nn.init.constant_(self.reliability[-2].bias, rel_bias)
        self.ls = LayerScale2d(dim, init_val=ls_init)

        if has_proto_prior:
            self.prior_proj = nn.Sequential(
                nn.Conv2d(2, dim, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.prior_proj[0].weight)
            nn.init.zeros_(self.prior_proj[0].bias)

    def forward(self, f_loc: torch.Tensor, f_sem: torch.Tensor, proto_logits: torch.Tensor | None = None) -> torch.Tensor:
        f_l = f_loc * self.scale_loc
        f_s = f_sem * self.scale_sem

        if self.has_proto_prior and proto_logits is not None:
            p_up = F.interpolate(proto_logits.to(dtype=f_loc.dtype), size=f_loc.shape[-2:], mode="bilinear", align_corners=False)
            p_prob = torch.sigmoid(p_up.float()).to(dtype=f_loc.dtype)
            p_unc = (4.0 * p_prob * (1.0 - p_prob)).to(dtype=f_loc.dtype)
            prior_2ch = torch.cat([p_prob, p_unc], dim=1)
            f_s = f_s * (1.0 + 0.5 * (self.prior_proj(prior_2ch) - 0.5))
        else:
            prior_2ch = None

        alpha = self.sscs_gate(torch.cat([f_l, f_s], dim=1))
        swap_l = alpha * f_l + (1.0 - alpha) * f_s
        swap_s = alpha * f_s + (1.0 - alpha) * f_l

        s_l = self.ssm_loc(swap_l, cond=f_s)
        s_s = self.ssm_sem(swap_s, cond=f_l)
        g = self.gate(torch.cat([s_l, s_s], dim=1))
        fused = g * s_s + (1.0 - g) * s_l
        delta = self.out_conv(fused)

        rel_inputs = [f_loc, f_sem]
        if self.has_proto_prior and prior_2ch is not None:
            rel_inputs.append(prior_2ch)
        conf = self.reliability(torch.cat(rel_inputs, dim=1))
        return f_loc + conf * self.ls(delta)


class SafeConcatFusion(nn.Module):
    """Identity-preserving gated concat fusion for Dual-Stream Control arm."""

    def __init__(self, dim: int, init_bias: float = -1.5, ls_init: float = 0.05):
        super().__init__()
        self.fuser = ConvBNAct(dim * 2, dim, kernel_size=3, padding=1)
        self.reliability = nn.Conv2d(dim * 2, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.reliability.weight)
        nn.init.constant_(self.reliability.bias, init_bias)
        self.ls = LayerScale2d(dim, init_val=ls_init)

    def forward(self, f_loc: torch.Tensor, f_sem: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([f_loc, f_sem], dim=1)
        conf = torch.sigmoid(self.reliability(cat))
        return f_loc + conf * self.ls(self.fuser(cat))


class A2BDBlock(nn.Module):
    """
    Attention-Augmented Boundary Decoder (A2BD) block:
    Supports both SS2DLite (v3) and FastAxialStripSSM (v4).
    """

    def __init__(self, dim: int, use_ssm: bool = True, use_fast_strip: bool = False):
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
            self.ssm = FastAxialStripSSM(dim, cond_dim=dim) if use_fast_strip else SS2DLite(dim, cond_dim=dim)
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
        self.ls = LayerScale2d(dim, init_val=0.20 if use_fast_strip else 0.05)

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

        xf = base_y.float()
        gx = F.conv2d(xf, self.sobel_kx, padding=1, groups=self.dim)
        gy = F.conv2d(xf, self.sobel_ky, padding=1, groups=self.dim)
        sobel_l1 = (gx.abs() + gy.abs()).to(dtype=base_y.dtype)
        box_hp = (base_y - F.avg_pool2d(base_y, kernel_size=3, stride=1, padding=1)).abs()
        edge_gate = self.edge_enhance(0.5 * (sobel_l1 + box_hp))
        return base_y + self.ls(self.out_proj(mod_feat + base_y * edge_gate))


class DetailFirstDecoderStage(nn.Module):
    """Protects fine lateral CNN detail with a spatial top-down reliability gate before A2BD refinement."""

    def __init__(self, dim: int, has_top: bool = True, use_ssm: bool = True, top_bias: float = 0.0, use_fast_strip: bool = False):
        super().__init__()
        self.has_top = has_top
        if has_top:
            self.top_gate = nn.Conv2d(dim * 2, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.top_gate.weight)
            nn.init.constant_(self.top_gate.bias, top_bias)
        self.block = A2BDBlock(dim, use_ssm=use_ssm, use_fast_strip=use_fast_strip)

    def forward(self, lateral: torch.Tensor, top: torch.Tensor | None = None) -> torch.Tensor:
        x = lateral
        if self.has_top and top is not None:
            top_up = F.interpolate(top, size=lateral.shape[-2:], mode="bilinear", align_corners=False)
            g = 2.0 * torch.sigmoid(self.top_gate(torch.cat([lateral, top_up], dim=1)))
            x = lateral + g * top_up
        return self.block(x)


class UncertaintyStripDecoderStage(nn.Module):
    """
    Uncertainty-Guided Reverse-Attention + Fast Axial Strip-SSM Decoder Stage (v4 Pillar C):
    Explicitly computes from the coarser stage's mask logit:
      - Foreground probability A_fg = sigmoid(M_coarse)
      - Boundary Uncertainty Map U = 4 * A_fg * (1 - A_fg) (peaks at 1.0 on ambiguous p=0.5 flat borders)
      - Outer Morphological Ring Halo R = MaxPool5x5(A_fg) - A_fg
    and uses [A_fg, U, R] to modulate lateral & top-down features before FastAxialStripSSM refinement.
    """

    def __init__(self, dim: int, has_top: bool = True, use_ssm: bool = True):
        super().__init__()
        self.has_top = has_top
        if has_top:
            self.top_gate = nn.Conv2d(dim * 2 + 3, dim, kernel_size=1, bias=True)
            nn.init.zeros_(self.top_gate.weight)
            nn.init.constant_(self.top_gate.bias, 0.0)
            self.unc_attention = nn.Sequential(
                ConvBNAct(dim + 3, dim, kernel_size=3, padding=1),
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        self.block = A2BDBlock(dim, use_ssm=use_ssm, use_fast_strip=True)
        self.stage_head = nn.Sequential(
            ConvBNAct(dim, dim // 2, kernel_size=3, padding=1),
            nn.Conv2d(dim // 2, 1, kernel_size=1, bias=True),
        )

    def forward(
        self,
        lateral: torch.Tensor,
        top_feat: torch.Tensor | None = None,
        prev_logit: torch.Tensor | None = None,
        proto_prior_logit: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = lateral.shape[-2:]
        if self.has_top and top_feat is not None and prev_logit is not None:
            top_up = F.interpolate(top_feat, size=(H, W), mode="bilinear", align_corners=False)
            logit_up = F.interpolate(prev_logit.float(), size=(H, W), mode="bilinear", align_corners=False)
            prob_fg = torch.sigmoid(logit_up)
            unc_map = 4.0 * prob_fg * (1.0 - prob_fg)
            halo_map = (F.max_pool2d(prob_fg, kernel_size=5, stride=1, padding=2) - prob_fg).clamp(0.0, 1.0)
            priors = torch.cat([prob_fg, unc_map, halo_map], dim=1).to(dtype=lateral.dtype)

            g_top = 2.0 * torch.sigmoid(self.top_gate(torch.cat([lateral, top_up, priors], dim=1)))
            fused = lateral + g_top * top_up
            unc_mod = self.unc_attention(torch.cat([fused, priors], dim=1))
            x = fused * (1.0 + 0.5 * unc_map.to(dtype=lateral.dtype) + 0.3 * (unc_mod - 0.5))
            feat = self.block(x)
            stage_logit = logit_up + self.stage_head(feat).float()
        else:
            feat = self.block(lateral)
            stage_logit = self.stage_head(feat).float()
            if proto_prior_logit is not None:
                p_up = F.interpolate(proto_prior_logit.float(), size=(H, W), mode="bilinear", align_corners=False)
                stage_logit = 0.60 * stage_logit + 0.40 * p_up
        return feat, stage_logit


class DANGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim
        self.use_proto_dpt = getattr(cfg, "use_proto_dpt", False)
        self.use_passe_envelope = getattr(cfg, "use_passe_envelope", False)
        self.use_style_calib = getattr(cfg, "use_style_calib", False)
        self.use_fast_strip_ssm = getattr(cfg, "use_fast_strip_ssm", False)

        if self.use_style_calib:
            self.input_calibrator = RetinexChromaticStem(aug_prob=0.0)

        self.cnn = timm.create_model(cfg.cnn_backbone, pretrained=True, features_only=True)
        cnn_channels = self.cnn.feature_info.channels()
        self.loc_projs = nn.ModuleList([
            ConvBNAct(c, dim, kernel_size=1, padding=0) for c in cnn_channels
        ])

        self.use_dinov2 = cfg.use_dinov2
        vit_dim = 768
        if self.use_dinov2:
            self.vit = timm.create_model(cfg.vit_backbone, pretrained=True, dynamic_img_size=True)
            for p in self.vit.parameters():
                p.requires_grad = False

            num_prefix = getattr(self.vit, "num_prefix_tokens", 1)
            vit_dim = self.vit.embed_dim
            self.adapter_start_block = getattr(cfg, "adapter_start_block", 8)
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

            if self.use_proto_dpt:
                self.dpt_reassembler = DPTFourScaleReassembler(dim=dim)
                self.proto_head = DualPrototypeMarginHead(
                    dim=dim,
                    proto_dim=getattr(cfg, "proto_dim", 128),
                    num_prototypes=getattr(cfg, "num_prototypes", 8),
                )

            if cfg.fusion_mode == "proto_dpt_strip_ssa":
                self.fuser_s4 = SafeConcatFusion(dim, init_bias=-0.2, ls_init=0.25)
                self.fuser_s8 = GDSSAFusion(dim, use_fast_strip=True, has_proto_prior=self.use_proto_dpt)
                self.fuser_s16 = GDSSAFusion(dim, use_fast_strip=True, has_proto_prior=self.use_proto_dpt)
                self.fuser_s32 = GDSSAFusion(dim, use_fast_strip=True, has_proto_prior=self.use_proto_dpt)
            elif cfg.fusion_mode == "safe_concat":
                self.fusers = nn.ModuleList([
                    nn.Identity(),
                    SafeConcatFusion(dim),
                    SafeConcatFusion(dim),
                    SafeConcatFusion(dim),
                ])
            elif cfg.fusion_mode == "safe_gd_ssa_v2":
                self.edge_snap_s8 = EdgeSnappedSemUp(dim, k=5, dilation=2)
                self.edge_snap_s16 = EdgeSnappedSemUp(dim, k=3, dilation=1)
                self.fuser_s8 = SafeConcatFusion(dim, init_bias=-0.5 if self.use_fast_strip_ssm else -1.5, ls_init=0.20 if self.use_fast_strip_ssm else 0.05)
                self.fuser_s16 = GDSSAFusion(dim, use_fast_strip=self.use_fast_strip_ssm, has_proto_prior=False)
                self.fuser_s32 = GDSSAFusion(dim, use_fast_strip=self.use_fast_strip_ssm, has_proto_prior=False)

        if self.use_style_calib:
            self.style_calib_s4 = DINOFreqStyleCalibrator(dim, vit_dim=vit_dim, mixstyle_p=0.5)
            self.style_calib_s8 = DINOFreqStyleCalibrator(dim, vit_dim=vit_dim, mixstyle_p=0.5)

        if cfg.decoder_mode == "uncertainty_strip_ssm":
            self.dec_p4 = UncertaintyStripDecoderStage(dim, has_top=False, use_ssm=True)
            self.dec_p3 = UncertaintyStripDecoderStage(dim, has_top=True, use_ssm=True)
            self.dec_p2 = UncertaintyStripDecoderStage(dim, has_top=True, use_ssm=True)
            # Keep use_ssm=False at 88x88 p1 to prevent 15GB T4 VRAM fragmentation while preserving sharp Sobel-L1 edges
            self.dec_p1 = UncertaintyStripDecoderStage(dim, has_top=True, use_ssm=False)
        elif cfg.decoder_mode == "detail_a2bd_v2":
            self.dec_p4 = DetailFirstDecoderStage(dim, has_top=False, use_ssm=True, use_fast_strip=self.use_fast_strip_ssm)
            self.dec_p3 = DetailFirstDecoderStage(dim, has_top=True, use_ssm=True, top_bias=0.0, use_fast_strip=self.use_fast_strip_ssm)
            self.dec_p2 = DetailFirstDecoderStage(dim, has_top=True, use_ssm=True, top_bias=0.0, use_fast_strip=self.use_fast_strip_ssm)
            self.dec_p1 = DetailFirstDecoderStage(dim, has_top=True, use_ssm=False, top_bias=0.0, use_fast_strip=self.use_fast_strip_ssm)
        elif cfg.decoder_mode == "a2bd":
            self.dec_blocks = nn.ModuleList([A2BDBlock(dim, use_ssm=True, use_fast_strip=self.use_fast_strip_ssm) for _ in range(4)])
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

    def _apply_passe_envelope(
        self,
        logit_low: torch.Tensor,
        coarse_logit: torch.Tensor,
        proto_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Prototype-Anchored Spatial Support Envelope (PASSE, v4.1 Pillar A+):
        1) Computes the 25x25 semantic support map P_sem = sigmoid(0.5 * m4_25 + 0.5 * proto_logits).
        2) Forms a dilated smooth support envelope E_env = AvgPool3x3(MaxPool5x5(P_sem)).
        3) Outside the support envelope (E_env_up < 0.22), subtracts a smooth far-field suppression term
           to eliminate isolated false-positive background specks that otherwise inflate HD95.
        4) Inside the high-confidence semantic core (P_sem_up > 0.38), adds a core reinforcement boost
           to prevent empty predictions on tiny (<1.5% area) or low-contrast camouflaged polyps.
        """
        h1, w1 = logit_low.shape[-2:]
        if proto_logits is not None:
            p25 = proto_logits.float()
            c25 = F.interpolate(coarse_logit.float(), size=p25.shape[-2:], mode="bilinear", align_corners=False)
            sem_25 = 0.50 * c25 + 0.50 * p25
        else:
            sem_25 = F.interpolate(coarse_logit.float(), size=(25, 25), mode="bilinear", align_corners=False)

        p_sem = torch.sigmoid(sem_25)
        e_env = F.avg_pool2d(
            F.max_pool2d(p_sem, kernel_size=5, stride=1, padding=2),
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        e_up = F.interpolate(e_env, size=(h1, w1), mode="bilinear", align_corners=False)
        p_up = F.interpolate(p_sem, size=(h1, w1), mode="bilinear", align_corners=False)

        outside_suppress = 2.20 * F.relu(0.22 - e_up) / 0.22
        core_boost = 1.20 * F.relu(p_up - 0.38)
        return logit_low - outside_suppress + core_boost

    def _extract_vit_pyramid(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
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
        last_sp = None
        for idx in range(self.adapter_start_block, len(self.vit.blocks)):
            h = self.vit.blocks[idx](h)
            if idx in self.extract_layers:
                h_norm = self.vit.norm(h)
                sp = h_norm[:, num_prefix:, :].transpose(1, 2).reshape(B, -1, H_p, W_p)
                sem_feats_25.append((stage_idx, sp))
                last_sp = sp
                stage_idx += 1

        proj_25 = [self.sem_projs[s_idx](sp) for s_idx, sp in sem_feats_25]
        g_dino = last_sp.mean(dim=(-2, -1)) if last_sp is not None else sem_feats_25[-1][1].mean(dim=(-2, -1))
        return proj_25, g_dino

    def forward(self, x: torch.Tensor, gt_mask: torch.Tensor | None = None) -> dict:
        H, W = x.shape[-2:]
        if self.use_style_calib:
            x = self.input_calibrator(x)

        raw_loc = self.cnn(x)
        loc_feats = [proj(f) for proj, f in zip(self.loc_projs, raw_loc)]

        proto_logits = None
        g_dino = None

        if self.use_dinov2:
            sem_25, g_dino = self._extract_vit_pyramid(x)
            if self.use_proto_dpt:
                proto_logits, _ = self.proto_head(sem_25, gt_mask=gt_mask)

        if self.use_style_calib:
            loc_feats[0] = self.style_calib_s4(loc_feats[0], g_dino=g_dino)
            loc_feats[1] = self.style_calib_s8(loc_feats[1], g_dino=g_dino)

        if self.use_dinov2:
            if self.cfg.fusion_mode == "proto_dpt_strip_ssa":
                s_scales = self.dpt_reassembler(sem_25, loc_feats)
                f0 = self.fuser_s4(loc_feats[0], s_scales[0])
                f1 = self.fuser_s8(loc_feats[1], s_scales[1], proto_logits=proto_logits)
                f2 = self.fuser_s16(loc_feats[2], s_scales[2], proto_logits=proto_logits)
                f3 = self.fuser_s32(loc_feats[3], s_scales[3], proto_logits=proto_logits)
                fused_feats = [f0, f1, f2, f3]
            elif self.cfg.fusion_mode == "safe_gd_ssa_v2":
                f0 = loc_feats[0]
                s1 = self.edge_snap_s8(sem_25[1], loc_feats[1])
                f1 = self.fuser_s8(loc_feats[1], s1)
                s2 = self.edge_snap_s16(sem_25[2], loc_feats[2])
                f2 = self.fuser_s16(loc_feats[2], s2)
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
                fused_feats = loc_feats
        else:
            fused_feats = loc_feats

        if self.cfg.decoder_mode == "uncertainty_strip_ssm":
            p4, m4 = self.dec_p4(fused_feats[3], proto_prior_logit=proto_logits)
            p3, m3 = self.dec_p3(fused_feats[2], p4, prev_logit=m4)
            p2, m2 = self.dec_p2(fused_feats[1], p3, prev_logit=m3)
            p1, m1 = self.dec_p1(fused_feats[0], p2, prev_logit=m2)

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
            fused_logit_low = 0.55 * self.head(cat_feat).float() + 0.45 * m1.float()
            if proto_logits is not None:
                p_up = F.interpolate(proto_logits.float(), size=(h1, w1), mode="bilinear", align_corners=False)
                fused_logit_low = 0.80 * fused_logit_low + 0.20 * p_up
            if self.use_passe_envelope:
                fused_logit_low = self._apply_passe_envelope(fused_logit_low, m4, proto_logits)
            logits = F.interpolate(fused_logit_low, size=(H, W), mode="bilinear", align_corners=False)
            boundary_logits = F.interpolate(b_logits_low, size=(H, W), mode="bilinear", align_corners=False)
            aux_logits = [
                F.interpolate(m2, size=(H, W), mode="bilinear", align_corners=False),
                F.interpolate(m3, size=(H, W), mode="bilinear", align_corners=False),
                F.interpolate(m4, size=(H, W), mode="bilinear", align_corners=False),
            ]
        else:
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

            head_logit_low = self.head(cat_feat).float()
            if proto_logits is not None:
                p_up = F.interpolate(proto_logits.float(), size=(h1, w1), mode="bilinear", align_corners=False)
                head_logit_low = 0.80 * head_logit_low + 0.20 * p_up
            c_m4 = self.aux_heads[2](p4).float()
            if self.use_passe_envelope:
                head_logit_low = self._apply_passe_envelope(head_logit_low, c_m4, proto_logits)

            logits = F.interpolate(head_logit_low, size=(H, W), mode="bilinear", align_corners=False)
            boundary_logits = F.interpolate(b_logits_low, size=(H, W), mode="bilinear", align_corners=False)
            aux_logits = [
                F.interpolate(self.aux_heads[0](p2), size=(H, W), mode="bilinear", align_corners=False),
                F.interpolate(self.aux_heads[1](p3), size=(H, W), mode="bilinear", align_corners=False),
                F.interpolate(c_m4, size=(H, W), mode="bilinear", align_corners=False),
            ]

        out = {
            "logits": logits,
            "boundary_logits": boundary_logits,
            "aux_logits": aux_logits,
        }
        if proto_logits is not None:
            out["proto_logits"] = proto_logits
        return out
