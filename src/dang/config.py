from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    # Experiment identification
    variant_name: str = "r12a_dang_med_v4_1_pranet_standard"
    seed: int = 42
    # External SOTA selector (dang/sota.py) and loss: "shared" = Eq. (9) of the paper, "official" = method's own loss
    arch: str = "dang"
    loss_mode: str = "shared"
    # Write binary predicted masks for this seed to $ORX_OUTPUT_DIR/pred_masks_seed<N>.zip (None = off)
    dump_pred_seed: int | None = 42
    # "dang" = Kvasir-only stress protocol; "pranet_standard" = literature split (dang/standard_protocol.py)
    protocol: str = "pranet_standard"
    std_epochs: int = 50
    seeds: tuple[int, ...] = (42, 43, 44)

    # Stress-test S1 (5% Few-Shot Kvasir = 40 images) + Full 100% (800 images)
    run_s1_10pct: bool = True
    s1_epochs: int = 40
    s1_warmup_epochs: int = 3

    # Architecture knobs
    cnn_backbone: str = "convnext_tiny.fb_in22k_ft_in1k_384"
    use_dinov2: bool = True
    vit_backbone: str = "vit_base_patch14_dinov2.lvd142m"
    peft_mode: str = "freq_moe_conv_lora"
    lora_rank: int = 4
    adapter_start_block: int = 8

    # DANG-Med v4.1 Breakthrough Pillars
    use_proto_dpt: bool = True           # Pillar A: 4-Scale DPT Reassembly + Supervised Prototype & Self-Support Margin Prior
    use_passe_envelope: bool = True      # Pillar A+: Prototype-Anchored Spatial Support Envelope (suppresses far-field FP specks)
    use_style_calib: bool = True         # Pillar B: Zero-Shot Retinex Chromatic Stem + DINO-Conditioned Style Calibrator
    use_fast_strip_ssm: bool = True      # Fast Multi-Scale Axial Strip-SSM (replaces 168-cat Python scan loop)
    fusion_mode: str = "proto_dpt_strip_ssa"   # "none" | "safe_concat" | "safe_gd_ssa_v2" | "proto_dpt_strip_ssa"
    decoder_mode: str = "uncertainty_strip_ssm"  # "fpn" | "detail_a2bd_v2" | "uncertainty_strip_ssm"
    embed_dim: int = 192
    proto_dim: int = 128
    num_prototypes: int = 8
    proto_loss_weight: float = 0.35

    # Loss & optimization knobs (Clean Protocol v3/v4)
    use_boundary_loss: bool = False
    use_boundary_feedback: bool = False
    boundary_loss_weight: float = 0.2
    deep_supervision: bool = True
    img_size: int = 352
    vit_img_size: int = 350          # 25x25 patches for patch_size=14
    batch_size: int = 16
    epochs: int = 20
    cnn_lr: float = 1.6e-4
    lr: float = 4.0e-4
    peft_lr: float = 1.0e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    num_workers: int = 2

    def to_dict(self):
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        return d


def get_config() -> ExperimentConfig:
    return ExperimentConfig()
