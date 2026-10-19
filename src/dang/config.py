from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    # Experiment identification
    variant_name: str = "v5a_dinov3_convnext_s_vitb"
    seed: int = 42
    # External SOTA selector (dang/sota.py) and loss: "shared" = Eq. (9) of the paper, "official" = method's own loss
    arch: str = "dang_v5"
    loss_mode: str = "shared"
    # Write binary predicted masks for this seed to $ORX_OUTPUT_DIR/pred_masks_seed<N>.zip (None = off)
    dump_pred_seed: int | None = 42
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)

    # Stress-test S1 (5% Few-Shot Kvasir = 40 images) + Full 100% (800 images)
    run_s1_10pct: bool = True
    s1_epochs: int = 40
    s1_warmup_epochs: int = 3

    # DANG v5 (dang/dang_v5.py)
    v5_local: str = "convnext_small.dinov3_lvd1689m"
    v5_vit: str = "vit_base_patch16_dinov3.lvd1689m"   # "" = no semantic stream
    v5_vit_size: int = 448                              # 28x28 patch grid at patch 16
    v5_vit_layers: tuple[int, ...] = (2, 5, 8, 11)
    v5_lora_blocks: int = 4
    v5_lora_rank: int = 8
    bg_loss_weight: float = 0.4

    # Architecture knobs
    cnn_backbone: str = "convnext_tiny.fb_in22k_ft_in1k_384"
    use_dinov2: bool = False
    vit_backbone: str = "vit_base_patch14_dinov2.lvd142m"
    peft_mode: str = "none"
    lora_rank: int = 4
    adapter_start_block: int = 8

    # DANG-Med v4.1 Breakthrough Pillars
    use_proto_dpt: bool = False
    use_passe_envelope: bool = False
    use_style_calib: bool = False
    use_fast_strip_ssm: bool = False
    fusion_mode: str = "none"
    decoder_mode: str = "fpn"
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
    cnn_lr: float = 1.2e-4
    lr: float = 4.0e-4
    peft_lr: float = 1.0e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    num_workers: int = 2

    def to_dict(self):
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        d["v5_vit_layers"] = list(self.v5_vit_layers)
        return d


def get_config() -> ExperimentConfig:
    return ExperimentConfig()
