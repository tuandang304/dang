from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    # Experiment identification
    variant_name: str = "r6d_clean_protocol_v3_full_dang_med_v3"
    seed: int = 42

    # Architecture knobs
    cnn_backbone: str = "convnext_tiny.fb_in22k_ft_in1k_384"
    use_dinov2: bool = True
    vit_backbone: str = "vit_base_patch14_dinov2.lvd142m"
    peft_mode: str = "freq_moe_conv_lora"
    lora_rank: int = 4
    adapter_start_block: int = 6
    fusion_mode: str = "safe_gd_ssa_v2"
    decoder_mode: str = "detail_a2bd_v2"
    embed_dim: int = 192

    # Loss & optimization knobs (Protocol v3)
    use_boundary_loss: bool = True
    use_boundary_feedback: bool = True
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
        return asdict(self)


def get_config() -> ExperimentConfig:
    return ExperimentConfig()
