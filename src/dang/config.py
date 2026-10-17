from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    # Experiment identification
    variant_name: str = "r2a_gd_ssa_fusion"
    seed: int = 42

    # Architecture knobs
    cnn_backbone: str = "convnext_tiny.fb_in22k_ft_in1k_384"
    use_dinov2: bool = True
    vit_backbone: str = "vit_base_patch14_dinov2.lvd142m"
    peft_mode: str = "freq_moe_conv_lora"          # "none", "linear_lora", "conv_lora", "freq_moe_conv_lora"
    lora_rank: int = 4
    fusion_mode: str = "gd_ssa"      # "none", "concat", "gd_ssa"
    decoder_mode: str = "fpn"        # "fpn", "a2bd"
    embed_dim: int = 192

    # Loss & optimization knobs
    use_boundary_loss: bool = False
    boundary_loss_weight: float = 0.4
    deep_supervision: bool = False
    img_size: int = 352
    vit_img_size: int = 350          # 25x25 patches for patch_size=14
    batch_size: int = 16
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    num_workers: int = 2

    def to_dict(self):
        return asdict(self)


def get_config() -> ExperimentConfig:
    return ExperimentConfig()
