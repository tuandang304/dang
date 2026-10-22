# Vendored official model code

Copied verbatim from the authors' repositories; the architectures are not modified.

| Directory | Repository | Commit | Files |
|---|---|---|---|
| `pranet/` | https://github.com/DengPingFan/PraNet | 95f12bd4e32ec954b5069654abae7de9473fd1f5 | `lib/PraNet_Res2Net.py`, `lib/Res2Net_v1b.py` |
| `polyp_pvt/` | https://github.com/DengPingFan/Polyp-PVT | cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4 | `lib/pvt.py`, `lib/pvtv2.py` |
| `pranet_v2/` | https://github.com/ai4colonoscopy/PraNet-V2 | ca08522ae77cbda6c5056609e7951b502dbb08f2 | `binary_seg/lib/pranet.py`, `pvtv2.py`, `Res2Net_v1b.py`, `nn/` (minus `nn/modules/tests/`), `LICENSE` |
| `segdino/` | https://github.com/script-Yang/segdino | c078ca0e2c7578304e80eaa4e9d3f7c6dd0f3c41 | `dpt.py`, `blocks.py`, `LICENSE` |

Only these edits were made, all outside the network definitions:

1. Package-relative imports: `from lib.pvtv2 import` → `from .pvtv2 import` (Polyp-PVT `pvt.py`, PraNet-V2 `pranet.py`) and
   `from lib.nn import` → `from .nn import` (PraNet-V2 `pranet.py`).
2. PraNet `Res2Net_v1b.res2net50_v1b_26w_4s`: the authors' machine-local checkpoint path is replaced by the authors' own
   commented-out `model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'])` (the same `res2net50_v1b_26w_4s-3cf99910.pth`).
3. SegDINO `dpt.py`: `from blocks import` → `from .blocks import`. The backbone is DINOv3 ViT-S/16 LVD-1689M through timm's
   port (`vit_small_patch16_dinov3.lvd1689m`) behind DINOv3's `get_intermediate_layers(x, n)` contract (final-norm'd patch
   tokens; layer 12 equals `forward_features` exactly) instead of `torch.hub.load(<dinov3 repo>, 'dinov3_vits16')`.
   Frozen via the authors' `lock_backbone()` as the paper states. 24.06M parameters, 2.48M trainable (paper: 2.21M; the
   0.26M difference is the decoder's ConvTranspose `proj`, which the code uses).

ImageNet weights are the files the official code expects: `res2net50_v1b_26w_4s-3cf99910.pth` (Res2Net authors' URL) and
`pvt_v2_b2.pth` (https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth), placed by `dang/bench/recipes.py`
at the relative paths the official constructors read (`./pretrained_pth/`, `./models/`).

The training loops, losses, optimisers, schedules and test-time output fusion are re-implemented line by line in
`dang/bench/recipes.py` from the official `MyTrain.py` / `Train.py` / `MyTrain_med.py` / `train_segdino.py` and test scripts at the same commits.
