# Vendored official model code

Copied verbatim from the authors' repositories; the architectures are not modified.

| Directory | Repository | Commit | Files |
|---|---|---|---|
| `pranet/` | https://github.com/DengPingFan/PraNet | 95f12bd4e32ec954b5069654abae7de9473fd1f5 | `lib/PraNet_Res2Net.py`, `lib/Res2Net_v1b.py` |
| `polyp_pvt/` | https://github.com/DengPingFan/Polyp-PVT | cd71fbf9fcd8c32cc0f33699f4e0186073c64dc4 | `lib/pvt.py`, `lib/pvtv2.py` |
| `pranet_v2/` | https://github.com/ai4colonoscopy/PraNet-V2 | ca08522ae77cbda6c5056609e7951b502dbb08f2 | `binary_seg/lib/pranet.py`, `pvtv2.py`, `Res2Net_v1b.py`, `nn/` (minus `nn/modules/tests/`), `LICENSE` |

Only these edits were made, all outside the network definitions:

1. Package-relative imports: `from lib.pvtv2 import` → `from .pvtv2 import` (Polyp-PVT `pvt.py`, PraNet-V2 `pranet.py`) and
   `from lib.nn import` → `from .nn import` (PraNet-V2 `pranet.py`).
2. PraNet `Res2Net_v1b.res2net50_v1b_26w_4s`: the authors' machine-local checkpoint path is replaced by the authors' own
   commented-out `model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'])` (the same `res2net50_v1b_26w_4s-3cf99910.pth`).

ImageNet weights are the files the official code expects: `res2net50_v1b_26w_4s-3cf99910.pth` (Res2Net authors' URL) and
`pvt_v2_b2.pth` (https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth), placed by `dang/bench/recipes.py`
at the relative paths the official constructors read (`./pretrained_pth/`, `./models/`).

The training loops, losses, optimisers, schedules and test-time output fusion are re-implemented line by line in
`dang/bench/recipes.py` from the official `MyTrain.py` / `Train.py` / `MyTrain_med.py` and test scripts at the same commits.
