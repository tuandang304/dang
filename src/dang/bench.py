"""Same-host deployment benchmark: latency, throughput, peak memory and FLOPs of every candidate and baseline.

All models are timed in one process on one device so host/clock differences cancel (training-run FPS differed by up
to ~34% between T4 hosts for the identical model). Per model: batch-1 GPU latency under fp16 autocast (median / p90
of CUDA-event timings), batch-1 fp32 latency, batch-8 fp16 throughput, peak allocated memory at batch 1, forward
GMACs at batch 1, and batch-1 fp32 CPU latency with 4 threads. Two full passes are run to expose timing drift.
"""

import copy
import json
import statistics as st
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

V5 = {"arch": "dang_v5", "v5_lora_blocks": 12, "v5_lora_rank": 16, "v5_vit_size": 448, "v5_vit_layers": (2, 5, 8, 11),
      "v5_vit_fp32": False, "v5_vit_fp32_fallback": False, "v5_local_out_indices": ()}
VARIANTS = {
    "V7-b ConvNeXt-S + ViT-B (fp16)": {**V5, "v5_local": "convnext_small.dinov3_lvd1689m", "v5_vit": "vit_base_patch16_dinov3.lvd1689m"},
    "V8-a ConvNeXt-S + ViT-S (fp32 ViT)": {**V5, "v5_local": "convnext_small.dinov3_lvd1689m", "v5_vit": "vit_small_patch16_dinov3.lvd1689m", "v5_vit_fp32": True},
    "V8-b ConvNeXt-T + ViT-S (fp32 ViT)": {**V5, "v5_local": "convnext_tiny.dinov3_lvd1689m", "v5_vit": "vit_small_patch16_dinov3.lvd1689m", "v5_vit_fp32": True},
    "V8-b ConvNeXt-T + ViT-S (fp16 ViT)": {**V5, "v5_local": "convnext_tiny.dinov3_lvd1689m", "v5_vit": "vit_small_patch16_dinov3.lvd1689m"},
    "V9-d ConvNeXt-T + ViT-S (fp16 ViT + finite guard)": {**V5, "v5_local": "convnext_tiny.dinov3_lvd1689m", "v5_vit": "vit_small_patch16_dinov3.lvd1689m", "v5_vit_fp32_fallback": True},
    "V8-c ConvNeXt-S only": {**V5, "v5_local": "convnext_small.dinov3_lvd1689m", "v5_vit": ""},
    "V9-a ConvNeXt-T + EUPE ViT-T (fp32 ViT)": {**V5, "v5_local": "convnext_tiny.dinov3_lvd1689m", "v5_vit": "vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m", "v5_vit_fp32": True},
    "V9-b MobileNetV4-M + ViT-S (fp32 ViT)": {**V5, "v5_local": "mobilenetv4_conv_medium.e500_r256_in1k", "v5_vit": "vit_small_patch16_dinov3.lvd1689m", "v5_vit_fp32": True, "v5_local_out_indices": (1, 2, 3, 4)},
    "V9-c ConvNeXt-T + ViT-S 352px (fp32 ViT)": {**V5, "v5_local": "convnext_tiny.dinov3_lvd1689m", "v5_vit": "vit_small_patch16_dinov3.lvd1689m", "v5_vit_fp32": True, "v5_vit_size": 352},
    "Polyp-PVT (PVTv2-B2)": {"arch": "polyp_pvt"},
    "PraNet-V2 (PVTv2-B2)": {"arch": "pranet_v2"},
    "EMCAD (PVTv2-B2)": {"arch": "emcad"},
    "PraNet (Res2Net-50)": {"arch": "pranet"},
}


def _gpu_latency(model, x, dtype, iters):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        for _ in range(50):
            model(x)
        torch.cuda.synchronize()
        for i in range(iters):
            starts[i].record()
            model(x)
            ends[i].record()
        torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return st.median(t), t[int(0.9 * (len(t) - 1))]


@torch.no_grad()
def bench_one(name, overrides, base_cfg, build_model, device):
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    model = build_model(cfg).eval()
    deploy = getattr(model, "is_deploy_param", lambda n: True)
    params = sum(p.numel() for n, p in model.named_parameters() if deploy(n)) / 1e6
    s = cfg.img_size
    rec = {"model": name, "params_m": round(params, 2)}

    model = model.to(device)
    x1 = torch.randn(1, 3, s, s, device=device)
    with FlopCounterMode(display=False) as fc:
        model(x1)
    rec["gmacs_bs1"] = round(fc.get_total_flops() / 2e9, 2)  # FlopCounterMode counts 2 FLOPs per multiply-accumulate

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    med, p90 = _gpu_latency(model, x1, torch.float16, 300)
    rec.update(gpu_fp16_bs1_ms_median=round(med, 2), gpu_fp16_bs1_ms_p90=round(p90, 2),
               gpu_fp16_bs1_fps=round(1000.0 / med, 1),
               gpu_peak_mem_mb_bs1=round(torch.cuda.max_memory_allocated(device) / 2**20, 1))
    med32, _ = _gpu_latency(model, x1, None, 100)
    rec["gpu_fp32_bs1_ms_median"] = round(med32, 2)
    x8 = torch.randn(8, 3, s, s, device=device)
    med8, _ = _gpu_latency(model, x8, torch.float16, 50)
    rec["gpu_fp16_bs8_img_per_s"] = round(8000.0 / med8, 1)

    model = model.cpu().float()
    torch.set_num_threads(4)
    xc = torch.randn(1, 3, s, s)
    for _ in range(3):
        model(xc)
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        model(xc)
        ts.append((time.perf_counter() - t0) * 1000)
    rec["cpu4_fp32_bs1_ms_median"] = round(st.median(ts), 1)
    del model
    torch.cuda.empty_cache()
    return rec


def run_benchmark(cfg, build_model, device):
    print(f"DEPLOY BENCHMARK | device={torch.cuda.get_device_name(device)} | torch={torch.__version__} | "
          f"input={cfg.img_size}x{cfg.img_size} | bs1 fp16 autocast, CUDA events, 50 warmup + 300 timed", flush=True)
    for rep in (1, 2):
        for name, ov in VARIANTS.items():
            try:
                rec = bench_one(name, ov, cfg, build_model, device)
            except Exception as e:  # keep benchmarking the rest
                rec = {"model": name, "error": repr(e)[:300]}
            rec["pass"] = rep
            print("BENCH_JSON: " + json.dumps(rec), flush=True)
