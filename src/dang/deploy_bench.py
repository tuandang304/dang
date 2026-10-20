"""Same-host deployment benchmark (D1 harness, ported to the benchmark line and extended for the v12 round).

All models are timed in one process on one device so host and clock differences cancel. Per model and per mode:
  * eager   : fp16 autocast, fp32 weights (the D1 protocol; identical numbers to D1 for the same model)
  * half    : weights and input cast to fp16 once, no autocast (removes the per-call re-cast of frozen weights)
  * graph   : the half model captured once in a CUDA graph and replayed (removes per-op launch overhead)
Batch-1 latency is the median / p90 of CUDA-event timings (50 warm-up + 300 timed), plus bs8 throughput in eager mode,
peak allocated memory at bs1, GMACs at bs1 and total params (frozen weights included). Two passes expose drift.
"""

import copy
import json
import statistics as st

import torch
from torch.utils.flop_counter import FlopCounterMode

V8B = {"arch": "dang_v5", "v5_local": "convnext_tiny.dinov3_lvd1689m", "v5_vit": "vit_small_patch16_dinov3.lvd1689m",
       "v5_vit_size": 448, "v5_vit_layers": (2, 5, 8, 11), "v5_lora_blocks": 12, "v5_lora_rank": 16,
       "v5_vit_fp32": True, "v12_trunk": "dual", "v12_cache_rope": True}
DW6 = (1, 3, 5, 7, 9, 11)   # latency does not depend on which blocks; training runs select them from data
VARIANTS = {
    "V8-b (reference)": V8B,
    "V12-a ViT trunk + conv detail stem": {**V8B, "v12_trunk": "vit"},
    "V12-b ViT trunk + token pyramid": {**V8B, "v12_trunk": "tpa"},
    "info: V12-b uncached RoPE (= V12-L)": {**V8B, "v12_trunk": "tpa", "v12_cache_rope": False},
    "info: V12-b + slim head": {**V8B, "v12_trunk": "tpa", "v12_head": "slim"},
    "info: V12-b without Conv-LoRA": {**V8B, "v12_trunk": "tpa", "v5_lora_blocks": 0},
    "info: V12-b + DWConv 6/12 + slim head": {**V8B, "v12_trunk": "tpa", "v12_dw_fixed": DW6, "v12_head": "slim"},
    "info: bare ViT-S@448 readout (floor)": {"_vit_floor": True},
    "DANG (R2-b, selected): token pyramid + SAD": {**V8B, "v12_trunk": "tpa", "v12_decoder": "sad", "v12_passe": False},
    "R2-c: token pyramid + fg/bg query decoder": {**V8B, "v12_trunk": "tpa", "v12_decoder": "query", "v12_passe": False},
    "PraNet (Res2Net-50)": {"arch": "pranet"},
    "Polyp-PVT (PVTv2-B2)": {"arch": "polyp_pvt"},
    "PraNet-V2 (PVTv2-B2)": {"arch": "pranet_v2"},
}


class ViTFloor(torch.nn.Module):
    """Lower bound for any ViT-S@448 design: resize + frozen DINOv3 ViT-S readout at depths 2/5/8/11, nothing else."""

    def __init__(self, cfg):
        super().__init__()
        import timm
        from dang.dang_v5 import CachedRope
        self.size = cfg.v5_vit_size
        self.vit = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True, dynamic_img_size=True)
        self.vit.rope = CachedRope(self.vit.rope)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.vit.forward_intermediates(x, indices=[2, 5, 8, 11], norm=True, output_fmt="NCHW",
                                              intermediates_only=True)[-1]


def _time(fn, iters):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return round(st.median(t), 2), round(t[int(0.9 * (len(t) - 1))], 2)


def _half_model(model):
    m = copy.deepcopy(model).half()
    if hasattr(m, "cfg"):
        m.cfg = copy.deepcopy(m.cfg)
        m.cfg.v5_vit_fp32 = False          # the fp32 island would mismatch fp16 weights
    return m


@torch.no_grad()
def bench_one(name, overrides, base_cfg, build_model, device):
    cfg = copy.deepcopy(base_cfg)
    for k, v in overrides.items():
        if not k.startswith("_"):
            setattr(cfg, k, v)
    model = (ViTFloor(cfg) if overrides.get("_vit_floor") else build_model(cfg)).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    s = cfg.img_size
    rec = {"model": name, "params_m": round(params, 2)}
    model = model.to(device)
    x1 = torch.randn(1, 3, s, s, device=device)
    with FlopCounterMode(display=False) as fc:
        model(x1)
    rec["gmacs_bs1"] = round(fc.get_total_flops() / 2e9, 2)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.float16):
        rec["eager_fp16_bs1_ms"], rec["eager_fp16_bs1_p90"] = _time(lambda: model(x1), 300)
        rec["peak_mem_mb_bs1"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        x8 = torch.randn(8, 3, s, s, device=device)
        med8, _ = _time(lambda: model(x8), 50)
        rec["eager_fp16_bs8_img_per_s"] = round(8000.0 / med8, 1)
        del x8

    try:
        hm = _half_model(model)
        xh = x1.half()
        out = hm(xh)
        logit = out["logits"] if isinstance(out, dict) else (out[0] if isinstance(out, (tuple, list)) else out)
        rec["half_output_finite"] = bool(torch.isfinite(logit).all())
        rec["half_bs1_ms"], rec["half_bs1_p90"] = _time(lambda: hm(xh), 300)
    except Exception as e:
        rec["half_error"] = repr(e)[:200]
        return rec
    try:
        static_x = xh.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                hm(static_x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hm(static_x)
        rec["graph_half_bs1_ms"], rec["graph_half_bs1_p90"] = _time(graph.replay, 300)
        del graph
    except Exception as e:  # dynamic control flow or host syncs prevent capture
        rec["graph_error"] = repr(e)[:200]
    del model, hm
    torch.cuda.empty_cache()
    return rec


def run_benchmark(cfg, build_model, device, variants=None):
    variants = variants or VARIANTS
    print(f"DEPLOY BENCHMARK | device={torch.cuda.get_device_name(device)} | torch={torch.__version__} | "
          f"input={cfg.img_size}x{cfg.img_size} | modes: eager fp16 autocast, half weights, CUDA-graph half | "
          f"50 warm-up + 300 timed", flush=True)
    for rep in (1, 2):
        for name, ov in variants.items():
            try:
                rec = bench_one(name, ov, cfg, build_model, device)
            except Exception as e:  # keep benchmarking the rest
                rec = {"model": name, "error": repr(e)[:300]}
            rec["pass"] = rep
            print("BENCH_JSON: " + json.dumps(rec), flush=True)
