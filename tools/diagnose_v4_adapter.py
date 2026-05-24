#!/usr/bin/env python3
r"""Ablation diagnostics for the V4 RepeatSkipConditionAdapter.

Goal
----
回答一个核心问题：V4 的 adapter 真的在起作用吗？
通过运行时替换 RepeatSkipConditionAdapter.forward，在小测试子集上重新跑模型，
对比下列模式的 SI-SDR / SDR：

  - full        : 原始 forward（不动）
  - no_delta    : 强制 delta=0（adapter 完全失活，应该等价于 baseline TCN-V2）
  - no_need_gate: need_gate=1（去掉 per-frame 选择性，让 delta 总是注入）
  - no_reli_gate: reliability_gate=1（去掉 per-channel 可靠性筛选）
  - cond_zero   : cond 输入置零（模拟"听不到 aux"）

如果 full 和 no_delta 差距很小，说明 adapter 没学到东西；
如果 no_delta 比 full 还好，说明 adapter 在拖累主干。
如果 cond_zero 比 full 还好，说明模型在忽略/反向使用 aux 信息。

同时打印学到的 adapter_scale_logit 经过 sigmoid 后的标量值（应该不再是 0.05 附近）。

Usage
-----
    python tools/diagnose_v4_adapter.py \
      -c chkpt/USEF-TCCTCN-V4-From-WRCD/wsj0-2mix/config.yaml \
      -p chkpt/USEF-TCCTCN-V4-From-WRCD/wsj0-2mix/temp_best.pth.tar \
      --max-utts 200
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import (
    get_test_path,
    load_pretrained_modules,
    reserve_gpu_memory,
    si_snr,
)
from models.model_USEF_TCCTCN_V4 import RepeatSkipConditionAdapter


# ---------------------------------------------------------------------------
# 五种 forward 模式
# 直接 monkey-patch 类方法，避免改写 nn.Module 结构。
# 每个版本严格对齐原 forward 的张量形状和 dtype。
# ---------------------------------------------------------------------------
def _interp_cond(skip, cond):
    if cond.size(-1) != skip.size(-1):
        cond = F.interpolate(cond, size=skip.size(-1), mode="nearest")
    return cond


def _forward_full(self, skip, cond, scale):
    cond = _interp_cond(skip, cond)
    cond_skip = self.cond_proj(cond)

    delta = torch.cat([skip, cond_skip], dim=1)
    delta = self.delta_in(delta)
    delta = self.delta_prelu(delta)
    delta = self.delta_norm(delta)
    delta = self.delta_out(delta)

    gate_input = torch.cat(
        [skip, cond_skip, torch.abs(skip - cond_skip), skip * cond_skip], dim=1
    )
    need_gate = self.need_gate_in(gate_input)
    need_gate = self.need_gate_prelu(need_gate)
    need_gate = self.need_gate_norm(need_gate)
    need_gate = torch.sigmoid(self.need_gate_out(need_gate))

    if self.reliability_mlp is None:
        reli = 1.0
    else:
        cond_float = cond_skip.float()
        cond_stats = torch.cat([cond_float.mean(-1), cond_float.std(-1, unbiased=False)], dim=1)
        reli = torch.sigmoid(self.reliability_mlp(cond_stats)).to(skip.dtype).unsqueeze(-1)

    return skip + scale * need_gate * reli * delta


def _forward_no_delta(self, skip, cond, scale):
    # adapter 失活：返回原始 skip
    return skip


def _forward_no_need_gate(self, skip, cond, scale):
    cond = _interp_cond(skip, cond)
    cond_skip = self.cond_proj(cond)
    delta = torch.cat([skip, cond_skip], dim=1)
    delta = self.delta_in(delta)
    delta = self.delta_prelu(delta)
    delta = self.delta_norm(delta)
    delta = self.delta_out(delta)
    if self.reliability_mlp is None:
        reli = 1.0
    else:
        cond_float = cond_skip.float()
        cond_stats = torch.cat([cond_float.mean(-1), cond_float.std(-1, unbiased=False)], dim=1)
        reli = torch.sigmoid(self.reliability_mlp(cond_stats)).to(skip.dtype).unsqueeze(-1)
    return skip + scale * reli * delta  # need_gate=1


def _forward_no_reli_gate(self, skip, cond, scale):
    cond = _interp_cond(skip, cond)
    cond_skip = self.cond_proj(cond)
    delta = torch.cat([skip, cond_skip], dim=1)
    delta = self.delta_in(delta)
    delta = self.delta_prelu(delta)
    delta = self.delta_norm(delta)
    delta = self.delta_out(delta)
    gate_input = torch.cat(
        [skip, cond_skip, torch.abs(skip - cond_skip), skip * cond_skip], dim=1
    )
    need_gate = self.need_gate_in(gate_input)
    need_gate = self.need_gate_prelu(need_gate)
    need_gate = self.need_gate_norm(need_gate)
    need_gate = torch.sigmoid(self.need_gate_out(need_gate))
    return skip + scale * need_gate * delta  # reliability=1


def _forward_cond_zero(self, skip, cond, scale):
    # 把 cond 直接置零再走完整 forward
    return _forward_full(self, skip, torch.zeros_like(cond), scale)


MODES = OrderedDict([
    ("full",          _forward_full),
    ("no_delta",      _forward_no_delta),
    ("no_need_gate",  _forward_no_need_gate),
    ("no_reli_gate",  _forward_no_reli_gate),
    ("cond_zero",     _forward_cond_zero),
])


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def local_path(path):
    if os.name == "nt":
        normalized = path.replace("\\", "/")
        if normalized.startswith("/mnt/") and len(normalized) > 6:
            drive = normalized[5].upper()
            rest = normalized[7:].replace("/", "\\")
            return "{}:\\{}".format(drive, rest)
    return path


def load_scp(path):
    with open(path, "r", encoding="utf-8") as f:
        return OrderedDict(line.strip().split(maxsplit=1) for line in f if line.strip())


def load_wav(path, fs, length=None):
    wav, _ = librosa.load(local_path(path), sr=fs)
    wav = wav.astype(np.float32, copy=False)
    if length is None:
        return wav
    if wav.shape[0] >= length:
        return wav[:length]
    return np.pad(wav, (0, length - wav.shape[0]), mode="constant")


def collect_utts(test_path, mix_scp_name, ref_scp_name, aux_scp_name, max_utts):
    mix = load_scp(os.path.join(test_path, mix_scp_name))
    ref = load_scp(os.path.join(test_path, ref_scp_name))
    aux = load_scp(os.path.join(test_path, aux_scp_name))
    keys = list(mix.keys())
    if max_utts and max_utts > 0:
        keys = keys[:max_utts]
    return [(k, mix[k], ref[k], aux[k]) for k in keys]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def patch_mode(mode_name):
    """把所有 RepeatSkipConditionAdapter 实例的 forward 替换为指定模式。"""
    RepeatSkipConditionAdapter.forward = MODES[mode_name]


def evaluate_mode(model, utts, device, fs, aux_duration):
    sisdris = []
    aux_len = int(aux_duration * fs) if aux_duration is not None else None

    with torch.no_grad():
        for _utt_id, mix_path, ref_path, aux_path in tqdm(utts, leave=False):
            mix_wav = load_wav(mix_path, fs)
            ref_wav = load_wav(ref_path, fs)
            aux_wav = load_wav(aux_path, fs, length=aux_len)

            mix_t = torch.from_numpy(mix_wav).unsqueeze(0).to(device)
            aux_t = torch.from_numpy(aux_wav).unsqueeze(0).to(device)

            est = model(mix_t, aux_t).squeeze().detach().cpu().numpy()

            end = min(est.size, mix_wav.size, ref_wav.size)
            sisdr_mix = si_snr(mix_wav[:end], ref_wav[:end])
            sisdr_est = si_snr(est[:end], ref_wav[:end])
            sisdris.append(sisdr_est - sisdr_mix)

    sisdris = np.asarray(sisdris, dtype=np.float64)
    sisdris = sisdris[np.isfinite(sisdris)]
    return {
        "n": int(sisdris.size),
        "si_sdri_mean": float(sisdris.mean()) if sisdris.size else float("nan"),
        "si_sdri_median": float(np.median(sisdris)) if sisdris.size else float("nan"),
        "si_sdri_std": float(sisdris.std()) if sisdris.size else float("nan"),
    }


def report_scalar_values(model):
    """打印模型里和 adapter 相关的标量参数，帮助判断学习状态。"""
    lines = []
    backend = model.tcn_backend
    scale = torch.sigmoid(backend.adapter_scale_logit).item()
    lines.append("Global adapter_scale (sigmoid of logit): {:.6f}".format(scale))
    lines.append("  (init was sigmoid(-3.0)={:.6f})".format(torch.sigmoid(torch.tensor(-3.0)).item()))

    for i, adapter in enumerate(backend.repeat_adapters):
        delta_w_norm = adapter.delta_out.weight.detach().float().norm().item()
        delta_b_norm = adapter.delta_out.bias.detach().float().norm().item()
        nb = adapter.need_gate_out.bias.detach().float().mean().item()
        lines.append(
            "Repeat {} | delta_out ||W||={:.4f} ||b||={:.4f} | need_gate bias mean={:+.4f}".format(
                i, delta_w_norm, delta_b_norm, nb,
            )
        )
        if adapter.reliability_mlp is not None:
            rb = adapter.reliability_mlp[-1].bias.detach().float().mean().item()
            lines.append("  reliability_mlp out bias mean = {:+.4f}".format(rb))
    return "\n".join(lines)


def main(args):
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:{}".format(args.device))
        torch.cuda.set_device(args.device)
    else:
        device = torch.device("cpu")
    print("INFO: device = {}".format(device))

    if device.type == "cuda":
        reserve_gpu_memory(device_id=args.device, size_in_gb=args.memory)

    with open(args.config, "r", encoding="utf-8") as f:
        config = load_hyperpyyaml(f.read())
    fs = int(config["sample_rate"])

    model = config["modules"]["masknet"]
    model = load_pretrained_modules(model, args.chkpt_path)
    model.to(device).eval()

    # 先打印学到的标量
    print("=" * 70)
    print("Learned adapter parameters")
    print("=" * 70)
    print(report_scalar_values(model))

    test_path = args.test_set or get_test_path(config, args)
    aux_duration = args.aux_duration if args.aux_duration is not None else config.get("valid_wake_aux_duration")
    utts = collect_utts(
        test_path=test_path,
        mix_scp_name=config["mix_scp"],
        ref_scp_name=config["ref_scp"],
        aux_scp_name=config["aux_scp"],
        max_utts=args.max_utts,
    )
    print("INFO: evaluating on {} utts from {}".format(len(utts), test_path))

    # 跑五种模式
    results = OrderedDict()
    for mode in MODES.keys():
        print("=" * 70)
        print("Mode: {}".format(mode))
        print("=" * 70)
        patch_mode(mode)
        res = evaluate_mode(model, utts, device, fs, aux_duration)
        results[mode] = res
        print(
            "  N={:<5d}  SI-SDRi mean={:+.4f}  median={:+.4f}  std={:.4f}".format(
                res["n"], res["si_sdri_mean"], res["si_sdri_median"], res["si_sdri_std"]
            )
        )

    # 汇总并写 csv
    full_mean = results["full"]["si_sdri_mean"]
    print("=" * 70)
    print("Delta vs full (mode - full)")
    print("=" * 70)
    print("{:<14s} {:>12s} {:>12s} {:>14s}".format("mode", "SI-SDRi", "vs full", "interpretation"))

    interp = {
        "full":         "baseline",
        "no_delta":     "adapter 失活；若≥full则 adapter 没贡献",
        "no_need_gate": "need_gate 强制开；若涨说明 gate 太保守",
        "no_reli_gate": "reliability 强制开；若涨说明 reli 太保守",
        "cond_zero":    "听不到 aux；若接近 full 说明模型没用上 aux",
    }
    rows = []
    for mode, res in results.items():
        delta = res["si_sdri_mean"] - full_mean
        print("{:<14s} {:>12.4f} {:>+12.4f}  {}".format(mode, res["si_sdri_mean"], delta, interp[mode]))
        rows.append({
            "mode": mode,
            "n": res["n"],
            "si_sdri_mean": res["si_sdri_mean"],
            "si_sdri_median": res["si_sdri_median"],
            "si_sdri_std": res["si_sdri_std"],
            "delta_vs_full": delta,
        })

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_csv = os.path.join(args.output_dir, "v4_adapter_diagnosis.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print("INFO: saved diagnostic table to {}".format(out_csv))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("V4 RepeatSkipConditionAdapter ablation diagnostics")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-p", "--chkpt-path", required=True)
    parser.add_argument("-t", "--test-set", default="")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--memory", type=float, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--aux-duration", type=float, default=None)
    parser.add_argument("--max-utts", type=int, default=200,
                        help="子集规模；200 条 ~ 几分钟就能跑完，足够看出趋势")
    parser.add_argument("--output-dir", type=str, default="eval_logs/USEF-TCCTCN-V4/wsj0-2mix/diagnosis_adapter")
    args = parser.parse_args()
    main(args)
