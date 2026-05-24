# Mamba-V3 改进说明

本文档记录 `USEF-Mamba-V3` 的设计动机、与 V2 的差异、Mamba-2 的引入、归一化讨论与配置参数。

## 1. 设计背景

`USEF-Mamba-V2` 已经把 Mamba 后端做对（双向 + skip aggregation + 简化 mask head），承担"离线非因果 Mamba TSE 基线"的角色。

V3 的目标是回答两个独立问题：

```text
1. 把 Mamba-1 换成 Mamba-2，TSE 上是否有可观察的增益？
2. 把全局归一化 (gLN) 换成帧级归一化 (fLN)，能否让模型同时具备 streaming 部署能力？
```

V3 同时改这两个变量，不再保持单变量纯净对比——理由是：把 Mamba-2 + fLN 一起验证更接近"streaming Mamba TSE"的最终目标，否则要单独再开 V4-Stream 等更多 config 文件。V2 仍然是 ablation 的对照组。

## 2. V3 的核心改进

### 2.1 Mamba-2 替换 Mamba-1

来源：Dao & Gu, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*, ICML 2024。

Mamba-2 的几个对当前任务有意义的差异：

```text
1. 更大的 d_state 在更低算力下可行：
   Mamba-1 默认 d_state=16，常用 64；
   Mamba-2 默认 d_state=128，借助 chunked scan 算法仍然高效。
2. 引入多头机制：
   inner dim = d_model × expand 必须能被 headdim 整除；
   d_model=256, expand=2, headdim=64 -> (256×2)/64 = 8 heads
3. GPU 利用率更高：chunked scan 在现代 GPU 上比 Mamba-1 的 selective scan 更快。
```

V3 代码层面只改一处：

```python
from mamba_ssm import Mamba2   # was: from mamba_ssm import Mamba

self.fwd = Mamba2(d_model=..., d_state=128, headdim=64, ...)
self.bwd = Mamba2(d_model=..., d_state=128, headdim=64, ...)
```

双向 + skip 结构保持与 V2 完全一致，仍然是：

```text
y = LayerNorm(x)
fwd = fwd_mamba2(y)
bwd = flip(bwd_mamba2(flip(y, dim=time)), dim=time)
out = x + dropout((fwd + bwd) * 1/sqrt(2))     # bidir 尺度修正
```

并在 backend 末端对 `skip_sum` 做 `1/sqrt(num_layers)` 缩放（V2/V3 同步生效）。这两项是把 Mamba 加深到 12 层时避免 mask logits 饱和的关键 — 详见 [`Mamba_V2_design.md`](Mamba_V2_design.md) §2.1, §2.2。

### 2.2 真正的 Frame-wise LN

V2 的 `norm: gln` 在整段序列上做归一化，结构上不可 streaming。V3 改成 `norm: fln`：

```text
fLN: 在每一帧内部跨 channel 做 LayerNorm
     - 完全无状态、与序列长度无关
     - 训练/推理一致性最好
     - DPMamba/SepMamba/TFGridNet 等 streaming 模型的标准选择
```

### 2.3 关于 cLN 的讨论与决定

cLN（Cumulative Layer Norm）是 Conv-TasNet (Luo 2019) 提出的 causal 替代品：

```text
在时刻 t，对 [0, t] 所有帧、所有 channel 做累积归一化。
推理时维护 (cum_sum, cum_sumsq, count)，每帧 O(C) 增量更新。
```

V3 仓库内**保留了真正的 cLN 实现**（[`models/local/normalization.py`](../models/local/normalization.py) `CumulativeLayerNorm`），但默认 config 选 fLN，原因如下：

```text
1. cLN 统计量随 t 变化，训练用 4 s 而推理可能 30 s，会引入分布漂移
2. cLN 早期帧统计极不稳定，会污染 Mamba 的 SSM 状态
3. cLN 与 Mamba 的"长程依赖建模"职责重叠，让 norm 抢 backbone 的活
4. 现代 streaming TSE 模型 (DPMamba/SepMamba/TFGridNet/MossFormer) 没有任何一个用 cLN
```

cLN 实现保留作为消融行：未来可以单独跑一组 `norm: cln` 验证以上判断，然后写进论文表格。

### 2.4 修复 select_norm 的潜在 alias bug

旧版 `select_norm` 把 `'cln'` 和 `'fln'` 都映射到同一个 `FramewiseLayerNorm`，等于无声地把 cLN 偷换成 fLN。当前所有 yaml 都没有用过 `norm: cln`，因此这个 bug 还没有真的破坏过 checkpoint，但 V3 想用真 cLN 时会立刻踩到。

V3 把 dispatch 函数做了拆分：

```python
def select_norm(norm, channels, ...):
    if norm in {"fln", ...}:
        return FramewiseLayerNorm(...)
    if norm in {"cln", "cumulative_ln"}:
        return CumulativeLayerNorm(...)    # V3 新增分支，路由到真 cLN
    if norm in {"gln", ...}:
        return GlobalLayerNorm(...)
    ...
```

`CumulativeLayerNorm` 类放在 `models/local/normalization.py` 里以便其他模型按需引用，但 `select_norm` 函数本身**仍由每个模型文件各自维护一份**——保留 SepFormer/TFGridNet 项目的代码组织习惯，避免跨文件耦合。

## 3. 与 V2 的对比

| 维度 | V2 | V3 |
|---|---|---|
| SSM block | Mamba (Mamba-1) | Mamba2 (Mamba-2) |
| `d_state` | 64 | 128 |
| 多头 | 不存在 | `headdim=64` → 8 heads |
| 归一化 | `gln`（全局） | `fln`（帧级） |
| Streamable | 否（gLN 是非因果） | 是（fLN 与序列长度无关） |
| 双向 + skip 结构 | 一致 | 一致 |
| 前端结构 | 与 TCN-V2 同构 | 与 TCN-V2 同构 |
| 前端 init 文件 | `for_tcn_v2.pth.tar` | `for_tcn_v2.pth.tar`（共用） |

## 4. 代码修改

### 4.1 新增模型文件

```text
models/model_USEF_Mamba_V3.py
```

主要包含：

```text
BidirectionalMamba2Layer   单层双向 Mamba-2 + LayerNorm + 残差
Mamba2BackendV3            堆叠 N 层 + per-layer skip 投影
Tar_Model                  与 V2 同型的 USEF 主结构
```

`Encoder/Decoder/FiLM` 直接 import `models.model_USEF_TCN_V2` 的同名类，保持与 V2 / TCN-V2 参数兼容。`select_norm` 由 V3 文件自己维护一份。

### 4.2 新增配置文件

```text
config/config-USEF-Mamba-V3.yaml
```

默认配置：

```yaml
continue_from: 'chkpt/init_frontend_from_sepformer_wsj0-2mix_for_tcn_v2.pth.tar'
freeze_frontend_warmup: True
frontend_warmup_epochs: 5
frontend_unfreeze_lr_scale: 0.1
separate_frontend_backend_lr: True
batch_size: 8
mamba_num_layers: 12
mamba_d_state: 128
mamba_d_conv: 4
mamba_expand: 2
mamba_headdim: 64
mamba_dropout: 0.0
skip_channels: 256
mask_activation: sigmoid
norm: fln
```

前端保护策略与 V2 共用同一套理由（见 [`Mamba_V2_design.md`](Mamba_V2_design.md) §4.2）：Mamba-2 backend 同样是随机初始化，前端是 SepFormer init，需要 freeze 5 epoch + lr scale 0.1 来避免漂移。

约束：`(out_channels × mamba_expand) % mamba_headdim == 0`。当前 `(256 × 2) / 64 = 8`，OK。改 encoder 维度时务必同步检查这个约束。

### 4.3 增补归一化模块

```text
models/local/normalization.py
```

在已有 `FramewiseLayerNorm` / `GlobalLayerNorm` 的基础上新增 `CumulativeLayerNorm`，使用向量化 `cumsum` 实现（训练时 batch 并行，推理时可退化成增量更新）。

## 5. 配置参数说明

### 5.1 Encoder / 前端参数

与 V2 相同，与 TCN-V2 baseline 对齐。详见 [`Mamba_V2_design.md`](Mamba_V2_design.md) §5.1。

### 5.2 Mamba-2 Backend 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `mamba_num_layers` | `12` | 堆叠双向 Mamba-2 层数（与 V2 保持一致，加深以提升非线性能力） |
| `mamba_d_state` | `128` | Mamba-2 默认（Mamba-1 用 64） |
| `mamba_d_conv` | `4` | Mamba 内部局部 conv 核大小 |
| `mamba_expand` | `2` | inner dim = d_model × expand |
| `mamba_headdim` | `64` | 头维度，必须能整除 `d_model × expand` |
| `mamba_dropout` | `0.0` | 层内 dropout（baseline 暂不开） |
| `skip_channels` | `256` | skip 投影通道数 |

backend 内部固定开启两项尺度修正（不暴露为 yaml 配置）：

```text
- (fwd + bwd) * 1/sqrt(2)：双向输出合并尺度修正
- skip_sum * 1/sqrt(num_layers)：skip aggregation 深度尺度修正
```

### 5.3 归一化与 Mask 激活

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `norm` | `fln` | 帧级 LN，streamable 默认 |
| `mask_activation` | `sigmoid` | 与 V2 一致 |

可选值：`fln`（默认） / `cln`（消融实验用） / `gln`（与 V2 严格对比时切换） / `gn` / `bn`。

## 6. 训练命令

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py --config config/config-USEF-Mamba-V3.yaml --dataset wsj0-2mix --device 0 --memory 0
```

V2 与 V3 的代码完全独立，可以两台服务器并行训练，分别拿到：

```text
- V2: 离线 + Mamba-1 + gLN 基线
- V3: streamable + Mamba-2 + fLN
```

## 7. 训练时建议观察的指标

```text
1. Train/Valid Loss：与 V2 同 epoch 比对，差距能反映"Mamba-2 vs Mamba-1 + fLN vs gLN"组合效应
2. 显存：Mamba-2 d_state=128 比 Mamba-1 d_state=64 略大，但 chunked scan 分摊后增量不大
3. 数值稳定性：Mamba-2 在 bfloat16 下个别版本有 NaN，发现后请切回 fp16 或 disable AMP 试一次
4. 最终 SI-SDRi：fLN 通常比 gLN 掉 0.3-0.5 dB；如果 V3 仍能持平甚至超过 V2，说明 Mamba-2 的增益足够覆盖 fLN 的代价
```

## 8. 后续路线

如果 V3 验证 Mamba-2 + fLN 仍然有竞争力，下一步可以：

```text
- Chunked bidirectional inference：训练保持当前形式，推理时切 chunk，每个 chunk 内部双向，
  延迟 = chunk 长度。这是 V3 在 paper 里"streaming wake-word TSE"卖点的核心实现。
- d_state 消融：尝试 d_state ∈ {64, 128, 256}，画曲线
- cLN 消融：把 norm 切到 cln 单跑一组，验证文中的判断
```

## 9. 与 V2 / 旧版 Mamba 的关系

| 模型 | 方向性 | Skip | Block | norm | Streamable | 期望 |
|---|---|---|---|---|---|---|
| 旧版 USEF-Mamba | 单向 | 无 | Mamba-1 | gLN | 否 | ~10 dB（已确认） |
| USEF-Mamba-V2 | 双向 | 有 | Mamba-1 | gLN | 否 | 17 dB+（待跑） |
| USEF-Mamba-V3 | 双向 | 有 | Mamba-2 | fLN | 是 | 持平 V2 同时 streamable |

V2 和 V3 不是替代关系，是两个不同的 paper 角色：

```text
V2: 证明"把 Mamba 做对"能在 TSE 上达到与 TCN/Transformer 同档表现
V3: 证明"在不显著掉点的前提下"模型可以做成 streamable，呼应 wake-word 部署故事
```
