# Mamba-V2 改进说明

本文档记录 `USEF-Mamba-V2` 的设计动机、模型结构、和旧版 `USEF-Mamba` 的差异、训练逻辑、代码修改与配置参数。

## 1. 设计背景

旧版 `models/model_USEF_Mamba.py` 在 wsj0-2mix 上只能跑到 ~10 dB SI-SDRi，远低于 SepFormer/TFGridNet 同期 baseline，也低于公开的 Mamba TSE 论文（DPMamba/SepMamba 等达到 19+ dB）。

诊断后确认这不是 Mamba 本身能力不够，而是旧实现里同时叠加了三个结构性问题：

```text
1. 单向 Mamba：只 forward，等价于 causal RNN
2. 没有 skip aggregation：mask 只用最后一层输出，信息瓶颈极窄
3. mask head 顺序错乱：out_conv1d 扩展通道 -> tanh × sigmoid 门控 -> end_conv1x1
```

外加旧的 scp 文件存在样本对齐 bug，进一步压低性能。

`USEF-Mamba-V2` 的目标是**把 Mamba 后端做对**，建立一个干净的 Mamba 基线，再决定后续是否需要更复杂的 dual-path 结构。

## 2. V2 的核心改进

V2 的整体流水线：

```text
input/aux -> encoder
mix_w/aux_w -> norm_m + conv1d1
USEF cross-attention fusion -> target_seq
FiLM(x_seq, target_seq)
[B, T, C]  bidirectional Mamba stack with per-layer skip projection
skip_sum -> PReLU -> norm -> mask_conv1x1 -> sigmoid mask
mask * encoder feature -> decoder
```

### 2.1 双向 Mamba

每一层都同时跑 forward Mamba 和 reversed Mamba，输出相加后再做残差：

```text
y = LayerNorm(x)
fwd = fwd_mamba(y)
bwd = flip(bwd_mamba(flip(y, dim=time)), dim=time)
out = x + dropout(fwd + bwd)
```

forward 与 backward Mamba **不共享参数和 SSM 状态**，因此一层双向 Mamba 的参数量约等于两层单向 Mamba。这是 DPMamba/SepMamba 在分离任务上验证有效的标准模式。对离线 TSE，能看到未来帧是 1-2 dB 的关键收益来源。

`fwd + bwd` 的合并做 `1/sqrt(2)` 缩放。理由：fwd 与 bwd 输出近似独立，原始相加会让方差变成 `2σ²`，反传时每个 Mamba 的 dt 参数收到的梯度也被放大 ~2×，bf16 下偶发 NaN。`/sqrt(2)` 把 post-merge 方差还原为 `σ²`，与 DPMamba 用 `*0.5` 是同类做法。

### 2.2 Per-Layer Skip Aggregation

每层都把当前输出投影到 skip 空间并累加：

```text
for each Mamba layer i:
    x = layer_i(x)              # residual stream, [B, T, C]
    skip_i = skip_proj_i(x)     # [B, T, skip_channels]
    skip_sum += skip_i
```

`skip_proj_i` 是 `nn.Linear`（不是 `Conv1d`），因为后端工作在 `[B, T, C]` 布局下，Linear 等价于 1×1 conv 但避免每层多两次 permute。

`skip_sum` 在返回前做 `1/sqrt(N)` 缩放（N = num_layers）。理由：N 个近似独立的 skip 直接相加方差变成 `N·σ²`，N=12 时 mask logits 早期容易把 sigmoid 推到饱和、训练初期不稳。`1/sqrt(N)` 等价于把每个 `skip_proj` 的 init 缩小 `1/sqrt(N)`，是纯数学层面的尺度修正，不引入新参数。深度加深时该项尤为关键。

mask head 只用 `skip_sum`，不再用最后一层 residual。Conv-TasNet 的论文里就指出过：纯 residual stream 对 mask 信息太瘦，per-layer skip 聚合是其性能的一半。

### 2.3 简化的 Mask Head

旧版 head：

```text
skip -> out_conv1d(扩展通道) -> tanh × sigmoid 门控 -> end_conv1x1 -> mask
```

V2 的 head：

```text
skip_sum -> PReLU -> norm -> mask_conv1x1 -> sigmoid -> mask
```

去掉了「先扩展通道再用 tanh×sigmoid 当 GLU」这一段——它在 Conv-TasNet/SepFormer 体系里都不存在，是旧版的冗余设计。

### 2.4 与 TCN-V2 对齐的接口

V2 的 `Encoder/Decoder/FiLM` 实现与 TCN-V2 完全一致：

```text
Encoder:  Conv1d(in=1, out=N_encoder_out, kernel=16, stride=8) + ReLU
Decoder:  ConvTranspose1d(in=N_encoder_out, out=1, kernel=16, stride=8)
FiLM:     two Linear(d_model, d_model), x * f1(aux) + f2(aux)
Fusion:   models.local.TransformerEncoderCross
```

这样做有两个好处：

```text
1. 公平 ablation：V2 vs TCN-V2 的差距完全归因于 backend
2. 前端可复用：直接加载 chkpt/init_frontend_from_sepformer_wsj0-2mix_for_tcn_v2.pth.tar
```

## 3. 与旧版 Mamba 的对比

| 维度 | 旧版 USEF-Mamba | USEF-Mamba-V2 |
|---|---|---|
| 方向性 | 单向 forward Mamba | bidirectional (forward + reversed) |
| Skip 路径 | 无 | 每层 1×1 投影累加 |
| Mask head | out_conv1d → tanh×sigmoid → end_conv1x1 | skip_sum → PReLU → norm → 1×1 → sigmoid |
| 参数共享 | 单向 | forward 和 backward Mamba 不共享 |
| 训练数据 | 旧 scp（疑似有对齐 bug） | 当前正确 scp |
| 期望表现 | ~10 dB SI-SDRi | 待跑出，预期 ~17 dB+ |

## 4. 代码修改

### 4.1 新增模型文件

```text
models/model_USEF_Mamba_V2.py
```

主要包含：

```text
BidirectionalMambaLayer    单层双向 Mamba + LayerNorm + 残差
MambaBackendV2             堆叠 N 层 + per-layer skip 投影
Tar_Model                  USEF 目标说话人提取主结构（encoder/fusion/FiLM/backend/mask head/decoder）
```

`Tar_Model.forward` 的关键 layout 变化：

```text
encoder/conv1d1 阶段     [B, C, T]
fusion/FiLM/backend 阶段 [B, T, C]   (Mamba native layout)
mask head 阶段           [B, C, T]
```

`fusion_norm` 仍然在 `[B, C, T]` 布局下做，与 TCN-V2 行为一致。

### 4.2 新增配置文件

```text
config/config-USEF-Mamba-V2.yaml
```

默认从 SepFormer 抽取的稳定前端初始化：

```yaml
continue_from: 'chkpt/init_frontend_from_sepformer_wsj0-2mix_for_tcn_v2.pth.tar'
freeze_frontend_warmup: True
frontend_warmup_epochs: 5
frontend_unfreeze_lr_scale: 0.1
separate_frontend_backend_lr: True
```

为什么这样配：Mamba backend 是随机初始化的，前端是 SepFormer 抽出来的稳定 init。如果一开始就端到端联合训练，Mamba 早期巨大的随机梯度会沿着 skip→fusion→conv1d1→encoder 反传，把 SepFormer init 推走（这是过去 V4 lr=5e-5 出现 catastrophic drift 的同类问题）。

```text
- freeze 5 epoch：让 backend 在前端不变的条件下先适应，避开最危险窗口
- scale 0.1：解冻后前端 lr=1e-5 vs backend lr=1e-4，前端只做轻微微调
- separate_frontend_backend_lr=True：保证 backend 学习率不被一起拉低
```

这个配置是保守版。如果跑出来发现 Mamba V2 比 TCN-V2 提升幅度小于预期，下一步可以试 `frontend_unfreeze_lr_scale: 0.3 / 0.5`，让前端有更大空间适配 Mamba 的特征分布需求。

为什么直接复用 `for_tcn_v2`：V2 的 Encoder/Decoder/FiLM/fusion_mdl 与 TCN-V2 同构，前端 checkpoint 的 key 集合完全一致，没必要单独再抽一份 `for_mamba_v2`。

### 4.3 不修改 Trainer

V2 没有引入新的训练机制（adapter / freeze_backbone / 分组学习率），完全沿用 V4 之前 trainer 已有的功能。SI-SNR loss + 全模型协同训练。

## 5. 配置参数说明

### 5.1 Encoder / 前端参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `N_encoder_out` | `256` | encoder 输出通道，与 TCN-V2 baseline 对齐 |
| `out_channels` | `256` | backend 内部通道数 |
| `encoder_kernel_size` | `16` | encoder 1D 卷积核 (2 ms @ 8 kHz) |
| `encoder_kernel_stride` | `8` | encoder stride (1 ms hop) |
| `embd` | `256` | aux 路径 embedding 维度 |

为什么不放大到 512：第一版 V2 的目的是干净对比 backend 的影响，引入 encoder 维度变化会污染 ablation 表。等 V2 baseline 跑稳，未来再做 N=512 的消融。

### 5.2 Mamba Backend 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `mamba_num_layers` | `12` | 堆叠层数；12 层双向 ≈ 24 层单向，加深以提升非线性能力 |
| `mamba_d_state` | `64` | SSM 隐状态维度（Mamba-1 默认 16，TSE 文献常用 64） |
| `mamba_d_conv` | `4` | Mamba 内部局部 conv 核大小（保留默认） |
| `mamba_expand` | `2` | inner dim = d_model × expand（保留默认） |
| `mamba_dropout` | `0.0` | 层内 dropout（baseline 暂不开） |
| `skip_channels` | `256` | skip 投影通道数 |

backend 内部固定开启两项尺度修正（不暴露为 yaml 配置）：

```text
- (fwd + bwd) * 1/sqrt(2)：双向输出合并尺度修正
- skip_sum * 1/sqrt(num_layers)：skip aggregation 深度尺度修正
```

加深到 12 层时，第二项尤其重要 — 否则 skip_sum 方差 ~ 12σ²，sigmoid 容易饱和。

### 5.3 归一化与 Mask 激活

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `norm` | `gln` | 全局 LN，离线非因果场景的标准选择 |
| `mask_activation` | `sigmoid` | TCN-V2 同款 |

V2 用 `gln`（global LN，整段序列做归一化）与 TCN-V2 baseline 行为一致。**V2 不承担 streamable 故事**——streamable 留给 V3。

### 5.4 Loss 设置

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `lambda_anti` | `0.0` | 关闭 anti-interferer loss |
| `lambda_confusion` | `0.0` | 关闭 confusion margin loss |

V2 baseline 只用 SI-SNR，等基线跑稳再考虑加辅助 loss。

## 6. 训练命令

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py --config config/config-USEF-Mamba-V2.yaml --dataset wsj0-2mix --device 0 --memory 0
```

## 7. 训练时建议观察的指标

```text
1. Train/Valid Loss 趋势：早期下降速度应当不慢于 TCN-V2，否则说明 Mamba 没起作用
2. 显存：12 层双向 Mamba @ d_model=256, d_state=64, batch=8 大致占 5-7 GB
3. 收敛 epoch：Mamba 通常比 TCN 收敛慢 10-20%，留 200 epoch 上限
4. 解冻 epoch (5)：观察 frontend grad norm 是否平稳，若爆炸先降 max_norm
5. 最终 SI-SDRi：目标 17 dB+；如果连 TCN-V2 baseline (15.92 dB @ 4s aux) 都打不过，说明 backend 实现仍有 bug
```

## 8. 后续路线

如果 V2 跑出来 17+ dB，可继续：

```text
- V3：换成 Mamba-2 + 真 cLN/fLN，验证 Mamba-2 在 TSE 上的增益（已实现）
- Dual-path Mamba：intra-chunk 双向 + inter-chunk 单向，针对长序列
- Chunked streaming：把 V2/V3 改成 chunked 推理形式，承担 streamable 故事
```

如果 V2 仍 < 15 dB，需要回到诊断：

```text
- 检查 mamba_ssm 版本与 dtype 兼容（bfloat16 在某些版本下有 NaN）
- Disable AMP 重训一遍排除数值问题
- 单步过 checkpoint：encoder → fusion → backend 每段输出范围是否合理
```
