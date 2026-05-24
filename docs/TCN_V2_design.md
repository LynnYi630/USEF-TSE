# TCN-V2 改进说明

本文档记录 `USEF-TCN-V2` 的设计动机、模型结构、代码实现、配置参数和实验结果。它是后续 `WRCD-V2`、`TCCTCN-V4` 的基础主干。

## 1. 设计背景

原始 USEF-TSE 框架使用 SepFormer / TFGridNet 作为主要时序建模后端，优点是分离质量强，缺点是模型较重、推理耗时较高，不利于人形机器人端侧部署。

当前轻量化目标可以概括为：

```text
保留 USEF 的目标说话人条件建模能力，
把重型后端替换为更轻、更快、更容易部署的 TCN 后端。
```

因此 `USEF-TCN-V2` 做的不是完全推翻 USEF，而是保留前端结构：

```text
encoder feature
    -> target enrollment / aux fusion
    -> FiLM target conditioning
    -> lightweight TCN backend
    -> mask estimation
    -> decoder waveform reconstruction
```

其中最核心的替换是：

```text
SepFormer / TFGridNet backend
    -> Conv-TasNet style TCN backend
```

直观理解：SepFormer / TFGridNet 更像“强表达能力的大后端”，TCN 更像“局部卷积 + 空洞卷积堆叠”的轻量后端。TCN 的计算路径更规则，主要由 `Conv1d`、depthwise convolution、PReLU 和 norm 组成，对端侧部署更友好。

## 2. 为什么选择 TCN

### 2.1 相比 SepFormer / TFGridNet

SepFormer / TFGridNet 的优势是长程建模能力强，但计算图更复杂，参数量和延迟都更高。TCN 的优势是：

```text
1. 结构简单，主要是 1D convolution
2. 支持 dilation 扩大感受野
3. 推理延迟低，RTF 更小
4. 参数量明显低于 SepFormer
5. 更容易做 causal / streaming 扩展
```

在当前 wsj0-2mix 评估中：

| 模型 | SDRi | SI-SDRi | PESQi | Latency | RTF | Param |
|---|---:|---:|---:|---:|---:|---:|
| USEF-SepFormer | 20.2243 | 19.9581 | 2.0838 | 0.1079 | 0.0187 | 29.1684M |
| USEF-TCN-V2 | 16.3869 | 15.9210 | 1.6758 | 0.0489 | 0.0086 | 12.9554M |

也就是说，TCN-V2 用大约 `44.4%` 的参数量，把平均推理延迟从 `0.1079s/utt` 降到 `0.0489s/utt`，RTF 从 `0.0187` 降到 `0.0086`。代价是 SI-SDRi 下降约 `4.04 dB`。

这符合 TCN-V2 的定位：

```text
优先获得轻量、低延迟、端侧友好的 USEF-TSE baseline，
再在此基础上继续做 WRCD / TCCTCN 等鲁棒性增强。
```

### 2.2 为什么第一版没有选择 Mamba

Mamba 有更强的序列建模潜力，但对当前目标来说有几个现实风险：

```text
1. 模型结构和训练稳定性更依赖实现细节
2. 与已有 USEF 前端和 mask 估计路径的耦合方式不如 TCN 直接
3. 端侧部署时对算子支持、kernel 优化和设备适配要求更高
4. 当前阶段更需要一个稳定、可解释、可诊断的轻量 baseline
```

TCN 的优点是工程风险低。它可以直接替换原始时序后端，并且每个模块都容易解释：

```text
1x1 conv -> PReLU -> norm -> depthwise temporal conv -> PReLU -> norm
```

因此 TCN-V2 更适合作为轻量化 USEF-TSE 的第一条主线。

## 3. V2 相比 V1 的核心改进

仓库中的旧版 TCN 对应 `models/model_USEF_TCN.py`，V2 对应：

```text
models/model_USEF_TCN_V2.py
config/config-USEF-TCN-V2.yaml
```

旧版 TCN 的后端基本是：

```text
TCNBlock:
    1x1 conv -> PReLU -> GLN -> depthwise conv -> PReLU -> GLN -> 1x1 conv
    -> residual add

TCNBackend:
    sequential(TCNBlock x 24)
```

旧版输出头还使用了类似 gated output 的路径：

```text
x -> output(tanh) * output_gate(sigmoid) -> end_conv1x1 -> ReLU mask
```

V2 的主要改动是改成更标准的 Conv-TasNet 风格 residual + skip aggregation：

```text
TCNBlockV2:
    1x1 in_proj
    -> PReLU
    -> norm
    -> depthwise separable temporal conv
    -> PReLU
    -> norm
    -> dropout
    -> residual branch + skip branch
```

整体输出路径变成：

```text
all block skips -> skip_sum -> PReLU -> norm -> mask_conv1x1 -> mask_activation
```

这一步很关键。V1 主要依赖最终 residual feature 估计 mask，而 V2 显式聚合所有 TCN block 的 skip 输出，让不同 dilation 层的时序信息直接参与 mask 预测。

可以把 V2 理解为：

```text
USEF target cue 负责告诉模型“要找谁”，
TCN residual path 负责逐层更新时序表征，
TCN skip path 负责把多尺度时序证据汇总到 mask head。
```

## 4. 模型结构

### 4.1 总体前向路径

`USEF-TCN-V2` 的 `Tar_Model.forward(input, aux)` 可以写成：

```text
input -> encoder -> mix_w
aux   -> encoder -> aux_w

mix_w / aux_w
    -> norm_m
    -> conv1d1
    -> USEF cross-attention fusion
    -> FiLM(x, target_seq)
    -> fusion_norm
    -> TCNBackendV2
    -> skip_sum
    -> skip_prelu
    -> skip_norm
    -> mask_conv1x1
    -> mask_activation
    -> mask * mix_w
    -> decoder
```

用更紧凑的学术表述可以写成：

```text
Encoder -> USEF Fusion -> FiLM -> TCN residual/skip stack
        -> skip_sum -> PReLU -> norm -> mask_conv1x1 -> ReLU mask -> Decoder
```

中文解释：

```text
Encoder 把 waveform 映射到可分离的时域特征；
USEF Fusion 和 FiLM 把目标说话人 aux 信息注入 mixture feature；
TCN stack 做轻量时序建模；
skip_sum 汇总多层时序信息；
mask head 估计目标说话人的时域 mask；
Decoder 把 mask 后的 encoder feature 还原成 waveform。
```

### 4.2 Encoder / Decoder

Encoder 是一个 1D convolution：

```text
waveform -> Conv1d(kernel_size=16, stride=8, out_channels=256) -> ReLU
```

Decoder 是对应的 transposed convolution：

```text
masked encoder feature -> ConvTranspose1d -> estimated waveform
```

当前配置：

```yaml
N_encoder_out: 256
kernel_size: 16
kernel_stride: 8
sample_rate: 8000
```

这是一种 Conv-TasNet 常见的时域 encoder/decoder 设计，不需要显式 STFT，适合端到端训练。

### 4.3 USEF Fusion + FiLM

TCN-V2 没有删除 USEF 的目标说话人条件建模，而是保留了：

```text
Fusion_mdl: TransformerEncoderCross
FiLM: feature-wise linear modulation
```

对应路径：

```text
x_seq, aux_seq -> fusion_mdl(x_seq, aux_seq) -> target_seq
x_seq -> FiLM(x_seq, target_seq)
```

FiLM 形式是：

```text
x = x * linear1(target_seq) + linear2(target_seq)
```

直观理解：`fusion_mdl` 从 aux 中提取目标说话人线索，`FiLM` 用这个线索对 mixture feature 做逐帧条件调制。TCN 后端接收到的不是原始 mixture feature，而是已经带有目标说话人条件的 feature。

### 4.4 TCNBlockV2

每个 TCN block 的路径是：

```text
x
 -> in_proj(1x1 conv)
 -> PReLU
 -> norm1
 -> depthwise separable Conv1d
 -> PReLU
 -> norm2
 -> dropout
 -> res_proj / skip_proj
```

对应输出：

```text
residual_out = residual + res_proj(y)
skip         = skip_proj(y)
```

其中 depthwise temporal convolution 使用 dilation：

```text
dilation = 1, 2, 4, 8, 16, 32, 64, 128
```

每个 repeat 重新从 dilation 1 开始。当前配置是：

```yaml
tcn_num_blocks: 8
tcn_num_repeats: 3
```

因此一共有：

```text
8 blocks/repeat * 3 repeats = 24 TCN blocks
```

这种 dilation 堆叠让 TCN 在保持卷积结构简单的同时获得较大的时间感受野。

### 4.5 Skip Aggregation Mask Head

V2 的 mask head 是：

```text
skip_sum -> PReLU -> norm -> mask_conv1x1 -> mask_activation
```

这里的 `skip_sum` 来自所有 block 的 skip 输出累加：

```text
for block in blocks:
    x, skip = block(x)
    skip_sum += skip
```

直观解释：每个 block 都从不同 dilation 尺度提取一部分有用信息，最后通过 `skip_sum` 汇总给 mask head。这样最终 mask 不只依赖最后一层 residual feature，而是能同时利用浅层细节和深层长上下文。

当前 TCN-V2 配置使用：

```yaml
mask_activation: relu
```

也就是无上界 mask。优点是模型可以放大某些 encoder feature，表达能力较强；缺点是如果后续加入更强 target cue 注入，mask 幅度可能更难控制。因此在 TCCTCN-V4 中才更倾向使用 `sigmoid` 稳定 adapter 训练。

## 5. 代码仓库实现

### 5.1 模型文件

主要文件：

```text
models/model_USEF_TCN_V2.py
```

核心类：

| 类/函数 | 作用 |
|---|---|
| `select_norm` | 根据配置选择 `fln/cln/gln/gn/bn` |
| `Encoder` | waveform 到 encoder feature |
| `Decoder` | masked feature 到 waveform |
| `FiLM` | target cue 条件调制 |
| `DepthwiseSeparableConv1d` | 支持 causal / non-causal 的 depthwise temporal conv |
| `TCNBlockV2` | residual + skip 的 TCN 基本块 |
| `TCNBackendV2` | 多 repeat、多 dilation 的 TCN 堆叠 |
| `Tar_Model` | USEF-TCN-V2 完整模型 |

### 5.2 配置文件

主要配置：

```text
config/config-USEF-TCN-V2.yaml
```

模型入口：

```yaml
MaskNet: !new:models.model_USEF_TCN_V2.Tar_Model
```

前端初始化：

```yaml
continue_from: 'chkpt/init_frontend_from_sepformer_wsj0-2mix_for_tcn_v2.pth.tar'
```

这表示 TCN-V2 从 SepFormer 相关前端权重初始化，主要继承：

```text
encoder
decoder
conv1d1
fusion_mdl
film
```

TCN 后端本身是新建训练的轻量后端。

### 5.3 训练入口

`train.py` 通过 HyperPyYAML 读取配置并实例化：

```text
config yaml -> Encoder / Decoder / Fusion_mdl / FiLM / MaskNet
```

因此 `tcn_norm`、`mask_activation` 等参数不是手动在代码里写死的，而是通过：

```yaml
MaskNet:
  tcn_norm: !ref <tcn_norm>
  mask_activation: !ref <mask_activation>
```

注入到 `Tar_Model.__init__()` 中。

## 6. 配置参数说明

### 6.1 训练参数

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `optimizer.lr` | `0.0001` | Adam 基础学习率 |
| `epochs` | `150` | 最大训练轮数 |
| `batch_size` | `4` | 训练 batch size |
| `valid_batch_size` | `1` | 验证 batch size |
| `use_amp` | `True` | 使用混合精度训练 |
| `amp_dtype` | `bfloat16` | AMP 数据类型 |
| `max_norm` | `5` | 梯度裁剪阈值 |
| `checkpoint` | `1` | checkpoint 保存间隔 |

### 6.2 前端初始化与学习率参数

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `continue_from` | `init_frontend_from_sepformer...pth.tar` | 从 SepFormer 前端初始化 |
| `freeze_frontend_warmup` | `False` | 当前 TCN-V2 不冻结前端 warmup |
| `frontend_warmup_epochs` | `3` | 若启用冻结，冻结 epoch 数 |
| `frontend_unfreeze_lr_scale` | `0.1` | 若启用冻结，解冻前端时缩放前端 LR |
| `separate_frontend_backend_lr` | `True` | optimizer 分成 frontend/backend 参数组 |

当前配置里 `freeze_frontend_warmup: False`，所以 `frontend_warmup_epochs` 和 `frontend_unfreeze_lr_scale` 不会实际触发解冻逻辑。保留这些参数主要是为了和后续 WRCD / TCCTCN 实验配置保持一致。

### 6.3 Encoder / USEF 参数

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `N_encoder_out` | `256` | encoder 输出通道 |
| `out_channels` | `256` | TCN 输入通道 |
| `kernel_size` | `16` | encoder/decoder kernel size |
| `kernel_stride` | `8` | decoder stride |
| `embd` | `256` | 目标条件表示维度 |
| `Fusion_mdl.num_layers` | `4` | cross transformer 层数 |
| `Fusion_mdl.nhead` | `8` | attention head 数 |
| `Fusion_mdl.d_ffn` | `1024` | FFN 隐层维度 |

### 6.4 TCN 参数

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `tcn_conv_channels` | `512` | TCN block 内部卷积通道数 |
| `tcn_skip_channels` | `256` | skip 分支通道数 |
| `tcn_kernel_size` | `3` | depthwise temporal conv kernel size |
| `tcn_num_blocks` | `8` | 每个 repeat 的 block 数 |
| `tcn_num_repeats` | `3` | repeat 数 |
| `tcn_norm` | `gln` | norm 类型 |
| `tcn_causal` | `False` | 是否使用 causal convolution |
| `tcn_dropout` | `0.0` | dropout 比例 |
| `mask_activation` | `relu` | mask 激活函数 |

当前 `tcn_causal: False`，表示 TCN 使用 non-causal padding，可以看到未来帧，因此更适合作为离线质量 baseline。如果要做严格 streaming，需要切到 causal，但通常会牺牲一部分分离质量。

## 7. Norm 与 Mask 设计说明

### 7.1 tcn_norm

`tcn_norm` 在当前实现中同时控制多个位置：

```text
encoder feature 后的 norm_m
fusion 后的 fusion_norm
TCN block 内部 norm1 / norm2
skip_sum 后的 skip_norm
```

当前配置使用：

```yaml
tcn_norm: gln
```

这是一种工程上简洁的写法，方便统一切换 norm 类型。它的潜在问题是，不同位置的 feature 分布并不完全一样：

```text
encoder feature: 更接近时域基表示
fusion feature: 已经过 target cue 调制
TCN block feature: residual 时序建模中间状态
skip feature: 靠近 mask head 的聚合表示
```

因此从研究角度看，未来可以拆成：

```text
input_norm / fusion_norm / tcn_norm / skip_norm
```

但 TCN-V2 阶段优先保持配置简单，先建立稳定轻量 baseline。

### 7.2 mask_activation

V2 支持：

```text
sigmoid / relu / softplus / none
```

当前 TCN-V2 使用：

```yaml
mask_activation: relu
```

ReLU mask 的含义是：

```text
mask >= 0，但没有上界
```

优点：

```text
1. 可以放大 encoder feature
2. 对纯 TCN-V2 来说表达能力更强
3. 当前实验中 TCN-V2 使用 ReLU 得到了较稳定结果
```

风险：

```text
如果后续在 TCN block 内部加入过强 target cue，
无上界 mask 更容易被推大，导致重建质量下降。
```

所以在 TCN-V2 中使用 ReLU 是合理的；在 TCCTCN-V4 中改用 sigmoid 是为了控制 adapter 引入后的训练风险，两者服务于不同实验目标。

## 8. 实验结果与定位

### 8.1 与 SepFormer 对比

| 模型 | SDRi | SI-SDRi | PESQi | Latency | RTF | Param |
|---|---:|---:|---:|---:|---:|---:|
| USEF-SepFormer | 20.2243 | 19.9581 | 2.0838 | 0.1079 | 0.0187 | 29.1684M |
| USEF-TCN-V2 | 16.3869 | 15.9210 | 1.6758 | 0.0489 | 0.0086 | 12.9554M |

结论：

```text
TCN-V2 明显降低参数量和推理时间，
但分离质量低于 SepFormer。
```

这说明 TCN-V2 不是为了直接超过 SepFormer，而是为了得到一个可以在机器人端侧部署的轻量 USEF baseline。

### 8.2 与 WRCD-V2 对比

| 模型 | SDRi | SI-SDRi | PESQi | Latency | RTF | Param |
|---|---:|---:|---:|---:|---:|---:|
| USEF-TCN-V2 | 16.3869 | 15.9210 | 1.6758 | 0.0489 | 0.0086 | 12.9554M |
| USEF-TCN-V2-WRCD-V2 | 16.7962 | 16.3953 | 1.7794 | 0.0482 | 0.0084 | 12.9554M |

WRCD-V2 没有改动 TCN-V2 模型结构，而是通过 short aux、teacher waveform distillation 和 dual-aux consistency 改进训练策略。因此参数量不变，但 SI-SDRi 提升 `+0.4743 dB`。

这也说明 TCN-V2 是一个可继续优化的轻量主干：

```text
TCN-V2 提供轻量结构；
WRCD-V2 改进短 aux 场景训练；
TCCTCN-V4 尝试进一步降低 target_confusion。
```

## 9. 当前已知问题

TCN-V2 的主要问题不是整体重建崩坏，而是坏样本里存在较明显的 target-confusion：

```text
模型输出有时更接近干扰说话人，而不是目标说话人。
```

原因可能是：

```text
1. target cue 主要在 USEF fusion / FiLM 阶段注入
2. TCN 后端内部没有持续使用目标说话人信息
3. 轻量后端容量有限，复杂重叠语音中可能依赖声学主导源
4. ReLU mask 表达能力强，但不显式约束目标/干扰人的相对接近程度
```

这就是后续 TCCTCN 系列的设计动机：

```text
让 target cue 更深入地参与 TCN 后端，
并尝试降低 target_confusion。
```

不过 TCCTCN-V2/V3 的诊断结果说明，target cue 不能过强注入，否则会牺牲重建质量。因此 TCCTCN-V4 才转向更保守的：

```text
repeat-level + skip-only + gated target correction
```

## 10. 推荐使用方式

### 10.1 训练

```bash
python train.py \
  --config config/config-USEF-TCN-V2.yaml \
  --device 0 \
  --memory 0
```

### 10.2 评估

```bash
python eval.py \
  --config chkpt/USEF-TCN-V2/wsj0-2mix/config.yaml \
  --chkpt-path chkpt/USEF-TCN-V2/wsj0-2mix/temp_best.pth.tar \
  --device 0 \
  --memory 0
```

如果评估短 aux / wake-word 场景，需要显式控制 aux duration，例如：

```bash
python eval.py \
  --config chkpt/USEF-TCN-V2/wsj0-2mix/config.yaml \
  --chkpt-path chkpt/USEF-TCN-V2/wsj0-2mix/temp_best.pth.tar \
  --device 0 \
  --memory 0 \
  --aux-duration 1.0
```

## 11. 设计总结

TCN-V2 的核心贡献是把 USEF-TSE 改造成一个轻量、低延迟、结构清晰的端侧 baseline：

```text
SepFormer / TFGridNet heavy backend
    -> Conv-TasNet style residual + skip TCN backend
```

它的关键结构是：

```text
USEF Fusion -> FiLM -> TCN blocks
TCN block: 1x1 -> PReLU -> norm -> depthwise conv -> PReLU -> norm -> residual + skip
Mask head: skip_sum -> PReLU -> norm -> mask_conv1x1 -> ReLU mask
```

从实验结果看，TCN-V2 相比 SepFormer 明显降低参数量、Latency 和 RTF，适合作为人形机器人端侧部署方向的轻量化主干。但它也暴露出 target-confusion 问题，因此后续 WRCD-V2 和 TCCTCN-V4 都是在 TCN-V2 这个主干上继续增强鲁棒性和目标说话人一致性。
