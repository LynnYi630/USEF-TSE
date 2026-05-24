# TCCTCN-V4 改进说明

本文档记录 `USEF-TCCTCN-V4` 的设计动机、模型结构、训练逻辑、代码修改和配置参数说明。

## 1. 设计背景

`USEF-TCN-V2` 的主要优点是速度快、参数量低、重建质量比较稳定，但诊断结果显示它在坏样本中存在较高比例的 `target_confusion`。也就是说，模型有时会把干扰说话人当成目标说话人输出。

`USEF-TCCTCN-V2/V3` 的设计初衷是缓解这个问题：把 USEF 的 target cue 注入 TCN 后端，让目标说话人信息在时序建模过程中持续参与。但最终诊断显示，V2/V3 虽然一定程度上压低了 target-confusion 风险，却引入了更严重的 `target_degradation` 和 `low_quality_or_silence`：

```text
TCN-V2:    target_confusion 比例偏高
TCCTCN-V2/V3: target_confusion 降低一些，但 target_degradation 明显增多
```

这说明原来的 dense conditioning 过强。V3 中 target cue 被注入到多个 TCN block 的多个分支：

```text
cond -> depthwise branch
cond -> residual branch
cond -> skip branch
```

这种做法容易让 target cue 过度干预 TCN 主干，导致 mask 或中间特征幅度失控，最终表现为重建质量下降。

V4 的核心目标是：

```text
保留 TCN-V2 的稳定重建能力，
只在有 target-confusion 风险时，
用低侵入方式引入 target cue 做纠偏。
```

## 2. V4 的核心改进

V4 从 `block-wise dense conditioning` 改为：

```text
repeat-level + skip-only + gated target correction
```

可以概括为：

```text
USEF fusion -> FiLM -> TCN repeat
                         -> repeat-level skip adapter
                         -> skip_sum -> PReLU -> norm -> mask_conv1x1 -> sigmoid mask
```

### 2.1 Repeat-Level Gate

V3 是每个 TCN block 都可以注入 target cue：

```text
block1 注入
block2 注入
...
block24 注入
```

V4 改成每个 repeat 只注入一次：

```text
repeat1 末尾注入一次
repeat2 末尾注入一次
repeat3 末尾注入一次
```

这样可以减少 target cue 的干预频率，避免条件信息在每一层不断累积并污染主干特征。

### 2.2 Skip-Only Gated Conditioning

V4 不再修改 residual 主干，也不再修改 depthwise 时序卷积分支，只在 repeat 聚合后的 skip 输出上做小幅 correction。

原因是：

```text
residual path: TCN 内部时序表征传播主干，污染后会影响后续所有层
depthwise path: TCN 内部局部时序建模位置，过强调制会破坏重建
skip path: 更靠近最终 mask 估计，适合做目标说话人纠偏
```

因此 V4 的注入位置更保守：

```text
skip_r = skip_r + alpha_r * delta_r
```

其中 `skip_r` 是第 `r` 个 repeat 的 skip 聚合结果，`delta_r` 是 target adapter 产生的修正项。

### 2.3 Need Gate * Reliability Gate

V4 的 adapter 不是无条件打开，而是通过两个 gate 控制：

```text
alpha_r = scale * need_gate_r * reliability_gate_r
```

其中：

```text
need_gate: 判断当前 skip 特征是否需要 target cue 纠偏
reliability_gate: 判断当前 target cue 是否可靠
scale: 全局可学习强度，控制 adapter 最大影响
```

直观理解：

```text
need 高、reliability 高：打开 target correction
need 低、reliability 高：当前不需要纠偏，少动
need 高、reliability 低：虽然需要纠偏，但 target cue 不可靠，也少动
```

### 2.4 可学习 Near-Zero Scale

V4 没有手动固定 `alpha_max=0.1/0.2`，而是使用一个全局可学习 scale：

```text
scale = sigmoid(adapter_scale_logit)
```

默认：

```yaml
adapter_scale_init_logit: -3.0
```

初始时：

```text
sigmoid(-3.0) ≈ 0.047
```

同时 adapter 的最后一层 `delta_out` 默认零初始化：

```yaml
zero_init_adapter_delta: True
```

因此训练初期 V4 基本等价于已有的稳定 TCN 主干。只有当 target adapter 能降低训练 loss 时，模型才会逐渐学会打开 correction。

### 2.5 Confusion Margin Loss

原来的 `anti loss` 只判断：

```text
est 像不像 interferer
```

V4 新增 `target-interferer margin loss`，直接判断：

```text
est 更像 target，还是更像 interferer
```

训练时有 clean target，因此可以构造 oracle interferer：

```text
interferer = mixture - target
```

然后计算：

```text
score_t = SI-SDR(est, target)
score_i = SI-SDR(est, interferer)

L_conf = ReLU(confusion_margin + score_i - score_t)
```

当模型输出更接近干扰说话人时，`L_conf` 会被触发。总 loss 为：

```text
L = L_sisdr + lambda_confusion * L_conf
```

这比单纯 anti loss 更贴近 target-confusion 的定义。

## 3. 代码仓库修改

### 3.1 新增模型文件

新增文件：

```text
models/model_USEF_TCCTCN_V4.py
```

主要包含三个模块：

```text
RepeatSkipConditionAdapter
RepeatLevelSkipTCNBackend
Tar_Model
```

#### RepeatSkipConditionAdapter

该模块实现 repeat-level skip-only gated target correction。

核心输入：

```text
skip: 当前 repeat 的 skip 聚合输出
cond: USEF fusion 产生的 frame-level target cue
scale: 全局 adapter 强度
```

内部主要路径：

```text
cond -> cond_proj -> cond_skip

[skip, cond_skip] -> delta branch -> delta

[skip, cond_skip, |skip-cond_skip|, skip*cond_skip]
    -> need gate branch -> need_gate

cond_skip 的 mean/std
    -> reliability MLP -> reliability_gate

output = skip + scale * need_gate * reliability_gate * delta
```

其中 `need_gate` 是逐帧逐通道 gate，`reliability_gate` 是根据 target cue 统计量生成的全局通道 gate。

#### RepeatLevelSkipTCNBackend

该模块基于 `TCNBlockV2` 构建 repeat-level 后端。

实现逻辑：

```text
for each repeat:
    repeat_skip = 0
    for each block in repeat:
        x, skip = block(x)
        repeat_skip += skip

    repeat_skip = adapter_r(repeat_skip, cond, scale)
    skip_sum += repeat_skip
```

注意：V4 保留了 flat `self.blocks = nn.ModuleList(blocks)` 命名方式，这样可以兼容 TCN-V2/WRCD-V2 的 checkpoint。旧 checkpoint 中已有的 TCN block 权重可以直接加载，新加入的 adapter 保持初始化状态。

#### Tar_Model

V4 的 `Tar_Model` 复用了 TCN-V2 的主要前向流程：

```text
input/aux -> encoder
mix_w/aux_w -> norm + conv1d1
USEF cross-attention fusion -> target_seq
FiLM(x_seq, target_seq)
TCN backend with repeat-level skip adapter
skip_sum -> PReLU -> norm -> mask_conv1x1 -> mask activation
mask * encoder feature -> decoder
```

与 V3 的区别是：`cond` 不再进入每个 TCN block 的 depthwise/residual/skip 分支，而是只在每个 repeat 的 skip 输出上做一次 gated correction。

### 3.2 新增配置文件

新增文件：

```text
config/config-USEF-TCCTCN-V4.yaml
```

该配置默认使用当前最稳定的 TCN/WRCD checkpoint 做初始化：

```yaml
continue_from: 'chkpt/USEF-TCN-V2-WRCD-V2/wsj0-2mix/temp_best.pth.tar'
reset_epoch_on_continue: True
load_optimizer_state: False
```

这不是断点续训，而是：

```text
加载旧模型权重
重置 epoch
重置 optimizer
重置随机数状态
从第 1 轮开始训练 V4
```

如果要从 SepFormer frontend 初始化开始训练，可以改成：

```yaml
continue_from: 'chkpt/init_frontend_from_sepformer_wsj0-2mix_for_tcctcn.pth.tar'
reset_epoch_on_continue: True
load_optimizer_state: False
freeze_frontend_warmup: True
frontend_warmup_epochs: 3
optimizer:
  lr: 0.0001
epochs: 200
```

### 3.3 修改 Trainer

修改文件：

```text
trainer/trainer.py
```

新增功能包括：

```text
1. reset_epoch_on_continue
2. load_optimizer_state
3. lambda_confusion / confusion_margin
4. confusion margin loss 日志输出
```

#### reset_epoch_on_continue

用于把旧 checkpoint 当作初始化，而不是断点续训。

```yaml
reset_epoch_on_continue: True
```

启用后：

```text
self.start_epoch = 0
不恢复 optimizer state
不恢复 RNG state
```

适合 V4 从稳定 TCN/WRCD 主干微调。

#### load_optimizer_state

控制是否加载旧 checkpoint 里的 optimizer 状态。

```yaml
load_optimizer_state: False
```

V4 推荐关闭，因为新模型加入了 adapter，旧 optimizer state 与新结构不完全匹配，而且当前目的不是继续旧实验，而是启动新实验。

#### confusion margin loss

新增训练项：

```text
if lambda_confusion > 0:
    interferer = mixture - target
    target_score = sisnr(est, target)
    interferer_score = sisnr(est, interferer)
    L_conf = ReLU(confusion_margin + interferer_score - target_score)
```

日志中会额外输出：

```text
Conf <value>
```

这样可以观察 target-confusion 约束是否在训练中持续触发。

## 4. 配置参数说明

### 4.1 训练初始化相关参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `continue_from` | `chkpt/USEF-TCN-V2-WRCD-V2/wsj0-2mix/temp_best.pth.tar` | 加载旧 checkpoint 权重作为初始化 |
| `reset_epoch_on_continue` | `True` | 不继承旧 checkpoint 的 epoch，从第 1 轮开始训练 |
| `load_optimizer_state` | `False` | 不继承旧 optimizer 状态 |
| `freeze_frontend_warmup` | `False` | 是否在训练初期冻结 encoder/decoder/fusion/FiLM |
| `frontend_warmup_epochs` | `0` | 冻结前端的 epoch 数 |
| `frontend_unfreeze_lr_scale` | `1.0` | 解冻前端时学习率缩放 |
| `separate_frontend_backend_lr` | `True` | 是否把 optimizer 分成 frontend/backend 两个参数组 |

默认配置使用 WRCD-V2 checkpoint 初始化，因此不冻结前端，而是全模型小学习率微调。

如果从 `init_frontend_from_sepformer_wsj0-2mix_for_tcctcn.pth.tar` 开始训练，建议启用：

```yaml
freeze_frontend_warmup: True
frontend_warmup_epochs: 10
frontend_unfreeze_lr_scale: 0.5
separate_frontend_backend_lr: True
```

启用 `separate_frontend_backend_lr` 后，optimizer 会分成：

```text
frontend: encoder / decoder / fusion_mdl / film
backend: 其余 TCN backend、adapter、mask head 等模块
```

此时解冻前端时：

```text
frontend lr = frontend lr * frontend_unfreeze_lr_scale
backend lr  保持不变
```

这样可以让预训练前端用更小学习率微调，同时让随机初始化的 TCN/V4 adapter 继续保持较大学习率学习。

### 4.2 TCN 主干相关参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `tcn_conv_channels` | `512` | TCN block 内部卷积通道数 |
| `tcn_skip_channels` | `256` | skip 分支通道数 |
| `tcn_kernel_size` | `3` | depthwise temporal convolution kernel size |
| `tcn_num_blocks` | `8` | 每个 repeat 内的 block 数 |
| `tcn_num_repeats` | `3` | repeat 数 |
| `tcn_norm` | `gln` | TCN 归一化方式 |
| `tcn_causal` | `False` | 是否使用 causal depthwise convolution |
| `tcn_dropout` | `0.0` | TCN dropout |
| `mask_activation` | `sigmoid` | mask 激活函数 |

第一版 V4 默认使用 non-causal + `gln`，目的是先验证 V4 adapter 是否有效，避免 causal 损失和 adapter 效果混在一起。

### 4.3 V4 Adapter 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `adapter_bottleneck_channels` | `256` | adapter delta 分支的 bottleneck 通道数 |
| `adapter_scale_init_logit` | `-3.0` | 全局 adapter scale 的 logit 初始化 |
| `adapter_scale_trainable` | `True` | 是否让全局 scale 可学习 |
| `use_reliability_gate` | `True` | 是否启用 reliability gate |
| `need_gate_init_bias` | `0.0` | need gate 输出层 bias 初始化 |
| `reliability_gate_init_bias` | `2.0` | reliability gate 输出层 bias 初始化 |
| `zero_init_adapter_delta` | `True` | 是否把 adapter 最后一层零初始化 |

重点参数解释：

```text
adapter_scale_init_logit = -3.0
scale = sigmoid(-3.0) ≈ 0.047
```

这让 adapter 初始强度很小。

```text
zero_init_adapter_delta = True
```

这让 adapter 初始输出 `delta≈0`，因此初始模型行为接近加载的稳定 TCN/WRCD 主干。

```text
reliability_gate_init_bias = 2.0
sigmoid(2.0) ≈ 0.88
```

这表示 reliability gate 初始时不强行关闭 target cue，主要约束由 near-zero delta 和 small scale 提供。训练后 reliability gate 可以根据 target cue 的统计稳定性调整各通道使用强度。

### 4.4 Target-Confusion Loss 参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `lambda_anti` | `0.0` | 是否启用旧 anti-interferer loss |
| `anti_sisnr_margin` | `0.0` | anti loss 的 margin |
| `lambda_confusion` | `0.05` | confusion margin loss 权重 |
| `confusion_margin` | `3.0` | 要求 target 分数比 interferer 分数高出的 margin |

V4 默认关闭 `lambda_anti`，只启用更直接的 `lambda_confusion`：

```text
L_conf = ReLU(confusion_margin + SI-SDR(est, interferer) - SI-SDR(est, target))
```

如果后续实验发现 target-confusion 仍然严重，可以再尝试：

```yaml
lambda_confusion: 0.1
```

但不建议一开始同时打开很强的 `lambda_anti` 和 `lambda_confusion`，否则可能再次牺牲重建质量。

## 5. 训练时建议观察的指标

训练日志中重点观察：

```text
Train Loss / Valid Loss
Sup
Conf
```

建议判断方式：

```text
1. Valid Loss 不能明显低于 TCN-V2/WRCD-V2 对应水平太多；
2. Conf 不应该一直很大，理想情况是前期触发、后期下降；
3. eval 后重点看 target_confusion 数量是否下降；
4. 同时确认 target_degradation 不要像 V2/V3 那样大幅增加。
```

如果 V4 有效，理想趋势是：

```text
相比 TCN-V2: target_confusion 下降
相比 TCCTCN-V2/V3: target_degradation 下降
整体 SI-SDRi 接近或超过 TCN-V2-WRCD-V2
```

## 6. 与 V2/V3 的关系

| 模型 | Target cue 注入方式 | 风险 |
|---|---|---|
| TCN-V2 | 只在 USEF fusion/FiLM 阶段使用 target cue | target_confusion 偏高 |
| TCCTCN-V2 | 每个 TCN block 内部注入 target cue | 容易扰动主干 |
| TCCTCN-V3 | depthwise/residual/skip 多分支注入 target cue | target_degradation 明显增多 |
| TCCTCN-V4 | repeat-level + skip-only + gated correction | 低侵入、可控纠偏 |

V4 不是简单增强 target cue，而是降低 target cue 的权限：

```text
从“每层主导特征变化”
改为“靠近输出 mask 的受控纠偏信号”
```

这与当前诊断结论一致：TCCTCN 的主要问题不是 target cue 没用，而是 target cue 在 V2/V3 中过度干预重建路径。
