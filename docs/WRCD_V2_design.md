# WRCD-V2 改进说明

本文档记录 `USEF-TCN-V2-WRCD-V2` 的设计动机、训练目标、数据流、代码实现和配置参数说明。

## 1. 设计背景

`USEF-TCN-V2` 的目标是把原始 USEF-TSE 框架轻量化，用 TCN 后端替代更重的 SepFormer / TFGridNet 后端。TCN-V2 在速度、参数量和端侧部署友好性上更有优势，但它仍然有两个现实问题：

```text
1. 训练和测试时的 enrollment / aux 使用方式不够贴近机器人端侧场景
2. 轻量 TCN 学生模型的重建质量和强 teacher 模型之间仍有差距
```

在机器人端侧部署里，目标说话人线索往往不是一整段干净长 enrollment，而更像一段较短的唤醒词、短句或局部参考语音。因此，模型需要在短 aux 条件下仍然稳定提取目标说话人。

WRCD 的设计目标可以概括为：

```text
用短 wake/enrollment 训练目标说话人提取模型，
同时用强 teacher 的 waveform 输出约束轻量学生模型，
并让同一目标说话人的不同 aux 裁剪得到一致输出。
```

在当前代码中，WRCD-V2 是一个**训练策略改进**，不是一个新的模型后端。它复用 `models/model_USEF_TCN_V2.py` 的 TCN-V2 模型结构，因此推理阶段参数量、延迟和 RTF 基本不增加。

## 2. WRCD-V2 的核心改进

WRCD-V2 包含三项训练侧改进：

```text
short wake aux / enrollment
waveform teacher distillation
dual-aux consistency
```

整体训练流可以写成：

```text
mixture, target, aux_full
    -> crop aux_1 as short wake aux
    -> crop aux_2 as consistency aux
    -> load teacher_est from offline cache

student_est_1 = model(mixture, aux_1)
student_est_2 = model(mixture, aux_2)   # no_grad / detach target

L = L_sup + lambda_kd * L_kd + lambda_cons * L_cons
```

其中：

```text
L_sup  = -SI-SDR(student_est_1, target)
L_kd   = -SI-SDR(student_est_1, teacher_est)
L_cons = mean(|student_est_1 - stopgrad(student_est_2)|)
```

### 2.1 Short Wake Aux / Enrollment

原始 TCN-V2 训练时会从 aux 语音中取较长片段。WRCD-V2 改为显式使用短 wake/enrollment：

```yaml
use_short_aux: True
train_wake_aux_duration: 1.0
valid_wake_aux_duration: 1.0
```

训练时，`dataset.data.tr_dataset` 会从同一目标说话人的 aux 语音中随机裁剪一段短片段：

```text
aux_full -> random crop -> aux_1
```

如果启用一致性约束，还会再裁剪第二段：

```text
aux_full -> random crop -> aux_2
```

这样训练出来的模型不再依赖长 enrollment，而是更适合端侧机器人“短唤醒词 / 短参考语音”的使用场景。

### 2.2 Waveform Teacher Distillation

WRCD 使用强 teacher 模型提前生成训练集上的 teacher waveform：

```text
teacher_est = teacher_model(mixture, aux)
```

这些输出通过 `tools/generate_teacher_cache.py` 离线缓存到磁盘：

```text
teacher_cache/<teacher-name>/<dataset>/<split>/<escaped-utt-id>.pt
```

训练学生模型时，不再在线运行 teacher，而是直接读取缓存：

```text
teacher_est = load_teacher_cache(utt_id)
```

然后加入 waveform-level KD：

```text
L_kd = -SI-SDR(student_est, teacher_est)
```

直观理解：`target` 是干净目标语音，提供严格监督；`teacher_est` 是强 teacher 在混合语音上的增强结果，包含 teacher 对混合语音、目标说话人线索和重建细节的经验。学生模型同时向 clean target 和 teacher waveform 靠拢，有助于轻量模型学习更稳的 mask / waveform 重建。

当前 `config/config-USEF-TCN-WRCD-V2.yaml` 中使用：

```yaml
teacher_cache_dir: teacher_cache/USEF-TFGridNet/wsj0-2mix/train
lambda_kd: 0.2
```

### 2.3 Dual-Aux Consistency

WRCD-V1 只有短 aux 和 teacher KD。WRCD-V2 新增了双 aux 一致性：

```yaml
use_wrcd_consistency: True
lambda_cons: 0.05
```

同一个训练样本会裁剪两段不同的短 aux：

```text
aux_1: 用于主前向和监督
aux_2: 用于一致性前向
```

模型分别输出：

```text
est_1 = model(mixture, aux_1)
est_2 = model(mixture, aux_2)
```

然后约束它们一致：

```text
L_cons = mean(|est_1 - stopgrad(est_2)|)
```

当前实现中，第二个 aux 的 forward 在 `torch.no_grad()` 下执行，并且 loss 中使用 `detach()`，因此一致性项只推动主分支 `est_1` 向另一段 aux 的输出靠近，不会双向反传。这种设计更稳，也能减少显存占用。

直观解释：

```text
同一个目标说话人，即使用不同短 aux，
模型也应该输出同一个目标语音。
```

这项约束针对的是短 enrollment 场景下的 aux 片段敏感性。它希望模型不要过度依赖某一段 aux 的局部语音内容，而是学习更稳定的说话人条件表示。

## 3. 代码仓库修改

### 3.1 配置文件

主要配置文件：

```text
config/config-USEF-TCN-WRCD-V2.yaml
```

关键开关：

```yaml
use_short_aux: True
use_wrcd: True
use_wrcd_consistency: True
teacher_cache_dir: teacher_cache/USEF-TFGridNet/wsj0-2mix/train
lambda_kd: 0.2
lambda_cons: 0.05
train_wake_aux_duration: 1.0
valid_wake_aux_duration: 1.0
```

WRCD-V2 仍然使用 TCN-V2 模型：

```yaml
MaskNet: !new:models.model_USEF_TCN_V2.Tar_Model
```

因此 WRCD-V2 不新增模型参数，推理时仍然是 TCN-V2 结构。

### 3.2 Teacher Cache 生成工具

新增/使用工具：

```text
tools/generate_teacher_cache.py
```

该脚本完成：

```text
1. 读取 mix.scp / aux.scp
2. 加载 teacher config 和 checkpoint
3. 对每个训练样本运行 teacher_model(mixture, aux)
4. 保存 teacher_est 到 .pt cache
```

典型命令：

```bash
python tools/generate_teacher_cache.py \
  --config config/config-USEF-TFGridNet.yaml \
  --chkpt-path chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar \
  --data-path data/train/wsj0-2mix \
  --output-dir teacher_cache/USEF-TFGridNet/wsj0-2mix/train \
  --device cuda:0 \
  --max-aux-duration 4 \
  --overwrite
```

生成后，在 WRCD-V2 配置中通过 `teacher_cache_dir` 指向该目录。

### 3.3 Dataset 修改

修改文件：

```text
dataset/data.py
```

新增逻辑包括：

```text
teacher_cache_path(cache_dir, utt_id)
tr_dataset(..., use_short_aux, use_wrcd, use_wrcd_consistency, teacher_cache_dir, ...)
_load_teacher_est(utt)
_crop_wake_aux(y)
```

训练样本返回格式根据开关变化：

```text
普通训练:
    mix_wav, target_wav, aux_wav, source_len

WRCD-V1:
    mix_wav, target_wav, aux_wav, source_len, teacher_wav

WRCD-V2:
    mix_wav, target_wav, aux_wav, source_len, teacher_wav, aux_wav_cons
```

其中：

```text
aux_wav:       第一段短 aux
aux_wav_cons:  第二段短 aux，用于一致性约束
teacher_wav:   离线 teacher waveform cache
```

### 3.4 Train 入口修改

修改文件：

```text
train.py
```

`train.py` 会把配置中的 WRCD 参数传给 `tr_dataset`：

```python
use_short_aux = config.get('use_short_aux', False)
use_wrcd = config.get('use_wrcd', False)
use_wrcd_consistency = config.get('use_wrcd_consistency', False)
teacher_cache_dir = config.get('teacher_cache_dir', None)
wake_aux_min_duration = ...
wake_aux_max_duration = ...
```

验证集使用：

```python
aux_duration = config.get('valid_wake_aux_duration', None)
```

因此训练和验证都可以使用短 aux，避免 train/valid 条件不一致。

### 3.5 Trainer 修改

修改文件：

```text
trainer/trainer.py
```

新增配置读取：

```python
self.use_wrcd = bool(config.get('use_wrcd', False))
self.use_wrcd_consistency = bool(config.get('use_wrcd_consistency', False))
self.lambda_kd = float(config.get('lambda_kd', 0.0))
self.lambda_cons = float(config.get('lambda_cons', 0.0))
```

新增 batch 解析逻辑：

```text
len(data) == 4: 普通训练
len(data) == 5: WRCD-V1, 带 teacher
len(data) == 6: WRCD-V2, 带 teacher 和第二段 aux
```

新增 loss：

```text
KD:
    metrics['kd'] = -SI-SDR(est, teacher)

Cons:
    metrics['cons'] = mean(|est(aux_1) - stopgrad(est(aux_2))|)
```

训练日志中对应字段：

```text
Sup  : 基础 clean target 监督
KD   : teacher waveform distillation
Cons : 双 aux 输出一致性
```

## 4. 配置参数说明

### 4.1 WRCD 开关参数

| 参数 | 默认/当前值 | 作用 |
|---|---:|---|
| `use_short_aux` | `True` | 训练时使用短 wake/enrollment aux |
| `use_wrcd` | `True` | 启用 WRCD teacher waveform distillation |
| `use_wrcd_consistency` | `True` | 启用 WRCD-V2 双 aux 一致性 |
| `teacher_cache_dir` | `teacher_cache/USEF-TFGridNet/wsj0-2mix/train` | teacher waveform cache 目录 |
| `lambda_kd` | `0.2` | teacher KD loss 权重 |
| `lambda_cons` | `0.05` | 双 aux consistency loss 权重 |

### 4.2 Wake Aux 参数

| 参数 | 默认/当前值 | 作用 |
|---|---:|---|
| `train_wake_aux_duration` | `1.0` | 训练时短 aux 长度，min=max 的简写 |
| `train_wake_aux_min_duration` | 未显式设置 | 训练时随机 aux 裁剪最短长度 |
| `train_wake_aux_max_duration` | 未显式设置 | 训练时随机 aux 裁剪最长长度 |
| `valid_wake_aux_duration` | `1.0` | 验证/评估时 aux 截断长度 |

当前配置使用固定 1 秒短 aux：

```yaml
train_wake_aux_duration: 1.0
valid_wake_aux_duration: 1.0
```

如果要做更强的数据增强，可以改成区间采样：

```yaml
train_wake_aux_min_duration: 0.5
train_wake_aux_max_duration: 2.0
valid_wake_aux_duration: 1.0
```

这样训练时模型会看到不同长度的 wake/enrollment，但验证仍保持固定长度，便于对比。

### 4.3 TCN-V2 主干参数

WRCD-V2 沿用 TCN-V2 后端：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `tcn_conv_channels` | `512` | TCN block 内部卷积通道数 |
| `tcn_skip_channels` | `256` | skip 分支通道数 |
| `tcn_kernel_size` | `3` | depthwise temporal convolution kernel size |
| `tcn_num_blocks` | `8` | 每个 repeat 的 block 数 |
| `tcn_num_repeats` | `3` | repeat 数 |
| `tcn_norm` | `gln` | TCN 归一化方式 |
| `tcn_causal` | `False` | 是否使用 causal TCN |
| `tcn_dropout` | `0.0` | TCN dropout |
| `mask_activation` | `sigmoid` | mask 激活函数 |

注意：WRCD-V2 的提升来自训练策略，不来自模型结构增大。因此参数量与 TCN-V2 保持一致。

## 5. V1 与 V2 的区别

| 版本 | short aux | teacher KD | dual-aux consistency | 主要目的 |
|---|---|---|---|---|
| WRCD-V1 | 有 | 有 | 无 | 让轻量模型适应短 wake aux，并向 teacher waveform 学习 |
| WRCD-V2 | 有 | 有 | 有 | 进一步降低模型对具体 aux 裁剪片段的敏感性 |

V2 的新增项只有：

```text
use_wrcd_consistency: True
lambda_cons: 0.05
第二段 aux_wav_cons
Cons loss
```

## 6. 训练与评估结果

在当前 wsj0-2mix 结果中，WRCD-V2 相比 TCN-V2 有小幅提升，且没有增加推理参数量：

| 模型 | SDRi | SI-SDRi | PESQi | Latency | RTF | Param |
|---|---:|---:|---:|---:|---:|---:|
| USEF-TCN-V2 | 16.3869 | 15.9210 | 1.6758 | 0.0489 | 0.0086 | 12.9554M |
| USEF-TCN-V2-WRCD-V2 | 16.7962 | 16.3953 | 1.7794 | 0.0482 | 0.0084 | 12.9554M |

可以看到：

```text
SI-SDRi: +0.4743 dB
SDRi:    +0.4093 dB
PESQi:   +0.1036
Param:   不变
RTF:     基本不变
```

这符合 WRCD-V2 的设计定位：它是训练侧增强，不是推理侧增大模型。

## 7. 推荐训练流程

### 7.1 生成 teacher cache

先用强 teacher 生成训练集 cache：

```bash
python tools/generate_teacher_cache.py \
  --config config/config-USEF-TFGridNet.yaml \
  --chkpt-path chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar \
  --data-path data/train/wsj0-2mix \
  --output-dir teacher_cache/USEF-TFGridNet/wsj0-2mix/train \
  --device cuda:0 \
  --max-aux-duration 4
```

### 7.2 训练 WRCD-V2

```bash
python train.py \
  --config config/config-USEF-TCN-WRCD-V2.yaml \
  --device 0 \
  --memory 0
```

### 7.3 评估

```bash
python eval.py \
  --config chkpt/USEF-TCN-V2-WRCD-V2/wsj0-2mix/config.yaml \
  --chkpt-path chkpt/USEF-TCN-V2-WRCD-V2/wsj0-2mix/temp_best.pth.tar \
  --device 0 \
  --memory 0 \
  --aux-duration 1.0
```

如果训练配置使用 `valid_wake_aux_duration: 1.0`，评估时也建议显式设置 `--aux-duration 1.0`，避免训练/评估条件不一致。

## 8. 训练时建议观察的指标

训练日志中重点看：

```text
Sup
KD
Cons
Valid Loss
```

含义：

```text
Sup  : 学生输出接近 clean target 的程度
KD   : 学生输出接近 teacher waveform 的程度
Cons : 两段不同 aux 输出的一致性
```

理想趋势：

```text
1. Sup 逐渐下降，对应 SI-SDR 变好
2. KD 不应长期过大，说明学生在靠近 teacher
3. Cons 前期可能较大，后期应降低或稳定
4. Valid Loss 不应因为 KD/Cons 牺牲 clean target 性能
```

如果出现：

```text
KD 很低但 Sup 变差
```

说明 teacher 约束可能过强，可以降低 `lambda_kd`。

如果出现：

```text
Cons 很低但 SI-SDRi / PESQ 下降
```

说明一致性约束可能让模型过度平滑或压制细节，可以降低 `lambda_cons`。

## 9. 设计总结

WRCD-V2 的核心不是改变模型结构，而是改变训练约束：

```text
short wake aux: 让训练条件贴近端侧机器人短参考语音场景
teacher KD:     用强模型 waveform 输出指导轻量 TCN 学生
aux consistency:让同一目标说话人的不同短 aux 得到稳定输出
```

因此它的优势是：

```text
推理时参数量不变
推理时延迟基本不变
训练时利用 teacher 和一致性提升鲁棒性
```

从当前 wsj0-2mix 结果看，WRCD-V2 在保持 `12.9554M` 参数和 `RTF≈0.0084` 的同时，将 TCN-V2 的 `SI-SDRi` 从 `15.9210 dB` 提升到 `16.3953 dB`，说明训练侧蒸馏和一致性约束对轻量 TSE 后端是有效的。
