# -*- coding: utf-8 -*-

import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from utils.losses import batchMean_sisnrLoss, sisnr


class Trainer(object):
    """训练主循环。

    这个 Trainer 目前承载了几条相互独立的实验改进：
    1. WRCD-V2：短 wake aux/enrollment、teacher waveform KD、双 aux 一致性约束。
    2. TCCTCN-V2：target-conditioned TCN 的 conditioning reset，以及 anti-interferer SI-SNR loss。
    3. TCCTCN-V4：把旧 checkpoint 作为初始化重新训练，以及 target-interferer margin loss。
    4. Valid chunking：validation 阶段按块跑长语音，避免 SepFormer/Cross-Attention OOM。

    """

    def __init__(self, chkpt_dir, data, model, optimizer, scheduler, logger, config):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

        # ===== 通用训练配置：所有模型共享 =====
        self.epochs = config['epochs']
        self.max_norm = config['max_norm']
        self.save_folder = chkpt_dir
        self.checkpoint = config['checkpoint']
        self.continue_from = config['continue_from']
        self.print_freq = config['print_freq']
        os.makedirs(self.save_folder, exist_ok=True)

        self.best_val_loss = float("inf")
        self.start_epoch = 0

        # ===== 通用迁移初始化：把旧 checkpoint 当作初始化，而不是断点续训 =====
        # reset_epoch_on_continue=True 时，continue_from 只加载模型权重；
        # epoch、optimizer、随机数状态都会重新开始，适合 V4 从 TCN/WRCD 稳定主干微调。
        self.reset_epoch_on_continue = bool(
            config.get('reset_epoch_on_continue', False)
        )
        self.load_optimizer_state = bool(
            config.get('load_optimizer_state', not self.reset_epoch_on_continue)
        )

        # ===== Mixed precision: keeps model forward fast while losses stay fp32 =====
        self.use_amp = bool(config.get('use_amp', False))
        self.amp_dtype = self._parse_amp_dtype(config.get('amp_dtype', 'float16'))
        self.grad_scaler = self._make_grad_scaler(
            self.use_amp and self.amp_dtype == torch.float16
        )

        # ===== Valid chunking：只影响 validation，不影响训练和 eval.py =====
        # 目的：CV 阶段数据集可以返回全长语音，但模型按 chunk 前向，降低 attention 显存峰值。
        # 默认使用训练 duration 作为 chunk 长度；如果配置 valid_chunk_duration <= 0，则关闭分块。
        self.sample_rate = int(config.get('sample_rate', 0))
        self.valid_chunk_duration = float(
            config.get('valid_chunk_duration', config.get('duration', 0))
        )
        self.valid_chunk_hop_duration = float(
            config.get('valid_chunk_hop_duration', self.valid_chunk_duration)
        )
        self.valid_chunk_len = 0
        self.valid_chunk_hop = 0
        self._init_valid_chunking()

        # ===== 通用迁移学习 warm-up：冻结/解冻前端 =====
        # 这块不专属于 WRCD 或 TCCTCN；用于继续训练时先保护 encoder/decoder/fusion/FiLM。
        self.freeze_frontend_warmup = bool(
            config.get('freeze_frontend_warmup', bool(self.continue_from))
        )
        self.frontend_warmup_epochs = int(config.get('frontend_warmup_epochs', 10))
        self.frontend_unfreeze_lr_scale = float(
            config.get('frontend_unfreeze_lr_scale', 0.1)
        )
        self.separate_frontend_backend_lr = bool(
            config.get('separate_frontend_backend_lr', False)
        )

        # ===== 永久冻结 backbone（V4 adapter-only 微调用）=====
        # 与 frontend warmup 不同：这是“训练全程都冻”，不是 warmup。
        # 触发条件：config.freeze_backbone=True 且 model 暴露 freeze_backbone() 方法。
        self.freeze_backbone_full = bool(config.get('freeze_backbone', False))

        # ===== WRCD-V2：teacher KD + wake aux consistency =====
        # use_wrcd: 训练 batch 会额外带 teacher waveform。
        # use_wrcd_consistency: 训练 batch 会再额外带第二个 wake aux，用于一致性约束。
        self.use_wrcd = bool(config.get('use_wrcd', False))
        self.use_wrcd_consistency = bool(config.get('use_wrcd_consistency', False))
        self.lambda_kd = float(config.get('lambda_kd', 0.0))
        self.lambda_cons = float(config.get('lambda_cons', 0.0))

        # ===== TCCTCN-V2/V4：target-confusion 相关训练项 =====
        # lambda_anti: 抑制模型靠近干扰说话人，缓解 target-swap。
        # lambda_confusion: 直接约束 est-target 分数高于 est-interferer 分数。
        # confusion_margin: 要求 target 相似度至少比 interferer 相似度高出的 margin。
        # reset_tc_conditioning_on_continue: 从旧 checkpoint 迁移时重置新加的 conditioning path。
        self.lambda_anti = float(config.get('lambda_anti', 0.0))
        self.anti_sisnr_margin = float(config.get('anti_sisnr_margin', 0.0))
        self.lambda_confusion = float(config.get('lambda_confusion', 0.0))
        self.confusion_margin = float(config.get('confusion_margin', 0.0))
        self.reset_tc_conditioning_on_continue = bool(
            config.get('reset_tc_conditioning_on_continue', False)
        )
        self.tc_cond_scale_init = float(config.get('tc_cond_scale_init', 1.0))
        self.tc_zero_init_cond_proj = bool(config.get('tc_zero_init_cond_proj', False))

        self._validate_config()

        if self.valid_chunk_len > 0:
            self.logger.info(
                'Validation chunking enabled: chunk %.2fs, hop %.2fs'
                % (self.valid_chunk_duration, self.valid_chunk_hop_duration)
            )

        if self.continue_from:
            self._load_checkpoint()

        if self.use_amp:
            self.logger.info(
                'AMP enabled for model forward (dtype=%s). Losses are computed in fp32.'
                % str(self.amp_dtype).replace('torch.', '')
            )

    # -------------------------------------------------------------------------
    # 配置校验与 checkpoint 加载
    # -------------------------------------------------------------------------
    def _init_valid_chunking(self):
        """[Valid chunking] 将秒级配置换算成采样点。"""
        if self.valid_chunk_duration <= 0:
            return
        if self.sample_rate <= 0:
            raise ValueError('sample_rate is required for valid chunking')

        self.valid_chunk_len = int(round(self.valid_chunk_duration * self.sample_rate))
        self.valid_chunk_hop = int(round(self.valid_chunk_hop_duration * self.sample_rate))
        if self.valid_chunk_len <= 0 or self.valid_chunk_hop <= 0:
            raise ValueError('valid chunk duration and hop must be positive')
        if self.valid_chunk_hop > self.valid_chunk_len:
            raise ValueError('valid_chunk_hop_duration must be <= valid_chunk_duration')

    def _validate_config(self):
        """配置合法性检查。"""
        if self.lambda_kd < 0:
            raise ValueError('lambda_kd must be >= 0')
        if self.lambda_cons < 0:
            raise ValueError('lambda_cons must be >= 0')
        if self.lambda_anti < 0:
            raise ValueError('lambda_anti must be >= 0')
        if self.lambda_confusion < 0:
            raise ValueError('lambda_confusion must be >= 0')
        if self.frontend_warmup_epochs < 0:
            raise ValueError('frontend_warmup_epochs must be >= 0')

    def _parse_amp_dtype(self, amp_dtype):
        amp_dtype = str(amp_dtype).lower()
        if amp_dtype in ('bf16', 'bfloat16'):
            return torch.bfloat16
        if amp_dtype in ('fp16', 'float16', 'half'):
            return torch.float16
        raise ValueError('amp_dtype must be "float16" or "bfloat16"')

    def _make_grad_scaler(self, enabled):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            try:
                return torch.amp.GradScaler('cuda', enabled=enabled)
            except TypeError:
                return torch.amp.GradScaler(enabled=enabled)
        return torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            return torch.amp.autocast(
                device_type='cuda',
                dtype=self.amp_dtype,
                enabled=True,
            )
        return torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=True)

    def _load_checkpoint(self):
        """通用 checkpoint 恢复；其中包含 TCCTCN-V2 的 conditioning reset。"""
        print('Loading checkpoint model %s' % self.continue_from)
        # PyTorch 2.6+ 默认 weights_only=True，旧 checkpoint 需要显式关闭。
        cont = torch.load(self.continue_from, weights_only=False)
        self.start_epoch = 0 if self.reset_epoch_on_continue else cont.get('epoch', 0)

        # 通用迁移学习：strict=False 允许只加载形状匹配的旧权重。
        # V4 新增 adapter 没有旧权重，会保持随机/零初始化；TCN 主干可从旧模型继承。
        self.model.load_state_dict(cont['model_state_dict'], strict=False)

        # ===== TCCTCN-V2：迁移旧模型时重置 target-conditioned path =====
        if self.reset_tc_conditioning_on_continue:
            self._reset_tc_conditioning()

        # 通用保护：如果 optimizer state 与当前模型不兼容，就跳过而不是中断训练。
        if (
            self.load_optimizer_state
            and 'optimizer_state' in cont
            and cont['optimizer_state'] is not None
        ):
            try:
                self.optimizer.load_state_dict(cont['optimizer_state'])
                print("optimizer state loaded")
            except ValueError as exc:
                print("optimizer state is incompatible; reinitializing optimizer")
                print("optimizer_state load error: {}".format(exc))
        else:
            print("未检测到优化器状态，将随机初始化优化器（适用于迁移学习启动）。")

        if self.reset_epoch_on_continue:
            print("checkpoint 仅作为初始化使用；epoch/optimizer/RNG 状态已重置")
            return

        if 'trandom_state' in cont and cont['trandom_state'] is not None:
            torch.set_rng_state(cont['trandom_state'])
        if 'nrandom_state' in cont and cont['nrandom_state'] is not None:
            np.random.set_state(cont['nrandom_state'])
        if (
            self.grad_scaler.is_enabled()
            and cont.get('amp_scaler_state') is not None
        ):
            self.grad_scaler.load_state_dict(cont['amp_scaler_state'])

    # -------------------------------------------------------------------------
    # TCCTCN-V2：target-conditioned path 初始化/重置
    # -------------------------------------------------------------------------
    def _reset_tc_conditioning(self):
        """[TCCTCN-V2] 重置新加的 conditioning path，避免旧 checkpoint 污染新结构。"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        backend = getattr(model, 'tcn_backend', None)
        blocks = getattr(backend, 'blocks', None)
        if blocks is None:
            return

        for block in blocks:
            if self.tc_zero_init_cond_proj:
                for name, module in block.named_modules():
                    if name.endswith('cond_proj') and isinstance(module, nn.Conv1d):
                        nn.init.zeros_(module.weight)
                        nn.init.zeros_(module.bias)

            cond_scale = getattr(block, 'cond_scale', None)
            if cond_scale is not None:
                with torch.no_grad():
                    cond_scale.fill_(self.tc_cond_scale_init)

        self.logger.info(
            'Reset TCCTCN conditioning path after checkpoint load '
            '(cond_scale_init=%.4f, zero_init_cond_proj=%s).'
            % (self.tc_cond_scale_init, self.tc_zero_init_cond_proj)
        )

    # -------------------------------------------------------------------------
    # 通用迁移学习：前端冻结/解冻 warm-up
    # -------------------------------------------------------------------------
    def _frontend_modules(self):
        """返回需要 warm-up 冻结的前端模块，兼容 DataParallel。"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        modules = []
        for name in ('encoder', 'decoder', 'conv1d1', 'fusion_mdl', 'fusion_norm', 'film'):
            module = getattr(model, name, None)
            if module is not None:
                modules.append(module)
        return modules

    def _grad_norm_for_params(self, params):
        """Return the pre-clipping L2 grad norm for a parameter iterable."""
        norms = []
        with torch.no_grad():
            for param in params:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if grad.is_sparse:
                    grad = grad.coalesce().values()
                norms.append(torch.linalg.vector_norm(grad.float(), 2).cpu())

            if not norms:
                return None
            return float(torch.linalg.vector_norm(torch.stack(norms), 2).item())

    def _module_grad_norms(self):
        """Collect pre-clipping grad norms for coarse model regions."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        groups = (
            ('enc', ('encoder',)),
            ('dec', ('decoder',)),
            ('pre', ('norm_m', 'conv1d1')),
            ('fusion', ('fusion_mdl', 'fusion_norm')),
            ('film', ('film',)),
            ('backend', ('backend', 'tcn_backend')),
            ('head', ('skip_prelu', 'skip_norm', 'mask_conv1x1')),
        )

        module_norms = []
        used_param_ids = set()
        for group_name, module_names in groups:
            params = []
            for module_name in module_names:
                module = getattr(model, module_name, None)
                if module is None:
                    continue
                for param in module.parameters():
                    param_id = id(param)
                    if param_id in used_param_ids:
                        continue
                    params.append(param)
                    used_param_ids.add(param_id)
            if params:
                module_norms.append((group_name, self._grad_norm_for_params(params)))

        other_params = [
            param for param in model.parameters()
            if id(param) not in used_param_ids
        ]
        other_norm = self._grad_norm_for_params(other_params)
        if other_norm is not None:
            module_norms.append(('other', other_norm))

        return module_norms

    def _clip_ratio_from_grad_norm(self, grad_norm):
        if grad_norm <= 0:
            return 1.0
        if not np.isfinite(grad_norm):
            return 0.0
        return min(1.0, float(self.max_norm) / (grad_norm + 1e-12))

    def _format_module_grad_norms(self, module_grad_norms):
        if not module_grad_norms:
            return 'ModuleGradNorms none'
        parts = []
        for name, value in module_grad_norms:
            if value is None:
                parts.append('%s=-' % name)
            else:
                parts.append('%s=%.3f' % (name, value))
        return 'ModuleGradNorms ' + ', '.join(parts)

    def _freeze_frontend(self):
        """[通用 warm-up] 冻结前端模块，仅训练后端或新加模块。"""
        if getattr(self, '_frontend_frozen', False):
            return

        for module in self._frontend_modules():
            for param in module.parameters():
                param.requires_grad = False

        self.logger.info("-> [状态切换] 前端模块已冻结，当前仅训练后端/新增模块。")
        self._frontend_frozen = True
        self._frontend_unfrozen = False

    def _unfreeze_frontend(self):
        """[通用 warm-up] 解冻前端模块，进入全网络端到端训练。"""
        if getattr(self, '_frontend_unfrozen', False):
            return

        for module in self._frontend_modules():
            for param in module.parameters():
                param.requires_grad = True

        self.logger.info("-> [状态切换] 前端模块已解冻，模型进入全网络协同微调阶段。")
        self._frontend_frozen = False
        self._frontend_unfrozen = True

    def _scale_lr_after_frontend_unfreeze(self):
        """解冻前端后的学习率调整。

        默认旧逻辑：所有 param_group 都乘 frontend_unfreeze_lr_scale。
        若启用 separate_frontend_backend_lr，则只调整 name='frontend' 的参数组，
        backend 组保持原学习率，避免新后端/adapter 学习变慢。
        """
        if self.separate_frontend_backend_lr:
            scaled = False
            for param_group in self.optimizer.param_groups:
                if param_group.get('name') != 'frontend':
                    continue
                param_group['lr'] *= self.frontend_unfreeze_lr_scale
                scaled = True

            if scaled:
                self.logger.info(
                    "-> [学习率调整] 解冻前端，仅 frontend 学习率乘以 %.4g，backend 学习率保持不变。"
                    % self.frontend_unfreeze_lr_scale
                )
            else:
                self.logger.info(
                    "-> [学习率调整] 未找到 frontend 参数组，跳过单独前端学习率调整。"
                )
            return

        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.frontend_unfreeze_lr_scale
        self.logger.info(
            "-> [学习率调整] 解冻前端，整体学习率已乘以 %.4g。"
            % self.frontend_unfreeze_lr_scale
        )

    def _apply_frontend_schedule(self, epoch):
        """[通用 warm-up] 每个 epoch 开始前决定是否冻结/解冻。"""
        if (
            self.freeze_frontend_warmup
            and self.frontend_warmup_epochs > 0
            and epoch < self.frontend_warmup_epochs
        ):
            self._freeze_frontend()
            return

        if (
            self.freeze_frontend_warmup
            and self.frontend_warmup_epochs > 0
            and epoch == self.frontend_warmup_epochs
        ):
            self._unfreeze_frontend()
            self._scale_lr_after_frontend_unfreeze()
            return

        if (
            self.freeze_frontend_warmup
            and self.frontend_warmup_epochs > 0
            and epoch > self.frontend_warmup_epochs
        ):
            self._unfreeze_frontend()
            return

        self._unfreeze_frontend()

    # -------------------------------------------------------------------------
    # WRCD-V2 / TCCTCN-V2：训练 batch 与 loss 计算
    # -------------------------------------------------------------------------
    def _unpack_train_batch(self, data):
        """解析训练 batch。

        基础 batch:
            mixture, source, embd, ilens

        WRCD-V2 teacher KD:
            + teacher

        WRCD-V2 consistency:
            + embd_cons
        """
        if len(data) == 6:
            mixture, source, embd, ilens, teacher, embd_cons = data
        elif len(data) == 5:
            mixture, source, embd, ilens, teacher = data
            embd_cons = None
        else:
            mixture, source, embd, ilens = data
            teacher = None
            embd_cons = None
        return mixture, source, embd, ilens, teacher, embd_cons

    def _move_train_batch_to_cuda(self, mixture, source, embd, ilens, teacher, embd_cons):
        mixture = mixture.cuda()
        source = source.cuda()
        embd = embd.cuda()
        ilens = ilens.cuda()
        if teacher is not None:
            teacher = teacher.cuda()
        if embd_cons is not None:
            embd_cons = embd_cons.cuda()
        return mixture, source, embd, ilens, teacher, embd_cons

    def _compute_train_loss(self, mixture, source, embd, teacher, embd_cons):
        """计算训练 loss，并显式分隔各实验项。"""
        with self._autocast():
            estimate_source = self.model(mixture, embd)

            # ===== WRCD-V2：第二个 wake aux 的一致性前向 =====
            estimate_source_cons = None
            if embd_cons is not None and self.lambda_cons > 0:
                with torch.no_grad():
                    estimate_source_cons = self.model(mixture, embd_cons)

        min_len = min(estimate_source.shape[1], source.shape[1])
        if teacher is not None:
            min_len = min(min_len, teacher.shape[1])
        if estimate_source_cons is not None:
            min_len = min(min_len, estimate_source_cons.shape[1])

        estimate_source = estimate_source[:, :min_len]
        source = source[:, :min_len]
        estimate_source_loss = estimate_source.float()
        source_loss = source.float()

        # ===== 基础监督项：所有实验共享 =====
        loss_sup = batchMean_sisnrLoss(estimate_source_loss, source_loss)
        loss = loss_sup
        metrics = {
            'sup': loss_sup,
            'kd': None,
            'cons': None,
            'anti': None,
            'conf': None,
        }

        # ===== WRCD-V2：teacher waveform KD =====
        if teacher is not None and self.lambda_kd > 0:
            teacher = teacher[:, :min_len].float()
            metrics['kd'] = batchMean_sisnrLoss(estimate_source_loss, teacher)
            loss = loss + self.lambda_kd * metrics['kd']

        # ===== WRCD-V2：双 wake aux 输出一致性 =====
        if estimate_source_cons is not None and self.lambda_cons > 0:
            estimate_source_cons = estimate_source_cons[:, :min_len].float()
            metrics['cons'] = torch.mean(
                torch.abs(estimate_source_loss - estimate_source_cons.detach())
            )
            loss = loss + self.lambda_cons * metrics['cons']

        # ===== TCCTCN-V2：anti-interferer SI-SNR，降低 target-swap 风险 =====
        if self.lambda_anti > 0:
            interferer = mixture[:, :min_len].float() - source_loss
            anti_score = sisnr(estimate_source_loss, interferer)
            metrics['anti'] = torch.relu(anti_score - self.anti_sisnr_margin).mean()
            loss = loss + self.lambda_anti * metrics['anti']

        # ===== TCCTCN-V4：target-interferer margin，直接抑制 target-confusion =====
        # anti loss 只问 est 像不像 interferer；这里进一步要求：
        # SI-SDR(est, target) >= SI-SDR(est, interferer) + confusion_margin。
        # 当输出更接近干扰人时，该项才明显触发。
        if self.lambda_confusion > 0:
            interferer = mixture[:, :min_len].float() - source_loss
            target_score = sisnr(estimate_source_loss, source_loss)
            interferer_score = sisnr(estimate_source_loss, interferer)
            metrics['conf'] = torch.relu(
                self.confusion_margin + interferer_score - target_score
            ).mean()
            loss = loss + self.lambda_confusion * metrics['conf']

        return loss, metrics

    def _log_train_iter(self, epoch, i, average_loss, loss, metrics, start, grad_stats):
        if grad_stats is None:
            clip_ratio = 1.0
            module_grad_text = self._format_module_grad_norms(None)
        else:
            clip_ratio = grad_stats['clip_ratio']
            module_grad_text = self._format_module_grad_norms(
                grad_stats['module_grad_norms']
            )

        if (
            metrics['kd'] is None
            and metrics['cons'] is None
            and metrics['anti'] is None
            and metrics['conf'] is None
        ):
            self.logger.info(
                'Epoch {0:3d} | Iter {1:5d} | Average Loss {2:3.3f} | '
                'Current Loss {3:3.6f} | ClipRatio {4:3.3f} | {5} | '
                '{6:5.1f} ms/batch'.format(
                    epoch + 1,
                    i + 1,
                    average_loss,
                    loss.item(),
                    clip_ratio,
                    module_grad_text,
                    1000 * (time.time() - start) / (i + 1),
                )
            )
            return

        kd_value = metrics['kd'].item() if metrics['kd'] is not None else 0.0
        cons_value = metrics['cons'].item() if metrics['cons'] is not None else 0.0
        anti_value = metrics['anti'].item() if metrics['anti'] is not None else 0.0
        conf_value = metrics['conf'].item() if metrics['conf'] is not None else 0.0
        self.logger.info(
            'Epoch {0:3d} | Iter {1:5d} | Average Loss {2:3.3f} | '
            'Current Loss {3:3.6f} | Sup {4:3.6f} | KD {5:3.6f} | '
            'Cons {6:3.6f} | Anti {7:3.6f} | Conf {8:3.6f} | '
            'ClipRatio {9:3.3f} | {10} | {11:5.1f} ms/batch'.format(
                epoch + 1,
                i + 1,
                average_loss,
                loss.item(),
                metrics['sup'].item(),
                kd_value,
                cons_value,
                anti_value,
                conf_value,
                clip_ratio,
                module_grad_text,
                1000 * (time.time() - start) / (i + 1),
            )
        )

    def _run_train_epoch(self, epoch):
        start = time.time()
        total_loss = 0
        num_finite_loss = 0
        total_sup_loss = 0
        num_sup_loss = 0
        self.last_train_sup_loss = None

        for i, data in enumerate(self.tr_loader):
            batch = self._unpack_train_batch(data)
            mixture, source, embd, ilens, teacher, embd_cons = (
                self._move_train_batch_to_cuda(*batch)
            )

            loss, metrics = self._compute_train_loss(
                mixture, source, embd, teacher, embd_cons
            )

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                self.logger.warning(
                    'Skipping non-finite train loss at epoch %d iter %d: %s'
                    % (epoch + 1, i + 1, str(loss.item()))
                )
                continue

            log_this_iter = i % self.print_freq == 0
            module_grad_norms = None

            self.optimizer.zero_grad(set_to_none=True)
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                if log_this_iter:
                    module_grad_norms = self._module_grad_norms()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_norm,
                )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                if log_this_iter:
                    module_grad_norms = self._module_grad_norms()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_norm,
                )
                self.optimizer.step()
            grad_norm = float(grad_norm.detach().cpu())
            grad_stats = None
            if log_this_iter:
                grad_stats = {
                    'clip_ratio': self._clip_ratio_from_grad_norm(grad_norm),
                    'module_grad_norms': module_grad_norms,
                }

            total_loss += loss.item()
            num_finite_loss += 1
            has_extra_metrics = (
                metrics['kd'] is not None
                or metrics['cons'] is not None
                or metrics['anti'] is not None
                or metrics['conf'] is not None
            )
            if has_extra_metrics:
                sup_loss = metrics['sup'].detach()
                if torch.isfinite(sup_loss):
                    total_sup_loss += sup_loss.item()
                    num_sup_loss += 1

            if log_this_iter:
                self._log_train_iter(
                    epoch,
                    i,
                    total_loss / max(num_finite_loss, 1),
                    loss,
                    metrics,
                    start,
                    grad_stats,
                )

        if num_sup_loss > 0:
            self.last_train_sup_loss = total_sup_loss / num_sup_loss

        return total_loss / max(num_finite_loss, 1)

    # -------------------------------------------------------------------------
    # Valid chunking：validation 长语音分块前向
    # -------------------------------------------------------------------------
    def _valid_chunk_starts(self, total_len):
        """[Valid chunking] 给定总长度，生成覆盖整句的 chunk 起点。"""
        if self.valid_chunk_len <= 0 or total_len <= self.valid_chunk_len:
            return [0]

        starts = list(range(0, total_len - self.valid_chunk_len + 1, self.valid_chunk_hop))
        tail_start = total_len - self.valid_chunk_len
        if starts[-1] != tail_start:
            starts.append(tail_start)
        return starts

    def _estimate_valid_source(self, mixture, embd):
        """[Valid chunking] validation 阶段按块估计，再对重叠区域平均。"""
        total_len = mixture.shape[1]
        if self.valid_chunk_len <= 0 or total_len <= self.valid_chunk_len:
            return self.model(mixture, embd)

        estimate = mixture.new_zeros(mixture.shape[0], total_len)
        weight = mixture.new_zeros(mixture.shape[0], total_len)
        for start in self._valid_chunk_starts(total_len):
            end = min(start + self.valid_chunk_len, total_len)
            chunk_estimate = self.model(mixture[:, start:end], embd)
            chunk_len = min(chunk_estimate.shape[1], end - start)
            estimate[:, start:start + chunk_len] += chunk_estimate[:, :chunk_len]
            weight[:, start:start + chunk_len] += 1

        return estimate / weight.clamp_min(1)

    def _run_valid_epoch(self, epoch):
        start = time.time()
        total_loss = 0

        for i, data in enumerate(self.cv_loader):
            mixture, source, embd, ilens = data
            mixture = mixture.cuda()
            source = source.cuda()
            embd = embd.cuda()
            ilens = ilens.cuda()

            with self._autocast():
                estimate_source = self._estimate_valid_source(mixture, embd)
            min_len = min(estimate_source.shape[1], source.shape[1])
            loss = batchMean_sisnrLoss(
                estimate_source[:, :min_len].float(),
                source[:, :min_len].float(),
            )

            total_loss += loss.item()

            if i % self.print_freq == 0:
                self.logger.info(
                    'Epoch {0:3d} | Iter {1:5d} | Average Valid Loss {2:3.3f} | '
                    'Current Valid Loss {3:3.6f} | {4:5.1f} ms/batch'.format(
                        epoch + 1,
                        i + 1,
                        total_loss / (i + 1),
                        loss.item(),
                        1000 * (time.time() - start) / (i + 1),
                    )
                )

        return total_loss / (i + 1)

    # -------------------------------------------------------------------------
    # 通用训练主循环
    # -------------------------------------------------------------------------
    def _save_checkpoint(self, file_path, epoch):
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'amp_scaler_state': (
                    self.grad_scaler.state_dict()
                    if self.grad_scaler.is_enabled()
                    else None
                ),
                'trandom_state': torch.get_rng_state(),
                'nrandom_state': np.random.get_state(),
            },
            file_path,
        )

    def _apply_backbone_freeze(self):
        """[V4-FrozenBackbone] 每个 epoch 开始前强制冻结 TCN backbone。

        与 frontend warmup 不同，这里是“训练全程冻”，不会解冻。
        要求 model 暴露 freeze_backbone() 方法。
        """
        if not self.freeze_backbone_full:
            return
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if not hasattr(model, 'freeze_backbone'):
            return
        # 仅首次冻结时打印日志，避免每个 epoch 刷屏。
        already_logged = getattr(self, '_backbone_freeze_logged', False)
        changed = model.freeze_backbone(freeze=True)
        if not already_logged:
            self.logger.info(
                '-> [Backbone Freeze] TCN backbone permanently frozen '
                '(%d params switched to requires_grad=False).' % changed
            )
            self._backbone_freeze_logged = True

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self._apply_frontend_schedule(epoch)
            self._apply_backbone_freeze()

            lr_parts = []
            for group_idx, param_group in enumerate(self.optimizer.param_groups):
                group_name = param_group.get('name', 'group%d' % group_idx)
                lr_parts.append('%s=%.6f' % (group_name, param_group['lr']))
            if any('name' in group for group in self.optimizer.param_groups):
                self.logger.info(
                    'epoch start Learning rate: %s' % ', '.join(lr_parts)
                )
            else:
                self.logger.info(
                    'epoch start Learning rate: %.6f'
                    % self.optimizer.param_groups[0]['lr']
                )
            self.logger.info("Training...")

            # ===== Training stage =====
            self.model.train()
            start = time.time()
            tr_loss = self._run_train_epoch(epoch)

            self.logger.info('-' * 85)
            self.logger.info(
                'Train Summary | End of Epoch {0:5d} | Time {1:.2f}s | '
                'Train Loss {2:.3f}'.format(
                    epoch + 1,
                    time.time() - start,
                    tr_loss,
                )
            )
            if self.last_train_sup_loss is not None:
                self.logger.info(
                    'Train Sup Summary | End of Epoch {0:5d} | '
                    'Train Sup Loss {1:.3f}'.format(
                        epoch + 1,
                        self.last_train_sup_loss,
                    )
                )
            self.logger.info('-' * 85)

            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder,
                    'epoch%d.pth.tar' % (epoch + 1),
                )
                self._save_checkpoint(file_path, epoch)
                self.logger.info('Saving checkpoint model to %s' % file_path)

            # ===== Validation stage：这里会走 Valid chunking =====
            self.logger.info('Cross validation...')
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_valid_epoch(epoch)

            self.logger.info('-' * 85)
            self.logger.info(
                'Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                'Valid Loss {2:.3f}'.format(
                    epoch + 1,
                    time.time() - start,
                    val_loss,
                )
            )
            self.logger.info('-' * 85)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_file_path = os.path.join(self.save_folder, 'temp_best.pth.tar')
                self._save_checkpoint(best_file_path, epoch)
                self.logger.info(
                    "Find better validated model, saving to %s" % best_file_path
                )

            self.scheduler.step(val_loss)
