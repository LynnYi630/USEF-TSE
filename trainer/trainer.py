# -*- coding: utf-8 -*-

import os
import time
import torch
import torch.nn as nn
import numpy as np

from utils.losses import batchMean_sisnrLoss

class Trainer(object):
    
    def __init__(self, chkpt_dir, data, model, optimizer, scheduler, logger, config):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training config
        self.epochs = config['epochs']
        self.max_norm = config['max_norm']
        self.logger = logger
        # save and load model
        self.save_folder = chkpt_dir
        self.checkpoint = config['checkpoint']
        self.continue_from = config['continue_from']
        # logging
        self.print_freq = config['print_freq']
        self.freeze_frontend_warmup = bool(
            config.get('freeze_frontend_warmup', bool(self.continue_from))
        )
        self.frontend_warmup_epochs = int(config.get('frontend_warmup_epochs', 10))
        self.frontend_unfreeze_lr_scale = float(
            config.get('frontend_unfreeze_lr_scale', 0.1)
        )
        if self.frontend_warmup_epochs < 0:
            raise ValueError('frontend_warmup_epochs must be >= 0')
        os.makedirs(self.save_folder, exist_ok=True)
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            # 加上 weights_only=False 规避 PyTorch 2.6 安全报错
            cont = torch.load(self.continue_from, weights_only=False) 
            self.start_epoch = cont.get('epoch', 0)
            
            # 核心：加上 strict=False 允许只加载部分匹配的权重（迁移学习必须）
            self.model.load_state_dict(cont['model_state_dict'], strict=False)
            
            # 安全检查：只有当存在优化器状态且不为空时才加载
            if 'optimizer_state' in cont and cont['optimizer_state'] is not None:
                self.optimizer.load_state_dict(cont['optimizer_state'])
                print("成功加载历史优化器状态。")
            else:
                print("未检测到优化器状态，将随机初始化优化器（适用于迁移学习启动）。")
                
            # 安全检查：随机数种子
            if 'trandom_state' in cont and cont['trandom_state'] is not None:
                torch.set_rng_state((cont['trandom_state']))
            if 'nrandom_state' in cont and cont['nrandom_state'] is not None:
                np.random.set_state((cont['nrandom_state']))

    # ===== [新增 1/2]：前端模块的冻结与解冻控制函数 =====
    def _freeze_frontend(self):
        """冻结前端模块，仅训练后端（如Mamba/TCN）"""
        if getattr(self, '_frontend_frozen', False):
            return
        modules_to_freeze = []
        if hasattr(self.model, 'encoder'):
            modules_to_freeze.extend([self.model.encoder, self.model.decoder, self.model.fusion_mdl, self.model.film])
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'encoder'):
            # 兼容 DataParallel 多卡并行模式
            modules_to_freeze.extend([self.model.module.encoder, self.model.module.decoder, self.model.module.fusion_mdl, self.model.module.film])
            
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
                
        self.logger.info("==================================================")
        self.logger.info("-> [状态切换] 前端模块已冻结，当前仅训练后端网络及掩码层。")
        self.logger.info("==================================================")
        self._frontend_frozen = True
        self._frontend_unfrozen = False

    def _unfreeze_frontend(self):
        """解冻前端模块，开启全网络端到端微调"""
        if getattr(self, '_frontend_unfrozen', False):
            return
        modules_to_unfreeze = []
        if hasattr(self.model, 'encoder'):
            modules_to_unfreeze.extend([self.model.encoder, self.model.decoder, self.model.fusion_mdl, self.model.film])
        elif hasattr(self.model, 'module') and hasattr(self.model.module, 'encoder'):
            modules_to_unfreeze.extend([self.model.module.encoder, self.model.module.decoder, self.model.module.fusion_mdl, self.model.module.film])
            
        for module in modules_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True
                
        self.logger.info("==================================================")
        self.logger.info("-> [状态切换] 前端模块已解冻，模型进入全网络协同微调阶段。")
        self.logger.info("==================================================")
        self._frontend_frozen = False
        self._frontend_unfrozen = True
    # ===================================================

    def _run_train_epoch(self, epoch):

        start = time.time()
        total_loss = 0
        data_loader = self.tr_loader

        for i, (data) in enumerate(data_loader):

            mixture, source, embd, ilens = data
            mixture = mixture.cuda()
            source = source.cuda()
            embd = embd.cuda()
            ilens = ilens.cuda()
            
            estimate_source = self.model(mixture, embd)

            loss = batchMean_sisnrLoss(estimate_source, source)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.max_norm)
            self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                self.logger.info('Epoch {0:3d} | Iter {1:5d} | Average Loss {2:3.3f} | '
                    'Current Loss {3:3.6f} | {4:5.1f} ms/batch'.format(
                        epoch + 1, i + 1, total_loss / (i + 1),
                        loss.item(), 1000 * (time.time() - start) / (i + 1)))

        return total_loss / (i + 1)
    
    def _run_valid_epoch(self, epoch):

        start = time.time()
        total_loss = 0
        data_loader = self.cv_loader

        for i, (data) in enumerate(data_loader):

            mixture, source, embd, ilens = data
            mixture = mixture.cuda()
            source = source.cuda()
            embd = embd.cuda()
            ilens = ilens.cuda()
            
            estimate_source = self.model(mixture, embd)
            min_len = min(estimate_source.shape[1], source.shape[1])
            loss = batchMean_sisnrLoss(estimate_source[:,:min_len], source[:,:min_len])

            total_loss += loss.item()

            if i % self.print_freq == 0:
                self.logger.info('Epoch {0:3d} | Iter {1:5d} | Average Valid Loss {2:3.3f} | '
                    'Current Valid Loss {3:3.6f} | {4:5.1f} ms/batch'.format(
                        epoch + 1, i + 1, total_loss / (i + 1),
                        loss.item(), 1000 * (time.time() - start) / (i + 1)))

        return total_loss / (i + 1)
    
    def train(self):
        # 前端冻结 warm-up 由 config 控制。

        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):

            # ===== [新增 2/2]：在每一个 Epoch 开始前，根据状态机判断是否需要解冻 =====
            if (
                self.freeze_frontend_warmup
                and self.frontend_warmup_epochs > 0
                and epoch < self.frontend_warmup_epochs
            ):
                self._freeze_frontend()
            elif (
                self.freeze_frontend_warmup
                and self.frontend_warmup_epochs > 0
                and epoch == self.frontend_warmup_epochs
            ):
                self._unfreeze_frontend()
                # 首次解冻时，将学习率衰减，防止破坏前端的预训练权重
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.frontend_unfreeze_lr_scale
                self.logger.info(
                    "-> [学习率调整] 解冻前端，整体学习率已乘以 %.4g。"
                    % self.frontend_unfreeze_lr_scale
                )
            elif (
                self.freeze_frontend_warmup
                and self.frontend_warmup_epochs > 0
                and epoch > self.frontend_warmup_epochs
            ):
                self._unfreeze_frontend() # 确保断点恢复后依然是解冻状态
            # ===================================================================

            else:
                self._unfreeze_frontend()

            optim_state = self.optimizer.state_dict()
            self.logger.info('epoch start Learning rate: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
            self.logger.info("Training...")
            
            # train stage
            self.model.train()
            start = time.time()
            tr_loss = self._run_train_epoch(epoch)

            # train log
            self.logger.info('-' * 85)
            self.logger.info('Train Summary | End of Epoch {0:5d} | Time {1:.2f}s | '
                'Train Loss {2:.3f}'.format(
                    epoch + 1, time.time() - start, tr_loss))
            self.logger.info('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, file_path)
                self.logger.info('Saving checkpoint model to %s' % file_path)

            # validation stage
            self.logger.info('Cross validation...')
            
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_valid_epoch(epoch)
                
            # val log
            self.logger.info('-' * 85)
            self.logger.info('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                'Valid Loss {2:.3f}'.format(
                    epoch + 1, time.time() - start, val_loss))
            self.logger.info('-' * 85)

            # save the temp_best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_file_path = os.path.join(
                    self.save_folder, 'temp_best.pth.tar')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, best_file_path)
                self.logger.info("Find better validated model, saving to %s" % best_file_path)
            
            self.scheduler.step(val_loss)
