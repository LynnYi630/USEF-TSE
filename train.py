import os
import sys
import torch
sys.path.append('../../..')
import numpy as np
import logging
import argparse
import time
import random
import shutil
import inspect

from hyperpyyaml import load_hyperpyyaml
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist

from trainer.trainer import Trainer
# from utils.freeze import *

from dataset.data import tr_dataset, te_dataset

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.manual_seed_all(SEED)

def reserve_gpu_memory(device_id=0, size_in_gb=10):
    """
    在训练开始前预占指定大小的 GPU 显存
    """
    print(f"尝试在 GPU {device_id} 上预占 {size_in_gb} GB 显存...")
    try:
        # 1 GB = 1024 * 1024 * 1024 Bytes
        # 创建一个类型为 int8 (1 byte) 的空张量
        dummy_tensor = torch.empty(size_in_gb * 1024**3, dtype=torch.int8, device=f'cuda:{device_id}')
        
        # 删除张量引用，显存会被释放到 PyTorch 的缓存池中
        del dummy_tensor
        
        # 注意：千万不要调用 torch.cuda.empty_cache()！
        # 否则显存会被立刻归还给操作系统，占座就失败了。
        
        print(f"成功预留 {size_in_gb} GB 显存。在 nvidia-smi 中应该已经能看到占用。")
    except RuntimeError as e:
        print(f"预占显存失败，可能是当前 GPU 剩余显存不足 {size_in_gb} GB。")
        print(f"错误信息: {e}")

def main(config, args):
    # 在你初始化模型和加载数据之前调用它
    reserve_gpu_memory(device_id=args.device, size_in_gb=args.memory)

    log_dir = os.path.join('logs', config['name'])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, time.strftime('%Y-%m-%d-%H%M.log',time.localtime(time.time()))
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    trainset = tr_dataset(
        mix_scp = os.path.join(config['train_path'], config['mix_scp']),
        ref_scp = os.path.join(config['train_path'], config['ref_scp']),
        aux_scp = os.path.join(config['train_path'], config['aux_scp']),
        dur = config['duration'],
        fs = config['sample_rate'],
        dataset_type = config.get('dataset_type', 'wsj0-2mix')
    )

    trainloader = DataLoader(
            dataset=trainset, 
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            sampler=None,
    )
    
    validset = te_dataset(
        mix_scp = os.path.join(config['valid_path'], config['mix_scp']),
        ref_scp = os.path.join(config['valid_path'], config['ref_scp']),
        aux_scp = os.path.join(config['valid_path'], config['aux_scp']),
        fs = config['sample_rate']
    )

    validloader = DataLoader(
            dataset=validset, 
            batch_size=config['valid_batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True,
            sampler=None,
    )
    
    data = {'tr_loader': trainloader, 'cv_loader': validloader}

    model = config['modules']['masknet']
    
    model.cuda()
    
    logger.info(model)
    logger.info('-' * 50)

    model = torch.nn.DataParallel(model)

    chkpt_dir = os.path.join('chkpt', config['name'])
    os.makedirs(chkpt_dir, exist_ok=True)

    shutil.copyfile(args.config, os.path.join(chkpt_dir, 'config.yaml'))
    shutil.copyfile(inspect.getmodule(config['MaskNet'].__class__).__file__, os.path.join(chkpt_dir, 'model.py'))

    model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = config['optimizer'](params=model_params)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    trainer = Trainer(chkpt_dir = chkpt_dir,
                      data = data,
                      model = model,
                      optimizer = optimizer,
                      scheduler= lr_scheduler,
                      logger = logger,
                      config = config,
                      
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speech Separation: transformer')
    parser.add_argument('--config', default='config/config.yaml', type=str,
                        help='config file path (default: None)')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--memory', default=24, type=int)

    args = parser.parse_args()

    # Read config of the whole system.
    assert os.path.isfile(args.config), "No such file: %s" % args.config
    with open(args.config, 'r') as f:
        config_strings = f.read()
    config = load_hyperpyyaml(config_strings)
    print('INFO: Loaded hparams from: {}'.format(args.config))

    main(config, args)
