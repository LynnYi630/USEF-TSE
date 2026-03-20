import os
import sys
sys.path.append('../../..')
import argparse
import torch
import numpy as np
from tqdm import tqdm
from mir_eval.separation import bss_eval_sources
from hyperpyyaml import load_hyperpyyaml
from pesq import pesq
from utils.average import AverageVal
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.utils.data import Dataset
import librosa
import logging # [新增] 引入日志模块
import time # [新增] 引入时间模块计算延迟

class te_dataset(Dataset):
    def __init__(self, mix_scp, ref_scp, aux_scp, fs):
        self.mix = {x.split()[0]:x.split()[1] for x in open(mix_scp)}
        self.ref = {x.split()[0]:x.split()[1] for x in open(ref_scp)}
        self.aux = {x.split()[0]:x.split()[1] for x in open(aux_scp)}
        assert len(self.mix) == len(self.ref) == len(self.aux)
        
        wav_id = []
        for l in open(mix_scp):
            wav_id.append(l.split()[0])
         
        self.wav_id = wav_id
        self.fs = fs
        self.len = len(self.mix)

    
    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            index, tlen = sample_idx, None
        elif len(sample_idx) == 2:
            index, tlen = sample_idx
        else:
            raise AssertionError
                
        utt = self.wav_id[index]
        mix_wav_path = self.mix[utt]
        target_wav_path = self.ref[utt]
        aux_wav_path = self.aux[utt]
        
        mix_wav, _ = librosa.load(mix_wav_path, sr=self.fs)
        target_wav, _ = librosa.load(target_wav_path, sr=self.fs)
        aux_wav, _ = librosa.load(aux_wav_path, sr=self.fs)
        
        mix_wav = torch.from_numpy(mix_wav)
        target_wav = torch.from_numpy(target_wav)
        aux_wav = torch.from_numpy(aux_wav)

        return mix_wav, target_wav, aux_wav
    
    def __len__(self):
        return self.len

def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        name = k.replace("module.", "").replace("convolution_", "convolution_module.")   # remove 'module.'
        state_dict[name] = v
    model.load_state_dict(state_dict)

    return model

def main(config, args):

    model = config['modules']['masknet']
    model = load_pretrained_modules(model,args.chkpt_path)
    model.cuda()
    model.eval()

    testset = te_dataset(
        mix_scp = os.path.join(config[args.test_set], config['mix_scp']),
        ref_scp = os.path.join(config[args.test_set], config['ref_scp']),
        aux_scp = os.path.join(config[args.test_set], config['aux_scp']),
        fs = config['sample_rate']
    )
    test_dataloader = DataLoader(
            dataset=testset, 
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=None,
        )  

    fs = config['sample_rate']

    with torch.no_grad():
        total_cnt = 1
        sdrs_mix = AverageVal()
        sdrs_est = AverageVal()

        pesqs_mix = AverageVal()
        pesqs_est = AverageVal()

        sisnrs_mix = AverageVal()
        sisnrs_est = AverageVal()

        # === [新增] 追踪延迟与 RTF ===
        latencies = AverageVal()
        rtfs = AverageVal()
        # ==========================

        logging.info("开始进行模型评估...")

        for i, (mix_wav, target_wav, emb_s1) in enumerate(test_dataloader):

            mix = mix_wav.cuda()
            embd = emb_s1.cuda()
            tar = target_wav.cuda()
            
            # === [新增] 测算单句推理延迟与实时率 ===
            torch.cuda.synchronize() # 强制等待数据完全载入显存
            start_time = time.time()
            
            est_source = model(mix, embd)
            
            torch.cuda.synchronize() # 强制等待 GPU 计算彻底完成
            end_time = time.time()
            
            # 计算延迟（秒）和当前音频的物理时长（秒）
            latency = end_time - start_time
            audio_duration = mix.shape[-1] / fs
            rtf = latency / audio_duration # RTF = 处理耗时 / 音频时长
            
            latencies.update(latency)
            rtfs.update(rtf)
            # ==================================

            est_source = est_source.squeeze().cpu().numpy()
        
            mix = mix.squeeze().cpu().numpy()
            tar = tar.squeeze().cpu().numpy()

            end = min(est_source.size, mix.size, tar.size)

            mix_wav = mix[:end]
            tar_wav = tar[:end]
            est_wav = est_source[:end]

            sdr_mix = bss_eval_sources(tar_wav, mix_wav)[0][0]
            sdr_est = bss_eval_sources(tar_wav, est_wav)[0][0]
            sdrs_mix.update(sdr_mix)
            sdrs_est.update(sdr_est)

            sisnr_mix = si_snr(mix_wav, tar_wav)
            sisnr_est = si_snr(est_wav, tar_wav)
            sisnrs_mix.update(sisnr_mix)
            sisnrs_est.update(sisnr_est)

            pesq_mix = pesq(fs, tar_wav, mix_wav, 'nb')
            pesq_est = pesq(fs, tar_wav, est_wav, 'nb')
            pesqs_mix.update(pesq_mix)
            pesqs_est.update(pesq_est)

            # [修改] 日志格式，加入耗时和 RTF 监控
            log_str = "utt{}\t SI-SNRi: {:.2f}\t PESQi: {:.2f}\t Latency: {:.3f}s\t RTF: {:.3f}".format(
                total_cnt, 
                sisnr_est - sisnr_mix, 
                pesq_est - pesq_mix,
                latency, 
                rtf
            )
            logging.info(log_str)
            total_cnt += 1
            
        # 跑完全部数据后，打印最终包含速度的体检报告
        logging.info("\n" + "="*50)
        logging.info("评估完成！最终平均指标如下:")
        logging.info(f"平均 SDR 提升: {sdrs_est.avg - sdrs_mix.avg:.4f} (原始: {sdrs_mix.avg:.4f} -> 提取后: {sdrs_est.avg:.4f})")
        logging.info(f"平均 SI-SNR 提升: {sisnrs_est.avg - sisnrs_mix.avg:.4f} (原始: {sisnrs_mix.avg:.4f} -> 提取后: {sisnrs_est.avg:.4f})")
        logging.info(f"平均 PESQ 提升: {pesqs_est.avg - pesqs_mix.avg:.4f} (原始: {pesqs_mix.avg:.4f} -> 提取后: {pesqs_est.avg:.4f})")
        logging.info("-" * 50)
        logging.info(f"平均推理延迟 (Latency): {latencies.avg:.4f} 秒/句")
        logging.info(f"平均实时率 (RTF): {rtfs.avg:.4f} (越小越好，< 1.0 表示满足实时要求)")
        logging.info("="*50 + "\n")

def si_snr(x, s, remove_dc=True):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speech Separation')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='config file path')
    parser.add_argument('-p', '--chkpt-path', type=str, default='',
                        help='path to the chosen checkpoint')
    parser.add_argument('-t', '--test-set', type=str, default='',
                        help='path to the test set')
    # [修改] 默认值改为 None，以便程序判断你是否通过命令行显式指定了名称
    parser.add_argument('-l', '--log-file', type=str, default=None,
                        help='path to save the evaluation log file')
    args = parser.parse_args()

    # ================= 新增的动态命名逻辑 =================
    if args.log_file is None:
        from pathlib import Path
        
        if args.config:
            config_path = Path(args.config)
            # 获取 config 文件所在的当前目录 (例如 chkpt/USEF-SepFormer/)
            config_dir = config_path.parent
            # 提取模型名称 (例如 USEF-SepFormer)
            model_name = config_path.parent.name 
        else:
            config_dir = Path('.')
            model_name = "unknown_model"
        
        # 提取数据集名称
        dataset_name = args.test_set if args.test_set else "unknown_dataset"
        
        # 自动拼接出日志文件名，并合并为完整的绝对/相对路径
        log_filename = f"eval_{model_name}_{dataset_name}.log"
        args.log_file = str(config_dir / dataset_name / log_filename)
    # =====================================================

    # 初始化 logging 配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(args.log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    for f in args.config, args.chkpt_path:
        assert os.path.isfile(f), "No such file: %s" % f

    with open(args.config, 'r') as f:
        config_strings = f.read()
    config = load_hyperpyyaml(config_strings)
    
    logging.info('INFO: Loaded hparams from: {}'.format(args.config))
    # 顺便在终端打印一下最终确定的日志保存路径，做到心中有数
    logging.info('INFO: Evaluation log will be automatically saved to: {}'.format(args.log_file))

    main(config, args)