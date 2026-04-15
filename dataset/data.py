import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import librosa

class tr_dataset(Dataset):
    def __init__(self, mix_scp, ref_scp, aux_scp, dur, fs, dataset_type):
        self.mix = {x.split()[0]:x.split()[1] for x in open(mix_scp)}
        self.ref = {x.split()[0]:x.split()[1] for x in open(ref_scp)}
        self.aux = {x.split()[0]:x.split()[1] for x in open(aux_scp)}
        assert len(self.mix) == len(self.ref) == len(self.aux)
        
        wav_id = []
        for l in open(mix_scp):
            wav_id.append(l.split()[0])
         
        self.wav_id = wav_id
        self.tlen = dur * fs
        
        self.fs = fs
        self.len = len(self.mix)
        self.dataset_type = dataset_type
    
    def _trun_wav(self, y, tlen, offset=0):
        if y.shape[0] < tlen:
            npad = tlen - y.shape[0]
            y = np.pad(y, (0, npad), mode='constant', constant_values=0)
        else:
            y = y[offset:offset+tlen]
        return y 
    
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
        # 根据数据集决定 exclude 策略
        if self.dataset_type == "wsj0-2mix":
            exclude = [utt.split('_')[0]+'.wav', utt.split('_')[2]+'.wav']
            aux_list = os.listdir(os.path.dirname(self.aux[utt]))
            valid_aux = [x for x in aux_list if x not in exclude]

        elif self.dataset_type == "libri2mix":
            # 1. 提取当前 Target 说话人的 ID
            spk_id = utt.split('-')[0]
            
            # 2. 提取当前文件的真实文件名，用于排除自身
            current_file = os.path.basename(target_wav_path)
            
            # 3. 获取 s1 目录下的所有文件
            aux_list = os.listdir(os.path.dirname(self.aux[utt]))
            
            # 4. 核心过滤：文件名必须以该 spk_id 开头，且不能是当前正在训练的这段音频
            valid_aux = [
                x for x in aux_list 
                if x.startswith(spk_id + '-') and x != current_file
            ]
            
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        # 安全的随机挑选逻辑（防崩溃兜底）
        if len(valid_aux) > 0:
            chosen_aux = random.choice(valid_aux)
        else:
            # 如果该说话人在这个目录下恰好只有这一段音频，只能拿自己当 aux
            chosen_aux = os.path.basename(self.aux[utt])
            
        aux_wav_path = os.path.join(os.path.dirname(self.aux[utt]), chosen_aux)

        mix_wav, _ = librosa.load(mix_wav_path, sr=self.fs)
        target_wav, _ = librosa.load(target_wav_path, sr=self.fs)
        aux_wav, _ = librosa.load(aux_wav_path, sr=self.fs)

        offset = random.randint(0, max(len(target_wav) - self.tlen, 0))
        target_wav = self._trun_wav(target_wav, self.tlen, offset)
        mix_wav = self._trun_wav(mix_wav, self.tlen, offset)

        offset = random.randint(0, max(len(aux_wav) - self.tlen + 8000, 0))
        aux_wav = self._trun_wav(aux_wav, self.tlen - 8000, offset) # aux_len = 3s
        
        mix_wav = torch.from_numpy(mix_wav)
        target_wav = torch.from_numpy(target_wav)
        aux_wav = torch.from_numpy(aux_wav)

        source_len = np.array([target_wav.shape[-1]])
        source_len = torch.from_numpy(source_len)

        return mix_wav, target_wav, aux_wav, source_len
    
    def __len__(self):
        return self.len


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

    def _trun_wav(self, y, tlen, offset=0):
        if y.shape[0] < tlen:
            npad = tlen - y.shape[0]
            y = np.pad(y, (0, npad), mode='constant', constant_values=0)
        else:
            y = y[offset:offset+tlen]
        return y 
    
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

        # === 新增：防止验证集 OOM 的安全截断 ===
        # 1. 将参考语音 (aux) 限制为最多 4 秒 (足够网络提取声纹)
        max_aux_len = 4 * self.fs
        if len(aux_wav) > max_aux_len:
            aux_wav = aux_wav[:max_aux_len]
            
        # 2. 将混合语音和目标语音限制为最多 4 秒 (控制交叉注意力的矩阵大小)
        max_mix_len = 4 * self.fs
        if len(mix_wav) > max_mix_len:
            mix_wav = mix_wav[:max_mix_len]
            target_wav = target_wav[:max_mix_len]
        # ======================================
        
        mix_wav = torch.from_numpy(mix_wav)
        target_wav = torch.from_numpy(target_wav)
        aux_wav = torch.from_numpy(aux_wav)

        source_len = np.array([target_wav.shape[-1]])
        source_len = torch.from_numpy(source_len)

        return mix_wav, target_wav, aux_wav, source_len
    
    def __len__(self):
        return self.len