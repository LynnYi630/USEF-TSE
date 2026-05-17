import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import librosa
from urllib.parse import quote


def teacher_cache_path(cache_dir, utt_id):
    return os.path.join(cache_dir, quote(utt_id, safe='') + '.pt')

class tr_dataset(Dataset):
    def __init__(
        self,
        mix_scp,
        ref_scp,
        aux_scp,
        dur,
        fs,
        dataset_type,
        use_short_aux=False,
        use_wrcd=False,
        use_wrcd_consistency=False,
        teacher_cache_dir=None,
        wake_aux_min_duration=None,
        wake_aux_max_duration=None,
    ):
        self.mix = {x.split()[0]:x.split()[1] for x in open(mix_scp)}
        self.ref = {x.split()[0]:x.split()[1] for x in open(ref_scp)}
        self.aux = {x.split()[0]:x.split()[1] for x in open(aux_scp)}
        assert len(self.mix) == len(self.ref) == len(self.aux)
        
        wav_id = []
        for l in open(mix_scp):
            wav_id.append(l.split()[0])
         
        self.wav_id = wav_id
        self.tlen = int(dur * fs)
        
        self.fs = fs
        self.len = len(self.mix)
        self.dataset_type = dataset_type
        self.use_wrcd = bool(use_wrcd)
        self.use_wrcd_consistency = bool(use_wrcd_consistency)
        if self.use_wrcd_consistency and not self.use_wrcd:
            raise ValueError('use_wrcd_consistency requires use_wrcd=True')
        self.use_short_aux = bool(use_short_aux)
        if self.use_wrcd_consistency and not self.use_short_aux:
            raise ValueError('use_wrcd_consistency requires use_short_aux=True')
        self.teacher_cache_dir = teacher_cache_dir
        self.wake_aux_min_len = int((wake_aux_min_duration or 1.0) * fs)
        self.wake_aux_max_len = int((wake_aux_max_duration or wake_aux_min_duration or 1.0) * fs)
        if self.wake_aux_min_len <= 0 or self.wake_aux_max_len <= 0:
            raise ValueError('wake aux duration must be positive')
        if self.wake_aux_min_len > self.wake_aux_max_len:
            raise ValueError('wake_aux_min_duration must be <= wake_aux_max_duration')
        if self.use_wrcd and not self.teacher_cache_dir:
            raise ValueError('teacher_cache_dir is required when use_wrcd=True')
        self.libri_aux_by_spk = (
            self._build_librimix_aux_index()
            if self.dataset_type == "libri2mix"
            else {}
        )
    
    def _trun_wav(self, y, tlen, offset=0):
        y = y[offset:offset + tlen]
        if y.shape[0] < tlen:
            npad = tlen - y.shape[0]
            y = np.pad(y, (0, npad), mode='constant', constant_values=0)
        return y

    def _load_teacher_est(self, utt):
        path = teacher_cache_path(self.teacher_cache_dir, utt)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'Missing WRCD teacher cache for {utt}: {path}'
            )
        item = torch.load(path, map_location='cpu', weights_only=False)
        teacher_est = item['teacher_est'] if isinstance(item, dict) else item
        if isinstance(teacher_est, torch.Tensor):
            teacher_est = teacher_est.detach().cpu().float().view(-1).numpy()
        else:
            teacher_est = np.asarray(teacher_est, dtype=np.float32).reshape(-1)
        return teacher_est

    def _crop_wake_aux(self, y):
        crop_len = random.randint(self.wake_aux_min_len, self.wake_aux_max_len)
        if y.shape[0] < crop_len:
            seg = np.pad(y, (0, crop_len - y.shape[0]), mode='constant')
        else:
            offset = random.randint(0, max(y.shape[0] - crop_len, 0))
            seg = y[offset:offset + crop_len]
        if seg.shape[0] < self.wake_aux_max_len:
            seg = np.pad(seg, (0, self.wake_aux_max_len - seg.shape[0]), mode='constant')
        return seg.astype(np.float32, copy=False)

    def _parse_librimix_source_utt(self, source_path):
        stem = os.path.splitext(os.path.basename(source_path))[0]
        parts = stem.split("_")
        if len(parts) != 2:
            raise ValueError(f"Unsupported Libri2Mix source filename: {source_path}")
        source_dir = os.path.basename(os.path.dirname(source_path))
        if source_dir == "s1":
            return parts[0]
        if source_dir == "s2":
            return parts[1]
        raise ValueError(f"Libri2Mix source path must be under s1 or s2: {source_path}")

    def _build_librimix_aux_index(self):
        aux_by_spk = {}
        seen = set()
        for source_path in self.ref.values():
            source_utt = self._parse_librimix_source_utt(source_path)
            spk_id = source_utt.split("-", 1)[0]
            key = (spk_id, source_utt, source_path)
            if key in seen:
                continue
            seen.add(key)
            aux_by_spk.setdefault(spk_id, []).append((source_utt, source_path))
        for spk_id in aux_by_spk:
            aux_by_spk[spk_id].sort(key=lambda item: (item[0], item[1]))
        return aux_by_spk

    def _choose_librimix_aux_path(self, target_wav_path):
        target_utt = self._parse_librimix_source_utt(target_wav_path)
        spk_id = target_utt.split("-", 1)[0]
        candidates = self.libri_aux_by_spk.get(spk_id, [])
        valid = [path for source_utt, path in candidates if source_utt != target_utt]
        if not valid:
            valid = [path for _, path in candidates]
        if not valid:
            raise RuntimeError(f"No Libri2Mix aux candidate for speaker {spk_id}")
        return random.choice(valid)
    
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
        # Choose aux according to each dataset layout.
        if self.dataset_type == "wsj0-2mix":
            exclude = [utt.split('_')[0]+'.wav', utt.split('_')[2]+'.wav']
            aux_list = os.listdir(os.path.dirname(self.aux[utt]))
            valid_aux = [x for x in aux_list if x not in exclude]
            if len(valid_aux) > 0:
                chosen_aux = random.choice(valid_aux)
            else:
                chosen_aux = os.path.basename(self.aux[utt])
            aux_wav_path = os.path.join(os.path.dirname(self.aux[utt]), chosen_aux)
        elif self.dataset_type == "libri2mix":
            aux_wav_path = self._choose_librimix_aux_path(target_wav_path)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        mix_wav, _ = librosa.load(mix_wav_path, sr=self.fs)
        target_wav, _ = librosa.load(target_wav_path, sr=self.fs)
        aux_wav, _ = librosa.load(aux_wav_path, sr=self.fs)

        mixture_offset = random.randint(0, max(len(target_wav) - self.tlen, 0))
        target_wav = self._trun_wav(target_wav, self.tlen, mixture_offset)
        mix_wav = self._trun_wav(mix_wav, self.tlen, mixture_offset)

        aux_wav_cons = None
        if self.use_short_aux:
            aux_wav_full = aux_wav
            aux_wav = self._crop_wake_aux(aux_wav_full)
            if self.use_wrcd_consistency:
                aux_wav_cons = self._crop_wake_aux(aux_wav_full)
        if self.use_wrcd:
            teacher_wav = self._load_teacher_est(utt)
            teacher_wav = self._trun_wav(teacher_wav, self.tlen, mixture_offset)
        if not self.use_short_aux:
            offset = random.randint(0, max(len(aux_wav) - self.tlen + 8000, 0))
            aux_wav = self._trun_wav(aux_wav, self.tlen - 8000, offset) # aux_len = 3s
        
        mix_wav = torch.from_numpy(mix_wav)
        target_wav = torch.from_numpy(target_wav)
        aux_wav = torch.from_numpy(aux_wav)

        source_len = np.array([target_wav.shape[-1]])
        source_len = torch.from_numpy(source_len)

        if self.use_wrcd:
            teacher_wav = torch.from_numpy(teacher_wav)
            if self.use_wrcd_consistency:
                aux_wav_cons = torch.from_numpy(aux_wav_cons)
                return mix_wav, target_wav, aux_wav, source_len, teacher_wav, aux_wav_cons
            return mix_wav, target_wav, aux_wav, source_len, teacher_wav

        return mix_wav, target_wav, aux_wav, source_len
    
    def __len__(self):
        return self.len


class te_dataset(Dataset):
    def __init__(self, mix_scp, ref_scp, aux_scp, fs, aux_duration=None):
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
        self.aux_len = int(aux_duration * fs) if aux_duration else None

    def _trun_wav(self, y, tlen, offset=0):
        y = y[offset:offset + tlen]
        if y.shape[0] < tlen:
            npad = tlen - y.shape[0]
            y = np.pad(y, (0, npad), mode='constant', constant_values=0)
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

        if self.aux_len is not None:
            aux_wav = self._trun_wav(aux_wav, self.aux_len, 0)
        
        mix_wav = torch.from_numpy(mix_wav)
        target_wav = torch.from_numpy(target_wav)
        aux_wav = torch.from_numpy(aux_wav)

        source_len = np.array([target_wav.shape[-1]])
        source_len = torch.from_numpy(source_len)

        return mix_wav, target_wav, aux_wav, source_len
    
    def __len__(self):
        return self.len
