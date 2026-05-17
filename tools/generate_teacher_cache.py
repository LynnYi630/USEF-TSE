#!/usr/bin/env python3
"""
Generate offline teacher waveform caches for WRCD training.

Why this script exists
----------------------
WRCD uses a pretrained separation/TSE model as a waveform teacher. Running that
teacher online during every training step is expensive, so this script runs the
teacher once over a training split and stores one .pt cache file per utterance.

This script:
  1. Reads a USEF-TSE style data directory containing:
       mix.scp
       aux.scp
     The exact filenames come from the selected config.
  2. Loads the teacher model from the config and checkpoint.
  3. Runs teacher_est = model(mixture, aux) for every utterance.
  4. Writes cache files under:
       <output-dir>/<url-escaped-utt-id>.pt

Use the generated output directory as teacher_cache_dir in WRCD configs.

To generate teacher caches:
    python3 tools/generate_teacher_cache.py \
      --config config/config-USEF-SepFormer.yaml \
      --chkpt-path chkpt/USEF-SepFormer/libri2mix/best.pth.tar \
      --data-path data/train/libri2mix \
      --output-dir teacher_cache/USEF-SepFormer/libri2mix/train \
      --device cuda:0 \
      --max-aux-duration 4 \
      --overwrite

Notes:
  - --data-path defaults to train_path from the config.
  - --output-dir defaults to teacher_cache/<model-name>/<dataset>/<split>.
  - --max-aux-duration caps the teacher reference/enrollment audio length.
  - Existing cache files are skipped unless --overwrite is set.
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import librosa
import torch
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.data import teacher_cache_path


def load_scp(path):
    items = OrderedDict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            items[parts[0]] = parts[1]
    return items


def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        name = k.replace('module.', '').replace(
            'convolution_', 'convolution_module.'
        )
        state_dict[name] = v
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'WARNING: missing teacher keys: {len(missing)}')
    if unexpected:
        print(f'WARNING: unexpected teacher keys: {len(unexpected)}')
    return model


def infer_default_output_dir(config, data_path):
    model_name = str(config.get('name', 'teacher')).strip('/').split('/')[0]
    dataset_name = str(config.get('dataset_type') or Path(data_path).name)
    split_name = Path(data_path).parent.name + '_' + Path(data_path).name
    return os.path.join('teacher_cache', model_name, dataset_name, split_name)


def crop_or_pad(wav, num_samples):
    if num_samples is None:
        return wav
    if wav.shape[0] < num_samples:
        return torch.nn.functional.pad(
            torch.from_numpy(wav), (0, num_samples - wav.shape[0])
        ).numpy()
    return wav[:num_samples]


def main(args):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = load_hyperpyyaml(f.read())

    data_path = args.data_path or config.get('train_path')
    if not data_path:
        raise ValueError('Use --data-path or provide train_path in the config')

    output_dir = args.output_dir or infer_default_output_dir(config, data_path)
    os.makedirs(output_dir, exist_ok=True)

    mix_scp = load_scp(os.path.join(data_path, config['mix_scp']))
    aux_scp = load_scp(os.path.join(data_path, config['aux_scp']))
    missing_aux = [utt for utt in mix_scp.keys() if utt not in aux_scp]
    if missing_aux:
        raise RuntimeError(f'{len(missing_aux)} utterances are missing aux paths')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')

    model = config['modules']['masknet']
    model = load_pretrained_modules(model, args.chkpt_path)
    model.to(device)
    model.eval()

    fs = config['sample_rate']
    written = 0
    skipped = 0

    with torch.no_grad():
        for utt, mix_path in tqdm(mix_scp.items(), desc='teacher cache'):
            cache_path = teacher_cache_path(output_dir, utt)
            if os.path.isfile(cache_path) and not args.overwrite:
                skipped += 1
                continue

            aux_path = aux_scp[utt]
            mix_wav, _ = librosa.load(mix_path, sr=fs)
            aux_wav, _ = librosa.load(aux_path, sr=fs)
            max_aux_samples = int(args.max_aux_duration * fs) if args.max_aux_duration else None
            aux_wav = crop_or_pad(aux_wav, max_aux_samples)

            mix = torch.from_numpy(mix_wav).float().unsqueeze(0).to(device)
            aux = torch.from_numpy(aux_wav).float().unsqueeze(0).to(device)
            teacher_est = model(mix, aux).squeeze(0).detach().cpu()

            torch.save(
                {
                    'utt_id': utt,
                    'teacher_est': teacher_est,
                    'sample_rate': fs,
                    'mix_num_samples': int(mix_wav.shape[0]),
                },
                cache_path,
            )
            written += 1

    print(f'Wrote {written} teacher cache files to {output_dir}')
    if skipped:
        print(f'Skipped {skipped} existing files; use --overwrite to regenerate')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate offline teacher outputs for WRCD')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--chkpt-path', required=True, type=str)
    parser.add_argument('--data-path', default=None, type=str)
    parser.add_argument('--output-dir', default=None, type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--max-aux-duration', default=None, type=float,
                        help='optional cap for teacher reference duration in seconds')
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
