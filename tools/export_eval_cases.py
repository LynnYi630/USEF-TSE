#!/usr/bin/env python3
r"""Export low-scoring eval cases as wav files for listening checks.

The script parses the text eval log lines that contain ``uttXXXX SI-SNRi`` and
copies each selected case's mixture, reference, and auxiliary wav files into a
debug directory. If a checkpoint is provided, it also exports the model estimate.

Usage examples:
    python tools/export_eval_cases.py \
      --config config/config-USEF-TCN-V2.yaml \
      --log-file logs/USEF-TCN-V2/wsj0-2mix/eval.log \
      --out-dir debug_eval_cases/tcn_v2_worst \
      --top-k 10

    python tools/export_eval_cases.py \
      --config config/config-USEF-TCN-V2.yaml \
      --log-file logs/USEF-TCN-V2/wsj0-2mix/eval.log \
      --utt-index 1 \
      --utt-index 42 \
      --chkpt-path chkpt/USEF-TCN-V2/wsj0-2mix/temp_best.pth.tar \
      --device cuda:0 \
      --aux-duration 4
"""

import argparse
import csv
import os
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def parse_eval_log(log_path):
    pattern = re.compile(
        r"utt(?P<idx>\d+)\s+SI-SNRi:\s+(?P<si>[-\d.]+)\s+"
        r"PESQi:\s+(?P<pesq>[-\d.]+)"
    )
    rows = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                rows.append(
                    {
                        "utt_index": int(match.group("idx")),
                        "si_snri": float(match.group("si")),
                        "pesqi": float(match.group("pesq")),
                    }
                )
    return rows


def read_scp(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            utt, wav_path = line.strip().split(maxsplit=1)
            entries.append((utt, wav_path))
    return entries


def parse_simple_config(path):
    keys = {
        "test_path",
        "sample_rate",
        "mix_scp",
        "ref_scp",
        "aux_scp",
    }
    config = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            if key not in keys:
                continue
            value = value.split("#", 1)[0].strip()
            value = value.strip("'\"")
            config[key] = int(value) if key == "sample_rate" else value
    missing = sorted(keys - set(config))
    if missing:
        raise ValueError(f"missing config keys: {missing}")
    return config


def local_path(path):
    if os.name == "nt":
        match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", path)
        if match:
            drive = match.group(1).upper()
            rest = match.group(2).replace("/", "\\")
            return f"{drive}:\\{rest}"
    return path


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def write_wav(path, wav, sample_rate, peak_normalize=False):
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.nan_to_num(wav)
    if peak_normalize:
        peak = float(np.max(np.abs(wav)))
        if peak > 1e-8:
            wav = 0.98 * wav / peak
    wav = np.clip(wav, -1.0, 1.0)
    wavfile.write(path, sample_rate, (wav * 32767.0).astype(np.int16))


def load_pretrained_modules(model, ckpt_path):
    import torch

    model_info = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = OrderedDict()
    for key, value in model_info["model_state_dict"].items():
        name = key.replace("module.", "").replace(
            "convolution_", "convolution_module."
        )
        state_dict[name] = value
    model.load_state_dict(state_dict)
    return model


def load_model(config_path, ckpt_path, device):
    import torch
    from hyperpyyaml import load_hyperpyyaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = load_hyperpyyaml(f.read())
    model = config["modules"]["masknet"]
    model = load_pretrained_modules(model, ckpt_path)
    model.to(device)
    model.eval()
    return model


def select_cases(rows, top_k, utt_indices):
    if utt_indices:
        wanted = set(utt_indices)
        selected = [row for row in rows if row["utt_index"] in wanted]
        missing = sorted(wanted - {row["utt_index"] for row in selected})
        if missing:
            raise ValueError(f"utt indices not found in log: {missing}")
        return sorted(selected, key=lambda row: row["si_snri"])
    return sorted(rows, key=lambda row: row["si_snri"])[:top_k]


def export_original_audio(case_dir, paths, sample_rate):
    for kind, path in paths.items():
        src = local_path(path)
        dst = case_dir / f"{kind}.wav"
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
            continue
        import librosa

        wav, _ = librosa.load(src, sr=sample_rate)
        write_wav(dst, wav, sample_rate)


def export_estimate(case_dir, model, paths, sample_rate, aux_duration, device):
    import librosa
    import torch

    mix_wav, _ = librosa.load(local_path(paths["mix"]), sr=sample_rate)
    aux_wav, _ = librosa.load(local_path(paths["aux"]), sr=sample_rate)
    if aux_duration is not None:
        aux_len = int(aux_duration * sample_rate)
        if aux_wav.shape[0] < aux_len:
            aux_wav = np.pad(aux_wav, (0, aux_len - aux_wav.shape[0]))
        else:
            aux_wav = aux_wav[:aux_len]

    mix = torch.from_numpy(mix_wav).unsqueeze(0).to(device)
    aux = torch.from_numpy(aux_wav).unsqueeze(0).to(device)
    with torch.no_grad():
        est = model(mix, aux).squeeze().detach().cpu().numpy()
    write_wav(case_dir / "est.wav", est, sample_rate)
    write_wav(case_dir / "est_peaknorm.wav", est, sample_rate, peak_normalize=True)
    np.save(case_dir / "est.npy", est.astype(np.float32, copy=False))


def main():
    parser = argparse.ArgumentParser(
        description="Export the worst eval cases as wav files for listening."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--out-dir", default="debug_eval_cases")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--utt-index", type=int, action="append", default=[])
    parser.add_argument("--chkpt-path", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aux-duration", type=float, default=None)
    args = parser.parse_args()

    config = parse_simple_config(args.config)

    test_path = config["test_path"]
    sample_rate = int(config["sample_rate"])
    scps = {
        "mix": read_scp(os.path.join(test_path, config["mix_scp"])),
        "ref": read_scp(os.path.join(test_path, config["ref_scp"])),
        "aux": read_scp(os.path.join(test_path, config["aux_scp"])),
    }

    rows = parse_eval_log(args.log_file)
    selected = select_cases(rows, args.top_k, args.utt_index)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    device = args.device
    if args.chkpt_path:
        import torch

        device = torch.device(args.device)
        model = load_model(args.config, args.chkpt_path, device)

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "utt_index",
                "utt_id",
                "si_snri",
                "pesqi",
                "mix",
                "ref",
                "aux",
                "case_dir",
            ],
        )
        writer.writeheader()

        for rank, row in enumerate(selected, start=1):
            idx = row["utt_index"] - 1
            utt_id, mix_path = scps["mix"][idx]
            ref_id, ref_path = scps["ref"][idx]
            aux_id, aux_path = scps["aux"][idx]
            if not (utt_id == ref_id == aux_id):
                raise ValueError(f"SCP id mismatch at utt{row['utt_index']}")

            case_dir = out_dir / (
                f"{rank:02d}_utt{row['utt_index']:04d}_"
                f"si{row['si_snri']:.2f}_{safe_name(utt_id)}"
            )
            case_dir.mkdir(parents=True, exist_ok=True)
            paths = {"mix": mix_path, "ref": ref_path, "aux": aux_path}
            export_original_audio(case_dir, paths, sample_rate)
            if model is not None:
                export_estimate(case_dir, model, paths, sample_rate, args.aux_duration, device)

            writer.writerow(
                {
                    "rank": rank,
                    "utt_index": row["utt_index"],
                    "utt_id": utt_id,
                    "si_snri": row["si_snri"],
                    "pesqi": row["pesqi"],
                    "mix": mix_path,
                    "ref": ref_path,
                    "aux": aux_path,
                    "case_dir": str(case_dir),
                }
            )

    print(f"Exported {len(selected)} cases to {out_dir}")
    print(f"Manifest: {manifest_path}")
    if model is None:
        print("No checkpoint was provided, so est.wav was not generated.")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    main()
