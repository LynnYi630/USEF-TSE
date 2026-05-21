#!/usr/bin/env python3
r"""Diagnose whether failed eval samples confuse target and interferer speakers.

The script selects low-scoring rows from eval_results.csv, reruns the model,
then compares the estimate against both the target source and inferred
interferer source.

Usage examples:
    python tools/diagnose_target_confusion.py \
      -c config/config-USEF-TCN-V2.yaml \
      -p chkpt/USEF-TCN-V2/wsj0-2mix/temp_best.pth.tar \
      --eval-results eval_logs/USEF-TCN-V2/wsj0-2mix/eval_results.csv \
      --metric si_sdri \
      --threshold 5 \
      --max-utts 50 \
      --output-dir eval_logs/USEF-TCN-V2/wsj0-2mix/diagnosis

    python tools/diagnose_target_confusion.py \
      -c config/config-USEF-TCN-V2.yaml \
      -p chkpt/USEF-TCN-V2/wsj0-2mix/temp_best.pth.tar \
      --eval-results eval_logs/USEF-TCN-V2/wsj0-2mix/eval_results.csv \
      --cpu \
      --aux-duration 4
"""

import argparse
import csv
import os
import sys
from collections import Counter, OrderedDict

import librosa
import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import get_test_path, load_pretrained_modules, reserve_gpu_memory, si_snr


def load_scp(path):
    with open(path, "r", encoding="utf-8") as f:
        return OrderedDict(line.strip().split(maxsplit=1) for line in f if line.strip())


def read_eval_rows(path, metric, threshold, max_utts):
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    selected = [row for row in rows if float(row[metric]) < threshold]
    selected.sort(key=lambda row: float(row[metric]))
    if max_utts and max_utts > 0:
        selected = selected[:max_utts]
    return selected


def truncate_or_pad(wav, length):
    if length is None:
        return wav
    if wav.shape[0] >= length:
        return wav[:length]
    return np.pad(wav, (0, length - wav.shape[0]), mode="constant")


def local_path(path):
    if os.name == "nt":
        normalized = path.replace("\\", "/")
        if normalized.startswith("/mnt/") and len(normalized) > 6:
            drive = normalized[5].upper()
            rest = normalized[7:].replace("/", "\\")
            return "{}:\\{}".format(drive, rest)
    return path


def load_wav(path, fs, length=None):
    wav, _ = librosa.load(local_path(path), sr=fs)
    return truncate_or_pad(wav.astype(np.float32, copy=False), length)


def infer_interferer_path(target_path):
    normalized = target_path.replace("\\", "/")
    if "/s1/" in normalized:
        return normalized.replace("/s1/", "/s2/")
    if "/s2/" in normalized:
        return normalized.replace("/s2/", "/s1/")
    raise ValueError("Cannot infer interferer path from target path: {}".format(target_path))


def source_side(path):
    normalized = path.replace("\\", "/")
    if "/s1/" in normalized:
        return "s1"
    if "/s2/" in normalized:
        return "s2"
    return "unknown"


def speakers_from_utt_id(utt_id, side):
    parts = utt_id.split("_")
    if len(parts) < 5:
        return "unknown", "unknown", "unknown"

    source1_spk = parts[0][:3]
    source2_spk = parts[2][:3]
    aux_spk = parts[4][:3]
    if side == "s1":
        return source1_spk, source2_spk, aux_spk
    if side == "s2":
        return source2_spk, source1_spk, aux_spk
    return "unknown", "unknown", aux_spk


def chunk_starts(total_len, chunk_len, chunk_hop):
    if total_len <= chunk_len:
        return [0]
    starts = list(range(0, total_len - chunk_len + 1, chunk_hop))
    tail_start = total_len - chunk_len
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def forward_chunked(model, mix, aux, chunk_len, chunk_hop):
    total_len = mix.shape[1]
    if chunk_len <= 0 or total_len <= chunk_len:
        return model(mix, aux)

    estimate = mix.new_zeros(mix.shape[0], total_len)
    weight = mix.new_zeros(mix.shape[0], total_len)
    for start in chunk_starts(total_len, chunk_len, chunk_hop):
        end = min(start + chunk_len, total_len)
        chunk_estimate = model(mix[:, start:end], aux)
        chunk_len_actual = min(chunk_estimate.shape[1], end - start)
        estimate[:, start:start + chunk_len_actual] += chunk_estimate[:, :chunk_len_actual]
        weight[:, start:start + chunk_len_actual] += 1
    return estimate / weight.clamp_min(1)


def classify(row, confusion_margin, weak_target_margin, low_quality_sisdr):
    target_score = row["est_target_si_sdr"]
    interferer_score = row["est_interferer_si_sdr"]
    best_score = max(target_score, interferer_score)
    target_margin = target_score - interferer_score

    if interferer_score >= target_score + confusion_margin:
        return "target_confusion"
    if best_score < low_quality_sisdr:
        return "low_quality_or_silence"
    if target_margin < weak_target_margin:
        return "weak_target_or_ambiguous"
    return "target_degradation"


def write_outputs(rows, args):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "target_confusion_diagnosis.csv")
    summary_path = os.path.join(args.output_dir, "target_confusion_summary.txt")

    fieldnames = [
        "utt_id",
        "source_side",
        "target_speaker",
        "interferer_speaker",
        "aux_speaker",
        "eval_metric",
        "eval_metric_value",
        "eval_sdri",
        "eval_si_sdri",
        "mix_target_si_sdr",
        "mix_interferer_si_sdr",
        "est_target_si_sdr",
        "est_interferer_si_sdr",
        "target_minus_interferer_si_sdr",
        "target_si_sdri",
        "interferer_si_sdri",
        "estimate_energy_db",
        "target_energy_db",
        "interferer_energy_db",
        "diagnosis",
        "mix_path",
        "target_path",
        "interferer_path",
        "aux_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["diagnosis"] for row in rows)
    lines = [
        "Target Confusion Diagnosis",
        "=" * 70,
        "Analyzed utterances: {}".format(len(rows)),
        "",
        "Class counts:",
    ]
    for key, value in counts.most_common():
        lines.append("  {}: {}".format(key, value))

    if rows:
        avg_eval = np.mean([row["eval_metric_value"] for row in rows])
        avg_target = np.mean([row["est_target_si_sdr"] for row in rows])
        avg_interferer = np.mean([row["est_interferer_si_sdr"] for row in rows])
        avg_margin = np.mean([row["target_minus_interferer_si_sdr"] for row in rows])
        lines.extend(
            [
                "",
                "Averages:",
                "  selected eval metric: {:.4f} dB".format(avg_eval),
                "  est vs target SI-SDR: {:.4f} dB".format(avg_target),
                "  est vs interferer SI-SDR: {:.4f} dB".format(avg_interferer),
                "  target - interferer SI-SDR margin: {:.4f} dB".format(avg_margin),
            ]
        )

    lines.extend(["", "Outputs:", "  {}".format(csv_path)])

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("Saved summary to: {}".format(summary_path))


def main(config, args):
    fs = int(config["sample_rate"])
    aux_len = int(args.aux_duration * fs) if args.aux_duration is not None else None

    test_path = get_test_path(config, args)
    mix_scp = load_scp(os.path.join(test_path, config["mix_scp"]))
    ref_scp = load_scp(os.path.join(test_path, config["ref_scp"]))
    aux_scp = load_scp(os.path.join(test_path, config["aux_scp"]))

    selected_rows = read_eval_rows(
        args.eval_results,
        metric=args.metric,
        threshold=args.threshold,
        max_utts=args.max_utts,
    )
    if not selected_rows:
        raise RuntimeError("No eval rows matched {} < {}".format(args.metric, args.threshold))

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:{}".format(args.device))
        torch.cuda.set_device(device)
        reserve_gpu_memory(device_id=args.device, size_in_gb=args.memory)
    else:
        device = torch.device("cpu")

    model = config["modules"]["masknet"]
    model = load_pretrained_modules(model, args.chkpt_path)
    model.to(device)
    model.eval()

    chunk_len = int(round(args.chunk_duration * fs)) if args.chunk_duration > 0 else 0
    chunk_hop = (
        int(round(args.chunk_hop_duration * fs))
        if args.chunk_hop_duration > 0
        else chunk_len
    )
    if chunk_len > 0 and (chunk_hop <= 0 or chunk_hop > chunk_len):
        raise ValueError("--chunk-hop-duration must be > 0 and <= --chunk-duration")

    diagnosis_rows = []
    with torch.no_grad():
        for eval_row in tqdm(selected_rows, desc="Diagnosing"):
            utt_id = eval_row["utt_id"]
            mix_path = mix_scp[utt_id]
            target_path = ref_scp[utt_id]
            aux_path = aux_scp[utt_id]
            interferer_path = infer_interferer_path(target_path)

            mix = load_wav(mix_path, fs)
            target = load_wav(target_path, fs)
            interferer = load_wav(interferer_path, fs)
            aux = load_wav(aux_path, fs, aux_len)

            end = min(mix.shape[0], target.shape[0], interferer.shape[0])
            mix = mix[:end]
            target = target[:end]
            interferer = interferer[:end]

            mix_t = torch.from_numpy(mix).unsqueeze(0).to(device)
            aux_t = torch.from_numpy(aux).unsqueeze(0).to(device)
            estimate = forward_chunked(model, mix_t, aux_t, chunk_len, chunk_hop)
            estimate_np = estimate.squeeze(0).detach().cpu().numpy()[:end]

            mix_target_si_sdr = si_snr(mix, target)
            mix_interferer_si_sdr = si_snr(mix, interferer)
            est_target_si_sdr = si_snr(estimate_np, target)
            est_interferer_si_sdr = si_snr(estimate_np, interferer)
            target_si_sdri = est_target_si_sdr - mix_target_si_sdr
            interferer_si_sdri = est_interferer_si_sdr - mix_interferer_si_sdr
            target_margin = est_target_si_sdr - est_interferer_si_sdr
            side = source_side(target_path)
            target_speaker, interferer_speaker, aux_speaker = speakers_from_utt_id(
                utt_id,
                side,
            )

            eps = 1e-12
            row = {
                "utt_id": utt_id,
                "source_side": side,
                "target_speaker": target_speaker,
                "interferer_speaker": interferer_speaker,
                "aux_speaker": aux_speaker,
                "eval_metric": args.metric,
                "eval_metric_value": float(eval_row[args.metric]),
                "eval_sdri": float(eval_row.get("sdri", np.nan)),
                "eval_si_sdri": float(eval_row.get("si_sdri", np.nan)),
                "mix_target_si_sdr": float(mix_target_si_sdr),
                "mix_interferer_si_sdr": float(mix_interferer_si_sdr),
                "est_target_si_sdr": float(est_target_si_sdr),
                "est_interferer_si_sdr": float(est_interferer_si_sdr),
                "target_minus_interferer_si_sdr": float(target_margin),
                "target_si_sdri": float(target_si_sdri),
                "interferer_si_sdri": float(interferer_si_sdri),
                "estimate_energy_db": float(10 * np.log10(np.mean(estimate_np ** 2) + eps)),
                "target_energy_db": float(10 * np.log10(np.mean(target ** 2) + eps)),
                "interferer_energy_db": float(10 * np.log10(np.mean(interferer ** 2) + eps)),
                "mix_path": mix_path,
                "target_path": target_path,
                "interferer_path": interferer_path,
                "aux_path": aux_path,
            }
            row["diagnosis"] = classify(
                row,
                confusion_margin=args.confusion_margin,
                weak_target_margin=args.weak_target_margin,
                low_quality_sisdr=args.low_quality_sisdr,
            )
            diagnosis_rows.append(row)

    write_outputs(diagnosis_rows, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Diagnose target-speaker confusion on failed eval samples")
    parser.add_argument("-c", "--config", required=True, help="config file path")
    parser.add_argument("-p", "--chkpt-path", required=True, help="checkpoint path")
    parser.add_argument(
        "--eval-results",
        default="eval_logs/USEF-TCN-V2/wsj0-2mix/eval_results.csv",
        help="eval_results.csv used to select failed utterances",
    )
    parser.add_argument(
        "-t",
        "--test-set",
        default="",
        help="test set key in config, or direct test set path",
    )
    parser.add_argument(
        "--metric",
        default="sdri",
        choices=["sdri", "si_sdri"],
        help="eval_results metric used to select failed samples",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-utts", type=int, default=0, help="0 means all matching rows")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--memory", type=float, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--aux-duration", type=float, default=None)
    parser.add_argument("--chunk-duration", type=float, default=0.0)
    parser.add_argument("--chunk-hop-duration", type=float, default=0.0)
    parser.add_argument("--confusion-margin", type=float, default=3.0)
    parser.add_argument("--weak-target-margin", type=float, default=3.0)
    parser.add_argument("--low-quality-sisdr", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        default="eval_logs/USEF-TCN-V2/wsj0-2mix/diagnosis",
        help="directory for diagnosis csv and summary",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = load_hyperpyyaml(f.read())

    main(config, args)
