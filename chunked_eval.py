import argparse
import csv
import os
import time

import numpy as np
import torch
from hyperpyyaml import load_hyperpyyaml
from mir_eval.separation import bss_eval_sources
from pesq import pesq
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval import (
    RunningAverage,
    build_summary_text,
    count_parameters_m,
    get_pesq_mode,
    get_test_path,
    infer_dataset_name,
    infer_model_name,
    load_pretrained_modules,
    mean_metric,
    percentile_metric,
    reserve_gpu_memory,
    si_snr,
    te_dataset,
)


def chunk_starts(total_len, chunk_len, chunk_hop):
    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    if chunk_hop <= 0:
        raise ValueError("chunk_hop must be positive")
    if chunk_hop > chunk_len:
        raise ValueError("chunk_hop must be <= chunk_len")
    if total_len <= chunk_len:
        return [0]

    starts = list(range(0, total_len - chunk_len + 1, chunk_hop))
    tail_start = total_len - chunk_len
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def forward_chunked(model, mix, embd, chunk_len, chunk_hop):
    total_len = mix.shape[1]
    if total_len <= chunk_len:
        return model(mix, embd)

    estimate = mix.new_zeros(mix.shape[0], total_len)
    weight = mix.new_zeros(mix.shape[0], total_len)

    for start in chunk_starts(total_len, chunk_len, chunk_hop):
        end = min(start + chunk_len, total_len)
        chunk_estimate = model(mix[:, start:end], embd)
        chunk_len_actual = min(chunk_estimate.shape[1], end - start)
        estimate[:, start:start + chunk_len_actual] += chunk_estimate[:, :chunk_len_actual]
        weight[:, start:start + chunk_len_actual] += 1

    return estimate / weight.clamp_min(1)


def measure_chunked_forward_latency(model, mix, embd, device, chunk_len, chunk_hop):
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize(device)
        starter.record()
        est_source = forward_chunked(model, mix, embd, chunk_len, chunk_hop)
        ender.record()
        torch.cuda.synchronize(device)

        latency_sec = starter.elapsed_time(ender) / 1000.0
    else:
        start_time = time.perf_counter()
        est_source = forward_chunked(model, mix, embd, chunk_len, chunk_hop)
        latency_sec = time.perf_counter() - start_time

    return est_source, latency_sec


def safe_path_part(value):
    value = str(value).strip().strip("/\\")
    for ch in '<>:"\\|?*':
        value = value.replace(ch, "_")
    value = value.replace("/", "_")
    return value or "unknown"


def infer_chunked_output_dir(config, args, test_path):
    if args.output_dir:
        return args.output_dir

    model_name = infer_model_name(config, args.config)
    dataset_name = infer_dataset_name(config, args, test_path)
    suffix = "chunked_{}s".format(safe_path_part(args.chunk_duration))
    if args.chunk_hop_duration != args.chunk_duration:
        suffix += "_hop_{}s".format(safe_path_part(args.chunk_hop_duration))
    return os.path.join("eval_logs", model_name, "{}_{}".format(dataset_name, suffix))


def save_results(results, summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    detail_csv = os.path.join(output_dir, "eval_results.csv")
    summary_csv = os.path.join(output_dir, "eval_summary.csv")
    summary_txt = os.path.join(output_dir, "eval_summary.txt")

    detail_fieldnames = [
        "utt_idx",
        "utt_id",
        "num_samples",
        "audio_duration_sec",
        "num_chunks",
        "chunk_duration_sec",
        "chunk_hop_duration_sec",
        "sdr_mix",
        "sdr_est",
        "sdri",
        "si_sdr_mix",
        "si_sdr_est",
        "si_sdri",
        "pesq_mix",
        "pesq_est",
        "pesqi",
        "latency_sec",
        "rtf",
    ]

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    summary_text = build_summary_text(summary)
    extra_lines = [
        "Chunked Evaluation",
        "Chunk duration:          {:.4f} sec".format(summary["chunk_duration_sec"]),
        "Chunk hop duration:      {:.4f} sec".format(summary["chunk_hop_duration_sec"]),
        "Average chunks / utt:    {:.4f}".format(summary["chunks_avg"]),
        "",
    ]
    summary_text = summary_text.rstrip() + "\n" + "\n".join(extra_lines)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return detail_csv, summary_csv, summary_txt


def main(config, args):
    fs = int(config["sample_rate"])
    pesq_mode = get_pesq_mode(fs, args.pesq_mode)
    chunk_len = int(round(args.chunk_duration * fs))
    chunk_hop = int(round(args.chunk_hop_duration * fs))
    if chunk_len <= 0 or chunk_hop <= 0:
        raise ValueError("--chunk-duration and --chunk-hop-duration must be positive")
    if chunk_hop > chunk_len:
        raise ValueError("--chunk-hop-duration must be <= --chunk-duration")

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.device))
        torch.cuda.set_device(args.device)
    else:
        device = torch.device("cpu")

    print("INFO: Using device: {}".format(device))
    print(
        "INFO: Chunked forward enabled: chunk {:.4f}s, hop {:.4f}s".format(
            args.chunk_duration,
            args.chunk_hop_duration,
        )
    )

    reserve_gpu_memory(device_id=args.device, size_in_gb=args.memory)

    model = config["modules"]["masknet"]
    model = load_pretrained_modules(model, args.chkpt_path)
    model.to(device)
    model.eval()

    parameter_m = count_parameters_m(model)

    test_path = get_test_path(config, args)
    output_dir = infer_chunked_output_dir(config, args, test_path)

    testset = te_dataset(
        mix_scp=os.path.join(test_path, config["mix_scp"]),
        ref_scp=os.path.join(test_path, config["ref_scp"]),
        aux_scp=os.path.join(test_path, config["aux_scp"]),
        fs=fs,
        aux_duration=args.aux_duration,
    )

    test_dataloader = DataLoader(
        dataset=testset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
    )

    os.makedirs(output_dir, exist_ok=True)
    progress_log_path = os.path.join(output_dir, "eval_progress.log")

    results = []

    sdrs_mix = RunningAverage()
    sdrs_est = RunningAverage()
    sdris = RunningAverage()
    sisdrs_mix = RunningAverage()
    sisdrs_est = RunningAverage()
    sisdris = RunningAverage()
    pesqs_mix = RunningAverage()
    pesqs_est = RunningAverage()
    pesqis = RunningAverage()
    latencies = RunningAverage()
    rtfs = RunningAverage()

    total_audio_duration = 0.0
    total_latency = 0.0

    print("INFO: Loaded test path: {}".format(test_path))
    print("INFO: Output dir: {}".format(output_dir))
    print("INFO: PESQ mode: {}".format(pesq_mode))
    print("INFO: Parameter(M): {:.4f}".format(parameter_m))

    with open(progress_log_path, "w", encoding="utf-8") as progress_f:
        with torch.no_grad():
            total_cnt = 1

            for i, (mix_wav, target_wav, emb_s1) in enumerate(
                tqdm(test_dataloader, desc="Chunked evaluating")
            ):
                utt_id = testset.wav_id[i]

                mix = mix_wav.to(device, non_blocking=True)
                embd = emb_s1.to(device, non_blocking=True)
                tar = target_wav.to(device, non_blocking=True)

                if i == 0 and args.speed_warmup > 0:
                    for _ in range(args.speed_warmup):
                        _ = forward_chunked(model, mix, embd, chunk_len, chunk_hop)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

                audio_duration = mix.shape[-1] / fs
                num_chunks = len(chunk_starts(mix.shape[1], chunk_len, chunk_hop))

                est_source, latency = measure_chunked_forward_latency(
                    model=model,
                    mix=mix,
                    embd=embd,
                    device=device,
                    chunk_len=chunk_len,
                    chunk_hop=chunk_hop,
                )

                rtf = latency / audio_duration
                total_audio_duration += float(audio_duration)
                total_latency += float(latency)
                latencies.update(latency)
                rtfs.update(rtf)

                est_source = est_source.squeeze().detach().cpu().numpy()
                mix_np = mix.squeeze().detach().cpu().numpy()
                tar_np = tar.squeeze().detach().cpu().numpy()

                end = min(est_source.size, mix_np.size, tar_np.size)
                mix_eval = mix_np[:end]
                tar_eval = tar_np[:end]
                est_eval = est_source[:end]

                sdr_mix = bss_eval_sources(
                    tar_eval[np.newaxis, :],
                    mix_eval[np.newaxis, :],
                )[0][0]
                sdr_est = bss_eval_sources(
                    tar_eval[np.newaxis, :],
                    est_eval[np.newaxis, :],
                )[0][0]
                sdri = sdr_est - sdr_mix
                sdrs_mix.update(sdr_mix)
                sdrs_est.update(sdr_est)
                sdris.update(sdri)

                si_sdr_mix = si_snr(mix_eval, tar_eval)
                si_sdr_est = si_snr(est_eval, tar_eval)
                si_sdri = si_sdr_est - si_sdr_mix
                sisdrs_mix.update(si_sdr_mix)
                sisdrs_est.update(si_sdr_est)
                sisdris.update(si_sdri)

                try:
                    pesq_mix = pesq(
                        fs,
                        tar_eval.astype(np.float32),
                        mix_eval.astype(np.float32),
                        pesq_mode,
                    )
                    pesq_est = pesq(
                        fs,
                        tar_eval.astype(np.float32),
                        est_eval.astype(np.float32),
                        pesq_mode,
                    )
                    pesqi = pesq_est - pesq_mix
                except Exception as e:
                    pesq_mix = np.nan
                    pesq_est = np.nan
                    pesqi = np.nan
                    warn_msg = "WARNING: PESQ failed at utt {} / {}: {}".format(
                        total_cnt,
                        utt_id,
                        str(e),
                    )
                    tqdm.write(warn_msg)
                    progress_f.write(warn_msg + "\n")

                pesqs_mix.update(pesq_mix)
                pesqs_est.update(pesq_est)
                pesqis.update(pesqi)

                row = {
                    "utt_idx": total_cnt,
                    "utt_id": utt_id,
                    "num_samples": end,
                    "audio_duration_sec": float(audio_duration),
                    "num_chunks": int(num_chunks),
                    "chunk_duration_sec": float(args.chunk_duration),
                    "chunk_hop_duration_sec": float(args.chunk_hop_duration),
                    "sdr_mix": float(sdr_mix),
                    "sdr_est": float(sdr_est),
                    "sdri": float(sdri),
                    "si_sdr_mix": float(si_sdr_mix),
                    "si_sdr_est": float(si_sdr_est),
                    "si_sdri": float(si_sdri),
                    "pesq_mix": float(pesq_mix),
                    "pesq_est": float(pesq_est),
                    "pesqi": float(pesqi),
                    "latency_sec": float(latency),
                    "rtf": float(rtf),
                }
                results.append(row)

                msg = (
                    "utt{}\t"
                    "chunks:{}\t"
                    "SDRi:({:.2f}){:.2f}\t"
                    "SI-SDRi:({:.2f}){:.2f}\t"
                    "PESQi:({:.2f}){:.2f}\t"
                    "Latency:({:.4f}){:.4f}s\t"
                    "RTF:({:.4f}){:.4f}"
                ).format(
                    total_cnt,
                    num_chunks,
                    sdris.val,
                    sdris.avg,
                    sisdris.val,
                    sisdris.avg,
                    pesqis.val,
                    pesqis.avg,
                    latencies.val,
                    latencies.avg,
                    rtfs.val,
                    rtfs.avg,
                )

                tqdm.write(msg)
                progress_f.write(msg + "\n")
                progress_f.flush()

                total_cnt += 1

    rtf_corpus = (
        total_latency / total_audio_duration
        if total_audio_duration > 0
        else np.nan
    )

    summary = {
        "num_utts": len(results),
        "sample_rate": fs,
        "pesq_mode": pesq_mode,
        "chunk_duration_sec": float(args.chunk_duration),
        "chunk_hop_duration_sec": float(args.chunk_hop_duration),
        "chunks_avg": mean_metric(results, "num_chunks"),
        "sdr_mix_avg": mean_metric(results, "sdr_mix"),
        "sdr_est_avg": mean_metric(results, "sdr_est"),
        "sdri_avg": mean_metric(results, "sdri"),
        "si_sdr_mix_avg": mean_metric(results, "si_sdr_mix"),
        "si_sdr_est_avg": mean_metric(results, "si_sdr_est"),
        "si_sdri_avg": mean_metric(results, "si_sdri"),
        "pesq_mix_avg": mean_metric(results, "pesq_mix"),
        "pesq_est_avg": mean_metric(results, "pesq_est"),
        "pesqi_avg": mean_metric(results, "pesqi"),
        "latency_avg_sec": mean_metric(results, "latency_sec"),
        "latency_median_sec": percentile_metric(results, "latency_sec", 50),
        "latency_p95_sec": percentile_metric(results, "latency_sec", 95),
        "rtf_avg": mean_metric(results, "rtf"),
        "rtf_corpus": float(rtf_corpus),
        "total_audio_duration_sec": float(total_audio_duration),
        "total_latency_sec": float(total_latency),
        "parameter_m": float(parameter_m),
    }

    detail_csv, summary_csv, summary_txt = save_results(
        results=results,
        summary=summary,
        output_dir=output_dir,
    )

    print(build_summary_text(summary))
    print("INFO: Chunk duration: {:.4f}s".format(args.chunk_duration))
    print("INFO: Chunk hop duration: {:.4f}s".format(args.chunk_hop_duration))
    print("INFO: Average chunks / utt: {:.4f}".format(summary["chunks_avg"]))
    print("INFO: Saved detailed results to: {}".format(detail_csv))
    print("INFO: Saved summary csv to: {}".format(summary_csv))
    print("INFO: Saved summary txt to: {}".format(summary_txt))
    print("INFO: Saved progress log to: {}".format(progress_log_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Chunked Target Speech Extraction Evaluation")

    parser.add_argument("-c", "--config", type=str, default="", help="config file path")
    parser.add_argument("-p", "--chkpt-path", type=str, default="", help="checkpoint path")
    parser.add_argument(
        "-t",
        "--test-set",
        type=str,
        default="",
        help="test set key in config, or direct test set path",
    )
    parser.add_argument("--device", type=int, default=0, help="cuda device id")
    parser.add_argument(
        "--memory",
        type=float,
        default=0,
        help="GPU memory size in GB to reserve before evaluation; 0 disables it",
    )
    parser.add_argument(
        "--aux-duration",
        type=float,
        default=None,
        help="optional fixed enrollment/aux duration in seconds",
    )
    parser.add_argument(
        "--pesq-mode",
        type=str,
        default="auto",
        choices=["auto", "nb", "wb"],
        help="PESQ mode. auto: 8k->nb, 16k->wb",
    )
    parser.add_argument(
        "--speed-warmup",
        type=int,
        default=3,
        help="number of warmup chunked forward passes before timing the first utterance",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="chunk length in seconds; default uses config valid_chunk_duration or duration",
    )
    parser.add_argument(
        "--chunk-hop-duration",
        type=float,
        default=None,
        help="chunk hop in seconds; default uses config valid_chunk_hop_duration or chunk duration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="optional explicit output directory",
    )

    args = parser.parse_args()

    for f in [args.config, args.chkpt_path]:
        assert os.path.isfile(f), "No such file: %s" % f

    with open(args.config, "r") as f:
        config_strings = f.read()

    config = load_hyperpyyaml(config_strings)

    if args.chunk_duration is None:
        args.chunk_duration = float(
            config.get("valid_chunk_duration", config.get("duration", 4.0))
        )
    if args.chunk_hop_duration is None:
        args.chunk_hop_duration = float(
            config.get("valid_chunk_hop_duration", args.chunk_duration)
        )

    print("INFO: Loaded hparams from: {}".format(args.config))
    main(config, args)
