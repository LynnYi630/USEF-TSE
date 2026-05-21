import os
import sys
sys.path.append('../../..')

import argparse
import csv
import time
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from mir_eval.separation import bss_eval_sources
from hyperpyyaml import load_hyperpyyaml
from pesq import pesq
import librosa


class RunningAverage:
    def __init__(self):
        self.val = np.nan
        self.avg = np.nan
        self.sum = 0.0
        self.count = 0

    def update(self, value):
        self.val = float(value)
        if np.isfinite(self.val):
            self.sum += self.val
            self.count += 1
            self.avg = self.sum / self.count


class te_dataset(Dataset):
    def __init__(self, mix_scp, ref_scp, aux_scp, fs, aux_duration=None):
        self.mix = {x.split()[0]: x.split()[1] for x in open(mix_scp)}
        self.ref = {x.split()[0]: x.split()[1] for x in open(ref_scp)}
        self.aux = {x.split()[0]: x.split()[1] for x in open(aux_scp)}

        assert len(self.mix) == len(self.ref) == len(self.aux)

        wav_id = []
        for l in open(mix_scp):
            wav_id.append(l.split()[0])

        self.wav_id = wav_id
        self.fs = fs
        self.len = len(self.mix)
        self.aux_len = int(aux_duration * fs) if aux_duration is not None else None

    def _trun_wav(self, y, tlen, offset=0):
        if y.shape[0] < tlen:
            npad = tlen - y.shape[0]
            y = np.pad(y, (0, npad), mode='constant', constant_values=0)
        else:
            y = y[offset:offset + tlen]
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

        return mix_wav, target_wav, aux_wav

    def __len__(self):
        return self.len


def reserve_gpu_memory(device_id=0, size_in_gb=0):
    """
    预占指定大小的 GPU 显存。

    原理：
    先申请一个大 tensor，再删除引用。
    PyTorch 通常会把这部分显存留在缓存池中，不立刻还给系统。
    注意：不要在这里调用 torch.cuda.empty_cache()。
    """
    if size_in_gb is None or size_in_gb <= 0:
        print("INFO: --memory <= 0, skip GPU memory reservation.")
        return

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, skip GPU memory reservation.")
        return

    print("INFO: Trying to reserve {} GB memory on cuda:{} ...".format(size_in_gb, device_id))

    try:
        device = torch.device("cuda:{}".format(device_id))
        dummy_tensor = torch.empty(
            int(size_in_gb * 1024 ** 3),
            dtype=torch.int8,
            device=device
        )

        del dummy_tensor

        print("INFO: Successfully reserved about {} GB GPU memory.".format(size_in_gb))
        print("INFO: Do not call torch.cuda.empty_cache(), otherwise reserved memory may be released.")

    except RuntimeError as e:
        print("WARNING: Failed to reserve {} GB GPU memory.".format(size_in_gb))
        print("WARNING: Error message: {}".format(str(e)))


def load_pretrained_modules(model, ckpt_path):
    # PyTorch 2.6+ 默认 weights_only=True，旧 checkpoint 可能加载失败
    model_info = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        name = k.replace("module.", "").replace("convolution_", "convolution_module.")
        state_dict[name] = v

    model.load_state_dict(state_dict)
    return model


def get_test_path(config, args):
    if args.test_set:
        if args.test_set in config:
            return config[args.test_set]
        if os.path.exists(args.test_set):
            return args.test_set

    if 'test_path' in config:
        return config['test_path']

    if config.get('dataset_type') in config:
        return config[config['dataset_type']]

    raise KeyError(
        "Cannot find test data path. "
        "Please pass --test-set, or set test_path in config."
    )


def _safe_path_part(value):
    value = str(value).strip().strip("/\\")
    for ch in '<>:"\\|?*':
        value = value.replace(ch, "_")
    value = value.replace("/", "_")
    return value or "unknown"


def infer_model_name(config, config_path):
    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    if config_stem.startswith("config-"):
        return _safe_path_part(config_stem[len("config-"):])
    if config_stem and config_stem != "config":
        return _safe_path_part(config_stem)

    if config.get("name"):
        return _safe_path_part(str(config["name"]).strip("/").split("/")[0])

    parent = os.path.basename(os.path.dirname(os.path.abspath(config_path)))
    return _safe_path_part(parent)


def infer_dataset_name(config, args, test_path):
    if config.get("dataset_type"):
        return _safe_path_part(config["dataset_type"])
    if args.test_set and args.test_set in config:
        return _safe_path_part(args.test_set)
    if config.get("name") and "/" in str(config["name"]):
        return _safe_path_part(str(config["name"]).strip("/").split("/")[-1])
    return _safe_path_part(os.path.basename(os.path.normpath(test_path)))


def infer_output_dir(config, args, test_path):
    if getattr(args, "output_dir", ""):
        return args.output_dir

    model_name = infer_model_name(config, args.config)
    dataset_name = infer_dataset_name(config, args, test_path)
    return os.path.join("eval_logs", model_name, dataset_name)


def get_pesq_mode(fs, pesq_mode):
    fs = int(fs)

    if pesq_mode != "auto":
        return pesq_mode

    if fs == 8000:
        return "nb"
    elif fs == 16000:
        return "wb"
    else:
        raise ValueError(
            "PESQ only supports fs=8000 or fs=16000, but got fs={}".format(fs)
        )


def si_snr(x, s, remove_dc=True):
    """
    Compute scale-invariant SNR.

    In many speech separation papers, this implementation is also reported as SI-SDR.
    x: enhanced / separated signal
    s: reference / target signal
    """
    eps = 1e-8

    def vec_l2norm(v):
        return np.linalg.norm(v, 2)

    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)

        t = np.inner(x_zm, s_zm) * s_zm / (vec_l2norm(s_zm) ** 2 + eps)
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / (vec_l2norm(s) ** 2 + eps)
        n = x - t

    return 20 * np.log10((vec_l2norm(t) + eps) / (vec_l2norm(n) + eps))


def count_parameters_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_forward_latency(model, mix, embd, device):
    """
    只测模型 forward 时间。

    不包含：
    - wav 读取
    - SDR / SI-SDR / PESQ 计算
    - CSV 写入
    """
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize(device)
        starter.record()

        est_source = model(mix, embd)

        ender.record()
        torch.cuda.synchronize(device)

        latency_sec = starter.elapsed_time(ender) / 1000.0
    else:
        start_time = time.perf_counter()
        est_source = model(mix, embd)
        latency_sec = time.perf_counter() - start_time

    return est_source, latency_sec


def mean_metric(results, key):
    vals = []

    for r in results:
        v = r.get(key, np.nan)
        try:
            v = float(v)
        except Exception:
            continue

        if np.isfinite(v):
            vals.append(v)

    if len(vals) == 0:
        return np.nan

    return float(np.mean(vals))


def percentile_metric(results, key, q):
    vals = []

    for r in results:
        v = r.get(key, np.nan)
        try:
            v = float(v)
        except Exception:
            continue

        if np.isfinite(v):
            vals.append(v)

    if len(vals) == 0:
        return np.nan

    return float(np.percentile(vals, q))


def fmt(x, n=4):
    try:
        x = float(x)
    except Exception:
        return str(x)

    if not np.isfinite(x):
        return "nan"

    return "{:.{}f}".format(x, n)


def build_summary_text(summary):
    lines = []

    lines.append("")
    lines.append("=" * 70)
    lines.append("Final Evaluation Summary")
    lines.append("=" * 70)

    lines.append("Number of utterances:      {}".format(summary["num_utts"]))
    lines.append("Sample rate:               {} Hz".format(summary["sample_rate"]))
    lines.append("PESQ mode:                 {}".format(summary["pesq_mode"]))

    lines.append("-" * 70)
    lines.append("Quality Metrics")
    lines.append("-" * 70)

    lines.append("SDR-mix avg:              {} dB".format(fmt(summary["sdr_mix_avg"])))
    lines.append("SDR-est avg:              {} dB".format(fmt(summary["sdr_est_avg"])))
    lines.append("SDRi avg:                 {} dB".format(fmt(summary["sdri_avg"])))

    lines.append("SI-SDR/SI-SNR-mix avg:    {} dB".format(fmt(summary["si_sdr_mix_avg"])))
    lines.append("SI-SDR/SI-SNR-est avg:    {} dB".format(fmt(summary["si_sdr_est_avg"])))
    lines.append("SI-SDRi/SI-SNRi avg:      {} dB".format(fmt(summary["si_sdri_avg"])))

    lines.append("PESQ-mix avg:             {}".format(fmt(summary["pesq_mix_avg"])))
    lines.append("PESQ-est avg:             {}".format(fmt(summary["pesq_est_avg"])))
    lines.append("PESQi avg:                {}".format(fmt(summary["pesqi_avg"])))

    lines.append("-" * 70)
    lines.append("Speed Metrics")
    lines.append("-" * 70)

    lines.append("Latency avg:              {} sec / utt".format(fmt(summary["latency_avg_sec"])))
    lines.append("Latency median:           {} sec / utt".format(fmt(summary["latency_median_sec"])))
    lines.append("Latency p95:              {} sec / utt".format(fmt(summary["latency_p95_sec"])))
    lines.append("RTF avg:                  {}".format(fmt(summary["rtf_avg"])))
    lines.append("RTF corpus:               {}".format(fmt(summary["rtf_corpus"])))
    lines.append("Total audio duration:     {} sec".format(fmt(summary["total_audio_duration_sec"])))
    lines.append("Total forward time:       {} sec".format(fmt(summary["total_latency_sec"])))

    lines.append("-" * 70)
    lines.append("Model Size")
    lines.append("-" * 70)

    lines.append("Parameter:                {} M".format(fmt(summary["parameter_m"])))

    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


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
        "rtf"
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

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return detail_csv, summary_csv, summary_txt


def main(config, args):
    fs = int(config["sample_rate"])
    pesq_mode = get_pesq_mode(fs, args.pesq_mode)

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.device))
        torch.cuda.set_device(args.device)
    else:
        device = torch.device("cpu")

    print("INFO: Using device: {}".format(device))

    # 恢复“占显存”功能
    # 建议在加载模型之前执行，让 PyTorch 先建立显存缓存池。
    reserve_gpu_memory(device_id=args.device, size_in_gb=args.memory)

    model = config["modules"]["masknet"]
    model = load_pretrained_modules(model, args.chkpt_path)
    model.to(device)
    model.eval()

    parameter_m = count_parameters_m(model)

    test_path = get_test_path(config, args)
    output_dir = infer_output_dir(config, args, test_path)

    testset = te_dataset(
        mix_scp=os.path.join(test_path, config["mix_scp"]),
        ref_scp=os.path.join(test_path, config["ref_scp"]),
        aux_scp=os.path.join(test_path, config["aux_scp"]),
        fs=fs,
        aux_duration=args.aux_duration
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
                tqdm(test_dataloader, desc="Evaluating")
            ):
                utt_id = testset.wav_id[i]

                mix = mix_wav.to(device, non_blocking=True)
                embd = emb_s1.to(device, non_blocking=True)
                tar = target_wav.to(device, non_blocking=True)

                # 只在第一条样本上做 warmup，warmup 不计入 latency
                if i == 0 and args.speed_warmup > 0:
                    for _ in range(args.speed_warmup):
                        _ = model(mix, embd)

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

                audio_duration = mix.shape[-1] / fs

                est_source, latency = measure_forward_latency(
                    model=model,
                    mix=mix,
                    embd=embd,
                    device=device
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

                mix_wav = mix_np[:end]
                tar_wav = tar_np[:end]
                est_wav = est_source[:end]

                # SDR
                # mir_eval 推荐输入 shape: [num_sources, num_samples]
                sdr_mix = bss_eval_sources(
                    tar_wav[np.newaxis, :],
                    mix_wav[np.newaxis, :]
                )[0][0]

                sdr_est = bss_eval_sources(
                    tar_wav[np.newaxis, :],
                    est_wav[np.newaxis, :]
                )[0][0]

                sdri = sdr_est - sdr_mix

                sdrs_mix.update(sdr_mix)
                sdrs_est.update(sdr_est)
                sdris.update(sdri)

                # SI-SDR / SI-SNR
                si_sdr_mix = si_snr(mix_wav, tar_wav)
                si_sdr_est = si_snr(est_wav, tar_wav)
                si_sdri = si_sdr_est - si_sdr_mix

                sisdrs_mix.update(si_sdr_mix)
                sisdrs_est.update(si_sdr_est)
                sisdris.update(si_sdri)

                # PESQ
                # pesq 包 API: pesq(fs, ref, deg, mode)
                try:
                    pesq_mix = pesq(
                        fs,
                        tar_wav.astype(np.float32),
                        mix_wav.astype(np.float32),
                        pesq_mode
                    )
                    pesq_est = pesq(
                        fs,
                        tar_wav.astype(np.float32),
                        est_wav.astype(np.float32),
                        pesq_mode
                    )
                    pesqi = pesq_est - pesq_mix

                except Exception as e:
                    pesq_mix = np.nan
                    pesq_est = np.nan
                    pesqi = np.nan

                    warn_msg = (
                        "WARNING: PESQ failed at utt {} / {}: {}".format(
                            total_cnt,
                            utt_id,
                            str(e)
                        )
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
                    "SDRi:({:.2f}){:.2f}\t"
                    "SI-SDRi:({:.2f}){:.2f}\t"
                    "PESQi:({:.2f}){:.2f}\t"
                    "Latency:({:.4f}){:.4f}s\t"
                    "RTF:({:.4f}){:.4f}"
                ).format(
                    total_cnt,

                    sdris.val, sdris.avg,
                    sisdris.val, sisdris.avg,
                    pesqis.val, pesqis.avg,

                    latencies.val, latencies.avg,
                    rtfs.val, rtfs.avg,
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
        output_dir=output_dir
    )

    print(build_summary_text(summary))

    print("INFO: Saved detailed results to: {}".format(detail_csv))
    print("INFO: Saved summary csv to: {}".format(summary_csv))
    print("INFO: Saved summary txt to: {}".format(summary_txt))
    print("INFO: Saved progress log to: {}".format(progress_log_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Target Speech Extraction Evaluation")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="",
        help="config file path"
    )

    parser.add_argument(
        "-p",
        "--chkpt-path",
        type=str,
        default="",
        help="path to the chosen checkpoint"
    )

    parser.add_argument(
        "-t",
        "--test-set",
        type=str,
        default="",
        help=(
            "test set key in config, or direct test set path. "
            "For old configs, this may be something like tt or test."
        )
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="cuda device id"
    )

    parser.add_argument(
        "--memory",
        type=float,
        default=0,
        help=(
            "GPU memory size in GB to reserve before evaluation. "
            "Use 0 to disable memory reservation."
        )
    )

    parser.add_argument(
        "--aux-duration",
        type=float,
        default=None,
        help="optional fixed enrollment/aux duration in seconds"
    )

    parser.add_argument(
        "--pesq-mode",
        type=str,
        default="auto",
        choices=["auto", "nb", "wb"],
        help="PESQ mode. auto: 8k->nb, 16k->wb"
    )

    parser.add_argument(
        "--speed-warmup",
        type=int,
        default=3,
        help="number of warmup forward passes before timing the first utterance"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="optional explicit output directory"
    )

    args = parser.parse_args()

    for f in [args.config, args.chkpt_path]:
        assert os.path.isfile(f), "No such file: %s" % f

    with open(args.config, "r") as f:
        config_strings = f.read()

    config = load_hyperpyyaml(config_strings)

    print("INFO: Loaded hparams from: {}".format(args.config))

    main(config, args)
