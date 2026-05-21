#!/usr/bin/env python3
r"""Analyze USEF eval_results.csv files across models and datasets.

The script intentionally uses only the Python standard library so it can run
in lightweight environments where torch/librosa are unavailable.

Usage examples:
    # Discover and summarize all eval_results.csv files under eval_logs/.
    python tools/analyze_eval_logs.py --root .

    # Analyze only one dataset discovered from eval_logs/<model>/<dataset>/.
    python tools/analyze_eval_logs.py --dataset wsj0-2mix --metric si_sdri

    # Compare selected logs and print the worst regressions.
    python tools/analyze_eval_logs.py \
      --eval-log SepFormer=eval_logs/USEF-SepFormer/wsj0-2mix/eval_results.csv \
      --eval-log TCN-V2=eval_logs/USEF-TCN-V2/wsj0-2mix/eval_results.csv \
      --compare SepFormer=TCN-V2 \
      --worst-k 10

    # Summarize a training log in addition to eval CSVs.
    python tools/analyze_eval_logs.py \
      --dataset wsj0-2mix \
      --train-log logs/USEF-TCCTCN-V2/wsj0-2mix/2026-05-15-0148.log
"""

import argparse
import csv
import fnmatch
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DISCOVERY_GLOB = "eval_logs/*/*/eval_results.csv"
DEFAULT_SUMMARY_METRICS = ("sdri", "si_sdri", "pesqi", "latency_sec", "rtf")
DEFAULT_BIN_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0)
TEXT_FIELDS = {"utt_id"}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def maybe_number(field, value):
    if field in TEXT_FIELDS:
        return value
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return value


def finite(values):
    return [x for x in values if isinstance(x, (int, float)) and math.isfinite(x)]


def mean(values):
    values = finite(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def percentile(values, pct):
    values = sorted(finite(values))
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def fmt(value, digits=3):
    if isinstance(value, int):
        return str(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def speaker_id(token):
    if "-" in token:
        return token.split("-", 1)[0]
    if len(token) >= 3:
        return token[:3]
    return token


def parse_utt_metadata(utt_id):
    """Best-effort metadata parser for WSJ0-2mix and Libri2Mix style IDs."""
    parts = str(utt_id).split("_")
    out = {
        "source1_spk": "unknown",
        "source2_spk": "unknown",
        "aux_spk": "unknown",
        "target_side": "unknown",
        "target_sir": float("nan"),
    }
    if len(parts) < 5:
        return out

    out["source1_spk"] = speaker_id(parts[0])
    out["source2_spk"] = speaker_id(parts[2])
    out["aux_spk"] = speaker_id(parts[4])

    sir1 = to_float(parts[1])
    sir2 = to_float(parts[3])
    if out["aux_spk"] == out["source1_spk"]:
        out["target_side"] = "s1"
        out["target_sir"] = sir1
    elif out["aux_spk"] == out["source2_spk"]:
        out["target_side"] = "s2"
        out["target_sir"] = sir2

    return out


def load_rows(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader, start=1):
            parsed = {key: maybe_number(key, value) for key, value in row.items()}
            utt_id = row.get("utt_id") or row.get("utt_idx") or str(index)
            parsed["utt_id"] = utt_id
            parsed.update(parse_utt_metadata(utt_id))
            rows[utt_id] = parsed
    return rows


def infer_log_name(path, root):
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path

    parts = rel.parts
    if len(parts) >= 4 and parts[-1] == "eval_results.csv" and parts[-4] == "eval_logs":
        return f"{parts[-3]}/{parts[-2]}"
    if path.name == "eval_results.csv" and path.parent.parent != path.parent:
        return f"{path.parent.parent.name}/{path.parent.name}"
    return str(rel)


def parse_eval_log_arg(item, root):
    if "=" in item:
        name, path_text = item.split("=", 1)
        name = name.strip()
        path = Path(path_text.strip())
    else:
        path = Path(item.strip())
        name = ""

    if not path.is_absolute():
        path = root / path
    if not name:
        name = infer_log_name(path, root)
    return name, path


def matches_any(value, patterns):
    if not patterns:
        return True
    return any(value == pattern or fnmatch.fnmatch(value, pattern) for pattern in patterns)


def discover_eval_logs(root, pattern, model_patterns, dataset_patterns):
    logs = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        model = path.parent.parent.name
        dataset = path.parent.name
        if not matches_any(model, model_patterns):
            continue
        if not matches_any(dataset, dataset_patterns):
            continue
        logs.append((infer_log_name(path, root), path))
    return logs


def unique_log_names(logs):
    counts = Counter(name for name, _ in logs)
    seen = Counter()
    out = []
    for name, path in logs:
        if counts[name] == 1:
            out.append((name, path))
            continue
        seen[name] += 1
        out.append((f"{name}#{seen[name]}", path))
    return out


def metric_values(rows, metric):
    return [row.get(metric, float("nan")) for row in rows.values()]


def available_metrics(rows_by_log):
    names = set()
    for rows in rows_by_log.values():
        for row in rows.values():
            for key, value in row.items():
                if key in TEXT_FIELDS or key.startswith("source") or key.endswith("_spk"):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value):
                    names.add(key)
    return names


def metric_label(metric):
    labels = {
        "sdri": "SDRi",
        "si_sdri": "SI-SDRi",
        "pesqi": "PESQi",
        "latency_sec": "latency",
        "rtf": "RTF",
    }
    return labels.get(metric, metric)


def print_table(headers, rows):
    if not rows:
        print("(no rows)")
        return
    headers = [str(header) for header in headers]
    rows = [[str(value) for value in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    fmt_row = " | ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt_row.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row.format(*row))


def summarize_log(name, rows, summary_metrics, primary_metric):
    items = list(rows.values())
    primary = metric_values(rows, primary_metric)
    row = [name, len(items)]

    for metric in summary_metrics:
        row.append(fmt(mean(metric_values(rows, metric))))

    row.extend(
        [
            fmt(percentile(primary, 5)),
            fmt(percentile(primary, 25)),
            fmt(percentile(primary, 50)),
            fmt(percentile(primary, 75)),
            fmt(percentile(primary, 95)),
            str(sum(1 for x in primary if isinstance(x, (int, float)) and math.isfinite(x) and x < 0)),
            str(sum(1 for x in primary if isinstance(x, (int, float)) and math.isfinite(x) and x < 5)),
            str(sum(1 for x in primary if isinstance(x, (int, float)) and math.isfinite(x) and x < 10)),
        ]
    )
    return row


def metric_bin(value, edges=DEFAULT_BIN_EDGES):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "unknown"
    if value < edges[0]:
        return f"<{fmt(edges[0], 0)}"
    for lo, hi in zip(edges, edges[1:]):
        if value < hi:
            return f"{fmt(lo, 0)}-{fmt(hi, 0)}"
    return f">={fmt(edges[-1], 0)}"


def sir_bin(sir):
    if not isinstance(sir, (int, float)) or not math.isfinite(sir):
        return "unknown"
    if sir < -3:
        return "< -3 dB"
    if sir < 0:
        return "[-3, 0) dB"
    if sir < 3:
        return "[0, 3) dB"
    return ">= 3 dB"


def summarize_bins(name, rows, metric, field, key_func):
    grouped = defaultdict(list)
    for row in rows.values():
        grouped[key_func(row.get(field))].append(row.get(metric, float("nan")))

    order = ["s1", "s2", "< -3 dB", "[-3, 0) dB", "[0, 3) dB", ">= 3 dB", "unknown"]
    keys = [key for key in order if key in grouped] + sorted(
        key for key in grouped if key not in order
    )
    table = []
    for key in keys:
        vals = grouped[key]
        table.append(
            [
                name,
                key,
                len(vals),
                fmt(mean(vals)),
                fmt(percentile(vals, 50)),
                str(sum(v < 5 for v in vals if isinstance(v, (int, float)) and math.isfinite(v))),
            ]
        )
    return table


def compare_logs(base_name, target_name, base_rows, target_rows, metric, delta_threshold):
    common = sorted(set(base_rows) & set(target_rows))
    deltas = []
    ranked = []
    for utt in common:
        base = base_rows[utt].get(metric, float("nan"))
        target = target_rows[utt].get(metric, float("nan"))
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (base, target)):
            continue
        delta = target - base
        deltas.append(delta)
        ranked.append((delta, utt, target, base))
    ranked.sort()
    return {
        "base": base_name,
        "target": target_name,
        "common": len(deltas),
        "delta_mean": mean(deltas),
        "delta_p05": percentile(deltas, 5),
        "delta_p50": percentile(deltas, 50),
        "delta_p95": percentile(deltas, 95),
        "worse": sum(d < -delta_threshold for d in deltas),
        "better": sum(d > delta_threshold for d in deltas),
        "worst": ranked,
    }


def pearson_for_metric(rows_a, rows_b, metric):
    pairs = []
    for utt in sorted(set(rows_a) & set(rows_b)):
        x = rows_a[utt].get(metric, float("nan"))
        y = rows_b[utt].get(metric, float("nan"))
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y)):
            pairs.append((x, y))
    if len(pairs) < 2:
        return float("nan")

    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mx = mean(xs)
    my = mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def resolve_name(token, names):
    if token in names:
        return token
    matches = [name for name in names if token in name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Cannot resolve log name {token!r}; available: {', '.join(names)}")
    raise ValueError(f"Ambiguous log name {token!r}; matches: {', '.join(matches)}")


def build_comparisons(args, names):
    if args.compare:
        pairs = []
        for item in args.compare:
            if "=" in item:
                base, target = item.split("=", 1)
            elif ":" in item:
                base, target = item.split(":", 1)
            else:
                raise ValueError("--compare must be BASE=TARGET or BASE:TARGET")
            pairs.append((resolve_name(base.strip(), names), resolve_name(target.strip(), names)))
        return pairs

    if args.baseline:
        base = resolve_name(args.baseline, names)
        return [(base, name) for name in names if name != base]

    pairs = []
    for i, base in enumerate(names):
        for target in names[i + 1:]:
            pairs.append((base, target))
    return pairs


def parse_training_log(path):
    train = []
    valid = []
    lrs = []
    anti_nonzero = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"epoch start Learning rate: ([0-9.]+)", line)
            if m:
                lrs.append(float(m.group(1)))

            m = re.search(
                r"Train Summary \| End of Epoch\s+(\d+) \| Time [^|]+ \| Train Loss (-?[0-9.]+)",
                line,
            )
            if m:
                train.append((int(m.group(1)), float(m.group(2))))

            m = re.search(
                r"Valid Summary \| End of Epoch\s+(\d+) \| Time [^|]+ \| Valid Loss (-?[0-9.]+)",
                line,
            )
            if m:
                valid.append((int(m.group(1)), float(m.group(2))))

            m = re.search(r"Anti ([0-9.]+)", line)
            if m and float(m.group(1)) != 0.0:
                anti_nonzero += 1
    return train, valid, lrs, anti_nonzero


def print_training_log_summary(path):
    train, valid, lrs, anti_nonzero = parse_training_log(path)
    print("\n=== Training log summary ===")
    print(f"log: {path}")
    print(f"train summaries: {len(train)}")
    print(f"valid summaries: {len(valid)}")
    print(f"learning-rate entries: {len(lrs)}")
    print(f"non-zero Anti log entries: {anti_nonzero}")
    if not train or not valid:
        return

    best_valid = min(valid, key=lambda x: x[1])
    print_table(
        ["item", "epoch", "loss", "approx SI-SNR dB"],
        [
            ["first train", train[0][0], fmt(train[0][1]), fmt(-train[0][1])],
            ["last train", train[-1][0], fmt(train[-1][1]), fmt(-train[-1][1])],
            ["first valid", valid[0][0], fmt(valid[0][1]), fmt(-valid[0][1])],
            ["last valid", valid[-1][0], fmt(valid[-1][1]), fmt(-valid[-1][1])],
            ["best valid", best_valid[0], fmt(best_valid[1]), fmt(-best_valid[1])],
        ],
    )

    valid_by_epoch = dict(valid)
    gaps = []
    for epoch, train_loss in train:
        if epoch in valid_by_epoch:
            gaps.append((epoch, -train_loss - (-valid_by_epoch[epoch])))
    if gaps:
        last_gap = gaps[-1]
        last10 = gaps[-10:]
        print(
            "train-valid SI-SNR gap: last epoch {:.3f} dB, last-10 mean {:.3f} dB".format(
                last_gap[1],
                sum(gap for _, gap in last10) / len(last10),
            )
        )


def parse_metric_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def build_argparser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument(
        "--glob",
        default=DEFAULT_DISCOVERY_GLOB,
        help=f"discovery glob relative to --root; default: {DEFAULT_DISCOVERY_GLOB}",
    )
    parser.add_argument(
        "--eval-log",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="explicit eval_results.csv to analyze; can be passed multiple times",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model name or glob filter used during discovery",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="dataset name or glob filter used during discovery",
    )
    parser.add_argument(
        "--metric",
        default="si_sdri",
        help="primary metric for percentiles, bins, and comparisons",
    )
    parser.add_argument(
        "--summary-metrics",
        default=",".join(DEFAULT_SUMMARY_METRICS),
        help="comma-separated metric means to print in the overall table",
    )
    parser.add_argument(
        "--baseline",
        default="",
        help="compare every other log against this log name or unique substring",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="BASE=TARGET",
        help="specific pairwise comparison; can be passed multiple times",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=3.0,
        help="absolute delta threshold for worse/better counts",
    )
    parser.add_argument(
        "--worst-k",
        type=int,
        default=0,
        help="print this many worst utterances for each comparison; default: 0",
    )
    parser.add_argument(
        "--train-log",
        default="",
        help="optional training log to summarize",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    root = Path(args.root)

    if args.eval_log:
        logs = [parse_eval_log_arg(item, root) for item in args.eval_log]
    else:
        logs = discover_eval_logs(root, args.glob, args.model, args.dataset)

    logs = unique_log_names(logs)
    missing = [str(path) for _, path in logs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing eval log(s): " + ", ".join(missing))
    if not logs:
        raise RuntimeError("No eval_results.csv files found. Use --eval-log or adjust --glob.")

    rows_by_log = {name: load_rows(path) for name, path in logs}
    names = list(rows_by_log)
    metrics_available = available_metrics(rows_by_log)
    summary_metrics = [
        metric for metric in parse_metric_list(args.summary_metrics)
        if metric in metrics_available
    ]
    if args.metric not in metrics_available:
        raise RuntimeError(
            f"Primary metric {args.metric!r} was not found. "
            f"Available numeric metrics: {', '.join(sorted(metrics_available))}"
        )

    print("\n=== Eval logs ===")
    print_table(["name", "path", "rows"], [[name, path, len(rows_by_log[name])] for name, path in logs])

    print("\n=== Overall distribution ===")
    headers = ["log", "N"]
    headers.extend(f"{metric_label(metric)} mean" for metric in summary_metrics)
    headers.extend(
        [
            f"{metric_label(args.metric)} p05",
            f"{metric_label(args.metric)} p25",
            f"{metric_label(args.metric)} med",
            f"{metric_label(args.metric)} p75",
            f"{metric_label(args.metric)} p95",
            f"{metric_label(args.metric)}<0",
            f"{metric_label(args.metric)}<5",
            f"{metric_label(args.metric)}<10",
        ]
    )
    print_table(
        headers,
        [summarize_log(name, rows_by_log[name], summary_metrics, args.metric) for name in names],
    )

    print(f"\n=== {metric_label(args.metric)} bins ===")
    bin_rows = []
    bin_order = ["<0", "0-5", "5-10", "10-15", "15-20", ">=20", "unknown"]
    for name in names:
        counts = Counter(metric_bin(row.get(args.metric)) for row in rows_by_log[name].values())
        for key in bin_order:
            if counts.get(key, 0):
                bin_rows.append([name, key, counts[key]])
    print_table(["log", "bin", "count"], bin_rows)

    has_target_side = any(
        row.get("target_side") in {"s1", "s2"}
        for rows in rows_by_log.values()
        for row in rows.values()
    )
    if has_target_side:
        print(f"\n=== {metric_label(args.metric)} by target side ===")
        side_rows = []
        for name in names:
            side_rows.extend(summarize_bins(name, rows_by_log[name], args.metric, "target_side", lambda x: x))
        print_table(["log", "target_side", "N", "mean", "median", "<5"], side_rows)

        print(f"\n=== {metric_label(args.metric)} by target SIR bin ===")
        sir_rows = []
        for name in names:
            sir_rows.extend(summarize_bins(name, rows_by_log[name], args.metric, "target_sir", sir_bin))
        print_table(["log", "target_sir", "N", "mean", "median", "<5"], sir_rows)

    comparison_pairs = build_comparisons(args, names)
    comparison_infos = []
    for base, target in comparison_pairs:
        info = compare_logs(
            base,
            target,
            rows_by_log[base],
            rows_by_log[target],
            args.metric,
            args.delta_threshold,
        )
        if info["common"] > 0:
            comparison_infos.append(info)

    if comparison_infos:
        print(f"\n=== Pairwise {metric_label(args.metric)} deltas ===")
        print_table(
            [
                "delta",
                "N",
                "mean",
                "p05",
                "median",
                "p95",
                f"<-{fmt(args.delta_threshold, 0)}",
                f">+{fmt(args.delta_threshold, 0)}",
                "corr",
            ],
            [
                [
                    f"{info['target']} - {info['base']}",
                    info["common"],
                    fmt(info["delta_mean"]),
                    fmt(info["delta_p05"]),
                    fmt(info["delta_p50"]),
                    fmt(info["delta_p95"]),
                    info["worse"],
                    info["better"],
                    fmt(pearson_for_metric(rows_by_log[info["base"]], rows_by_log[info["target"]], args.metric)),
                ]
                for info in comparison_infos
            ],
        )

        if args.worst_k > 0:
            print(f"\n=== Worst {metric_label(args.metric)} regressions ===")
            for info in comparison_infos:
                print(f"\n{info['target']} - {info['base']} worst {args.worst_k}")
                print_table(
                    ["utt_id", info["target"], info["base"], "delta", "target_sir", "target_side"],
                    [
                        [
                            utt,
                            fmt(target_value),
                            fmt(base_value),
                            fmt(delta),
                            fmt(rows_by_log[info["target"]][utt].get("target_sir")),
                            rows_by_log[info["target"]][utt].get("target_side", "unknown"),
                        ]
                        for delta, utt, target_value, base_value in info["worst"][:args.worst_k]
                    ],
                )
    else:
        print("\n=== Pairwise deltas ===")
        print("(no comparison pairs had common utterance IDs)")

    if args.train_log:
        train_log = Path(args.train_log)
        if not train_log.is_absolute():
            train_log = root / train_log
        if train_log.is_file():
            print_training_log_summary(train_log)
        else:
            print(f"\nTraining log not found: {train_log}", file=sys.stderr)


if __name__ == "__main__":
    main()
