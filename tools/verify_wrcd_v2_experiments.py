#!/usr/bin/env python3
r"""Run local eval/diagnosis for WRCD-V2 confusion-margin experiments.

This script does not train. It assumes experiment checkpoints have already
been copied back under chkpt/<experiment>/wsj0-2mix/temp_best.pth.tar.

Usage examples:
    # Print the commands that would run.
    python tools/verify_wrcd_v2_experiments.py

    # Run aux=1s eval, full-aux eval, diagnosis, and final analysis.
    python tools/verify_wrcd_v2_experiments.py --run

    # Re-run even if eval_results.csv already exists.
    python tools/verify_wrcd_v2_experiments.py --run --overwrite
"""

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DATASET = "wsj0-2mix"
STANDARD_AUX_DURATION = 1.0


@dataclass(frozen=True)
class EvalTarget:
    label: str
    config: Path
    checkpoint: Path
    model_dir: str


BASELINES = [
    EvalTarget(
        label="TCN-V2",
        config=Path("chkpt/USEF-TCN-V2/wsj0-2mix/config.yaml"),
        checkpoint=Path("chkpt/USEF-TCN-V2/wsj0-2mix/temp_best.pth.tar"),
        model_dir="USEF-TCN-V2",
    ),
    EvalTarget(
        label="WRCD-V2",
        config=Path("chkpt/USEF-TCN-V2-WRCD-V2/wsj0-2mix/config.yaml"),
        checkpoint=Path("chkpt/USEF-TCN-V2-WRCD-V2/wsj0-2mix/temp_best.pth.tar"),
        model_dir="USEF-TCN-V2-WRCD-V2",
    ),
    EvalTarget(
        label="TFGridNet",
        config=Path("chkpt/USEF-TFGridNet/config.yaml"),
        checkpoint=Path("chkpt/USEF-TFGridNet/wsj0-2mix/temp_best.pth.tar"),
        model_dir="USEF-TFGridNet",
    ),
]


EXPERIMENTS = [
    EvalTarget(
        label="CONF005",
        config=Path("chkpt/USEF-TCN-V2-WRCD-V2-CONF005/wsj0-2mix/config.yaml"),
        checkpoint=Path("chkpt/USEF-TCN-V2-WRCD-V2-CONF005/wsj0-2mix/temp_best.pth.tar"),
        model_dir="USEF-TCN-V2-WRCD-V2-CONF005",
    ),
    EvalTarget(
        label="CONF010",
        config=Path("chkpt/USEF-TCN-V2-WRCD-V2-CONF010/wsj0-2mix/config.yaml"),
        checkpoint=Path("chkpt/USEF-TCN-V2-WRCD-V2-CONF010/wsj0-2mix/temp_best.pth.tar"),
        model_dir="USEF-TCN-V2-WRCD-V2-CONF010",
    ),
    EvalTarget(
        label="CONF005-AUXAUG",
        config=Path("chkpt/USEF-TCN-V2-WRCD-V2-CONF005-AUXAUG/wsj0-2mix/config.yaml"),
        checkpoint=Path("chkpt/USEF-TCN-V2-WRCD-V2-CONF005-AUXAUG/wsj0-2mix/temp_best.pth.tar"),
        model_dir="USEF-TCN-V2-WRCD-V2-CONF005-AUXAUG",
    ),
]


def eval_dir(target: EvalTarget, mode: str) -> Path:
    return Path("eval_logs") / target.model_dir / f"{DATASET}_{mode}"


def eval_csv(target: EvalTarget, mode: str) -> Path:
    return eval_dir(target, mode) / "eval_results.csv"


def diagnosis_dir(target: EvalTarget, mode: str) -> Path:
    return eval_dir(target, mode) / "diagnosis_si_sdri_lt5_all"


def command_to_text(command):
    return " ".join(str(part) for part in command)


def run_or_print(command, args):
    print(command_to_text(command))
    if args.run:
        subprocess.run(command, check=True)


def should_skip(path: Path, overwrite: bool) -> bool:
    return path.is_file() and not overwrite


def build_eval_command(target: EvalTarget, mode: str, args):
    command = [
        sys.executable,
        "eval.py",
        "--config",
        str(target.config),
        "--chkpt-path",
        str(target.checkpoint),
        "--device",
        str(args.device),
        "--memory",
        str(args.memory),
        "--output-dir",
        str(eval_dir(target, mode)),
    ]
    if mode == "aux1s":
        command.extend(["--aux-duration", str(STANDARD_AUX_DURATION)])
    return command


def build_diagnosis_command(target: EvalTarget, mode: str, args):
    return [
        sys.executable,
        "tools/diagnose_target_confusion.py",
        "--config",
        str(target.config),
        "--chkpt-path",
        str(target.checkpoint),
        "--eval-results",
        str(eval_csv(target, mode)),
        "--metric",
        "si_sdri",
        "--threshold",
        "5",
        "--max-utts",
        "0",
        "--device",
        str(args.device),
        "--memory",
        str(args.memory),
        "--output-dir",
        str(diagnosis_dir(target, mode)),
    ]


def build_analysis_command(targets, mode):
    command = [
        sys.executable,
        "tools/analyze_eval_logs.py",
        "--metric",
        "si_sdri",
        "--baseline",
        "TCN-V2",
        "--worst-k",
        "10",
    ]
    for target in targets:
        csv_path = eval_csv(target, mode)
        if csv_path.is_file():
            command.extend(["--eval-log", f"{target.label}={csv_path}"])
    return command


def read_diagnosis_counts(path: Path):
    csv_path = path / "target_confusion_diagnosis.csv"
    if not csv_path.is_file():
        return {}
    counts = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get("diagnosis", "unknown")
            counts[key] = counts.get(key, 0) + 1
    return counts


def count_low_metric_rows(path: Path, metric: str = "si_sdri", threshold: float = 5.0) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                value = float(row.get(metric, "nan"))
            except ValueError:
                continue
            if value < threshold:
                count += 1
    return count


def print_diagnosis_summary(targets, mode):
    print(f"\n=== Diagnosis summary ({mode}) ===")
    for target in targets:
        counts = read_diagnosis_counts(diagnosis_dir(target, mode))
        if counts:
            items = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            print(f"{target.label}: {items}")
        else:
            print(f"{target.label}: diagnosis not found")


def select_targets(args):
    labels = set(args.target or [])
    targets = BASELINES + EXPERIMENTS
    if not labels:
        return targets
    selected = [target for target in targets if target.label in labels or target.model_dir in labels]
    missing = labels - {target.label for target in selected} - {target.model_dir for target in selected}
    if missing:
        raise ValueError(f"Unknown target(s): {', '.join(sorted(missing))}")
    return selected


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", action="store_true", help="execute commands instead of printing them")
    parser.add_argument("--overwrite", action="store_true", help="rerun eval/diagnosis even when outputs exist")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--memory", type=float, default=0)
    parser.add_argument(
        "--mode",
        choices=["aux1s", "fullaux", "both"],
        default="both",
        help="which eval protocol to run",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="limit to a target label, e.g. CONF005; can be repeated",
    )
    args = parser.parse_args()

    targets = select_targets(args)
    modes = ["aux1s", "fullaux"] if args.mode == "both" else [args.mode]

    for target in targets:
        if not target.config.is_file() or not target.checkpoint.is_file():
            print(f"[SKIP] {target.label}: missing {target.config} or {target.checkpoint}")
            continue

        for mode in modes:
            csv_path = eval_csv(target, mode)
            if should_skip(csv_path, args.overwrite):
                print(f"[SKIP] {target.label} {mode}: {csv_path} already exists")
            else:
                run_or_print(build_eval_command(target, mode, args), args)

            if csv_path.is_file() or args.run:
                diag_csv = diagnosis_dir(target, mode) / "target_confusion_diagnosis.csv"
                low_rows = count_low_metric_rows(csv_path)
                if low_rows == 0 and csv_path.is_file():
                    print(f"[SKIP] {target.label} {mode}: no si_sdri < 5 rows")
                    continue
                if should_skip(diag_csv, args.overwrite):
                    print(f"[SKIP] {target.label} {mode}: {diag_csv} already exists")
                else:
                    run_or_print(build_diagnosis_command(target, mode, args), args)

    for mode in modes:
        analysis_command = build_analysis_command(targets, mode)
        if len(analysis_command) > 9:
            run_or_print(analysis_command, args)
        print_diagnosis_summary(targets, mode)


if __name__ == "__main__":
    main()
