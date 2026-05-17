#!/usr/bin/env python3
"""
Convert a source checkpoint into an initialization checkpoint for a target model.

Why this script exists
----------------------
When transferring from USEF-SepFormer to a different backend such as TCN,
TCCTCN, CausalTCN, or Mamba, only some parameters are actually reusable. The
valid subset is determined by the target model's state_dict: a source tensor is
kept only when the normalized key exists in the target model and the tensor
shape is exactly the same.

Typical usage:
    python3 tools/convert_checkpoint_for_target.py \
      --source-chkpt chkpt/USEF-SepFormer/wsj0-2mix/temp_best.pth.tar \
      --target-config config/config-USEF-TCN.yaml \
      --output chkpt/init_frontend_from_sepformer_wsj0-2mix_for_tcn.pth.tar

For a stricter frontend-only conversion:
    python3 tools/convert_checkpoint_for_target.py \
      --source-chkpt chkpt/USEF-SepFormer/wsj0-2mix/temp_best.pth.tar \
      --target-config config/config-USEF-TCCTCN-V2.yaml \
      --output chkpt/init_frontend_from_sepformer_wsj0-2mix_for_tcctcn.pth.tar \
      --include-prefix encoder decoder conv1d1 fusion_mdl fusion_norm film
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from hyperpyyaml import load_hyperpyyaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return load_hyperpyyaml(f.read())


def get_by_dotted_path(obj: MutableMapping[str, Any], dotted_path: str) -> Any:
    cur: Any = obj
    for part in dotted_path.split("."):
        if not isinstance(cur, MutableMapping) or part not in cur:
            raise KeyError(f"Cannot find '{dotted_path}' in target config")
        cur = cur[part]
    return cur


def extract_state_dict(ckpt: Any, state_key: str) -> MutableMapping[str, Any]:
    if isinstance(ckpt, MutableMapping):
        if state_key in ckpt:
            state = ckpt[state_key]
        elif all(hasattr(v, "shape") for v in ckpt.values()):
            state = ckpt
        else:
            raise KeyError(
                f"Cannot find state key '{state_key}'. Available keys: {list(ckpt.keys())}"
            )
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(ckpt)!r}")

    if not isinstance(state, MutableMapping):
        raise TypeError(f"Checkpoint state is not a mapping: {type(state)!r}")
    return state


def normalize_key(
    key: str,
    strip_module_prefix: bool,
    speechbrain_key_compat: bool,
    replacements: Iterable[Tuple[str, str]],
) -> str:
    if strip_module_prefix:
        while key.startswith("module."):
            key = key[len("module.") :]

    if speechbrain_key_compat:
        key = re.sub(r"convolution_(?!module)", "convolution_module.", key)

    for old, new in replacements:
        key = key.replace(old, new)

    return key


def parse_replacements(items: Iterable[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Replacement must be OLD=NEW, got: {item!r}")
        old, new = item.split("=", 1)
        if old == "":
            raise ValueError(f"Replacement OLD side cannot be empty: {item!r}")
        parsed.append((old, new))
    return parsed


def prefix_match(key: str, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        prefix = prefix.rstrip(".")
        if key == prefix or key.startswith(prefix + "."):
            return True
    return False


def shape_of(value: Any) -> Optional[List[int]]:
    if hasattr(value, "shape"):
        return list(value.shape)
    return None


def top_prefix(key: str) -> str:
    return key.split(".", 1)[0]


def convert_state_dict(
    source_state: MutableMapping[str, Any],
    target_state: MutableMapping[str, Any],
    include_prefixes: List[str],
    exclude_prefixes: List[str],
    strip_module_prefix: bool,
    speechbrain_key_compat: bool,
    replacements: List[Tuple[str, str]],
) -> Tuple[OrderedDict, Dict[str, Any]]:
    converted = OrderedDict()
    seen_normalized = set()

    report: Dict[str, Any] = {
        "source_keys": len(source_state),
        "target_keys": len(target_state),
        "matched_keys": [],
        "duplicate_normalized_keys": [],
        "skipped_by_include": [],
        "skipped_by_exclude": [],
        "skipped_missing_in_target": [],
        "skipped_shape_mismatch": [],
    }

    for source_key, source_value in source_state.items():
        key = normalize_key(
            source_key,
            strip_module_prefix=strip_module_prefix,
            speechbrain_key_compat=speechbrain_key_compat,
            replacements=replacements,
        )

        if key in seen_normalized:
            report["duplicate_normalized_keys"].append(
                {"source_key": source_key, "normalized_key": key}
            )
        seen_normalized.add(key)

        if include_prefixes and not prefix_match(key, include_prefixes):
            report["skipped_by_include"].append(key)
            continue

        if exclude_prefixes and prefix_match(key, exclude_prefixes):
            report["skipped_by_exclude"].append(key)
            continue

        target_value = target_state.get(key)
        if target_value is None:
            report["skipped_missing_in_target"].append(key)
            continue

        source_shape = shape_of(source_value)
        target_shape = shape_of(target_value)
        if source_shape != target_shape:
            report["skipped_shape_mismatch"].append(
                {
                    "key": key,
                    "source_shape": source_shape,
                    "target_shape": target_shape,
                }
            )
            continue

        converted[key] = source_value.detach().cpu() if hasattr(source_value, "detach") else source_value
        report["matched_keys"].append(key)

    matched_by_prefix: Dict[str, int] = {}
    for key in report["matched_keys"]:
        prefix = top_prefix(key)
        matched_by_prefix[prefix] = matched_by_prefix.get(prefix, 0) + 1

    report["matched_key_count"] = len(report["matched_keys"])
    report["matched_by_prefix"] = dict(sorted(matched_by_prefix.items()))
    report["skipped_by_include_count"] = len(report["skipped_by_include"])
    report["skipped_by_exclude_count"] = len(report["skipped_by_exclude"])
    report["skipped_missing_in_target_count"] = len(report["skipped_missing_in_target"])
    report["skipped_shape_mismatch_count"] = len(report["skipped_shape_mismatch"])
    report["duplicate_normalized_key_count"] = len(report["duplicate_normalized_keys"])

    return converted, report


def default_report_path(output_path: Path) -> Path:
    return Path(str(output_path) + ".report.json")


def save_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")


def print_summary(report: Dict[str, Any]) -> None:
    print("[SUMMARY]")
    print(f"  source keys:                 {report['source_keys']}")
    print(f"  target keys:                 {report['target_keys']}")
    print(f"  matched keys:                {report['matched_key_count']}")
    print(f"  skipped by include filter:   {report['skipped_by_include_count']}")
    print(f"  skipped by exclude filter:   {report['skipped_by_exclude_count']}")
    print(f"  skipped missing in target:   {report['skipped_missing_in_target_count']}")
    print(f"  skipped shape mismatch:      {report['skipped_shape_mismatch_count']}")
    print(f"  duplicate normalized keys:   {report['duplicate_normalized_key_count']}")
    print("  matched by top-level prefix:")
    for prefix, count in report["matched_by_prefix"].items():
        print(f"    {prefix}: {count}")

    mismatches = report["skipped_shape_mismatch"][:10]
    if mismatches:
        print("  first shape mismatches:")
        for item in mismatches:
            print(
                "    {key}: source={source_shape}, target={target_shape}".format(
                    **item
                )
            )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a partial initialization checkpoint by keeping source tensors "
            "that are loadable by a target model."
        )
    )
    parser.add_argument("--source-chkpt", required=True, type=Path)
    parser.add_argument("--target-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--target-module",
        default="modules.masknet",
        help="Dotted config path to the target nn.Module; default: modules.masknet",
    )
    parser.add_argument(
        "--source-state-key",
        default="model_state_dict",
        help="Key containing the source state_dict; default: model_state_dict",
    )
    parser.add_argument(
        "--include-prefix",
        nargs="*",
        default=[],
        help="Only keep keys under these top-level/dotted prefixes. Default keeps all matching keys.",
    )
    parser.add_argument(
        "--exclude-prefix",
        nargs="*",
        default=[],
        help="Drop keys under these top-level/dotted prefixes after include filtering.",
    )
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Additional key string replacement. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-strip-module-prefix",
        action="store_true",
        help="Do not strip leading DataParallel 'module.' prefixes.",
    )
    parser.add_argument(
        "--no-speechbrain-key-compat",
        action="store_true",
        help="Do not apply the convolution_ -> convolution_module. key compatibility rewrite.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path for JSON conversion report. Default: <output>.report.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print/report the conversion without writing the checkpoint.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow saving a checkpoint with zero matched tensors.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    if not args.source_chkpt.is_file():
        raise FileNotFoundError(args.source_chkpt)
    if not args.target_config.is_file():
        raise FileNotFoundError(args.target_config)

    replacements = parse_replacements(args.replace)

    print(f"[INFO] source checkpoint: {args.source_chkpt}")
    print(f"[INFO] target config:     {args.target_config}")
    print(f"[INFO] target module:     {args.target_module}")

    source_ckpt = torch.load(args.source_chkpt, map_location="cpu", weights_only=False)
    source_state = extract_state_dict(source_ckpt, args.source_state_key)

    target_config = load_config(args.target_config)
    target_model = get_by_dotted_path(target_config, args.target_module)
    target_state = target_model.state_dict()

    converted, report = convert_state_dict(
        source_state=source_state,
        target_state=target_state,
        include_prefixes=args.include_prefix,
        exclude_prefixes=args.exclude_prefix,
        strip_module_prefix=not args.no_strip_module_prefix,
        speechbrain_key_compat=not args.no_speechbrain_key_compat,
        replacements=replacements,
    )

    report.update(
        {
            "source_checkpoint": str(args.source_chkpt),
            "target_config": str(args.target_config),
            "target_module": args.target_module,
            "output": str(args.output),
            "include_prefixes": args.include_prefix,
            "exclude_prefixes": args.exclude_prefix,
            "replacements": [{"old": old, "new": new} for old, new in replacements],
            "strip_module_prefix": not args.no_strip_module_prefix,
            "speechbrain_key_compat": not args.no_speechbrain_key_compat,
        }
    )

    print_summary(report)

    if len(converted) == 0 and not args.allow_empty:
        raise RuntimeError("No tensors matched. Use --allow-empty only if this is expected.")

    report_path = args.report_path or default_report_path(args.output)
    save_report(report_path, report)
    print(f"[INFO] wrote conversion report: {report_path}")

    if args.dry_run:
        print("[INFO] dry run; checkpoint was not written")
        return

    new_ckpt = {
        "epoch": 0,
        "model_state_dict": converted,
        "optimizer_state": None,
        "optimizer_state_dict": None,
        "best_val_loss": float("inf"),
        "conversion_report": {
            "source_checkpoint": str(args.source_chkpt),
            "target_config": str(args.target_config),
            "matched_key_count": report["matched_key_count"],
            "matched_by_prefix": report["matched_by_prefix"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, args.output)
    print(f"[OK] wrote converted checkpoint: {args.output}")


if __name__ == "__main__":
    main()
