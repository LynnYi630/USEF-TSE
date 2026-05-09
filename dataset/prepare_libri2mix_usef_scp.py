#!/usr/bin/env python3
"""
Prepare USEF-TSE style SCP files for Libri2Mix.

Why this script exists
----------------------
ZBang/USEF-TSE was written around WSJ0-style auxiliary speech directories.
The current dataset/data.py handles Libri2Mix directly via dataset_type:
it builds a speaker index from ref.scp and samples same-speaker aux audio
from the original s1/s2 source wavs, so this script does not need to create
an aux_pool symlink tree.

This script:
  1. Reads Libri2Mix directories:
       Libri2Mix/wav8k/min/{train-100,train-360,dev,test}/{mix_clean,s1,s2}
     or the corresponding wav16k / max / mix_both variants.
  2. Writes USEF-TSE compatible:
       <out-dir>/<subset>/mix.scp
       <out-dir>/<subset>/ref.scp
       <out-dir>/<subset>/aux.scp

aux.scp points directly to original Libri2Mix s1/s2 source wav files.
No auxiliary wavs are copied or symlinked.

Recommended:
  - Use this to train/evaluate a new Libri2Mix USEF-SepFormer model.
  - Do NOT use this to reproduce ZBang's WSJ0-2mix checkpoint results.

To generate scp files:
    python3 prepare_libri2mix_usef_scp.py generate \
      --librimix-root /home/wangyi/datasets/Libri2Mix \
      --out-dir /home/wangyi/datasets/libri2mix_usef_scp \
      --subsets train-360 dev test \
      --sample-rate 8000 \
      --mix-mode min \
      --task sep_clean \
      --aux-policy hash \
      --seed 3407 \
      --check-exists
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple


Task = Literal["sep_clean", "sep_noisy"]
Policy = Literal["hash", "first", "random"]


@dataclass(frozen=True)
class MixtureItem:
    subset: str
    utt1: str
    utt2: str
    mix_path: Path
    s1_path: Path
    s2_path: Path

    @property
    def mix_stem(self) -> str:
        return f"{self.utt1}_{self.utt2}"


@dataclass(frozen=True)
class AuxCandidate:
    utt: str
    path: Path


def speaker_id(utt_id: str) -> str:
    """LibriSpeech utterance id format is speaker-chapter-utterance."""
    return utt_id.split("-", 1)[0]


def resolve_librimix_root(path: Path) -> Path:
    """
    Accept either:
      /path/to/Libri2Mix
    or:
      /path/to/root_containing/Libri2Mix
    """
    path = path.expanduser().resolve()
    if path.name in {"Libri2Mix", "Libri3Mix"}:
        return path
    candidate = path / "Libri2Mix"
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find Libri2Mix under {path}. "
        "Pass either /path/to/Libri2Mix or its parent directory."
    )


def parse_mix_stem(stem: str) -> Tuple[str, str]:
    parts = stem.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Expected Libri2Mix 2-speaker filename stem '<utt1>_<utt2>', got: {stem}"
        )
    return parts[0], parts[1]


def source_utt_from_librimix_path(path: Path) -> str:
    utt1, utt2 = parse_mix_stem(path.stem)
    source_dir = path.parent.name
    if source_dir == "s1":
        return utt1
    if source_dir == "s2":
        return utt2
    raise ValueError(f"Libri2Mix source path must be under s1 or s2: {path}")


def scan_subset(
    libri_root: Path,
    sample_rate: int,
    mix_mode: str,
    subset: str,
    task: Task,
) -> List[MixtureItem]:
    wav_dir = f"wav{sample_rate // 1000}k"
    subset_dir = libri_root / wav_dir / mix_mode / subset
    mix_name = "mix_clean" if task == "sep_clean" else "mix_both"

    mix_dir = subset_dir / mix_name
    s1_dir = subset_dir / "s1"
    s2_dir = subset_dir / "s2"

    for d in (mix_dir, s1_dir, s2_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Missing directory: {d}")

    items: List[MixtureItem] = []
    for mix_path in sorted(mix_dir.glob("*.wav")):
        utt1, utt2 = parse_mix_stem(mix_path.stem)
        s1_path = s1_dir / mix_path.name
        s2_path = s2_dir / mix_path.name
        if not s1_path.is_file():
            raise FileNotFoundError(f"Missing source file: {s1_path}")
        if not s2_path.is_file():
            raise FileNotFoundError(f"Missing source file: {s2_path}")
        items.append(
            MixtureItem(
                subset=subset,
                utt1=utt1,
                utt2=utt2,
                mix_path=mix_path.absolute(),
                s1_path=s1_path.absolute(),
                s2_path=s2_path.absolute(),
            )
        )
    if not items:
        raise RuntimeError(f"No wav files found in {mix_dir}")
    return items


def build_aux_index(
    items_by_subset: Dict[str, List[MixtureItem]],
) -> Dict[str, Dict[str, List[AuxCandidate]]]:
    """Build pool[subset][speaker] -> original Libri2Mix source wavs."""
    pool: Dict[str, Dict[str, List[AuxCandidate]]] = {}
    for subset, items in items_by_subset.items():
        subset_pool: Dict[str, List[AuxCandidate]] = {}
        seen = set()
        for item in items:
            entries = [
                (item.utt1, item.s1_path),
                (item.utt2, item.s2_path),
            ]
            for utt, src in entries:
                spk = speaker_id(utt)
                key = (spk, utt, src)
                if key in seen:
                    continue
                seen.add(key)
                subset_pool.setdefault(spk, []).append(
                    AuxCandidate(utt=utt, path=src)
                )

        pool[subset] = {
            spk: sorted(candidates, key=lambda item: (item.utt, str(item.path)))
            for spk, candidates in subset_pool.items()
        }

    return pool


def choose_aux(
    candidates: Sequence[AuxCandidate],
    exclude_utts: Sequence[str],
    key_seed: str,
    policy: Policy,
    seed: int,
) -> AuxCandidate:
    excluded = set(exclude_utts)
    available = sorted(
        [candidate for candidate in candidates if candidate.utt not in excluded],
        key=lambda item: (item.utt, str(item.path)),
    )
    if not available:
        raise RuntimeError(
            f"No available aux candidate after excluding {exclude_utts}. "
            "This target speaker may have only one utterance in the selected subset."
        )

    if policy == "first":
        return available[0]

    if policy == "random":
        rng = random.Random(f"{seed}:{key_seed}")
        return rng.choice(available)

    if policy == "hash":
        digest = hashlib.sha1(f"{seed}:{key_seed}".encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(available)
        return available[idx]

    raise ValueError(f"Unknown aux policy: {policy}")


def write_scp_lines(path: Path, lines: Iterable[Tuple[str, Path]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for key, wav_path in lines:
            f.write(f"{key} {wav_path}\n")
            count += 1
    return count


def generate_subset_scp(
    subset: str,
    items: Sequence[MixtureItem],
    pool: Dict[str, List[AuxCandidate]],
    out_dir: Path,
    aux_policy: Policy,
    seed: int,
) -> Dict[str, int]:
    mix_lines: List[Tuple[str, Path]] = []
    ref_lines: List[Tuple[str, Path]] = []
    aux_lines: List[Tuple[str, Path]] = []

    for item in items:
        spk1 = speaker_id(item.utt1)
        spk2 = speaker_id(item.utt2)

        if spk1 not in pool:
            raise RuntimeError(f"No aux candidate for speaker {spk1} in subset {subset}")
        if spk2 not in pool:
            raise RuntimeError(f"No aux candidate for speaker {spk2} in subset {subset}")

        exclude_utts = [item.utt1, item.utt2]

        aux1 = choose_aux(
            pool[spk1],
            exclude_utts=exclude_utts,
            key_seed=f"{subset}:{item.mix_stem}:s1",
            policy=aux_policy,
            seed=seed,
        )
        aux2 = choose_aux(
            pool[spk2],
            exclude_utts=exclude_utts,
            key_seed=f"{subset}:{item.mix_stem}:s2",
            policy=aux_policy,
            seed=seed,
        )

        # WSJ0-like key: parts[0] and parts[2] are the two original utterance ids.
        key1 = f"{item.utt1}_0_{item.utt2}_0_{aux1.utt}"
        key2 = f"{item.utt1}_0_{item.utt2}_0_{aux2.utt}"

        mix_lines.extend([(key1, item.mix_path), (key2, item.mix_path)])
        ref_lines.extend([(key1, item.s1_path), (key2, item.s2_path)])
        aux_lines.extend([(key1, aux1.path), (key2, aux2.path)])

    subset_out = out_dir / subset
    return {
        "mix": write_scp_lines(subset_out / "mix.scp", mix_lines),
        "ref": write_scp_lines(subset_out / "ref.scp", ref_lines),
        "aux": write_scp_lines(subset_out / "aux.scp", aux_lines),
    }


def read_scp(path: Path) -> Dict[str, Path]:
    data: Dict[str, Path] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"{path}:{lineno}: expected '<key> <path>'")
            key, wav = fields
            if key in data:
                raise ValueError(f"{path}:{lineno}: duplicate key {key}")
            data[key] = Path(wav)
    return data


def validate_scp_dir(scp_dir: Path, check_exists: bool = False) -> Dict[str, int]:
    mix = read_scp(scp_dir / "mix.scp")
    ref = read_scp(scp_dir / "ref.scp")
    aux = read_scp(scp_dir / "aux.scp")

    if set(mix) != set(ref) or set(mix) != set(aux):
        raise AssertionError(
            f"Key mismatch: mix={len(mix)}, ref={len(ref)}, aux={len(aux)}"
        )

    bad: List[str] = []
    for key in mix:
        parts = key.split("_")
        if len(parts) < 5:
            bad.append(f"{key}: expected key '<utt1>_0_<utt2>_0_<auxutt>'")
            continue

        utt1 = parts[0]
        utt2 = parts[2]

        ref_path = ref[key]
        aux_path = aux[key]

        try:
            target_utt = source_utt_from_librimix_path(ref_path)
            aux_utt = source_utt_from_librimix_path(aux_path)
        except ValueError as exc:
            bad.append(f"{key}: {exc}")
            continue

        target_spk = speaker_id(target_utt)
        aux_spk = speaker_id(aux_utt)

        if aux_spk != target_spk:
            bad.append(
                f"{key}: aux speaker mismatch, target={target_spk}, aux={aux_spk}, aux={aux_path}"
            )

        if aux_utt in {utt1, utt2}:
            bad.append(f"{key}: aux uses a current mixture utterance: {aux_path}")

        if check_exists:
            for name, p in (("mix", mix[key]), ("ref", ref_path), ("aux", aux_path)):
                if not p.is_file():
                    bad.append(f"{key}: missing {name} path: {p}")

    if bad:
        sample = "\n".join(bad[:20])
        more = "" if len(bad) <= 20 else f"\n... and {len(bad) - 20} more"
        raise AssertionError(f"Validation failed with {len(bad)} errors:\n{sample}{more}")

    return {"num_keys": len(mix), "mix": len(mix), "ref": len(ref), "aux": len(aux)}


def cmd_generate(args: argparse.Namespace) -> None:
    libri_root = resolve_librimix_root(Path(args.librimix_root))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    items_by_subset: Dict[str, List[MixtureItem]] = {}
    for subset in args.subsets:
        items_by_subset[subset] = scan_subset(
            libri_root=libri_root,
            sample_rate=args.sample_rate,
            mix_mode=args.mix_mode,
            subset=subset,
            task=args.task,
        )

    pool_by_subset = build_aux_index(items_by_subset=items_by_subset)

    counts: Dict[str, Dict[str, int]] = {}
    for subset, items in items_by_subset.items():
        counts[subset] = generate_subset_scp(
            subset=subset,
            items=items,
            pool=pool_by_subset[subset],
            out_dir=out_dir,
            aux_policy=args.aux_policy,
            seed=args.seed,
        )

    validation: Dict[str, Dict[str, int]] = {}
    for subset in args.subsets:
        validation[subset] = validate_scp_dir(
            out_dir / subset,
            check_exists=args.check_exists,
        )

    meta = {
        "librimix_root": str(libri_root),
        "out_dir": str(out_dir),
        "sample_rate": args.sample_rate,
        "mix_mode": args.mix_mode,
        "task": args.task,
        "subsets": args.subsets,
        "aux_policy": args.aux_policy,
        "seed": args.seed,
        "counts": counts,
        "validation": validation,
        "notes": [
            "mix/ref/aux contain two USEF-TSE samples per Libri2Mix mixture.",
            "aux.scp points directly to original Libri2Mix s1/s2 source wavs.",
            "Training uses dataset_type=libri2mix to sample same-speaker aux audio in dataset/data.py.",
            "For config, set train_path/dev_path/test_path to the corresponding subset directories.",
        ],
    }
    with (out_dir / "prepare_libri2mix_usef_scp_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))


def cmd_validate(args: argparse.Namespace) -> None:
    result = validate_scp_dir(Path(args.scp_dir), check_exists=args.check_exists)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare USEF-TSE style mix/ref/aux scp files for Libri2Mix."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate Libri2Mix USEF-TSE scp files.")
    gen.add_argument("--librimix-root", required=True, help="Path to Libri2Mix or its parent.")
    gen.add_argument("--out-dir", required=True, help="Output directory.")
    gen.add_argument(
        "--subsets",
        nargs="+",
        default=["train-100", "dev", "test"],
        choices=["train-100", "train-360", "dev", "test"],
        help="Libri2Mix subsets to process.",
    )
    gen.add_argument("--sample-rate", type=int, default=8000, choices=[8000, 16000])
    gen.add_argument("--mix-mode", default="min", choices=["min", "max"])
    gen.add_argument(
        "--task",
        default="sep_clean",
        choices=["sep_clean", "sep_noisy"],
        help="sep_clean uses mix_clean; sep_noisy uses mix_both.",
    )
    gen.add_argument(
        "--aux-policy",
        default="hash",
        choices=["hash", "first", "random"],
        help="How to deterministically choose aux utterances.",
    )
    gen.add_argument("--seed", type=int, default=3407)
    gen.add_argument("--check-exists", action="store_true")
    gen.set_defaults(func=cmd_generate)

    val = sub.add_parser("validate", help="Validate one generated scp directory.")
    val.add_argument("--scp-dir", required=True, help="Directory containing mix.scp/ref.scp/aux.scp.")
    val.add_argument("--check-exists", action="store_true")
    val.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
