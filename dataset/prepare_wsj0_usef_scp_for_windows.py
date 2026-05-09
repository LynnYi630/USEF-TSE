#!/usr/bin/env python3
"""
Prepare SCP files for ZBang/USEF-TSE on WSJ0-2mix.

Windows/WSL note
----------------
This variant keeps the original output format, but avoids unnecessary per-file
Path.resolve() calls and prints progress while scanning/validating datasets on
/mnt/<drive>, where file metadata checks can be noticeably slower.

This script has three subcommands:

1) rewrite
   Rewrites the original author's scp files to your local absolute paths.
   Use this for reproducing the HuggingFace checkpoint evaluation.

2) generate
   Generates mix/ref/aux scp files from local wsj0-2mix and wsj0 directories.
   Use this for your own training or ablation experiments, not for exact
   reproduction of the author's released checkpoint.

3) validate
   Checks that mix/ref/aux scp files have matching keys and reasonable
   target-speaker/aux-speaker consistency.

Recommended for reproducing the author's released checkpoint:

    python prepare_wsj0_usef_scp_for_windows.py rewrite \
        --src-dir /mnt/d/wsl/USEF-TSE-ZBang/data/test/wsj0-2mix \
        --out-dir wsj0-2mix_abs \
        --wsj0-root /mnt/d/Datasets/wsj0 \
        --wsj02mix-root /mnt/d/Datasets/wsj0-2mix

Add --check-exists only when you need a full file-existence audit. On WSL
paths under /mnt/<drive>, that audit can be slow because it performs one
metadata check per mix/ref/aux wav path.

Then set config test path to data/test/wsj0-2mix_abs.

To generate new scp files:
	python3 prepare_wsj0_usef_scp_for_windows.py generate \
  		--wsj0-root /mnt/d/Datasets/wsj0 \
  		--wsj02mix-root /mnt/d/Datasets/wsj0-2mix \
  		--out-dir data/generated_scp \
  		--splits tr cv tt \
  		--mix-mode min \
  		--aux-policy hash \
  		--seed 3407 \
  		--key-style author \
  		--path-style absolute \
  		--check-exists \
  		--strict
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCP_NAMES = ("mix.scp", "ref.scp", "aux.scp")
AUX_NAME_CANDIDATES = ("aux.scp", "au_.scp", "au.scp")


@dataclass(frozen=True)
class ScpEntry:
    key: str
    path: str


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def read_scp(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Bad scp line in {path}:{lineno}: {line!r}")
            key, wav_path = parts
            if key in data:
                raise ValueError(f"Duplicate key {key!r} in {path}:{lineno}")
            data[key] = wav_path
    return data


def read_scp_ordered(path: Path) -> List[ScpEntry]:
    entries: List[ScpEntry] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Bad scp line in {path}:{lineno}: {line!r}")
            key, wav_path = parts
            if key in seen:
                raise ValueError(f"Duplicate key {key!r} in {path}:{lineno}")
            seen.add(key)
            entries.append(ScpEntry(key, wav_path))
    return entries


def write_scp(path: Path, entries: Iterable[ScpEntry]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(f"{entry.key} {entry.path}\n")
            n += 1
    return n


def find_aux_scp(src_dir: Path) -> Path:
    for name in AUX_NAME_CANDIDATES:
        p = src_dir / name
        if p.is_file():
            return p

    # Some downloaded files may be renamed by the shell/browser, for example
    # au_(1).scp. Accept a single obvious aux candidate, but fail if ambiguous.
    globbed = sorted(src_dir.glob("au*.scp"))
    if len(globbed) == 1:
        return globbed[0]

    tried = list(AUX_NAME_CANDIDATES) + ["au*.scp"]
    raise FileNotFoundError(
        f"Cannot find aux scp in {src_dir}. Tried: {', '.join(tried)}"
    )


def lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving every filesystem component."""
    path = path.expanduser()
    if path.is_absolute():
        return path
    return Path.cwd() / path


def absolutize_path(path_str: str, wsj0_root: Path, wsj02mix_root: Path) -> str:
    """Map original author prefixes to local absolute paths."""
    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    if path_str.startswith("wsj0_8k/"):
        rel = path_str[len("wsj0_8k/") :]
        return str(wsj0_root / rel)

    if path_str.startswith("wsj0-mix/"):
        rel = path_str[len("wsj0-mix/") :]
        return str(wsj02mix_root / rel)

    # Also support paths that already begin with the internal WSJ0-2mix tree.
    if path_str.startswith("2speakers/"):
        return str(wsj02mix_root / path_str)

    # Fallback: keep relative to cwd as an absolute path. This makes missing
    # unexpected prefixes visible during --check-exists.
    return str(lexical_absolute(Path(path_str)))


def rewrite_one_scp(src: Path, dst: Path, wsj0_root: Path, wsj02mix_root: Path) -> int:
    entries = read_scp_ordered(src)
    rewritten = [
        ScpEntry(e.key, absolutize_path(e.path, wsj0_root=wsj0_root, wsj02mix_root=wsj02mix_root))
        for e in entries
    ]
    return write_scp(dst, rewritten)


def command_rewrite(args: argparse.Namespace) -> None:
    wsj0_root = lexical_absolute(Path(args.wsj0_root))
    wsj02mix_root = lexical_absolute(Path(args.wsj02mix_root))
    out_dir = lexical_absolute(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.src_dir:
        src_dir = lexical_absolute(Path(args.src_dir))
        mix_scp = src_dir / "mix.scp"
        ref_scp = src_dir / "ref.scp"
        aux_scp = find_aux_scp(src_dir)
    else:
        mix_scp = lexical_absolute(Path(args.mix_scp))
        ref_scp = lexical_absolute(Path(args.ref_scp))
        aux_scp = lexical_absolute(Path(args.aux_scp))

    for p in (mix_scp, ref_scp, aux_scp):
        if not p.is_file():
            raise FileNotFoundError(p)

    counts = {
        "mix.scp": rewrite_one_scp(mix_scp, out_dir / "mix.scp", wsj0_root, wsj02mix_root),
        "ref.scp": rewrite_one_scp(ref_scp, out_dir / "ref.scp", wsj0_root, wsj02mix_root),
        "aux.scp": rewrite_one_scp(aux_scp, out_dir / "aux.scp", wsj0_root, wsj02mix_root),
    }

    print(f"[OK] wrote rewritten scp files to {out_dir}")
    for name, n in counts.items():
        print(f"  {name}: {n} lines")

    validate_triplet(
        out_dir / "mix.scp",
        out_dir / "ref.scp",
        out_dir / "aux.scp",
        check_exists=args.check_exists,
        strict=args.strict,
        max_examples=args.max_examples,
    )


def is_wav_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".wav"


def build_wsj0_index(wsj0_root: Path) -> Dict[str, List[Path]]:
    """Index WSJ0 utterances by the first 3 chars of the wav stem."""
    if not wsj0_root.is_dir():
        raise NotADirectoryError(wsj0_root)

    index: Dict[str, List[Path]] = {}
    n_wavs = 0
    for wav in wsj0_root.rglob("*.wav"):
        spk = wav.stem[:3]
        index.setdefault(spk, []).append(wav)
        n_wavs += 1
        if n_wavs % 10000 == 0:
            print(f"[INFO] indexed {n_wavs} WSJ0 wavs ...", flush=True)

    for spk in index:
        index[spk].sort(key=lambda x: str(x))

    if not index:
        raise RuntimeError(f"No wav files found under {wsj0_root}")
    return index


def choose_aux(
    candidates: Sequence[Path],
    mixture_utts: Sequence[str],
    salt: str,
    seed: int,
    policy: str,
) -> Path:
    exclude = {u + ".wav" for u in mixture_utts}
    valid = [p for p in candidates if p.name not in exclude]
    if not valid:
        raise RuntimeError(
            f"No valid aux candidates for salt={salt!r}; excluded={sorted(exclude)}"
        )

    if policy == "first":
        return valid[0]
    if policy == "random":
        rng = random.Random(f"{seed}:{salt}")
        return rng.choice(valid)
    if policy == "hash":
        h = hashlib.sha1(f"{seed}:{salt}".encode("utf-8")).hexdigest()
        idx = int(h, 16) % len(valid)
        return valid[idx]
    raise ValueError(f"Unknown aux policy: {policy}")


def path_with_prefix(path: Path, root: Path, prefix_mode: str, prefix_name: str) -> str:
    if prefix_mode == "absolute":
        if not path.is_absolute():
            path = path.resolve()
        return str(path)
    if not path.is_absolute():
        path = path.resolve()
    if not root.is_absolute():
        root = root.resolve()
    rel = path.relative_to(root)
    if prefix_mode == "author":
        return str(Path(prefix_name) / rel)
    if prefix_mode == "relative":
        return str(rel)
    raise ValueError(f"Unknown prefix mode: {prefix_mode}")


def parse_mixture_stem(stem: str) -> Tuple[str, str, str, str]:
    # WSJ0-2mix min/max filename style:
    #   utt1_snr1_utt2_snr2.wav
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Cannot parse mixture stem: {stem!r}")
    utt1 = parts[0]
    snr1 = parts[1]
    utt2 = parts[2]
    snr2 = "_".join(parts[3:])
    return utt1, snr1, utt2, snr2


def command_generate(args: argparse.Namespace) -> None:
    wsj0_root = Path(args.wsj0_root).expanduser().resolve()
    wsj02mix_root = Path(args.wsj02mix_root).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()

    print(f"[INFO] indexing WSJ0 wavs under {wsj0_root} ...", flush=True)
    speaker_index = build_wsj0_index(wsj0_root)
    print(f"[OK] indexed {sum(len(v) for v in speaker_index.values())} WSJ0 wavs from {len(speaker_index)} speakers")

    for split in args.splits:
        split_dir = wsj02mix_root / "2speakers" / args.sample_rate_dir / args.mix_mode / split
        mix_dir = split_dir / "mix"
        s1_dir = split_dir / "s1"
        s2_dir = split_dir / "s2"
        for d in (mix_dir, s1_dir, s2_dir):
            if not d.is_dir():
                raise NotADirectoryError(d)

        mix_entries: List[ScpEntry] = []
        ref_entries: List[ScpEntry] = []
        aux_entries: List[ScpEntry] = []

        mix_wavs = sorted(mix_dir.glob("*.wav"), key=lambda p: p.name)
        if not mix_wavs:
            raise RuntimeError(f"No mix wavs found in {mix_dir}")
        print(f"[INFO] {split}: generating from {len(mix_wavs)} mixtures ...", flush=True)

        for mix_idx, mix_wav in enumerate(mix_wavs, 1):
            utt1, _snr1, utt2, _snr2 = parse_mixture_stem(mix_wav.stem)
            s1_wav = s1_dir / mix_wav.name
            s2_wav = s2_dir / mix_wav.name
            if not s1_wav.is_file() or not s2_wav.is_file():
                raise FileNotFoundError(f"Missing s1/s2 for {mix_wav.name}")

            for target_idx, target_utt, ref_wav in (
                (1, utt1, s1_wav),
                (2, utt2, s2_wav),
            ):
                spk = target_utt[:3]
                if spk not in speaker_index:
                    raise RuntimeError(f"No WSJ0 candidates for speaker {spk!r} from {target_utt!r}")
                aux_wav = choose_aux(
                    speaker_index[spk],
                    mixture_utts=(utt1, utt2),
                    salt=f"{split}:{mix_wav.stem}:s{target_idx}:{target_utt}",
                    seed=args.seed,
                    policy=args.aux_policy,
                )

                if args.key_style == "author":
                    key = f"{mix_wav.stem}_{aux_wav.stem}"
                elif args.key_style == "index":
                    key = f"{mix_wav.stem}_{target_idx}"
                else:
                    raise ValueError(f"Unknown key style: {args.key_style}")

                mix_entries.append(
                    ScpEntry(
                        key,
                        path_with_prefix(mix_wav, wsj02mix_root, args.path_style, "wsj0-mix"),
                    )
                )
                ref_entries.append(
                    ScpEntry(
                        key,
                        path_with_prefix(ref_wav, wsj02mix_root, args.path_style, "wsj0-mix"),
                    )
                )
                aux_entries.append(
                    ScpEntry(
                        key,
                        path_with_prefix(aux_wav, wsj0_root, args.path_style, "wsj0_8k"),
                    )
                )
            if mix_idx % 5000 == 0:
                print(f"[INFO] {split}: processed {mix_idx}/{len(mix_wavs)} mixtures ...", flush=True)

        out_dir = out_root / split
        write_scp(out_dir / "mix.scp", mix_entries)
        write_scp(out_dir / "ref.scp", ref_entries)
        write_scp(out_dir / "aux.scp", aux_entries)
        print(f"[OK] {split}: wrote {len(mix_entries)} target-speaker samples to {out_dir}")

        validate_triplet(
            out_dir / "mix.scp",
            out_dir / "ref.scp",
            out_dir / "aux.scp",
            check_exists=(args.path_style == "absolute" and args.check_exists),
            strict=args.strict,
            max_examples=args.max_examples,
        )


def split_key_base_and_aux(key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (mix_base, utt1, utt2) if key looks like author/index style."""
    parts = key.split("_")
    if len(parts) < 4:
        return None, None, None
    utt1 = parts[0]
    utt2 = parts[2]
    mix_base = "_".join(parts[:4])
    return mix_base, utt1, utt2


def normalize_for_checks(path_str: str) -> str:
    return path_str.replace("\\", "/")


def validate_triplet(
    mix_scp: Path,
    ref_scp: Path,
    aux_scp: Path,
    check_exists: bool,
    strict: bool,
    max_examples: int,
) -> None:
    mix_ordered = read_scp_ordered(mix_scp)
    mix = {e.key: e.path for e in mix_ordered}
    ref = read_scp(ref_scp)
    aux = read_scp(aux_scp)

    errors: List[str] = []
    warnings: List[str] = []

    if set(mix) != set(ref) or set(mix) != set(aux):
        missing_ref = sorted(set(mix) - set(ref))[:max_examples]
        missing_aux = sorted(set(mix) - set(aux))[:max_examples]
        extra_ref = sorted(set(ref) - set(mix))[:max_examples]
        extra_aux = sorted(set(aux) - set(mix))[:max_examples]
        errors.append(
            "Key sets differ. "
            f"missing_ref={missing_ref}, missing_aux={missing_aux}, "
            f"extra_ref={extra_ref}, extra_aux={extra_aux}"
        )

    if check_exists:
        print(f"[INFO] validating {len(mix_ordered)} keys with file-existence checks ...", flush=True)
    else:
        print(f"[INFO] validating {len(mix_ordered)} keys ...", flush=True)

    for idx, e in enumerate(mix_ordered, 1):
        key = e.key
        if key not in ref or key not in aux:
            continue
        mix_base, utt1, utt2 = split_key_base_and_aux(key)
        if mix_base is not None:
            mix_stem = Path(mix[key]).stem
            ref_stem = Path(ref[key]).stem
            if mix_stem != mix_base:
                errors.append(f"{key}: mix stem {mix_stem!r} != key base {mix_base!r}")
            if ref_stem != mix_base:
                errors.append(f"{key}: ref stem {ref_stem!r} != key base {mix_base!r}")

            ref_path_norm = normalize_for_checks(ref[key])
            if "/s1/" in ref_path_norm:
                target_utt = utt1
            elif "/s2/" in ref_path_norm:
                target_utt = utt2
            else:
                target_utt = None
                warnings.append(f"{key}: cannot infer s1/s2 from ref path {ref[key]!r}")

            if target_utt is not None:
                target_spk = target_utt[:3]
                aux_stem = Path(aux[key]).stem
                aux_spk = aux_stem[:3]
                if target_spk != aux_spk:
                    errors.append(
                        f"{key}: aux speaker mismatch target={target_spk!r}, aux={aux_spk!r}, aux={aux[key]!r}"
                    )

        if check_exists:
            for label, p in (("mix", mix[key]), ("ref", ref[key]), ("aux", aux[key])):
                if not Path(p).is_file():
                    errors.append(f"{key}: {label} file missing: {p}")

        if len(errors) >= max_examples:
            break
        progress_interval = 500 if check_exists else 20000
        if idx % progress_interval == 0:
            print(f"[INFO] validated {idx}/{len(mix_ordered)} keys ...", flush=True)

    print(f"[CHECK] keys: mix={len(mix)}, ref={len(ref)}, aux={len(aux)}")
    if warnings:
        print(f"[WARN] {len(warnings)} warnings. Showing up to {max_examples}:")
        for w in warnings[:max_examples]:
            print(f"  - {w}")

    if errors:
        print(f"[FAIL] {len(errors)} validation errors. Showing up to {max_examples}:")
        for err in errors[:max_examples]:
            print(f"  - {err}")
        if strict:
            raise SystemExit(2)
    else:
        print("[OK] validation passed")


def command_validate(args: argparse.Namespace) -> None:
    validate_triplet(
        Path(args.mix_scp).expanduser().resolve(),
        Path(args.ref_scp).expanduser().resolve(),
        Path(args.aux_scp).expanduser().resolve(),
        check_exists=args.check_exists,
        strict=args.strict,
        max_examples=args.max_examples,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare mix/ref/aux scp files for ZBang/USEF-TSE on WSJ0-2mix."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("rewrite", help="Rewrite original author scp paths to local absolute paths.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--src-dir", help="Directory containing mix.scp/ref.scp/aux.scp or au_.scp")
    src.add_argument("--mix-scp", help="Input mix.scp")
    p.add_argument("--ref-scp", help="Input ref.scp; required if --mix-scp is used")
    p.add_argument("--aux-scp", help="Input aux.scp/au_.scp; required if --mix-scp is used")
    p.add_argument("--out-dir", required=True, help="Output directory; writes mix.scp/ref.scp/aux.scp")
    p.add_argument("--wsj0-root", default="/mnt/d/Datasets/wsj0")
    p.add_argument("--wsj02mix-root", default="/mnt/d/Datasets/wsj0-2mix")
    p.add_argument("--check-exists", action="store_true")
    p.add_argument("--strict", action="store_true", help="Exit non-zero on validation errors")
    p.add_argument("--max-examples", type=int, default=20)
    p.set_defaults(func=command_rewrite)

    p = sub.add_parser("generate", help="Generate new scp files from local WSJ0-2mix/WSJ0 directories.")
    p.add_argument("--wsj0-root", default="/mnt/d/Datasets/wsj0")
    p.add_argument("--wsj02mix-root", default="/mnt/d/Datasets/wsj0-2mix")
    p.add_argument("--out-dir", required=True, help="Output root; writes one subdir per split")
    p.add_argument("--splits", nargs="+", default=["tr", "cv", "tt"], choices=["tr", "cv", "tt"])
    p.add_argument("--mix-mode", default="min", choices=["min", "max"])
    p.add_argument("--sample-rate-dir", default="wav8k", choices=["wav8k"])
    p.add_argument("--aux-policy", default="hash", choices=["hash", "first", "random"])
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--key-style", default="author", choices=["author", "index"], help="author => base_auxStem; index => base_1/base_2")
    p.add_argument("--path-style", default="absolute", choices=["absolute", "author", "relative"])
    p.add_argument("--check-exists", action="store_true")
    p.add_argument("--strict", action="store_true", help="Exit non-zero on validation errors")
    p.add_argument("--max-examples", type=int, default=20)
    p.set_defaults(func=command_generate)

    p = sub.add_parser("validate", help="Validate an existing mix/ref/aux scp triplet.")
    p.add_argument("--mix-scp", required=True)
    p.add_argument("--ref-scp", required=True)
    p.add_argument("--aux-scp", required=True)
    p.add_argument("--check-exists", action="store_true")
    p.add_argument("--strict", action="store_true", help="Exit non-zero on validation errors")
    p.add_argument("--max-examples", type=int, default=20)
    p.set_defaults(func=command_validate)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.cmd == "rewrite" and args.mix_scp:
        if not args.ref_scp or not args.aux_scp:
            parser.error("rewrite with --mix-scp also requires --ref-scp and --aux-scp")
    args.func(args)


if __name__ == "__main__":
    main()
