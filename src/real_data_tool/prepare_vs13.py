from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from real_data_tool.audio_utils import AudioSpec, load_and_normalize, write_wav


def parse_speed_file(path: Path) -> float:
    """Parse VS13 annotation file.

    Expected format: "<speed> <something>" (space separated).
    We use the first float as speed in km/h.
    """
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Empty annotation file: {path}")
    parts = text.split()
    return float(parts[0])


def discover_pairs(root: Path) -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []
    for vehicle_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for wav_path in sorted(vehicle_dir.glob("*.wav")):
            stem = wav_path.stem
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                continue
            pairs.append((wav_path, txt_path, vehicle_dir.name))
    if not pairs:
        raise RuntimeError(f"No .wav/.txt pairs found under: {root}")
    return pairs


def convert_dataset(
    raw_dir: Path,
    out_dir: Path,
    csv_path: Path,
    spec: AudioSpec,
    direction: str,
    limit: int,
    dry_run: bool,
) -> None:
    pairs = discover_pairs(raw_dir)

    if limit > 0:
        pairs = pairs[:limit]

    records = []
    for idx, (wav_path, txt_path, vehicle_name) in enumerate(pairs):
        speed = parse_speed_file(txt_path)

        # This repo expects loc1..loc6; VS13 has a single recording setup -> loc1.
        loc = "loc1"
        vtype = "car"  # VS13 contains passenger vehicles
        direc = direction

        dst_rel = Path(f"{loc}_cut") / vtype / direc / f"vs13_{vehicle_name}_{idx:06d}_speed-{int(round(speed)):03d}.wav"
        dst = out_dir / dst_rel

        if not dry_run:
            audio = load_and_normalize(wav_path, spec)
            write_wav(dst, audio, spec.sampling_rate)

        records.append(
            {
                "path": str(dst_rel).replace("\\", "/"),
                "index": idx,
                "speed": float(speed),
                "vehicle_type": vtype,
                "direction": direc,
                "vehicle_name": vehicle_name,
                "file_raw": wav_path.name,
                "annotation_raw": txt_path.name,
            }
        )

    df = pd.DataFrame.from_records(records)
    if not dry_run:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    print(f"VS13: wrote {len(df)} rows")
    print(f"  out_dir: {out_dir}")
    print(f"  csv: {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert VS13 dataset into this repo's dataset layout.")
    p.add_argument("--raw-dir", default="./data/raw/VS13")
    p.add_argument("--out-dir", default="./data/processed/real/vs13")
    p.add_argument("--csv", default="./data/processed/real/vs13/vs13.csv")
    p.add_argument("--sampling-rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=6.0, help="Target duration in seconds (crop/pad center)")
    p.add_argument("--direction", choices=["left", "right"], default="right")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N pairs (0 = no limit)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = AudioSpec(sampling_rate=args.sampling_rate, duration_sec=args.duration, mono=True)

    convert_dataset(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        csv_path=Path(args.csv),
        spec=spec,
        direction=args.direction,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
