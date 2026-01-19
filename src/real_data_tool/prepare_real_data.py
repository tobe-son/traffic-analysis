from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from real_data_tool.prepare_idmt_traffic import convert_dataset as convert_idmt
from real_data_tool.prepare_vs13 import convert_dataset as convert_vs13
from real_data_tool.audio_utils import AudioSpec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare real datasets under data/raw for this repo.")
    sub = p.add_subparsers(dest="dataset", required=True)

    p_idmt = sub.add_parser("idmt", help="IDMT_Traffic (vehicle-type classification compatible layout)")
    p_idmt.add_argument("--raw-dir", default="./data/raw/IDMT_Traffic")
    p_idmt.add_argument("--out-dir", default="./data/processed/real/idmt_traffic")
    p_idmt.add_argument("--csv", default="./data/processed/real/idmt_traffic/idmt_traffic.csv")
    p_idmt.add_argument("--sampling-rate", type=int, default=16000)
    p_idmt.add_argument("--duration", type=float, default=6.0)
    p_idmt.add_argument("--mic", default="")
    p_idmt.add_argument("--channel", default="")
    p_idmt.add_argument("--vehicle-scheme", choices=["car_vs_cv"], default="car_vs_cv")
    p_idmt.add_argument("--unknown-speed", choices=["drop", "zero"], default="drop")
    p_idmt.add_argument("--limit", type=int, default=0)
    p_idmt.add_argument("--dry-run", action="store_true")

    p_vs13 = sub.add_parser("vs13", help="VS13 (speed estimation layout)")
    p_vs13.add_argument("--raw-dir", default="./data/raw/VS13")
    p_vs13.add_argument("--out-dir", default="./data/processed/real/vs13")
    p_vs13.add_argument("--csv", default="./data/processed/real/vs13/vs13.csv")
    p_vs13.add_argument("--sampling-rate", type=int, default=16000)
    p_vs13.add_argument("--duration", type=float, default=6.0)
    p_vs13.add_argument("--direction", choices=["left", "right"], default="right")
    p_vs13.add_argument("--limit", type=int, default=0)
    p_vs13.add_argument("--dry-run", action="store_true")

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.dataset == "idmt":
        spec = AudioSpec(sampling_rate=args.sampling_rate, duration_sec=args.duration, mono=True)
        mic = args.mic if str(args.mic).strip() else None
        channel = args.channel if str(args.channel).strip() else None
        convert_idmt(
            raw_dir=Path(args.raw_dir),
            out_dir=Path(args.out_dir),
            csv_path=Path(args.csv),
            spec=spec,
            mic_filter=mic,
            channel_filter=channel,
            vehicle_scheme=args.vehicle_scheme,
            unknown_speed=args.unknown_speed,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        return

    if args.dataset == "vs13":
        spec = AudioSpec(sampling_rate=args.sampling_rate, duration_sec=args.duration, mono=True)
        convert_vs13(
            raw_dir=Path(args.raw_dir),
            out_dir=Path(args.out_dir),
            csv_path=Path(args.csv),
            spec=spec,
            direction=args.direction,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        return

    raise RuntimeError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    main(sys.argv[1:])
