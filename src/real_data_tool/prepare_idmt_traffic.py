from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from real_data_tool.audio_utils import AudioSpec, load_and_normalize, write_wav


@dataclass(frozen=True)
class IdmtMapping:
    location_to_loc: Dict[str, str]


VEHICLE_CODE_TO_NAME = {
    "B": "bus",
    "C": "car",
    "M": "motorcycle",
    "T": "truck",
}


def parse_filename(name: str) -> Optional[dict]:
    """Parse IDMT-Traffic file naming convention.

    Returns metadata dict or None if the filename is not a passing-vehicle recording.
    """
    if not name.lower().endswith(".wav"):
        return None
    stem = name[:-4]

    # background
    if stem.endswith("-BG"):
        return None

    parts = stem.split("_")
    if len(parts) != 9:
        return None

    date_time, location, speed_kmh, sample_pos, daytime, weather, vehicle_direction, mic, channel = parts
    if len(vehicle_direction) != 2:
        return None
    vehicle_code, direction_code = vehicle_direction[0], vehicle_direction[1]

    speed_raw = speed_kmh.replace("Kmh", "")
    if speed_raw == "unknown":
        speed = None
    else:
        try:
            speed = float(speed_raw)
        except ValueError:
            speed = None

    direction = {"L": "left", "R": "right"}.get(direction_code)
    vehicle = VEHICLE_CODE_TO_NAME.get(vehicle_code)

    if direction is None or vehicle is None:
        return None

    return {
        "file": name,
        "location": location,
        "speed": speed,
        "vehicle_code": vehicle_code,
        "vehicle": vehicle,
        "direction": direction,
        "microphone": mic,
        "channel": channel,
        "daytime": daytime,
        "weather": weather,
        "sample_pos": sample_pos,
        "date_time": date_time,
    }


def discover_locations(meta: Iterable[dict]) -> IdmtMapping:
    locations = sorted({row["location"] for row in meta})
    mapping: Dict[str, str] = {}
    for idx, loc in enumerate(locations):
        if idx >= 6:
            break
        mapping[loc] = f"loc{idx + 1}"
    return IdmtMapping(location_to_loc=mapping)


def map_vehicle_to_project(vehicle_code: str, scheme: str) -> str:
    if scheme == "car_vs_cv":
        # Map into the repo's existing binary label space.
        # C/M => car-like, B/T => commercial/heavy
        return "car" if vehicle_code in {"C", "M"} else "cv"
    raise ValueError(f"Unknown scheme: {scheme}")


def build_rows(
    raw_dir: Path,
    mic_filter: Optional[str],
    channel_filter: Optional[str],
    vehicle_scheme: str,
    unknown_speed: str,
    limit: int,
) -> Tuple[List[dict], IdmtMapping]:
    audio_dir = raw_dir / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"audio directory not found: {audio_dir}")

    parsed: List[dict] = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        meta = parse_filename(wav_path.name)
        if meta is None:
            continue
        if mic_filter and meta["microphone"] != mic_filter:
            continue
        if channel_filter and meta["channel"] != channel_filter:
            continue
        if meta["speed"] is None:
            if unknown_speed == "drop":
                continue
            if unknown_speed == "zero":
                meta["speed"] = 0.0
            else:
                raise ValueError("unknown_speed must be one of: drop, zero")
        meta["vehicle_type"] = map_vehicle_to_project(meta["vehicle_code"], scheme=vehicle_scheme)
        parsed.append(meta)

        if limit > 0 and len(parsed) >= limit:
            break

    if not parsed:
        raise RuntimeError("No usable IDMT_Traffic files found after filtering.")

    mapping = discover_locations(parsed)
    return parsed, mapping


def convert_dataset(
    raw_dir: Path,
    out_dir: Path,
    csv_path: Path,
    spec: AudioSpec,
    mic_filter: Optional[str],
    channel_filter: Optional[str],
    vehicle_scheme: str,
    unknown_speed: str,
    limit: int,
    dry_run: bool,
) -> None:
    rows, mapping = build_rows(
        raw_dir=raw_dir,
        mic_filter=mic_filter,
        channel_filter=channel_filter,
        vehicle_scheme=vehicle_scheme,
        unknown_speed=unknown_speed,
        limit=limit,
    )

    records = []
    index = 0

    for row in rows:
        loc = mapping.location_to_loc.get(row["location"], "loc1")
        vtype = row["vehicle_type"]
        direc = row["direction"]
        src = raw_dir / "audio" / row["file"]

        dst_rel = Path(f"{loc}_cut") / vtype / direc / f"idmt_{index:06d}_speed-{int(round(float(row['speed']))):03d}.wav"
        dst = out_dir / dst_rel

        if not dry_run:
            audio = load_and_normalize(src, spec)
            write_wav(dst, audio, spec.sampling_rate)

        records.append(
            {
                "path": str(dst_rel).replace("\\", "/"),
                "index": index,
                "speed": float(row["speed"]),
                "vehicle_type": vtype,
                "direction": direc,
                "location_raw": row["location"],
                "microphone": row["microphone"],
                "channel": row["channel"],
                "file_raw": row["file"],
            }
        )
        index += 1

    df = pd.DataFrame.from_records(records)
    if not dry_run:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    print(f"IDMT_Traffic: wrote {len(df)} rows")
    print(f"  out_dir: {out_dir}")
    print(f"  csv: {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert IDMT-Traffic into this repo's dataset layout.")
    p.add_argument("--raw-dir", default="./data/raw/IDMT_Traffic")
    p.add_argument("--out-dir", default="./data/processed/real/idmt_traffic")
    p.add_argument("--csv", default="./data/processed/real/idmt_traffic/idmt_traffic.csv")
    p.add_argument("--sampling-rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=6.0, help="Target duration in seconds (crop/pad center)")
    p.add_argument("--mic", default="", help="Filter microphone type (e.g., SE, ME). Empty to keep all.")
    p.add_argument("--channel", default="", help="Filter channel (e.g., CH12, CH34). Empty to keep all.")
    p.add_argument("--vehicle-scheme", choices=["car_vs_cv"], default="car_vs_cv")
    p.add_argument("--unknown-speed", choices=["drop", "zero"], default="drop")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N usable files (0 = no limit)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = AudioSpec(sampling_rate=args.sampling_rate, duration_sec=args.duration, mono=True)

    mic = args.mic if str(args.mic).strip() else None
    channel = args.channel if str(args.channel).strip() else None

    convert_dataset(
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


if __name__ == "__main__":
    main()
