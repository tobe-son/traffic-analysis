"""Organize raw simulation audio into cut.py-compatible directories.

This script reads train/val metadata CSV files for each location and copies
FLAC files into location/category/direction folders expected by cut.py.
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Mapping from CSV column name to (category, direction) folder pair
LABEL_TARGETS: Dict[str, Tuple[str, str]] = {
    "car_left": ("car", "left"),
    "car_right": ("car", "right"),
    "cv_left": ("cv", "left"),
    "cv_right": ("cv", "right"),
}

# Inverse mapping for looking up the original label from (category, direction).
TARGET_LABELS: Dict[Tuple[str, str], str] = {
    target: label for label, target in LABEL_TARGETS.items()
}

LOCATIONS: Iterable[str] = (f"loc{i}" for i in range(1, 7))
AUDIO_SUFFIX = ".flac"
# Aim for this many files per category by backfilling with multi-label rows when needed.
TARGET_PER_CLASS = 50


@dataclass(frozen=True)
class AudioEntry:
    audio_path: Path
    source_name: Path
    split: str | None
    labels: tuple[str, ...]


def _is_positive(value: str | None) -> bool:
    """Return True when a CSV label cell indicates at least one event."""

    if value is None or value == "":
        return False

    try:
        return float(value) > 0
    except ValueError:
        return False


def _load_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _build_destination_name(
    split: str | None,
    source_path: Path,
    extra_suffix: str | None = None,
) -> str:
    split_prefix = (split or "").strip() or source_path.parent.name
    stem = source_path.stem
    base_name = f"{split_prefix}_{stem}"
    if extra_suffix:
        base_name = f"{base_name}_{extra_suffix}"
    return f"{base_name}{AUDIO_SUFFIX}"


def _clear_destination_dirs(loc_dir: Path) -> None:
    """Remove previously generated category directories for a fresh rebuild."""

    categories = {category for category, _ in LABEL_TARGETS.values()}
    cleared = []

    for category in sorted(categories):
        target_dir = loc_dir / category
        if target_dir.exists():
            shutil.rmtree(target_dir)
            cleared.append(category)

    if cleared:
        removed = ", ".join(cleared)
        print(f"[INFO] Reset {loc_dir.name}: removed {removed}")


def _collect_entries(
    csv_path: Path,
    loc_dir: Path,
    skipped: Dict[str, int],
) -> tuple[list[AudioEntry], list[AudioEntry]]:
    single_label: list[AudioEntry] = []
    multi_label: list[AudioEntry] = []

    for row in _load_rows(csv_path):
        relative_audio = row.get("path")
        if not relative_audio:
            skipped["missing_path"] += 1
            continue

        audio_path = (loc_dir / relative_audio).resolve()
        if audio_path.suffix.lower() != AUDIO_SUFFIX or not audio_path.exists():
            print(f"[WARN] Missing source audio: {audio_path}")
            skipped["missing_audio"] += 1
            continue

        labels = tuple(
            label for label in LABEL_TARGETS if _is_positive(row.get(label))
        )

        if not labels:
            skipped["none"] += 1
            continue

        entry = AudioEntry(
            audio_path=audio_path,
            source_name=Path(relative_audio),
            split=row.get("split"),
            labels=labels,
        )

        if len(labels) == 1:
            single_label.append(entry)
        else:
            multi_label.append(entry)

    return single_label, multi_label


def _copy_audio(
    entry: AudioEntry,
    label: str,
    loc_dir: Path,
    stats: Dict[Tuple[str, str], int],
    extra_suffix: str | None = None,
) -> Path:
    category, direction = LABEL_TARGETS[label]
    destination_dir = loc_dir / category / direction
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_name = _build_destination_name(
        entry.split,
        entry.source_name,
        extra_suffix,
    )
    destination_path = destination_dir / destination_name

    shutil.copy2(entry.audio_path, destination_path)
    stats[(category, direction)] += 1
    return destination_path



def process_location(base_dir: Path) -> None:
    loc_dir = base_dir.resolve()
    if not loc_dir.exists():
        return

    stats: Dict[Tuple[str, str], int] = defaultdict(int)
    skipped: Dict[str, int] = {
        "none": 0,
        "missing_path": 0,
        "missing_audio": 0,
        "single_excess": 0,
        "multi_unused": 0,
    }
    written: Dict[Tuple[str, str], list[AudioEntry]] = defaultdict(list)
    duplicate_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    _clear_destination_dirs(loc_dir)

    single_entries: list[AudioEntry] = []
    multi_entries: list[AudioEntry] = []

    for csv_name in ("train.csv", "val.csv"):
        csv_path = loc_dir / csv_name
        if csv_path.exists():
            singles, multis = _collect_entries(csv_path, loc_dir, skipped)
            single_entries.extend(singles)
            multi_entries.extend(multis)
        else:
            print(f"[INFO] Skipped missing CSV: {csv_path}")
    for entry in single_entries:
        label = entry.labels[0]
        category, direction = LABEL_TARGETS[label]
        target_key = (category, direction)
        if stats[target_key] >= TARGET_PER_CLASS:
            skipped["single_excess"] += 1
            continue
        _copy_audio(entry, label, loc_dir, stats)
        written[target_key].append(entry)

    for entry in multi_entries:
        used = False
        for label in entry.labels:
            category, direction = LABEL_TARGETS[label]
            if stats[(category, direction)] >= TARGET_PER_CLASS:
                continue
            _copy_audio(entry, label, loc_dir, stats)
            used = True
            written[(category, direction)].append(entry)
        if not used:
            skipped["multi_unused"] += 1

    for target_key in sorted(TARGET_LABELS):
        if stats.get(target_key, 0) >= TARGET_PER_CLASS:
            continue

        pool = written[target_key]
        if not pool:
            if stats.get(target_key, 0) == 0:
                category, direction = target_key
                print(
                    f"[INFO] Unable to reach target for {loc_dir.name}/{category}_{direction}: no seed audio."
                )
            continue

        label = TARGET_LABELS[target_key]
        while stats.get(target_key, 0) < TARGET_PER_CLASS and pool:
            index = duplicate_counts[target_key] % len(pool)
            entry = pool[index]
            duplicate_counts[target_key] += 1
            suffix = f"dup{duplicate_counts[target_key]:03d}"
            _copy_audio(entry, label, loc_dir, stats, extra_suffix=suffix)

    skip_summary = []
    if skipped["none"]:
        skip_summary.append(f"skipped_none={skipped['none']}")
    if skipped["missing_path"]:
        skip_summary.append(f"missing_path={skipped['missing_path']}")
    if skipped["missing_audio"]:
        skip_summary.append(f"missing_audio={skipped['missing_audio']}")
    if skipped["single_excess"]:
        skip_summary.append(f"single_excess={skipped['single_excess']}")
    if skipped["multi_unused"]:
        skip_summary.append(f"unused_multi_rows={skipped['multi_unused']}")

    if stats:
        summary_parts = []
        for (category, direction), count in sorted(stats.items()):
            dup_count = duplicate_counts.get((category, direction), 0)
            if dup_count:
                summary_parts.append(
                    f"{category}_{direction}={count}(+{dup_count} dup)"
                )
            else:
                summary_parts.append(f"{category}_{direction}={count}")
        summary_parts.extend(skip_summary)
        print(f"[DONE] {loc_dir.name}: " + ", ".join(summary_parts))
    elif skip_summary:
        print(f"[WARN] No audio copied for {loc_dir.name}. " + ", ".join(skip_summary))
    else:
        print(f"[WARN] No audio copied for {loc_dir.name}.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    for loc in LOCATIONS:
        process_location(project_root / loc)
