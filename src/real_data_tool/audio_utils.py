from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioSpec:
    sampling_rate: int = 16000
    duration_sec: float = 6.0
    mono: bool = True


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    # shape: (n_samples, n_channels)
    return data, sr


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    if data.shape[1] == 1:
        return data[:, 0]
    return data.mean(axis=1)


def resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    if x.size == 0:
        return x
    duration = x.shape[0] / float(orig_sr)
    target_len = int(round(duration * target_sr))
    if target_len <= 1:
        return np.zeros((0,), dtype=np.float32)

    xp = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    fp = x.astype(np.float32)
    xq = np.linspace(0.0, duration, num=target_len, endpoint=False)
    y = np.interp(xq, xp, fp).astype(np.float32)
    return y


def fix_length_center(x: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    n = x.shape[0]
    if n == target_len:
        return x.astype(np.float32)
    if n > target_len:
        start = (n - target_len) // 2
        return x[start:start + target_len].astype(np.float32)
    # pad
    pad_left = (target_len - n) // 2
    pad_right = target_len - n - pad_left
    return np.pad(x.astype(np.float32), (pad_left, pad_right), mode="constant")


def load_and_normalize(path: Path, spec: AudioSpec) -> np.ndarray:
    data, sr = read_audio(path)
    if spec.mono:
        x = to_mono(data)
    else:
        x = data[:, 0]
    x = resample_linear(x, orig_sr=sr, target_sr=spec.sampling_rate)
    target_len = int(round(spec.duration_sec * spec.sampling_rate))
    x = fix_length_center(x, target_len)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x / peak
    return x.astype(np.float32)


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="PCM_16")
