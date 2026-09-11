"""Channel-safe, anti-aliased preprocessing for controlled input experiments.

Arrays use (samples, channels). No per-channel normalization is applied.
This module is independent of the legacy training loader.
"""
import math
import numpy as np
from scipy.signal import resample_poly


def prepare(x, source_rate, target_rate, mode='stereo'):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty samples x channels')
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError('Sample rates must be positive')
    if mode == 'stereo':
        if x.shape[1] != 2:
            raise ValueError('Stereo requires two actual recorded channels')
    elif mode == 'mean':
        x = x.mean(axis=1, keepdims=True)
    elif mode == 'left':
        x = x[:, :1]
    elif mode == 'right':
        if x.shape[1] != 2:
            raise ValueError('Right channel requires stereo recording')
        x = x[:, 1:2]
    else:
        raise ValueError('Unknown channel mode')
    divisor = math.gcd(int(source_rate), int(target_rate))
    return resample_poly(x, target_rate // divisor, source_rate // divisor, axis=0)


def grouped_folds(rows, group_key):
    """Leave one group out; never silently invent missing group identities."""
    if not rows or any(not row.get(group_key) for row in rows):
        raise ValueError('Every recording needs a verified group identity')
    groups = sorted({row[group_key] for row in rows})
    if len(groups) < 2:
        raise ValueError('At least two independent groups required')
    ids = [row['recording_id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate recording IDs: resolve paired microphones/crops first')
    return [{'held_out': group,
             'train': [r['recording_id'] for r in rows if r[group_key] != group],
             'test': [r['recording_id'] for r in rows if r[group_key] == group]}
            for group in groups]


def fit_scale(train):
    """Fit only on training data; apply the returned values to validation/test."""
    train = np.asarray(train)
    if not train.size or not np.isfinite(train).all():
        raise ValueError('Invalid training input')
    low, high = float(train.min()), float(train.max())
    return low, high - low if high > low else 1.0
