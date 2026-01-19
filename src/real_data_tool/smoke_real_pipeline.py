from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from learn_tool.settings import output_settings, prepare_dataloader
from encoder.base_model import Encoder_Small


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test dataloading + one forward pass on real datasets.")
    p.add_argument("--data-csv", required=True)
    p.add_argument("--main-data-dir", required=True)
    p.add_argument("--data-selection", default="loc1")
    p.add_argument("--representation", choices=["spectrogram", "waveform"], default="spectrogram")
    p.add_argument("--mel", choices=["ON", "OFF"], default="OFF")
    p.add_argument("--sampling-rate", type=int, default=16000)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--test-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = output_settings()

    train_loader, val_loader, fmin, fmax = prepare_dataloader(
        logger=logger,
        hop_length=args.hop_length,
        batch_size=args.batch_size,
        data_csv_path=args.data_csv,
        main_data_dir=args.main_data_dir,
        n_fft=args.n_fft,
        data_num=args.data_selection,
        mel=args.mel,
        sampling_rate=args.sampling_rate,
        test_split=args.test_split,
        seed=args.seed,
        representation=args.representation,
    )

    logger.info("Loaded train=%d val=%d", len(train_loader.dataset), len(val_loader.dataset))
    logger.info("Feature min/max before normalisation: %.6f / %.6f", fmin, fmax)

    batch = next(iter(train_loader))
    x, speed, vtype, direc, loc = batch
    logger.info("Batch shapes: x=%s speed=%s", tuple(x.shape), tuple(speed.shape))
    logger.info(
        "Labels sample: speed=%s vtype=%s direc=%s loc=%s",
        speed[:5].tolist(),
        vtype[:5].tolist(),
        direc[:5].tolist(),
        loc[:5].tolist(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder_Small().to(device)
    with torch.no_grad():
        y = model(x.to(device))
    logger.info("Forward ok: encoder output=%s on %s", tuple(y.shape), device)


if __name__ == "__main__":
    main()
