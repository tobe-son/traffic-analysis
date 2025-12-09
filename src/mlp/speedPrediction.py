"""Pretrained encoder + MLP speed regression with a unified CLI."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from learn_tool.settings import output_settings, prepare_dataloader
from metric.utils import describe_continuous, global_average_pool, set_global_seed
from encoder.base_model import Encoder_Original, Encoder_Small, Encoder_Wave1D
from encoder.new_model import Encoder_ResNet, Encoder_VGG11


ModelEntry = Tuple[str, Type[torch.nn.Module], str]

MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "wave1d": ("1D Conv encoder (waveform)", Encoder_Wave1D, "waveform"),
    "small": ("Small CNN encoder", Encoder_Small, "spectrogram"),
    "original": ("Original CNN encoder", Encoder_Original, "spectrogram"),
    "vgg11": ("VGG11-based encoder", Encoder_VGG11, "spectrogram"),
    "resnet": ("Residual CNN encoder", Encoder_ResNet, "spectrogram"),
}

LOSS_REGISTRY: Dict[str, Type[nn.Module]] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.SmoothL1Loss,
}


class SpeedRegressor(nn.Module):
    """Wrap an encoder with a lightweight regression head."""

    def __init__(self, encoder: nn.Module, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = encoder
        self.regressor = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        pooled = global_average_pool(features)
        flat = pooled.view(pooled.size(0), -1)
        return self.regressor(flat).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an encoder for speed regression.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="small")
    parser.add_argument("--encoder-weights", required=True, help="Path to pretrained encoder weights (.pth)")
    parser.add_argument("--data-csv", default="./data/processed/datasets/data_1-6.csv")
    parser.add_argument("--main-data-dir", default="./data/processed/datasets")
    parser.add_argument("--data-selection", default="loc1-6")
    parser.add_argument("--representation", choices=["waveform", "spectrogram"], default=None)
    parser.add_argument("--mel", choices=["ON", "OFF"], default="OFF")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=None, help="Learning rate for encoder parameters (defaults to lr)")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--loss", choices=LOSS_REGISTRY.keys(), default="mse")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true", help="Plot prediction scatter for train/val sets")
    return parser.parse_args()


def resolve_representation(model_key: str, override: Optional[str]) -> str:
    if override is not None:
        return override
    return MODEL_REGISTRY[model_key][2]


def resolve_hop_length(model_key: str, override: Optional[int]) -> int:
    if override is not None:
        return override
    return 512 if model_key in {"wave1d", "small"} else 160


def build_model(args: argparse.Namespace, device: torch.device) -> SpeedRegressor:
    _, model_cls, _ = MODEL_REGISTRY[args.model]
    encoder = model_cls().to(device)
    state_dict = torch.load(args.encoder_weights, map_location=device)
    encoder.load_state_dict(state_dict)
    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
    model = SpeedRegressor(encoder, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    return model


def make_dataloaders(args, logger, representation: str, hop_length: int):
    train_loader, val_loader, feature_min, feature_max = prepare_dataloader(
        logger=logger,
        hop_length=hop_length,
        batch_size=args.batch_size,
        data_csv_path=args.data_csv,
        main_data_dir=args.main_data_dir,
        n_fft=args.n_fft,
        data_num=args.data_selection,
        mel=args.mel,
        sampling_rate=args.sampling_rate,
        test_split=args.test_split,
        seed=args.seed,
        representation=representation,
    )
    all_speeds = np.concatenate(
        (
            train_loader.dataset.tensors[1].cpu().numpy(),
            val_loader.dataset.tensors[1].cpu().numpy(),
        )
    )
    stats = describe_continuous(all_speeds)
    logger.info(
        "Speed stats -> count=%d min=%.2f max=%.2f mean=%.2f std=%.2f",
        stats.get("count", 0),
        stats.get("min", float("nan")),
        stats.get("max", float("nan")),
        stats.get("mean", float("nan")),
        stats.get("std", float("nan")),
    )
    logger.info(
        "feature range after normalisation: min=%.6f max=%.6f",
        feature_min,
        feature_max,
    )
    return train_loader, val_loader


def select_parameters(model: SpeedRegressor, base_lr: float, encoder_lr: Optional[float]) -> Iterable[dict]:
    if encoder_lr is None:
        encoder_lr = base_lr
    params = []
    if any(p.requires_grad for p in model.encoder.parameters()):
        params.append({"params": model.encoder.parameters(), "lr": encoder_lr})
    params.append({"params": model.regressor.parameters(), "lr": base_lr})
    return params


def train_one_epoch(model, loader, optimizer, criterion, device, epoch_label: str) -> float:
    model.train()
    running = 0.0
    count = 0
    for batch in tqdm(loader, desc=epoch_label):
        data, speed, *_ = batch
        data = data.to(device)
        speed = speed.to(device)
        preds = model(data)
        loss = criterion(preds, speed)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * data.size(0)
        count += data.size(0)
    return running / max(count, 1)


def evaluate(model, loader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total = 0.0
    count = 0
    all_preds: list[float] = []
    all_targets: list[float] = []
    with torch.no_grad():
        for data, speed, *_ in loader:
            data = data.to(device)
            speed = speed.to(device)
            preds = model(data)
            loss = criterion(preds, speed)
            total += loss.item() * data.size(0)
            count += data.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(speed.cpu().tolist())
    return total / max(count, 1), np.asarray(all_targets), np.asarray(all_preds)


def save_predictions(path: str, actual: np.ndarray, predicted: np.ndarray, logger) -> None:
    df = pd.DataFrame({"original_speed": actual, "predicted_speed": predicted})
    df.to_csv(path, index=False)
    logger.info("Saved predictions to %s", path)


def plot_scatter(actual: np.ndarray, predicted: np.ndarray, path: str, title: str, logger) -> None:
    if actual.size == 0:
        logger.warning("Skip scatter plot: no samples available.")
        return
    min_val = float(min(actual.min(), predicted.min()))
    max_val = float(max(actual.max(), predicted.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.6, edgecolors="none", label="Samples")
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
    if actual.size > 2:
        try:
            coeffs = np.polyfit(actual, predicted, deg=min(3, actual.size - 1))
            poly_fn = np.poly1d(coeffs)
            sort_idx = np.argsort(actual)
            sorted_actual = actual[sort_idx]
            fitted = poly_fn(sorted_actual)
            residuals = predicted - poly_fn(actual)
            sigma = float(np.std(residuals, ddof=1))
            plt.plot(sorted_actual, fitted, "b-", linewidth=2, label="Poly fit")
            plt.fill_between(sorted_actual, fitted - sigma, fitted + sigma, color="blue", alpha=0.2, label="±1σ")
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.warning("Polynomial fit skipped: %s", exc)
    plt.xlabel("Actual Speed [km/h]")
    plt.ylabel("Predicted Speed [km/h]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    logger.info("Saved scatter plot to %s", path)


def plot_loss_curve(train_losses: list[float], val_losses: list[float], output_dir: str, logger) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train")
    plt.plot(epochs, val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(path)
    plt.close()
    logger.info("Saved loss curve to %s", path)


def main() -> None:
    args = parse_args()

    representation = resolve_representation(args.model, args.representation)
    hop_length = resolve_hop_length(args.model, args.hop_length)

    logger = output_settings()
    logger.info("Using pretrained encoder weights: %s", args.encoder_weights)

    hyperparams = {
        "MODEL": args.model,
        "DATA_SELECTION": args.data_selection,
        "DATA_CSV_PATH": args.data_csv,
        "MAIN_DATA_DIR": args.main_data_dir,
        "REPRESENTATION": representation,
        "MEL": args.mel,
        "SAMPLING_RATE": args.sampling_rate,
        "N_FFT": args.n_fft,
        "HOP_LENGTH": hop_length,
        "BATCH_SIZE": args.batch_size,
        "EPOCHS": args.epochs,
        "LR": args.lr,
        "ENCODER_LR": args.encoder_lr if args.encoder_lr is not None else args.lr,
        "FREEZE_ENCODER": args.freeze_encoder,
        "LOSS": args.loss,
        "HIDDEN_DIM": args.hidden_dim,
        "DROPOUT": args.dropout,
        "TEST_SPLIT": args.test_split,
        "SEED": args.seed,
    }
    logger.info("Hyperparameters:")
    for key, value in hyperparams.items():
        logger.info("%s: %s", key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    set_global_seed(args.seed)

    train_loader, val_loader = make_dataloaders(args, logger, representation, hop_length)

    model = build_model(args, device)
    criterion = LOSS_REGISTRY[args.loss]()
    optimizer = optim.Adam(select_parameters(model, args.lr, args.encoder_lr))

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = math.inf
    best_epoch = 0
    best_path = os.path.join(logger.output_dir, f"best_speed_regressor_{args.loss}.pth")
    last_path = os.path.join(logger.output_dir, f"last_speed_regressor_{args.loss}.pth")

    for epoch in range(args.epochs):
        label = f"Train {epoch + 1}/{args.epochs}"
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, label)
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logger.info(
            "Epoch %d/%d -> train_loss=%.4f val_loss=%.4f",
            epoch + 1,
            args.epochs,
            train_loss,
            val_loss,
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            logger.info("Updated best model (epoch %d, val_loss=%.4f)", best_epoch, best_val)

    torch.save(model.state_dict(), last_path)
    logger.info("Saved last model to %s", last_path)

    plot_loss_curve(train_losses, val_losses, logger.output_dir, logger)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        logger.info("Loaded best checkpoint from epoch %d", best_epoch)

    logger.info("Evaluating best model on training set…")
    train_loss, train_actual, train_pred = evaluate(model, train_loader, criterion, device)
    train_csv = os.path.join(logger.output_dir, f"train_predictions_{args.loss}.csv")
    save_predictions(train_csv, train_actual, train_pred, logger)

    logger.info("Evaluating best model on validation set…")
    val_loss, val_actual, val_pred = evaluate(model, val_loader, criterion, device)
    val_csv = os.path.join(logger.output_dir, f"val_predictions_{args.loss}.csv")
    save_predictions(val_csv, val_actual, val_pred, logger)

    logger.info("Final losses -> train=%.4f val=%.4f", train_loss, val_loss)

    if args.visualize:
        plot_scatter(train_actual, train_pred, os.path.join(logger.output_dir, f"train_scatter_{args.loss}.png"), "Train Predictions", logger)
        plot_scatter(val_actual, val_pred, os.path.join(logger.output_dir, f"val_scatter_{args.loss}.png"), "Validation Predictions", logger)


if __name__ == "__main__":
    main()
