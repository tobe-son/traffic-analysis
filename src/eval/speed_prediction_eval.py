"""Evaluate a trained speed-regression model (encoder + MLP) without training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

if os.environ.get("MPLBACKEND") is None:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from learn_tool.settings import output_settings, prepare_dataloader
from metric.utils import describe_continuous, global_average_pool, set_global_seed
from encoder.base_model import Encoder_Original, Encoder_Small, Encoder_Wave1D
from encoder.new_model import Encoder_ResNet, Encoder_ResNet18, Encoder_ResNet50, Encoder_VGG11


ModelEntry = Tuple[str, Type[torch.nn.Module], str]

MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "wave1d": ("1D Conv encoder (waveform)", Encoder_Wave1D, "waveform"),
    "small": ("Small CNN encoder", Encoder_Small, "spectrogram"),
    "original": ("Original CNN encoder", Encoder_Original, "spectrogram"),
    "vgg11": ("VGG11-based encoder", Encoder_VGG11, "spectrogram"),
    "resnet": ("Residual CNN encoder (legacy)", Encoder_ResNet, "spectrogram"),
    "resnet18": ("ResNet-18 encoder", Encoder_ResNet18, "spectrogram"),
    "resnet50": ("ResNet-50 encoder", Encoder_ResNet50, "spectrogram"),
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
    parser = argparse.ArgumentParser(description="Evaluate a trained speed regressor.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="small")
    parser.add_argument("--model-weights", required=True, help="Path to trained speed regressor weights (.pth)")
    parser.add_argument("--data-csv", default="./data/processed/datasets/data_1-6.csv")
    parser.add_argument("--main-data-dir", default="./data/processed/datasets")
    parser.add_argument("--data-selection", default="loc1-6")
    parser.add_argument("--representation", choices=["waveform", "spectrogram"], default=None)
    parser.add_argument("--mel", choices=["ON", "OFF"], default="OFF")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--loss", choices=LOSS_REGISTRY.keys(), default="mse")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true", help="Plot prediction scatter for train/val sets")
    parser.add_argument("--output-dir", default=None, help="Directory to save metrics, predictions, and plots")
    parser.add_argument(
        "--optuna-params",
        default=None,
        help="Path to optuna_best.json to auto-fill batch_size, hop_length, n_fft, mel",
    )
    return parser.parse_args()


def resolve_representation(model_key: str, override: Optional[str]) -> str:
    if override is not None:
        return override
    return MODEL_REGISTRY[model_key][2]


def resolve_hop_length(model_key: str, override: Optional[int]) -> int:
    if override is not None:
        return override
    return 512 if model_key in {"wave1d", "small"} else 160


def load_optuna_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "best_params" in data and isinstance(data["best_params"], dict):
        return data["best_params"]
    return data


def apply_optuna_params(args: argparse.Namespace, params: dict, logger) -> None:
    mapping = {
        "batch_size": ("batch_size", "--batch-size"),
        "hop_length": ("hop_length", "--hop-length"),
        "n_fft": ("n_fft", "--n-fft"),
        "mel": ("mel", "--mel"),
    }
    argv = set(sys.argv[1:])
    for key, (arg_name, flag) in mapping.items():
        if key in params and flag not in argv:
            setattr(args, arg_name, params[key])
            logger.info("Applied optuna param %s=%s", key, params[key])


def build_model(args: argparse.Namespace, device: torch.device) -> SpeedRegressor:
    _, model_cls, _ = MODEL_REGISTRY[args.model]
    encoder = model_cls().to(device)
    model = SpeedRegressor(encoder, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    state_dict = torch.load(args.model_weights, map_location=device)
    model.load_state_dict(state_dict)
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
            degree = min(3, actual.size - 1)
            coeffs = np.polyfit(actual, predicted, deg=degree)
            poly_fn = np.poly1d(coeffs)
            sort_idx = np.argsort(actual)
            sorted_actual = actual[sort_idx]
            fitted = poly_fn(sorted_actual)
            residuals = predicted - poly_fn(actual)
            sigma = float(np.std(residuals, ddof=1))
            plt.plot(sorted_actual, fitted, "b-", linewidth=2, label=f"Poly fit (deg={degree})")
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


def compute_regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    if actual.size == 0:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "me": float("nan"),
            "r2": float("nan"),
            "mape": float("nan"),
        }
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    me = float(np.mean(errors))
    denom = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1.0 - np.sum(errors**2) / denom) if denom > 0 else float("nan")
    nonzero = actual != 0
    if np.any(nonzero):
        mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100.0)
    else:
        mape = float("nan")
    return {
        "count": int(actual.size),
        "mae": mae,
        "rmse": rmse,
        "me": me,
        "r2": r2,
        "mape": mape,
    }


def write_metrics(path: str, payload: dict, logger) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved metrics to %s", path)


def main() -> None:
    args = parse_args()

    logger = output_settings(args.output_dir)
    logger.info("Using model checkpoint: %s", args.model_weights)

    if args.optuna_params:
        optuna_path = Path(args.optuna_params)
        if optuna_path.exists():
            optuna_params = load_optuna_params(optuna_path)
            apply_optuna_params(args, optuna_params, logger)
        else:
            logger.warning("Optuna params file not found: %s", optuna_path)

    representation = resolve_representation(args.model, args.representation)
    hop_length = resolve_hop_length(args.model, args.hop_length)

    hyperparams = {
        "MODEL": args.model,
        "MODEL_WEIGHTS": os.path.abspath(args.model_weights),
        "DATA_SELECTION": args.data_selection,
        "DATA_CSV_PATH": args.data_csv,
        "MAIN_DATA_DIR": args.main_data_dir,
        "REPRESENTATION": representation,
        "MEL": args.mel,
        "SAMPLING_RATE": args.sampling_rate,
        "N_FFT": args.n_fft,
        "HOP_LENGTH": hop_length,
        "BATCH_SIZE": args.batch_size,
        "LOSS": args.loss,
        "HIDDEN_DIM": args.hidden_dim,
        "DROPOUT": args.dropout,
        "TEST_SPLIT": args.test_split,
        "SEED": args.seed,
        "OUTPUT_DIR": logger.output_dir,
    }
    logger.info("Evaluation settings:")
    for key, value in hyperparams.items():
        logger.info("%s: %s", key, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    set_global_seed(args.seed)

    train_loader, val_loader = make_dataloaders(args, logger, representation, hop_length)

    model = build_model(args, device)
    criterion = LOSS_REGISTRY[args.loss]()

    logger.info("Evaluating model on training set…")
    train_loss, train_actual, train_pred = evaluate(model, train_loader, criterion, device)
    train_csv = os.path.join(logger.output_dir, f"train_predictions_{args.loss}.csv")
    save_predictions(train_csv, train_actual, train_pred, logger)
    train_metrics = compute_regression_metrics(train_actual, train_pred)
    logger.info(
        "Train metrics -> count=%d MAE=%.4f RMSE=%.4f ME=%.4f R2=%.4f MAPE=%.2f%%",
        train_metrics["count"],
        train_metrics["mae"],
        train_metrics["rmse"],
        train_metrics["me"],
        train_metrics["r2"],
        train_metrics["mape"],
    )

    logger.info("Evaluating model on validation set…")
    val_loss, val_actual, val_pred = evaluate(model, val_loader, criterion, device)
    val_csv = os.path.join(logger.output_dir, f"val_predictions_{args.loss}.csv")
    save_predictions(val_csv, val_actual, val_pred, logger)
    val_metrics = compute_regression_metrics(val_actual, val_pred)
    logger.info(
        "Val metrics -> count=%d MAE=%.4f RMSE=%.4f ME=%.4f R2=%.4f MAPE=%.2f%%",
        val_metrics["count"],
        val_metrics["mae"],
        val_metrics["rmse"],
        val_metrics["me"],
        val_metrics["r2"],
        val_metrics["mape"],
    )

    metrics_payload = {
        "settings": hyperparams,
        "train": {"loss": float(train_loss), "metrics": train_metrics},
        "val": {"loss": float(val_loss), "metrics": val_metrics},
    }
    metrics_path = os.path.join(logger.output_dir, "evaluation_metrics.json")
    write_metrics(metrics_path, metrics_payload, logger)

    logger.info("Final losses -> train=%.4f val=%.4f", train_loss, val_loss)

    if args.visualize:
        plot_scatter(
            train_actual,
            train_pred,
            os.path.join(logger.output_dir, f"train_scatter_{args.loss}.png"),
            "Train Predictions",
            logger,
        )
        plot_scatter(
            val_actual,
            val_pred,
            os.path.join(logger.output_dir, f"val_scatter_{args.loss}.png"),
            "Validation Predictions",
            logger,
        )


if __name__ == "__main__":
    main()
