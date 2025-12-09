"""Log-Ratio loss を用いた連続値メトリックラーニングの共通化スクリプト。"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE
import umap  # type: ignore
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from learn_tool.settings import output_settings, prepare_dataloader
from learn_tool.visualize import LatentSpaceVisualizer
from encoder.base_model import Encoder_Original, Encoder_Small, Encoder_Wave1D
from encoder.new_model import Encoder_ResNet, Encoder_VGG11
from loss.LogRatioLoss import LogRatioLoss
from metric.utils import (describe_continuous, extract_embedding,
                          set_global_seed)


ModelEntry = Tuple[str, Type[torch.nn.Module]]

MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "wave1d": ("1D Conv encoder (waveform)", Encoder_Wave1D),
    "small": ("Small CNN encoder", Encoder_Small),
    "original": ("Original CNN encoder", Encoder_Original),
    "vgg11": ("VGG11-based encoder", Encoder_VGG11),
    "resnet": ("Residual CNN encoder", Encoder_ResNet),
}

DEFAULT_REPRESENTATION = {
    "wave1d": "waveform",
    "small": "spectrogram",
    "original": "spectrogram",
    "vgg11": "spectrogram",
    "resnet": "spectrogram",
}

DEFAULT_HOP_LENGTH = {
    "wave1d": 512,
    "small": 512,
    "original": 160,
    "vgg11": 160,
    "resnet": 160,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LogRatioLoss を用いて任意のエンコーダで連続値のメトリックラーニングを行うスクリプト"
    )
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="vgg11", help="使用するエンコーダ")
    parser.add_argument("--data-selection", default="loc1-6", help="使用するデータ識別子 (例: loc1, loc1-6)")
    parser.add_argument("--data-csv", default="./data/processed/datasets/data_1-6.csv", help="メタデータ CSV のパス")
    parser.add_argument("--main-data-dir", default="./data/processed/datasets", help="音声データのベースディレクトリ")
    parser.add_argument("--mel", choices=["ON", "OFF"], default="OFF", help="メルスペクトログラムを使用するか")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=None, help="STFT の hop length。未指定時はモデル推奨値を使用")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--norm-p", type=float, default=2.0, help="LogRatioLoss の Minkowski 次数")
    parser.add_argument("--eps", type=float, default=1e-6, help="LogRatioLoss の安定化項")
    parser.add_argument("--representation", choices=["waveform", "spectrogram"], default=None, help="入力表現。未指定時はモデル推奨を使用")
    parser.add_argument("--visualization", choices=["t-SNE", "PCA", "LDA", "UMAP", "MDS"], default="t-SNE")
    parser.add_argument("--dimension", type=int, default=2, help="潜在空間の可視化次元")
    parser.add_argument("--save-latent-space", action="store_true", help="潜在空間を CSV として保存する")
    parser.add_argument("--show-plot", action="store_true", help="可視化をインタラクティブ表示する")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-dir", default=None, help="CircleLoss などで学習済みの出力ディレクトリ")
    parser.add_argument("--resume-checkpoint", choices=["best", "last"], default="best", help="resume-dir 内で読むチェックポイント種別")
    parser.add_argument("--init-encoder-state", default=None, help="直接指定するエンコーダ重みファイル (.pth)")
    parser.add_argument("--init-optimizer-state", default=None, help="直接指定するオプティマイザ状態ファイル (.pth)")
    parser.add_argument("--load-optimizer", action="store_true", help="指定したチェックポイントからオプティマイザも復元する")
    return parser.parse_args()


def log_hyperparameters(logger, params) -> None:
    logger.info("Hyperparameters:")
    for key, value in params.items():
        logger.info("%s: %s", key, value)


def resolve_representation(model_key: str, override: str | None) -> str:
    if override is not None:
        return override
    return DEFAULT_REPRESENTATION.get(model_key, "spectrogram")


def resolve_hop_length(model_key: str, override: int | None) -> int:
    if override is not None:
        return override
    return DEFAULT_HOP_LENGTH.get(model_key, 512)


def resolve_checkpoint_paths(
    args: argparse.Namespace,
    model_key: str,
    logger,
) -> Tuple[Optional[Path], Optional[Path]]:
    if args.init_encoder_state:
        encoder_path = Path(args.init_encoder_state).expanduser()
        optimizer_path = (
            Path(args.init_optimizer_state).expanduser()
            if args.init_optimizer_state
            else None
        )
        return encoder_path, optimizer_path

    if args.resume_dir:
        base = Path(args.resume_dir).expanduser()
        if not base.exists():
            logger.warning("resume-dir %s が存在しません。", base)
            return None, None

        candidates = []
        checkpoint_tag = args.resume_checkpoint
        if checkpoint_tag:
            candidates.append(base / f"{checkpoint_tag}_encoder_{model_key}.pth")
        candidates.extend(
            [
                base / f"best_encoder_{model_key}.pth",
                base / f"last_encoder_{model_key}.pth",
            ]
        )

        encoder_path = next((path for path in candidates if path.exists()), None)
        if encoder_path is None:
            logger.warning("指定ディレクトリに %s 用のエンコーダ重みが見つかりませんでした。", model_key)
            return None, None

        optimizer_path = base / f"optimizer_{model_key}.pth"
        if not optimizer_path.exists():
            optimizer_path = None

        return encoder_path, optimizer_path

    return None, None


def log_ratio_loss_for_batch(model, batch, device, criterion):
    data, speed, *_ = batch
    if data.size(0) < 2:
        return None

    embeddings = extract_embedding(model, data.to(device))
    speeds = speed.to(device)
    gt_dist = torch.abs(speeds[0] - speeds[1:])
    if gt_dist.numel() == 0:
        return None

    return criterion(embeddings, gt_dist)


def train_one_epoch(model, loader, optimizer, device, criterion, epoch_label: str) -> float:
    model.train()
    total_loss = 0.0
    usable_batches = 0
    for batch in tqdm(loader, desc=epoch_label):
        optimizer.zero_grad()
        loss = log_ratio_loss_for_batch(model, batch, device, criterion)
        if loss is None or not torch.isfinite(loss):
            continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        usable_batches += 1

    if usable_batches == 0:
        raise RuntimeError("No valid batches produced a LogRatioLoss signal. Increase batch size or ensure diverse speeds.")

    return total_loss / usable_batches


def evaluate(model, loader, device, criterion) -> float:
    model.eval()
    total_loss = 0.0
    usable_batches = 0
    with torch.no_grad():
        for batch in loader:
            loss = log_ratio_loss_for_batch(model, batch, device, criterion)
            if loss is None or not torch.isfinite(loss):
                continue
            total_loss += loss.item()
            usable_batches += 1

    if usable_batches == 0:
        return float("nan")

    return total_loss / usable_batches


def collect_embeddings(model, loaders, device):
    model.eval()
    latent_chunks = []
    speeds = []
    vehicle_types = []
    directions = []
    locations = []

    with torch.no_grad():
        for loader in loaders:
            for data, speed, vtype, direc, loc in loader:
                embeddings = extract_embedding(model, data.to(device))
                latent_chunks.append(embeddings.cpu())
                speeds.append(speed.numpy())
                vehicle_types.append(vtype.numpy())
                directions.append(direc.numpy())
                locations.append(loc.numpy())

    if not latent_chunks:
        raise RuntimeError("No embeddings were collected from the provided loaders.")

    latent_matrix = torch.cat(latent_chunks, dim=0).numpy()
    speeds = np.concatenate(speeds)
    vehicle_types = np.concatenate(vehicle_types)
    directions = np.concatenate(directions)
    locations = np.concatenate(locations)
    return latent_matrix, speeds, vehicle_types, directions, locations


def reduce_latent_space(latent_matrix, classes, method, components, seed, logger):
    logger.info("%s による次元削減を開始します。", method)

    if method == "t-SNE":
        reducer = TSNE(n_components=components, random_state=seed)
        return reducer.fit_transform(latent_matrix)

    if method == "PCA":
        reducer = PCA(n_components=components, random_state=seed)
        return reducer.fit_transform(latent_matrix)

    if method == "LDA":
        reducer = LinearDiscriminantAnalysis(
            n_components=min(components, len(np.unique(classes)) - 1)
        )
        return reducer.fit_transform(latent_matrix, classes)

    if method == "UMAP":
        reducer = umap.UMAP(n_components=components, random_state=seed)
        return reducer.fit_transform(latent_matrix)

    if method == "MDS":
        reducer = MDS(n_components=components, random_state=seed)
        return reducer.fit_transform(latent_matrix)

    raise ValueError(f"Unsupported visualization method: {method}")


def save_latent_results(latent_matrix, reduced, output_dir, components, logger) -> None:
    latent_spaces_file = os.path.join(output_dir, "latent_spaces.csv")
    pd.DataFrame(latent_matrix).to_csv(latent_spaces_file, index=False)
    logger.info("latent_spaces を %s に保存しました。", latent_spaces_file)

    if components == 2:
        columns = ["Dimension_1", "Dimension_2"]
        filename = "latent_2d.csv"
    elif components == 3:
        columns = ["Dimension_1", "Dimension_2", "Dimension_3"]
        filename = "latent_3d.csv"
    else:
        columns = [f"Dimension_{idx + 1}" for idx in range(components)]
        filename = "latent_reduced.csv"

    reduced_file = os.path.join(output_dir, filename)
    pd.DataFrame(reduced, columns=columns).to_csv(reduced_file, index=False)
    logger.info("reduced latent を %s に保存しました。", reduced_file)


def save_metadata(speeds, vehicle_types, directions, locations, output_dir, logger) -> None:
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata_df = pd.DataFrame({
        "speed": speeds,
        "vehicle_type": vehicle_types,
        "direction": directions,
        "location": locations,
    })
    metadata_df.to_csv(metadata_path, index=False)
    logger.info("メタデータを %s に保存しました。", metadata_path)


def log_speed_summary(logger, speeds) -> None:
    stats = describe_continuous(speeds)
    if stats["count"] == 0:
        logger.warning("No speed information available for summary stats.")
        return
    logger.info(
        "Speed stats -> count=%d min=%.2f max=%.2f mean=%.2f std=%.2f",
        stats["count"],
        stats["min"],
        stats["max"],
        stats["mean"],
        stats["std"],
    )


def main() -> None:
    args = parse_args()

    model_key = args.model
    description, model_cls = MODEL_REGISTRY[model_key]
    representation = resolve_representation(model_key, args.representation)
    hop_length = resolve_hop_length(model_key, args.hop_length)

    logger = output_settings()

    hyperparameters = {
        "MODEL": model_key,
        "MODEL_DESCRIPTION": description,
        "DATA_SELECTION": args.data_selection,
        "DATA_CSV_PATH": args.data_csv,
        "MAIN_DATA_DIR": args.main_data_dir,
        "MEL": args.mel,
        "SAMPLING_RATE": args.sampling_rate,
        "N_FFT": args.n_fft,
        "HOP_LENGTH": hop_length,
        "TEST_DATASET_PERCENTAGE": args.test_split,
        "BATCH_SIZE": args.batch_size,
        "EPOCHS": args.epochs,
        "LEARNING_RATE": args.lr,
        "NORM_P": args.norm_p,
        "EPS": args.eps,
        "REPRESENTATION": representation,
        "VISUALIZATION": args.visualization,
        "DIMENSION_LATENT_SPACE": args.dimension,
        "SAVE_LATENT_SPACE": args.save_latent_space,
        "SEED": args.seed,
        "RESUME_DIR": args.resume_dir,
        "RESUME_CHECKPOINT": args.resume_checkpoint,
        "INIT_ENCODER_STATE": args.init_encoder_state,
        "INIT_OPTIMIZER_STATE": args.init_optimizer_state,
        "LOAD_OPTIMIZER": args.load_optimizer,
    }
    log_hyperparameters(logger, hyperparameters)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    set_global_seed(args.seed)

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
    logger.info(
        "データの準備が完了しました。feature_min=%.6f feature_max=%.6f",
        feature_min,
        feature_max,
    )

    model = model_cls().to(device)
    criterion = LogRatioLoss(p=args.norm_p, eps=args.eps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    encoder_ckpt, optimizer_ckpt = resolve_checkpoint_paths(args, model_key, logger)
    if encoder_ckpt is not None and encoder_ckpt.exists():
        state = torch.load(encoder_ckpt, map_location=device)
        model.load_state_dict(state)
        logger.info("Loaded encoder weights from %s", encoder_ckpt)
    elif encoder_ckpt is not None:
        logger.warning("指定したエンコーダ重み %s が見つかりませんでした。", encoder_ckpt)

    if args.load_optimizer:
        if optimizer_ckpt is not None and optimizer_ckpt.exists():
            opt_state = torch.load(optimizer_ckpt, map_location=device)
            optimizer.load_state_dict(opt_state)
            logger.info("Loaded optimizer state from %s", optimizer_ckpt)
        elif optimizer_ckpt is not None:
            logger.warning("指定したオプティマイザ状態 %s が見つかりませんでした。", optimizer_ckpt)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(args.epochs):
        epoch_label = f"Train Epoch {epoch + 1}/{args.epochs}"
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch_label)
        val_loss = evaluate(model, val_loader, device, criterion)

        logger.info(
            "Epoch [%d/%d] train_loss=%.4f val_loss=%s",
            epoch + 1,
            args.epochs,
            train_loss,
            f"{val_loss:.4f}" if math.isfinite(val_loss) else "nan",
        )

        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(logger.output_dir, f"best_encoder_{model_key}.pth"))
            torch.save(optimizer.state_dict(), os.path.join(logger.output_dir, f"optimizer_{model_key}.pth"))
            logger.info("Best model updated (epoch %d, val_loss=%.4f)", best_epoch, best_val_loss)

    torch.save(model.state_dict(), os.path.join(logger.output_dir, f"last_encoder_{model_key}.pth"))
    logger.info("Last model checkpoint saved. Best epoch=%s", best_epoch if best_epoch > 0 else "N/A")

    latent_matrix, speeds, vehicle_types, directions, locations = collect_embeddings(
        model,
        [train_loader, val_loader],
        device,
    )
    logger.info("潜在空間の抽出が完了しました。")

    classes = (vehicle_types * 2) + directions
    reduced_latent = reduce_latent_space(
        latent_matrix,
        classes,
        args.visualization,
        args.dimension,
        args.seed,
        logger,
    )

    if args.save_latent_space:
        save_latent_results(latent_matrix, reduced_latent, logger.output_dir, args.dimension, logger)

    save_metadata(speeds, vehicle_types, directions, locations, logger.output_dir, logger)
    log_speed_summary(logger, speeds)

    if args.dimension in (2, 3):
        visualizer = LatentSpaceVisualizer(
            dimension_latent_space=args.dimension,
            latent_data=reduced_latent,
            speeds=np.asarray(speeds),
            vehicle_types=np.asarray(vehicle_types),
            directions=np.asarray(directions),
            locations=np.asarray(locations),
            output_dir=logger.output_dir,
            visualization=args.visualization,
        )
        visualizer.visualize_all(show_plot=args.show_plot)
    else:
        logger.warning("Latent visualization is skipped because dimension is not 2 or 3.")


if __name__ == "__main__":
    main()