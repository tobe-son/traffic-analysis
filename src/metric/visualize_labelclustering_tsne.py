"""t-SNE visualization for LabelClustering using a trained encoder.

This script loads a trained encoder (.pth) and runs embedding extraction +
visualization (t-SNE by default). Optionally it can resolve the best encoder
from an Optuna study (requires that HPO saved checkpoints).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from learn_tool.settings import output_settings, prepare_dataloader
from learn_tool.visualize import LatentSpaceVisualizer
from metric.LabelClustering import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_REPRESENTATION,
    MODEL_REGISTRY,
    collect_embeddings,
    compute_stats_by_label,
    reduce_latent_space,
    resolve_hop_length,
    resolve_representation,
    save_latent_results,
    save_metadata,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE visualization for LabelClustering encoders")

    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="vgg11")
    parser.add_argument("--data-selection", default="loc1-6")
    parser.add_argument("--data-csv", default="./data/processed/datasets/data_1-6.csv")
    parser.add_argument("--main-data-dir", default="./data/processed/datasets")
    parser.add_argument("--mel", choices=["ON", "OFF"], default="OFF")

    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--representation", choices=["waveform", "spectrogram"], default=None)
    parser.add_argument("--visualization", choices=["t-SNE", "PCA", "LDA", "UMAP", "MDS"], default="t-SNE")
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--save-latent-space", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-weights", default=None, help="Path to encoder .pth")
    parser.add_argument("--study-name", default=None, help="Optuna study name to load best encoder")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (e.g., sqlite:///outputs/hpo.db)")

    return parser.parse_args()


def _resolve_weights_path(args: argparse.Namespace) -> Path:
    if args.encoder_weights:
        return Path(args.encoder_weights).expanduser()

    if args.study_name and args.storage:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
        best_trial = study.best_trial
        output_dir = Path(str(best_trial.user_attrs.get("output_dir", "")))
        if output_dir:
            candidate = output_dir / f"best_encoder_{args.model}.pth"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Best encoder checkpoint not found in best trial output_dir. "
            "Run HPO with --save-checkpoints (or --export-best-weights) first."
        )

    raise ValueError("--encoder-weights or (--study-name and --storage) must be provided.")


def main() -> None:
    args = parse_args()

    model_key = args.model
    _description, model_cls = MODEL_REGISTRY[model_key]
    representation = resolve_representation(model_key, args.representation)
    hop_length = resolve_hop_length(model_key, args.hop_length)

    weights_path = _resolve_weights_path(args)
    if not weights_path.exists():
        raise FileNotFoundError(f"Encoder weights not found: {weights_path}")

    logger = output_settings()
    logger.info("Using encoder weights: %s", weights_path)

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
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    latent_matrix, speeds, vehicle_types, directions, locations = collect_embeddings(
        model,
        [train_loader, val_loader],
        device,
    )
    logger.info("潜在空間の抽出が完了しました。")

    if args.max_points and args.max_points > 0 and latent_matrix.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(latent_matrix.shape[0], size=args.max_points, replace=False)
        keep.sort()
        latent_matrix = latent_matrix[keep]
        speeds = speeds[keep]
        vehicle_types = vehicle_types[keep]
        directions = directions[keep]
        locations = locations[keep]
        logger.info("Downsampled embeddings to %d points for visualization.", args.max_points)

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

    speed_stats = compute_stats_by_label(speeds, classes)
    for label, mean_speed in sorted(speed_stats.items()):
        logger.info("Class %d 平均速度: %.2f", label, mean_speed)

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
