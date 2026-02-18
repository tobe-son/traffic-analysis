"""Project real-data embeddings into a simulation-trained openTSNE space.

Flow:
1) Extract encoder embeddings from simulation and real datasets.
2) Fit openTSNE on simulation embeddings.
3) Transform real embeddings into the fitted simulation space.
4) Save projected coordinates and overlay plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

if os.environ.get("MPLBACKEND") is None:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openTSNE import TSNE

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from learn_tool.settings import output_settings, prepare_dataloader
from metric import LabelClustering as circle_backend
from metric import LabelClustering_arcface as arcface_backend
from metric.utils import set_global_seed

logger = logging.getLogger("deep_metric_eval_sim2real_tsne")

BACKENDS = {
    "circle": circle_backend,
    "arcface": arcface_backend,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit openTSNE on simulation embeddings and transform real embeddings"
    )

    parser.add_argument("--loss-type", choices=BACKENDS.keys(), default="circle")
    parser.add_argument("--model", choices=circle_backend.MODEL_REGISTRY.keys(), default="vgg11")
    parser.add_argument("--encoder-weights", required=True, help="Path to encoder .pth")
    parser.add_argument("--output-dir", default=None, help="Output directory for projections and plots")

    parser.add_argument("--sim-data-selection", default="loc1-6")
    parser.add_argument("--sim-data-csv", default="./data/processed/datasets/data_1-6.csv")
    parser.add_argument("--sim-main-data-dir", default="./data/processed/datasets")

    parser.add_argument("--real-data-selection", default="loc1-6")
    parser.add_argument("--real-data-csv", default="./data/processed/real/vs13/vs13.csv")
    parser.add_argument("--real-main-data-dir", default="./data/processed/real/vs13")

    parser.add_argument("--representation", choices=["waveform", "spectrogram"], default=None)
    parser.add_argument("--mel", choices=["ON", "OFF"], default="OFF")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--optuna-params",
        default=None,
        help="Path to optuna_best_params.json to auto-fill batch_size, hop_length, n_fft, mel",
    )

    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--show-plot", action="store_true")

    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluation.log"

    logger = logging.getLogger("deep_metric_eval_sim2real_tsne")
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.output_dir = str(output_dir)

    return logger


def load_optuna_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "best_params" in data and isinstance(data["best_params"], dict):
        return data["best_params"]
    return data


def apply_optuna_params(args: argparse.Namespace, params: dict, logger: logging.Logger) -> None:
    mapping = {
        "batch_size": "batch_size",
        "hop_length": "hop_length",
        "n_fft": "n_fft",
        "mel": "mel",
    }
    for key, arg_name in mapping.items():
        if key in params:
            setattr(args, arg_name, params[key])
            logger.info("Applied optuna param %s=%s", key, params[key])


def _build_projection_frame(
    projected: np.ndarray,
    speeds: np.ndarray,
    vehicle_types: np.ndarray,
    directions: np.ndarray,
    locations: np.ndarray,
    domain: str,
) -> pd.DataFrame:
    columns = {f"dim_{idx + 1}": projected[:, idx] for idx in range(projected.shape[1])}
    columns.update(
        {
            "speed": speeds,
            "vehicle_type": vehicle_types,
            "direction": directions,
            "location": locations,
            "domain": np.full(projected.shape[0], domain),
        }
    )
    return pd.DataFrame(columns)


def _save_sim_fourclass_plot(
    sim_xy: np.ndarray,
    sim_vehicle_type: np.ndarray,
    sim_direction: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    classes = (sim_vehicle_type.astype(int) * 2) + sim_direction.astype(int)
    class_styles = {
        0: {"label": "car/right", "color": "#1f77b4"},
        1: {"label": "car/left", "color": "#ff7f0e"},
        2: {"label": "cv/right", "color": "#2ca02c"},
        3: {"label": "cv/left", "color": "#d62728"},
    }

    for class_id, style in class_styles.items():
        mask = classes == class_id
        if np.any(mask):
            ax.scatter(
                sim_xy[mask, 0],
                sim_xy[mask, 1],
                s=50,
                c=style["color"],
                alpha=0.9,
                edgecolors="none",
                label=style["label"],
            )

    #ax.set_title("Simulation embedding distribution (4 classes)")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_sim_real_overlay_plot(
    sim_xy: np.ndarray,
    real_xy: np.ndarray,
    real_vehicle_type: np.ndarray,
    real_direction: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    class_styles = {
        0: {"label": "real car/right", "color": "#1f77b4"},
        1: {"label": "real car/left", "color": "#ff7f0e"},
        2: {"label": "real cv/right", "color": "#2ca02c"},
        3: {"label": "real cv/left", "color": "#d62728"},
    }

    ax.scatter(
        sim_xy[:, 0],
        sim_xy[:, 1],
        s=50,
        c="black",
        alpha=0.55,
        edgecolors="none",
        label="sim",
    )

    real_classes = (real_vehicle_type.astype(int) * 2) + real_direction.astype(int)
    for class_id, style in class_styles.items():
        mask = real_classes == class_id
        if np.any(mask):
            ax.scatter(
                real_xy[mask, 0],
                real_xy[mask, 1],
                s=50,
                c=style["color"],
                alpha=0.9,
                edgecolors="none",
                label=style["label"],
            )

    #ax.set_title("Real projection over simulation space")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _collect_domain_embeddings(
    *,
    logger: logging.Logger,
    args: argparse.Namespace,
    collect_embeddings,
    model,
    device,
    hop_length: int,
    representation: str,
    data_selection: str,
    data_csv: str,
    main_data_dir: str,
    domain_name: str,
) -> Dict[str, np.ndarray]:
    train_loader, val_loader, feature_min, feature_max = prepare_dataloader(
        logger=logger,
        hop_length=hop_length,
        batch_size=args.batch_size,
        data_csv_path=data_csv,
        main_data_dir=main_data_dir,
        n_fft=args.n_fft,
        data_num=data_selection,
        mel=args.mel,
        sampling_rate=args.sampling_rate,
        test_split=args.test_split,
        seed=args.seed,
        representation=representation,
    )
    logger.info(
        "%s data ready. feature_min=%.6f feature_max=%.6f",
        domain_name,
        feature_min,
        feature_max,
    )

    latent_matrix, speeds, vehicle_types, directions, locations = collect_embeddings(
        model,
        [train_loader, val_loader],
        device,
    )
    latent_matrix = normalize(latent_matrix, norm="l2")
    logger.info("%s embedding extraction done: %s", domain_name, latent_matrix.shape)

    return {
        "latent": latent_matrix,
        "speed": speeds,
        "vehicle_type": vehicle_types,
        "direction": directions,
        "location": locations,
    }


def main() -> None:
    global logger
    args = parse_args()

    if args.dimension < 2:
        raise ValueError("--dimension must be >= 2")

    backend = BACKENDS[args.loss_type]
    model_registry = backend.MODEL_REGISTRY
    resolve_representation = backend.resolve_representation
    resolve_hop_length = backend.resolve_hop_length
    collect_embeddings = backend.collect_embeddings

    weights_path = Path(args.encoder_weights).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"Encoder weights not found: {weights_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
        logger = setup_logger(output_dir)
    else:
        logger = output_settings()
        output_dir = Path(logger.output_dir)

    if args.optuna_params:
        optuna_path = Path(args.optuna_params).expanduser()
        if not optuna_path.exists():
            raise FileNotFoundError(f"Optuna params not found: {optuna_path}")
        params = load_optuna_params(optuna_path)
        apply_optuna_params(args, params, logger)

    model_key = args.model
    _description, model_cls = model_registry[model_key]
    representation = resolve_representation(model_key, args.representation)
    hop_length = resolve_hop_length(model_key, args.hop_length)

    logger.info("Using encoder weights: %s", weights_path)
    logger.info("Loss type: %s", args.loss_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    set_global_seed(args.seed)

    model = model_cls().to(device)
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    sim = _collect_domain_embeddings(
        logger=logger,
        args=args,
        collect_embeddings=collect_embeddings,
        model=model,
        device=device,
        hop_length=hop_length,
        representation=representation,
        data_selection=args.sim_data_selection,
        data_csv=args.sim_data_csv,
        main_data_dir=args.sim_main_data_dir,
        domain_name="sim",
    )

    real = _collect_domain_embeddings(
        logger=logger,
        args=args,
        collect_embeddings=collect_embeddings,
        model=model,
        device=device,
        hop_length=hop_length,
        representation=representation,
        data_selection=args.real_data_selection,
        data_csv=args.real_data_csv,
        main_data_dir=args.real_main_data_dir,
        domain_name="real",
    )

    logger.info("Fitting openTSNE on sim embeddings...")
    tsne = TSNE(
        n_components=args.dimension,
        perplexity=args.perplexity,
        metric="cosine",
        random_state=args.seed,
        n_jobs=args.n_jobs,
        initialization="pca",
        negative_gradient_method="fft",
    )
    sim_embedding = tsne.fit(sim["latent"])
    sim_proj = np.asarray(sim_embedding)

    logger.info("Transforming real embeddings into sim-trained space...")
    real_proj = np.asarray(sim_embedding.transform(real["latent"]))

    sim_df = _build_projection_frame(
        sim_proj,
        sim["speed"],
        sim["vehicle_type"],
        sim["direction"],
        sim["location"],
        "sim",
    )
    real_df = _build_projection_frame(
        real_proj,
        real["speed"],
        real["vehicle_type"],
        real["direction"],
        real["location"],
        "real",
    )
    combined_df = pd.concat([sim_df, real_df], ignore_index=True)

    sim_csv = output_dir / "sim_projection.csv"
    real_csv = output_dir / "real_projection.csv"
    combined_csv = output_dir / "sim_real_projection.csv"
    sim_df.to_csv(sim_csv, index=False)
    real_df.to_csv(real_csv, index=False)
    combined_df.to_csv(combined_csv, index=False)
    logger.info("Saved sim projection: %s", sim_csv)
    logger.info("Saved real projection: %s", real_csv)
    logger.info("Saved combined projection: %s", combined_csv)

    if args.dimension == 2:
        sim_plot = output_dir / "sim_distribution_4class.png"
        overlay_plot = output_dir / "sim_black_real_4class_overlay.png"
        _save_sim_fourclass_plot(
            sim_proj,
            sim["vehicle_type"],
            sim["direction"],
            sim_plot,
        )
        _save_sim_real_overlay_plot(
            sim_proj,
            real_proj,
            real["vehicle_type"],
            real["direction"],
            overlay_plot,
        )
        logger.info("Saved sim-class plot: %s", sim_plot)
        logger.info("Saved sim-real overlay plot: %s", overlay_plot)
        if args.show_plot:
            plt.show()
    else:
        logger.info("Skipped plotting because dimension=%d (only 2D plot is implemented).", args.dimension)

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
