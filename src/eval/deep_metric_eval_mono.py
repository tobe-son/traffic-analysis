"""Evaluation script for mono data: vehicle-type-only metrics.

Outputs:
- Silhouette Coefficient / Davies-Bouldin Index
- Recall@k
- NMI (Normalized Mutual Information) via KMeans
- k-NN classification accuracy
- t-SNE and UMAP visualizations (categorized by location, vehicle, no_speed, speed_only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import normalize

if os.environ.get("MPLBACKEND") is None:
    import matplotlib

    matplotlib.use("Agg")

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from learn_tool.settings import output_settings, prepare_dataloader
from learn_tool.visualize import LatentSpaceVisualizer
from metric import LabelClustering as circle_backend
from metric import LabelClustering_arcface as arcface_backend
from metric.utils import set_global_seed
from loss.arcface import ArcFaceLayer

logger = logging.getLogger("deep_metric_eval_mono")

BACKENDS = {
    "circle": circle_backend,
    "arcface": arcface_backend,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep metric-learning evaluation for mono data (vehicle-type-only)"
    )

    parser.add_argument("--loss-type", choices=BACKENDS.keys(), default="circle")
    parser.add_argument("--model", choices=circle_backend.MODEL_REGISTRY.keys(), default="vgg11")
    parser.add_argument("--data-selection", default="loc1-6")
    parser.add_argument("--data-csv", default="./data/processed/datasets/data_1-6.csv")
    parser.add_argument("--main-data-dir", default="./data/processed/datasets")
    parser.add_argument("--mel", choices=["ON", "OFF"], default="OFF")

    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--representation", choices=["waveform", "spectrogram"], default=None)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--save-latent-space", action="store_true")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-weights", required=True, help="Path to encoder .pth")
    parser.add_argument("--output-dir", default=None, help="Output directory for metrics/plots")
    parser.add_argument(
        "--optuna-params",
        default=None,
        help="Path to optuna_best_params.json to auto-fill batch_size, hop_length, n_fft, mel",
    )
    parser.add_argument("--metric-fc-weights", default=None, help="Path to ArcFace metric_fc weights (.pth)")
    parser.add_argument("--arcface-s", type=float, default=64.0, help="ArcFace scaling factor s")
    parser.add_argument("--arcface-m", type=float, default=0.5, help="ArcFace margin m")

    parser.add_argument("--recall-k", default="1,2,4,8,16", help="Comma-separated recall@k list")
    parser.add_argument("--knn-k", default="1,2,3", help="Comma-separated k-NN k list")
    parser.add_argument("--head-topk", default="1,2,3", help="Comma-separated ArcFace head top-k list")
    parser.add_argument(
        "--speed-bins",
        default="0,40,60,80,100",
        help="Comma-separated speed bin edges (last bin is +inf)",
    )

    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluation.log"

    logger = logging.getLogger("deep_metric_eval_mono")
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
        "arcface_s": "arcface_s",
        "arcface_m": "arcface_m",
    }
    for key, arg_name in mapping.items():
        if key in params:
            setattr(args, arg_name, params[key])
            logger.info("Applied optuna param %s=%s", key, params[key])


def parse_int_list(value: str) -> List[int]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in items]


def parse_float_list(value: str) -> List[float]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [float(v) for v in items]


def bin_speeds(speeds: np.ndarray, bins: List[float]) -> np.ndarray:
    bins_full = list(bins) + [float("inf")]
    labels = np.digitize(speeds, bins_full, right=True)
    labels = labels - 1
    labels = labels.astype(int)
    labels[np.isnan(speeds)] = -1
    return labels


def _valid_label_mask(labels: np.ndarray) -> np.ndarray:
    return labels >= 0


def compute_recall_at_k(embeddings: np.ndarray, labels: np.ndarray, k_list: Iterable[int]) -> Dict[str, float]:
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return {f"recall@{k}": float("nan") for k in k_list}

    max_k = min(max(k_list), n_samples - 1)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine")
    nn.fit(embeddings)
    _distances, indices = nn.kneighbors(embeddings, return_distance=True)

    results: Dict[str, float] = {}
    for k in k_list:
        k_eff = min(k, n_samples - 1)
        hits = []
        for i in range(n_samples):
            neighbors = indices[i, 1 : k_eff + 1]
            hits.append(np.any(labels[neighbors] == labels[i]))
        results[f"recall@{k}"] = float(np.mean(hits)) if hits else float("nan")
    return results


def compute_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: Iterable[int],
    seed: int,
) -> Dict[str, float]:
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return {f"knn_acc@{k}": float("nan") for k in k_list}

    n_classes = len(np.unique(labels))
    stratify = labels if n_classes > 1 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=0.3,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=0.3,
            random_state=seed,
            stratify=None,
        )

    results: Dict[str, float] = {}
    for k in k_list:
        k_eff = min(k, len(y_train))
        if k_eff < 1:
            results[f"knn_acc@{k}"] = float("nan")
            continue
        clf = KNeighborsClassifier(n_neighbors=k_eff, metric="cosine", algorithm="brute")
        clf.fit(x_train, y_train)
        acc = clf.score(x_test, y_test)
        results[f"knn_acc@{k}"] = float(acc)

    return results


def compute_knn_topk_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: Iterable[int],
    seed: int,
) -> Dict[str, float]:
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return {f"knn_top{k}": float("nan") for k in k_list}

    n_classes = len(np.unique(labels))
    if n_classes < 1:
        return {f"knn_top{k}": float("nan") for k in k_list}

    stratify = labels if n_classes > 1 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=0.3,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=0.3,
            random_state=seed,
            stratify=None,
        )

    max_k = min(max(k_list), n_classes, len(y_train))
    if max_k < 1:
        return {f"knn_top{k}": float("nan") for k in k_list}

    clf = KNeighborsClassifier(n_neighbors=max_k, metric="cosine", algorithm="brute")
    clf.fit(x_train, y_train)

    proba = clf.predict_proba(x_test)
    classes = clf.classes_
    order = np.argsort(-proba, axis=1)
    y_test = np.asarray(y_test)

    results: Dict[str, float] = {}
    for k in k_list:
        k_eff = min(k, order.shape[1])
        if k_eff < 1:
            results[f"knn_top{k}"] = float("nan")
            continue
        topk_classes = classes[order[:, :k_eff]]
        hits = (topk_classes == y_test[:, None]).any(axis=1)
        results[f"knn_top{k}"] = float(np.mean(hits))

    return results


def compute_nmi(embeddings: np.ndarray, labels: np.ndarray, seed: int) -> float:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return float("nan")
    kmeans = KMeans(n_clusters=unique_labels.size, n_init="auto", random_state=seed)
    clusters = kmeans.fit_predict(embeddings)
    return float(normalized_mutual_info_score(labels, clusters))


def compute_cluster_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return {"silhouette": float("nan"), "dbi": float("nan")}

    try:
        silhouette = float(silhouette_score(embeddings, labels, metric="cosine"))
    except Exception:
        silhouette = float("nan")

    try:
        dbi = float(davies_bouldin_score(embeddings, labels))
    except Exception:
        dbi = float("nan")

    return {"silhouette": silhouette, "dbi": dbi}


def evaluate_label_set(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list_recall: List[int],
    k_list_knn: List[int],
    seed: int,
) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "name": name,
        "n_samples": int(embeddings.shape[0]),
        "n_classes": int(np.unique(labels).size),
    }

    metrics.update(compute_cluster_metrics(embeddings, labels))
    metrics["nmi"] = compute_nmi(embeddings, labels, seed)
    metrics["recall"] = compute_recall_at_k(embeddings, labels, k_list_recall)
    metrics["knn"] = compute_knn_accuracy(embeddings, labels, k_list_knn, seed)
    metrics["knn_topk"] = compute_knn_topk_accuracy(embeddings, labels, k_list_knn, seed)

    return metrics


def write_metrics(output_dir: Path, results: Dict[str, Dict[str, object]]) -> None:
    json_path = output_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rows = []
    for name, metrics in results.items():
        base = {
            "label_set": name,
            "n_samples": metrics.get("n_samples"),
            "n_classes": metrics.get("n_classes"),
            "silhouette": metrics.get("silhouette"),
            "dbi": metrics.get("dbi"),
            "nmi": metrics.get("nmi"),
        }
        recall = metrics.get("recall", {}) or {}
        knn = metrics.get("knn", {}) or {}
        knn_topk = metrics.get("knn_topk", {}) or {}
        head = metrics.get("head", {}) or {}
        for key, value in recall.items():
            base[key] = value
        for key, value in knn.items():
            base[key] = value
        for key, value in knn_topk.items():
            base[key] = value
        for key, value in head.items():
            base[key] = value
        rows.append(base)

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "metrics.csv", index=False)


def save_embeddings(output_dir: Path, embeddings: np.ndarray) -> None:
    import pandas as pd

    path = output_dir / "embeddings.csv"
    pd.DataFrame(embeddings).to_csv(path, index=False)


def _compute_topk_metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topk_list: List[int],
    correct_topk: Dict[int, int],
    correct_top1: int,
) -> int:
    preds = logits.argmax(dim=1)
    correct_top1 += (preds == labels).sum().item()

    max_k = max(topk_list)
    topk = torch.topk(logits, k=max_k, dim=1).indices
    for k in topk_list:
        hits = (topk[:, :k] == labels.unsqueeze(1)).any(dim=1)
        correct_topk[k] += hits.sum().item()

    return correct_top1


def _nan_head_metrics(topk_list: Iterable[int], suffix: str) -> Dict[str, float]:
    results: Dict[str, float] = {f"head_acc_{suffix}": float("nan")}
    if suffix == "cos":
        results["head_acc"] = float("nan")
    for k in topk_list:
        if suffix == "cos":
            results[f"head_top{k}"] = float("nan")
        results[f"head_top{k}_{suffix}"] = float("nan")
    return results


def compute_arcface_head_metrics_cos(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric_fc: ArcFaceLayer,
    device: torch.device,
    topk_list: Iterable[int],
    batch_size: int = 512,
) -> Dict[str, float]:
    metric_fc.eval()
    total = labels.shape[0]
    if total == 0:
        return {"head_acc_cos": float("nan")}

    topk_list = sorted(set(int(k) for k in topk_list))
    correct_top1 = 0
    correct_topk = {k: 0 for k in topk_list}

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_emb = torch.from_numpy(embeddings[start:end]).to(device)
            batch_labels = torch.from_numpy(labels[start:end]).to(device)
            logits = F.linear(F.normalize(batch_emb), F.normalize(metric_fc.weight))
            correct_top1 = _compute_topk_metrics_from_logits(
                logits,
                batch_labels,
                topk_list,
                correct_topk,
                correct_top1,
            )

    results: Dict[str, float] = {
        "head_acc": float(correct_top1 / total),
        "head_acc_cos": float(correct_top1 / total),
    }
    for k in topk_list:
        results[f"head_top{k}"] = float(correct_topk[k] / total)
        results[f"head_top{k}_cos"] = float(correct_topk[k] / total)
    return results


def compute_arcface_head_metrics_arcface(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric_fc: ArcFaceLayer,
    device: torch.device,
    topk_list: Iterable[int],
    batch_size: int = 512,
) -> Dict[str, float]:
    metric_fc.eval()
    total = labels.shape[0]
    if total == 0:
        return {"head_acc_arcface": float("nan")}

    topk_list = sorted(set(int(k) for k in topk_list))
    correct_top1 = 0
    correct_topk = {k: 0 for k in topk_list}

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_emb = torch.from_numpy(embeddings[start:end]).to(device)
            batch_labels = torch.from_numpy(labels[start:end]).to(device)
            logits = metric_fc(batch_emb, batch_labels)
            correct_top1 = _compute_topk_metrics_from_logits(
                logits,
                batch_labels,
                topk_list,
                correct_topk,
                correct_top1,
            )

    results: Dict[str, float] = {
        "head_acc_arcface": float(correct_top1 / total),
    }
    for k in topk_list:
        results[f"head_top{k}_arcface"] = float(correct_topk[k] / total)
    return results


def compute_arcface_head_metrics_mono_cos(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric_fc: ArcFaceLayer,
    device: torch.device,
    topk_list: Iterable[int],
    num_vehicle_classes: int,
    batch_size: int = 512,
) -> Dict[str, float]:
    metric_fc.eval()
    total = labels.shape[0]
    if total == 0 or num_vehicle_classes < 1:
        return {"head_acc_cos": float("nan")}

    topk_list = sorted(set(int(k) for k in topk_list))
    max_k = min(max(topk_list), num_vehicle_classes)

    correct_top1 = 0
    correct_topk = {k: 0 for k in topk_list}
    group_size = int(metric_fc.weight.shape[0] // num_vehicle_classes)

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_emb = torch.from_numpy(embeddings[start:end]).to(device)
            batch_labels = torch.from_numpy(labels[start:end]).to(device)

            logits_full = F.linear(F.normalize(batch_emb), F.normalize(metric_fc.weight))
            logits_grouped = logits_full.view(-1, num_vehicle_classes, group_size)
            logits = torch.logsumexp(logits_grouped, dim=2)

            preds = logits.argmax(dim=1)
            correct_top1 += (preds == batch_labels).sum().item()

            topk = torch.topk(logits, k=max_k, dim=1).indices
            for k in topk_list:
                hits = (topk[:, :k] == batch_labels.unsqueeze(1)).any(dim=1)
                correct_topk[k] += hits.sum().item()

    results: Dict[str, float] = {
        "head_acc": float(correct_top1 / total),
        "head_acc_cos": float(correct_top1 / total),
    }
    for k in topk_list:
        results[f"head_top{k}"] = float(correct_topk[k] / total)
        results[f"head_top{k}_cos"] = float(correct_topk[k] / total)
    return results


def visualize_embeddings(
    output_dir: Path,
    latent_matrix: np.ndarray,
    speeds: np.ndarray,
    vehicle_types: np.ndarray,
    locations: np.ndarray,
    seed: int,
    dimension: int,
    max_points: int,
    show_plot: bool,
    reduce_latent_space,
) -> None:
    base_embeddings = latent_matrix
    base_speeds = speeds
    base_vehicle_types = vehicle_types
    base_locations = locations

    if max_points and max_points > 0 and latent_matrix.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(latent_matrix.shape[0], size=max_points, replace=False)
        keep.sort()
        base_embeddings = latent_matrix[keep]
        base_speeds = speeds[keep]
        base_vehicle_types = vehicle_types[keep]
        base_locations = locations[keep]

    for method in ("t-SNE", "UMAP"):
        reduced = reduce_latent_space(
            base_embeddings,
            base_vehicle_types,
            method,
            dimension,
            seed,
            logger,
        )

        if dimension in (2, 3):
            visualizer = LatentSpaceVisualizer(
                dimension_latent_space=dimension,
                latent_data=reduced,
                speeds=np.asarray(base_speeds),
                vehicle_types=np.asarray(base_vehicle_types),
                directions=None,
                locations=np.asarray(base_locations),
                output_dir=str(output_dir),
                visualization=method,
            )
            visualizer.visualize_all(show_plot=show_plot)


def main() -> None:
    global logger
    args = parse_args()

    backend = BACKENDS[args.loss_type]
    model_registry = backend.MODEL_REGISTRY
    resolve_representation = backend.resolve_representation
    resolve_hop_length = backend.resolve_hop_length
    collect_embeddings = backend.collect_embeddings
    reduce_latent_space = backend.reduce_latent_space
    save_metadata = backend.save_metadata

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
    logger.info("Data ready. feature_min=%.6f feature_max=%.6f", feature_min, feature_max)

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
    logger.info("Embedding extraction done: %s", latent_matrix.shape)
    if latent_matrix.ndim >= 2:
        logger.info("Embedding dimension: %d", latent_matrix.shape[1])

    latent_matrix = normalize(latent_matrix, norm="l2")
    logger.info("Applied L2 normalization to embeddings.")

    speed_bins = parse_float_list(args.speed_bins)
    speed_labels = bin_speeds(speeds, speed_bins)

    label_sets = {
        "vehicle": vehicle_types,
        "location": locations,
        "speed_bin": speed_labels,
    }

    recall_k = parse_int_list(args.recall_k)
    knn_k = parse_int_list(args.knn_k)
    head_topk = parse_int_list(args.head_topk)

    results: Dict[str, Dict[str, object]] = {}
    for name, labels in label_sets.items():
        mask = _valid_label_mask(labels)
        labels_valid = labels[mask]
        embeddings_valid = latent_matrix[mask]

        if embeddings_valid.shape[0] == 0:
            logger.warning("Skipping %s (no valid labels)", name)
            continue

        results[name] = evaluate_label_set(
            name=name,
            embeddings=embeddings_valid,
            labels=labels_valid,
            k_list_recall=recall_k,
            k_list_knn=knn_k,
            seed=args.seed,
        )

    if args.metric_fc_weights:
        metric_fc_path = Path(args.metric_fc_weights).expanduser()
        if not metric_fc_path.exists():
            raise FileNotFoundError(f"ArcFace metric_fc weights not found: {metric_fc_path}")

        num_classes = int(np.max(vehicle_types)) + 1 if vehicle_types.size else 0
        try:
            state = torch.load(metric_fc_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(metric_fc_path, map_location=device)

        weight_shape = state.get("weight", None)
        if isinstance(weight_shape, torch.Tensor):
            num_classes = int(weight_shape.shape[0])

        if num_classes > 0:
            metric_fc = ArcFaceLayer(
                in_features=int(latent_matrix.shape[1]),
                num_classes=num_classes,
                s=args.arcface_s,
                m=args.arcface_m,
            ).to(device)
            metric_fc.load_state_dict(state)
            metric_fc.eval()

            mask = _valid_label_mask(vehicle_types)
            num_vehicle_classes = int(np.max(vehicle_types[mask])) + 1 if mask.any() else 0
            if num_vehicle_classes > 0 and num_classes != num_vehicle_classes:
                if num_classes % num_vehicle_classes == 0:
                    logger.info(
                        "Aggregating ArcFace head logits from %d classes into %d vehicle classes",
                        num_classes,
                        num_vehicle_classes,
                    )
                    head_metrics_cos = compute_arcface_head_metrics_mono_cos(
                        embeddings=latent_matrix[mask],
                        labels=vehicle_types[mask].astype(np.int64),
                        metric_fc=metric_fc,
                        device=device,
                        topk_list=head_topk,
                        num_vehicle_classes=num_vehicle_classes,
                        batch_size=max(128, args.batch_size),
                    )
                    head_metrics_arcface = _nan_head_metrics(head_topk, "arcface")
                else:
                    logger.warning(
                        "ArcFace head evaluation skipped: num_classes=%d is not divisible by vehicle classes=%d",
                        num_classes,
                        num_vehicle_classes,
                    )
                    head_metrics_cos = _nan_head_metrics(head_topk, "cos")
                    head_metrics_arcface = _nan_head_metrics(head_topk, "arcface")
            else:
                head_metrics_cos = compute_arcface_head_metrics_cos(
                    embeddings=latent_matrix[mask],
                    labels=vehicle_types[mask].astype(np.int64),
                    metric_fc=metric_fc,
                    device=device,
                    topk_list=head_topk,
                    batch_size=max(128, args.batch_size),
                )
                head_metrics_arcface = compute_arcface_head_metrics_arcface(
                    embeddings=latent_matrix[mask],
                    labels=vehicle_types[mask].astype(np.int64),
                    metric_fc=metric_fc,
                    device=device,
                    topk_list=head_topk,
                    batch_size=max(128, args.batch_size),
                )

            head_metrics = {**head_metrics_cos, **head_metrics_arcface}

            if "vehicle" in results:
                results["vehicle"].setdefault("head", {}).update(head_metrics)
            else:
                results["vehicle"] = {
                    "name": "vehicle",
                    "n_samples": int(mask.sum()),
                    "n_classes": int(num_classes),
                    "head": head_metrics,
                }
        else:
            logger.warning("ArcFace head evaluation skipped: no valid classes")
    elif args.loss_type == "arcface":
        logger.warning("ArcFace head evaluation skipped: --metric-fc-weights not provided")

    write_metrics(output_dir, results)

    if args.save_latent_space:
        save_embeddings(output_dir, latent_matrix)
        logger.info("Saved embeddings to %s", output_dir / "embeddings.csv")

    save_metadata(speeds, vehicle_types, directions, locations, str(output_dir), logger)

    visualize_embeddings(
        output_dir=output_dir,
        latent_matrix=latent_matrix,
        speeds=speeds,
        vehicle_types=vehicle_types,
        locations=locations,
        seed=args.seed,
        dimension=args.dimension,
        max_points=args.max_points,
        show_plot=args.show_plot,
        reduce_latent_space=reduce_latent_space,
    )

    logger.info("Mono evaluation complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
