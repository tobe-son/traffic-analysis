"""ArcFace Loss を用いたエンコーダによるメトリックラーニングスクリプト。"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Type

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE
import umap  # type: ignore
from tqdm import tqdm

from loss.arcface import ArcFaceLayer

from learn_tool.settings import output_settings, prepare_dataloader
from learn_tool.visualize import LatentSpaceVisualizer
from encoder.base_model import Encoder_Original, Encoder_Small, Encoder_Wave1D
from encoder.new_model import Encoder_ResNet, Encoder_ResNet18, Encoder_ResNet50, Encoder_VGG11
from loss.circle_loss import CircleLoss, convert_label_to_similarity
from metric.utils import compute_stats_by_label, extract_embedding, set_global_seed


ModelEntry = Tuple[str, Type[torch.nn.Module]]

MODEL_REGISTRY: Dict[str, ModelEntry] = {
    "wave1d": ("1D Conv encoder (waveform)", Encoder_Wave1D),
    "small": ("Small CNN encoder", Encoder_Small),
    "original": ("Original CNN encoder", Encoder_Original),
    "vgg11": ("VGG11-based encoder", Encoder_VGG11),
    "resnet": ("Residual CNN encoder (legacy)", Encoder_ResNet),
    "resnet18": ("ResNet-18 encoder", Encoder_ResNet18),
    "resnet50": ("ResNet-50 encoder", Encoder_ResNet50),
}

DEFAULT_REPRESENTATION = {
    "wave1d": "waveform",
    "small": "spectrogram",
    "original": "spectrogram",
    "vgg11": "spectrogram",
    "resnet": "spectrogram",
    "resnet18": "spectrogram",
    "resnet50": "spectrogram",
}

DEFAULT_HOP_LENGTH = {
    "wave1d": 512,
    "small": 512,
    "original": 160,
    "vgg11": 160,
    "resnet": 160,
    "resnet18": 160,
    "resnet50": 160,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Circle Loss を用いて任意のエンコーダでメトリックラーニングを行うスクリプト"
    )
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="vgg11", help="使用するエンコーダ")
    parser.add_argument("--data-selection", default="loc1-6", help="使用するデータ識別子 (例: loc1, loc1-6)")
    parser.add_argument("--data-csv", default="./data/processed/datasets/data_1-6.csv", help="メタデータ CSV のパス")
    parser.add_argument("--main-data-dir", default="./data/processed/datasets", help="音声データのベースディレクトリ")
    parser.add_argument("--mel", choices=["ON", "OFF"], default="OFF", help="メルスペクトログラムを使用するか")
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=None, help="STFT の hop length。未指定時はモデルに応じた推奨値を使用")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--encoder-weights",
        default=None,
        help="Load encoder weights (.pth) before training / for --no-train mode.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip CircleLoss training and only extract embeddings + visualize.",
    )
    parser.add_argument("--margin", type=float, default=0.25, help="CircleLoss のマージン値")
    parser.add_argument("--gamma", type=float, default=80.0, help="CircleLoss のスケーリング係数")
    parser.add_argument("--representation", choices=["waveform", "spectrogram"], default=None, help="入力表現。未指定時はモデル推奨を使用")
    parser.add_argument("--visualization", choices=["t-SNE", "PCA", "LDA", "UMAP", "MDS"], default="t-SNE")
    parser.add_argument("--dimension", type=int, default=2, help="潜在空間の可視化次元")
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Downsample embeddings before visualization (0 = no limit). Useful for t-SNE speed.",
    )
    parser.add_argument("--save-latent-space", action="store_true", help="潜在空間を CSV として保存する")
    parser.add_argument("--show-plot", action="store_true", help="3D 可視化を表示する")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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


def circle_loss_for_batch(model, batch, device, criterion):
    data, _speed, vtype, direc, _loc = batch
    data = data.to(device)
    labels = (vtype.to(device) * 2) + direc.to(device)
    embeddings = extract_embedding(model, data)
    sp, sn = convert_label_to_similarity(embeddings, labels)
    if sp.numel() == 0 or sn.numel() == 0:
        return None
    return criterion(sp, sn)


def train_one_epoch(model, loader, optimizer, device, criterion, epoch_label: str) -> float:
    model.train()
    total_loss = 0.0
    usable_batches = 0
    for batch in tqdm(loader, desc=epoch_label):
        optimizer.zero_grad()
        loss = circle_loss_for_batch(model, batch, device, criterion)
        if loss is None:
            continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        usable_batches += 1

    if usable_batches == 0:
        raise RuntimeError("No valid batches produced a CircleLoss signal. Increase batch size or ensure label diversity.")

    return total_loss / usable_batches


def evaluate(model, loader, device, criterion) -> float:
    model.eval()
    total_loss = 0.0
    usable_batches = 0
    with torch.no_grad():
        for batch in loader:
            loss = circle_loss_for_batch(model, batch, device, criterion)
            if loss is None:
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
    
def get_embedding_dim(model, device, args):
    """ダミー入力を用いてモデルの出力次元数を自動取得する"""
    model.eval()
    # モデルごとの入力形状に合わせてダミーデータを作成
    # audio系モデルと仮定して (Batch, 1, Length) or (Batch, Length)
    if args.representation == "waveform":
         dummy_input = torch.randn(2, 1, 16000).to(device) # 仮の波形長
    else:
         dummy_input = torch.randn(2, 1, 64, 512).to(device) # 仮のスペクトログラム

    with torch.no_grad():
        try:
            output = extract_embedding(model, dummy_input)
            return output.shape[1]
        except Exception as e:
            # 失敗した場合はモデル定義から推測するか、エラーを出して手動設定を促す
            print(f"Warning: Could not infer embedding dim automatically: {e}")
            return 512 # Default fallback

# ArcFace用の学習ステップ関数
def train_one_epoch_arcface(model, metric_fc, loader, optimizer, device, criterion, epoch_label: str) -> tuple[float, float]:
    model.train()
    metric_fc.train() # ArcFaceレイヤーも学習モード
    
    total_loss = 0.0
    usable_batches = 0
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(loader, desc=epoch_label):
        data, _speed, vtype, direc, _loc = batch
        data = data.to(device)
        
        # ラベル生成: (vtype * 2) + direc => 0, 1, 2, 3 の4クラス想定
        labels = (vtype.to(device) * 2) + direc.to(device)
        labels = labels.long()

        optimizer.zero_grad()
        
        # 1. エンコーダで特徴抽出
        # extract_embedding はグローバル平均プーリング + L2 正規化を含む
        # ArcFace の in_features と整合する次元で統一する
        embeddings = extract_embedding(model, data)

        # 2. ArcFace Layer (Metric FC) で Logits 計算
        logits = metric_fc(embeddings, labels)
        
        # 3. Cross Entropy Loss
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.numel()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        usable_batches += 1

    avg_loss = total_loss / usable_batches
    accuracy = (total_correct / total_samples) if total_samples > 0 else float("nan")
    return avg_loss, accuracy

def evaluate_arcface(model, metric_fc, loader, device, criterion, num_classes: int) -> tuple[float, float, np.ndarray]:
    model.eval()
    metric_fc.eval()
    total_loss = 0.0
    usable_batches = 0
    total_correct = 0
    total_samples = 0
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for batch in loader:
            data, _speed, vtype, direc, _loc = batch
            data = data.to(device)
            labels = (vtype.to(device) * 2) + direc.to(device)
            labels = labels.long()

            embeddings = extract_embedding(model, data)

            logits = metric_fc(embeddings, labels)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.numel()
            for t, p in zip(labels.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy()):
                conf_mat[int(t), int(p)] += 1
            
            total_loss += loss.item()
            usable_batches += 1

    if usable_batches == 0:
        return float("nan"), float("nan"), conf_mat
    avg_loss = total_loss / usable_batches
    accuracy = (total_correct / total_samples) if total_samples > 0 else float("nan")
    return avg_loss, accuracy, conf_mat


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
        "ENCODER_WEIGHTS": args.encoder_weights or "(none)",
        "NO_TRAIN": args.no_train,
        "MARGIN": args.margin,
        "GAMMA": args.gamma,
        "REPRESENTATION": representation,
        "VISUALIZATION": args.visualization,
        "DIMENSION_LATENT_SPACE": args.dimension,
        "MAX_POINTS": args.max_points,
        "SAVE_LATENT_SPACE": args.save_latent_space,
        "SEED": args.seed,
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
    if args.encoder_weights:
        weights_path = Path(args.encoder_weights)
        try:
            state = torch.load(weights_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
        logger.info("Loaded encoder weights: %s", weights_path)

    if args.no_train and not args.encoder_weights:
        raise ValueError("--no-train requires --encoder-weights to be specified.")
    
    # 1. 埋め込み次元数の特定
    # Encoderの出力サイズがわからないとArcFaceの行列Wが作れないため
    embedding_dim = get_embedding_dim(model, device, args)
    logger.info(f"Detected embedding dimension: {embedding_dim}")

    # 2. クラス数の定義
    # 現在のロジック (vtype*2 + direc) では 4クラス です。
    # もしデータセットによって変わる場合は args から渡すか、メタデータから計算してください。
    NUM_CLASSES = 4 
    
    # 3. ArcFace Layer の初期化
    # s=64.0, m=0.50 は論文推奨値
    metric_fc = ArcFaceLayer(in_features=embedding_dim, num_classes=NUM_CLASSES, s=64.0, m=0.5).to(device)
    
    # 4. Loss関数の変更
    # ArcFaceの出力はLogitsなので、標準的なCrossEntropyLossを使います
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # 5. Optimizerの設定
    # 重要: ArcFaceのパラメータ(metric_fc.parameters)も最適化対象に含める
    # 論文ではSGD (momentum=0.9, weight_decay=5e-4) が推奨されていますが、
    # 収束が遅い場合は AdamW などでも可。ここでは論文準拠の設定例を示します。
    # optimizer = optim.SGD([
    #     {'params': model.parameters()},
    #     {'params': metric_fc.parameters()}
    # ], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # もしAdamを使いたい場合:
    optimizer = optim.Adam([
        {'params': model.parameters()}, 
        {'params': metric_fc.parameters()}
    ], lr=args.lr)

    if not args.no_train:
        best_val_loss = float("inf")
        best_epoch = -1

        for epoch in range(args.epochs):
            epoch_label = f"Train Epoch {epoch + 1}/{args.epochs}"
            train_loss, train_acc = train_one_epoch_arcface(
                model, metric_fc, train_loader, optimizer, device, criterion, epoch_label
            )
            val_loss, val_acc, conf_mat = evaluate_arcface(
                model, metric_fc, val_loader, device, criterion, NUM_CLASSES
            )

            logger.info(
                "Epoch [%d/%d] train_loss=%.4f train_acc=%.2f%% val_loss=%s val_acc=%s",
                epoch + 1,
                args.epochs,
                train_loss,
                train_acc * 100 if math.isfinite(train_acc) else float("nan"),
                f"{val_loss:.4f}" if math.isfinite(val_loss) else "nan",
                f"{val_acc * 100:.2f}%" if math.isfinite(val_acc) else "nan",
            )

            if np.any(conf_mat):
                logger.info("Validation confusion matrix:\n%s", conf_mat)

            if math.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(logger.output_dir, f"best_encoder_{model_key}.pth"))
                torch.save(metric_fc.state_dict(), os.path.join(logger.output_dir, f"best_metric_fc_{model_key}.pth"))
                torch.save(optimizer.state_dict(), os.path.join(logger.output_dir, f"optimizer_{model_key}.pth"))
                logger.info("Best model updated (epoch %d, val_loss=%.4f)", best_epoch, best_val_loss)

        torch.save(model.state_dict(), os.path.join(logger.output_dir, f"last_encoder_{model_key}.pth"))
        torch.save(metric_fc.state_dict(), os.path.join(logger.output_dir, f"last_metric_fc_{model_key}.pth"))
        logger.info("Last model checkpoint saved. Best epoch=%s", best_epoch if best_epoch > 0 else "N/A")
    else:
        logger.info("Skipping training (--no-train). Extracting embeddings only.")

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