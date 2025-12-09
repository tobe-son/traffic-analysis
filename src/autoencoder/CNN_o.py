"""
2次元CNNオートエンコーダでスペクトログラムを再構成し、潜在空間を可視化するスクリプト。
learn_tool 内の共通ユーティリティを利用して処理を簡素化する。
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE
import umap
from tqdm import tqdm

# src をモジュール検索パスへ追加（単体実行対応）
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from learn_tool.settings import output_settings, prepare_dataloader
from learn_tool.visualize import LatentSpaceVisualizer
from encoder.base_model import AutoEncoder_Original


# ハイパーパラメータ
ENCODER_MODEL = 'CNN_original'
DATA_SELECTION = 'loc1-6'
DATA_CSV_PATH = './data/processed/datasets/data_1-6.csv'
MAIN_DATA_DIR = './data/processed/datasets'
MEL = 'OFF'
SAMPLING_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
TEST_DATASET_PERCENTAGE = 0.2
BATCH_SIZE = 32
SEED = 42
LEARNING_RATE = 0.001
EPOCHS = 100
REPRESENTATION = 'spectrogram'  # CNN ではスペクトログラムを使用
SAVE_LATENT_SPACE = 'n'
DIMENSION_LATENT_SPACE = 2
VISUALIZATION = 't-SNE'  # t-SNE, PCA, LDA, UMAP, MDS


def log_hyperparameters(logger, params) -> None:
    logger.info('Hyperparameters:')
    for key, value in params.items():
        logger.info('%s: %s', key, value)


def train_one_epoch(model, loader, criterion, optimizer, device, label):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc=label):
        data = batch[0].to(device)
        optimizer.zero_grad()
        reconstruction = model(data)
        loss = criterion(reconstruction, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            data = batch[0].to(device)
            reconstruction = model(data)
            loss = criterion(reconstruction, data)
            running_loss += loss.item()
    return running_loss / max(len(loader), 1)


def extract_latent_spaces(model, loaders, device):
    """潜在空間とメタ情報をまとめて取得する。"""

    model.eval()
    latent_chunks = []
    speeds = []
    vehicle_types = []
    directions = []
    locations = []

    with torch.no_grad():
        for loader in loaders:
            for data, speed, vtype, direc, loc in loader:
                inputs = data.to(device)
                latent = model.enc(inputs).view(inputs.size(0), -1)
                latent_chunks.append(latent.cpu())
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


def reduce_latent_space(latent_matrix, full_classes, method, components, seed, logger):
    """選択された手法で潜在空間の次元を削減する。"""

    logger.info('%s による次元削減を開始します。', method)

    if method == 't-SNE':
        reducer = TSNE(n_components=components, random_state=seed)
        return reducer.fit_transform(latent_matrix)

    if method == 'PCA':
        reducer = PCA(n_components=components, random_state=seed)
        return reducer.fit_transform(latent_matrix)

    if method == 'LDA':
        reducer = LinearDiscriminantAnalysis(n_components=min(components, len(np.unique(full_classes)) - 1))
        return reducer.fit_transform(latent_matrix, full_classes)

    if method == 'UMAP':
        reducer = umap.UMAP(n_components=components, random_state=seed)
        return reducer.fit_transform(latent_matrix)

    if method == 'MDS':
        reducer = MDS(n_components=components, random_state=seed)
        return reducer.fit_transform(latent_matrix)

    raise ValueError(f"Unsupported visualization method: {method}")


def save_latent_results(latent_matrix, reduced, output_dir, components, logger) -> None:
    latent_spaces_file = os.path.join(output_dir, 'latent_spaces.csv')
    pd.DataFrame(latent_matrix).to_csv(latent_spaces_file, index=False)
    logger.info('latent_spaces を %s に保存しました。', latent_spaces_file)

    if components == 2:
        columns = ['Dimension_1', 'Dimension_2']
        filename = 'latent_2d.csv'
    elif components == 3:
        columns = ['Dimension_1', 'Dimension_2', 'Dimension_3']
        filename = 'latent_3d.csv'
    else:
        columns = [f'Dimension_{idx+1}' for idx in range(components)]
        filename = 'latent_reduced.csv'

    reduced_file = os.path.join(output_dir, filename)
    pd.DataFrame(reduced, columns=columns).to_csv(reduced_file, index=False)
    logger.info('reduced latent を %s に保存しました。', reduced_file)


def save_metadata(speeds, vehicle_types, directions, locations, output_dir, logger) -> None:
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    metadata_df = pd.DataFrame({
        'speed': speeds,
        'vehicle_type': vehicle_types,
        'direction': directions,
        'location': locations,
    })
    metadata_df.to_csv(metadata_path, index=False)
    logger.info('メタデータを %s に保存しました。', metadata_path)


def main() -> None:
    logger = output_settings()

    hyperparameters = {
        'ENCODER_MODEL': ENCODER_MODEL,
        'DATA_SELECTION': DATA_SELECTION,
        'DATA_CSV_PATH': DATA_CSV_PATH,
        'MEL': MEL,
        'SAMPLING_RATE': SAMPLING_RATE,
        'N_FFT': N_FFT,
        'HOP_LENGTH': HOP_LENGTH,
        'TEST_DATASET_PERCENTAGE': TEST_DATASET_PERCENTAGE,
        'BATCH_SIZE': BATCH_SIZE,
        'SEED': SEED,
        'LEARNING_RATE': LEARNING_RATE,
        'EPOCHS': EPOCHS,
        'REPRESENTATION': REPRESENTATION,
        'VISUALIZATION': VISUALIZATION,
        'DIMENSION_LATENT_SPACE': DIMENSION_LATENT_SPACE,
    }
    log_hyperparameters(logger, hyperparameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: %s', device)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # スペクトログラムを生成し DataLoader を構築
    train_loader, val_loader, feature_min, feature_max = prepare_dataloader(
        logger=logger,
        hop_length=HOP_LENGTH,
        batch_size=BATCH_SIZE,
        data_csv_path=DATA_CSV_PATH,
        main_data_dir=MAIN_DATA_DIR,
        n_fft=N_FFT,
        data_num=DATA_SELECTION,
        mel=MEL,
        sampling_rate=SAMPLING_RATE,
        test_split=TEST_DATASET_PERCENTAGE,
        seed=SEED,
        representation=REPRESENTATION,
    )
    logger.info('データの準備が完了しました。feature_min=%.6f feature_max=%.6f', feature_min, feature_max)

    model = AutoEncoder_Original().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            f'Train Epoch {epoch + 1}/{EPOCHS}',
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        logger.info('Epoch [%d/%d] train_loss=%.4f val_loss=%.4f', epoch + 1, EPOCHS, train_loss, val_loss)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), os.path.join(logger.output_dir, f'best_model_{ENCODER_MODEL}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(logger.output_dir, f'optimizer_{ENCODER_MODEL}.pth'))
            logger.info('Best model updated with validation loss %.4f', best_valid_loss)

    # 潜在空間を取得（スペクトログラム共通ロジック）
    latent_matrix, speeds, vehicle_types, directions, locations = extract_latent_spaces(
        model,
        [train_loader, val_loader],
        device,
    )
    logger.info('潜在空間の抽出が完了しました。')

    full_classes = (vehicle_types * 2) + directions
    reduced_latent = reduce_latent_space(
        latent_matrix,
        full_classes,
        VISUALIZATION,
        DIMENSION_LATENT_SPACE,
        SEED,
        logger,
    )

    if SAVE_LATENT_SPACE.lower() == 'y':
        save_latent_results(latent_matrix, reduced_latent, logger.output_dir, DIMENSION_LATENT_SPACE, logger)

    save_metadata(speeds, vehicle_types, directions, locations, logger.output_dir, logger)

    visualizer = LatentSpaceVisualizer(
        dimension_latent_space=DIMENSION_LATENT_SPACE,
        latent_data=reduced_latent,
        speeds=np.asarray(speeds),
        vehicle_types=np.asarray(vehicle_types),
        directions=np.asarray(directions),
        locations=np.asarray(locations),
        output_dir=logger.output_dir,
        visualization=VISUALIZATION,
    )
    visualizer.visualize_all(show_plot=False)


if __name__ == '__main__':
    main()
