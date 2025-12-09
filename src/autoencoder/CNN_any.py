"""
任意のCNNオートエンコーダを選択して学習・可視化を行う汎用スクリプト。
learn_tool 内の共通ユーティリティと encoder.* のモデル実装を再利用する。
python src/autoencoder/CNN_any.py --model small みたいに指定する。
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

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

# src ディレクトリをモジュール探索パスに追加（単体実行時の import 対応）
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
	sys.path.append(str(SRC_ROOT))

from learn_tool.settings import output_settings, prepare_dataloader
from learn_tool.visualize import LatentSpaceVisualizer
from encoder.base_model import AutoEncoder_Small, AutoEncoder_Original, AutoEncoder_Wave1D
from encoder.new_model import AutoEncoder_VGG11, AutoEncoder_ResNet


MODEL_REGISTRY: Dict[str, Tuple[str, nn.Module]] = {
	"wave1d": ("1D Conv AutoEncoder (waveform)", AutoEncoder_Wave1D),
	"small": ("Small CNN AutoEncoder", AutoEncoder_Small),
	"original": ("Original CNN AutoEncoder", AutoEncoder_Original),
	"vgg11": ("VGG11-based AutoEncoder", AutoEncoder_VGG11),
	"resnet": ("Residual CNN AutoEncoder", AutoEncoder_ResNet),
}

DEFAULT_REPRESENTATION = {
	"wave1d": "waveform",
	"small": "spectrogram",
	"original": "spectrogram",
	"vgg11": "spectrogram",
	"resnet": "spectrogram",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train and visualise latent space of selectable CNN autoencoders.")
	parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="small", help="モデル種別を選択")
	parser.add_argument("--data-selection", default="loc1-6", help="使用するデータ識別子 (例: loc1, loc1-6)")
	parser.add_argument("--data-csv", default="./data/processed/datasets/data_1-6.csv", help="メタデータ CSV のパス")
	parser.add_argument("--main-data-dir", default="./data/processed/datasets", help="音声データのベースディレクトリ")
	parser.add_argument("--mel", choices=["ON", "OFF"], default="OFF", help="メルスペクトログラムを使用するか")
	parser.add_argument("--sampling-rate", type=int, default=16000)
	parser.add_argument("--n-fft", type=int, default=1024)
	parser.add_argument("--hop-length", type=int, default=512)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--representation", choices=["waveform", "spectrogram"], default=None, help="入力表現の指定。未指定時はモデル既定値を使用")
	parser.add_argument("--visualization", choices=["t-SNE", "PCA", "LDA", "UMAP", "MDS"], default="t-SNE")
	parser.add_argument("--dimension", type=int, default=2, help="潜在空間の可視化次元 (2 or 3 を推奨)")
	parser.add_argument("--save-latent-space", action="store_true", help="潜在空間（CSV）を保存する")
	parser.add_argument("--test-split", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def log_hyperparameters(logger, params) -> None:
	logger.info("Hyperparameters:")
	for key, value in params.items():
		logger.info("%s: %s", key, value)


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
	"""潜在空間を取得し、付随するメタ情報をまとめる。"""

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


def resolve_representation(model_key: str, override: str | None) -> str:
	if override is not None:
		return override
	return DEFAULT_REPRESENTATION.get(model_key, "spectrogram")


def main() -> None:
	args = parse_args()

	model_name = args.model
	if model_name not in MODEL_REGISTRY:
		raise ValueError(f"Unknown model key: {model_name}")

	description, model_cls = MODEL_REGISTRY[model_name]
	representation = resolve_representation(model_name, args.representation)

	logger = output_settings()

	hyperparameters = {
		"MODEL": model_name,
		"MODEL_DESCRIPTION": description,
		"DATA_SELECTION": args.data_selection,
		"DATA_CSV_PATH": args.data_csv,
		"MAIN_DATA_DIR": args.main_data_dir,
		"MEL": args.mel,
		"SAMPLING_RATE": args.sampling_rate,
		"N_FFT": args.n_fft,
		"HOP_LENGTH": args.hop_length,
		"TEST_DATASET_PERCENTAGE": args.test_split,
		"BATCH_SIZE": args.batch_size,
		"EPOCHS": args.epochs,
		"LEARNING_RATE": args.lr,
		"REPRESENTATION": representation,
		"VISUALIZATION": args.visualization,
		"DIMENSION_LATENT_SPACE": args.dimension,
		"SAVE_LATENT_SPACE": args.save_latent_space,
		"SEED": args.seed,
	}
	log_hyperparameters(logger, hyperparameters)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info("Using device: %s", device)

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	if device.type == "cuda":
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	# NOTE: モデルに応じて波形/スペクトログラムを生成
	train_loader, val_loader, feature_min, feature_max = prepare_dataloader(
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
		representation=representation,
	)
	logger.info(
		"データの準備が完了しました。feature_min=%.6f feature_max=%.6f",
		feature_min,
		feature_max,
	)

	model = model_cls().to(device)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	best_valid_loss = float("inf")

	for epoch in range(args.epochs):
		train_loss = train_one_epoch(
			model,
			train_loader,
			criterion,
			optimizer,
			device,
			f"Train Epoch {epoch + 1}/{args.epochs}",
		)
		val_loss = evaluate(model, val_loader, criterion, device)
		logger.info(
			"Epoch [%d/%d] train_loss=%.4f val_loss=%.4f",
			epoch + 1,
			args.epochs,
			train_loss,
			val_loss,
		)

		if val_loss < best_valid_loss:
			best_valid_loss = val_loss
			torch.save(model.state_dict(), os.path.join(logger.output_dir, f"best_model_{model_name}.pth"))
			torch.save(optimizer.state_dict(), os.path.join(logger.output_dir, f"optimizer_{model_name}.pth"))
			logger.info("Best model updated with validation loss %.4f", best_valid_loss)

	latent_matrix, speeds, vehicle_types, directions, locations = extract_latent_spaces(
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
		visualizer.visualize_all(show_plot=False)
	else:
		logger.warning("Latent visualization is skipped because dimension is not 2 or 3.")


if __name__ == "__main__":
	main()
