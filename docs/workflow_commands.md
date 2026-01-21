# 学習・最適化・実データ推論/評価：コマンド集

このドキュメントは、本リポジトリでの **学習 → Optuna最適化 → 実データで推論/評価** を、コマンド単位で再現できるようにまとめたものです。

想定フロー（確認済み）:
- simulation データで学習（複数アーキテクチャ）
- Optuna を用いて学習を最適化・評価
- 実データで推論・評価（VS13: 速度推定 / IDMT: 車種分類）

> 注意: GUI無し環境（SSH/サーバ等）で `qt.qpa.plugin ... wayland` が出る場合があります。本リポジトリは Matplotlib を既定で `Agg` にする修正済みですが、念のため以下のように `MPLBACKEND=Agg` を付けてもOKです。

---

## 0. 環境

```bash
cd /home/tobeson/traffic-analysis
conda activate traf_ana
```

---

## 1. simulation データの前処理（6秒切り出し + CSV結合）

### 1-1) 6秒切り出し（12〜18秒）

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/sim_data_tool/cut.py
```

出力: `data/processed/datasets/loc{1..6}_cut/<car|cv>/<left|right>/*.flac`

### 1-2) メタCSV結合

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/sim_data_tool/combain.py
```

出力: `data/processed/datasets/data_1-6.csv`

---

## 2. simulation で学習（複数アーキテクチャ）

ここでは **(A) AutoEncoder** と **(B) CircleLoss（車種×方向）** の2系統の入口を用意しています。

### 2-A) AutoEncoder 学習（`CNN_any.py`）

例（モデルを切り替えて学習）:

```bash
# small
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/autoencoder/CNN_any.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --visualization PCA

# vgg11
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/autoencoder/CNN_any.py \
  --model vgg11 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --visualization PCA

# resnet（legacy）
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/autoencoder/CNN_any.py \
  --model resnet \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --visualization PCA
```

成果物: `outputs/<日付>/<時刻>/` 配下に `training.log` や各種png/CSVが保存されます。

### 2-B) CircleLoss（車種分類のための表現学習）

`LabelClustering.py` は車種（car/cv）と方向（left/right）を同時に分離する潜在表現を学習します。

```bash
# small
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/metric/LabelClustering.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --epochs 100 \
  --batch-size 64 \
  --visualization PCA

# ResNet-18 / ResNet-50（追加済み）
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/metric/LabelClustering.py \
  --model resnet18 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --epochs 100 \
  --batch-size 64 \
  --visualization PCA

MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/metric/LabelClustering.py \
  --model resnet50 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --epochs 100 \
  --batch-size 32 \
  --visualization PCA
```

成果物:
- `outputs/<日付>/<時刻>/best_encoder_<model>.pth`
- `outputs/<日付>/<時刻>/last_encoder_<model>.pth`
- `outputs/<日付>/<時刻>/*.png`（潜在空間可視化）

---

## 3. Optuna による学習最適化（AutoEncoder）

`src/hyper_optimizer/ho_autoencoder.py` は `src/autoencoder/CNN_any.py` を呼び出して、検証損失を最小化する探索を行います。

### 3-1) ローカルsqliteに保存して実行

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_autoencoder.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 30 \
  --hpo-config ./configs/optuna_autoencoder_hpo.json \
  --storage sqlite:///outputs/hpo_autoencoder.db \
  --study-name ae_small_loc1-6 \
  --pruner median \
  --n-jobs 1
```

### 3-2) 探索範囲（このスクリプトが探索するもの）

- `batch_size`: 16/32/48/64
- `lr`: 1e-4〜5e-3（log）
- `epochs`: `--min-epochs`〜`--max-epochs`
- スペクトログラム時: `n_fft`, `hop_length`, `mel`

---

## 4. 実データの前処理（VS13/IDMT）

この工程で、実データを `learn_tool.settings.prepare_dataloader` が読める形式へ変換します。

### 4-1) IDMT_Traffic（車種分類向け）

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/real_data_tool/prepare_real_data.py idmt \
  --raw-dir ./data/raw/IDMT_Traffic \
  --out-dir ./data/processed/real/idmt_traffic \
  --csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --duration 6.0 \
  --sampling-rate 16000
```

### 4-2) VS13（速度推定向け）

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/real_data_tool/prepare_real_data.py vs13 \
  --raw-dir ./data/raw/VS13 \
  --out-dir ./data/processed/real/vs13 \
  --csv ./data/processed/real/vs13/vs13.csv \
  --duration 6.0 \
  --sampling-rate 16000
```

---

## 5. 実データで推論・評価

### 5-A) IDMT（車種分類）

IDMT の `idmt_traffic.csv` を使って、実データ上で CircleLoss 学習（≒表現学習）を回し、潜在空間/メタデータを出力します。

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/metric/LabelClustering.py \
  --model small \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --data-selection loc1-6 \
  --epochs 20 \
  --batch-size 64 \
  --visualization PCA
```

（別アーキテクチャ例）:

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/metric/LabelClustering.py \
  --model resnet18 \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --data-selection loc1-6 \
  --epochs 20 \
  --batch-size 64 \
  --visualization PCA
```

（学習せずに t-SNE だけ見たい場合: 推論/可視化モード）:

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/metric/LabelClustering.py \
  --model small \
  --encoder-weights ./outputs/<日付>/<時刻>/best_encoder_small.pth \
  --no-train \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --data-selection loc1-6 \
  --batch-size 64 \
  --visualization t-SNE \
  --dimension 2 \
  --max-points 3000
```

※IDMT のサンプル数が多い場合、t-SNE は重くなりやすいので `--max-points` で間引くのがおすすめです。

### 5-B) VS13（速度推定）

`speedPrediction.py` は、**事前学習済み encoder weights**（例: `best_encoder_small.pth`）を読み込み、MLP回帰（＋必要ならencoder微調整）を行います。

```bash
# 例: 直前の LabelClustering の出力 encoder を利用
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/<日付>/<時刻>/best_encoder_small.pth \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --data-selection loc1 \
  --epochs 100 \
  --batch-size 32 \
  --loss mse

# encoder を固定したい場合
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/<日付>/<時刻>/best_encoder_small.pth \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --data-selection loc1 \
  --epochs 100 \
  --batch-size 32 \
  --freeze-encoder \
  --loss mse
```

成果物:
- `outputs/<日付>/<時刻>/loss_curve.png`
- `outputs/<日付>/<時刻>/train_predictions_<loss>.csv`
- `outputs/<日付>/<時刻>/val_predictions_<loss>.csv`

---

## 6. 最小スモークテスト（変換後に「読めるか」だけ確認）

```bash
/home/tobeson/miniconda3/envs/traf_ana/bin/python src/real_data_tool/smoke_real_pipeline.py \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --data-selection loc1 \
  --representation spectrogram \
  --mel OFF \
  --batch-size 8
```
