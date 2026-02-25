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
cd ~/traffic-analysis  # または実際のリポジトリパスへ変更してください
conda activate traf_ana
```

---

## 1. simulation データの前処理（6秒切り出し + CSV結合）

### 1-1) 6秒切り出し（12〜18秒）

```bash
MPLBACKEND=Agg python src/sim_data_tool/cut.py
```

出力: `data/processed/datasets/loc{1..6}_cut/<car|cv>/<left|right>/*.flac`

### 1-2) メタCSV結合

```bash
MPLBACKEND=Agg python src/sim_data_tool/combain.py
```

出力: `data/processed/datasets/data_1-6.csv`

---

## 2. simulation で学習（複数アーキテクチャ）

ここでは **(A) AutoEncoder** と **(B) CircleLoss（車種×方向）** の2系統の入口を用意しています。

### 2-A) AutoEncoder 学習（`CNN_any.py`）

例（モデルを切り替えて学習）:

```bash
# small
MPLBACKEND=Agg python src/autoencoder/CNN_any.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --visualization PCA

# vgg11
MPLBACKEND=Agg python src/autoencoder/CNN_any.py \
  --model vgg11 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --visualization PCA

# resnet（legacy）
MPLBACKEND=Agg python src/autoencoder/CNN_any.py \
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
MPLBACKEND=Agg python src/metric/LabelClustering.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --epochs 100 \
  --batch-size 64 \
  --visualization PCA

# ResNet-18 / ResNet-50（追加済み）
MPLBACKEND=Agg python src/metric/LabelClustering.py \
  --model resnet18 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --epochs 100 \
  --batch-size 64 \
  --visualization PCA

MPLBACKEND=Agg python src/metric/LabelClustering.py \
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
MPLBACKEND=Agg python src/hyper_optimizer/ho_autoencoder.py \
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

### 3-3) Optuna（CircleLoss/距離学習: 車種分類の表現学習）

`src/hyper_optimizer/ho_labelclustering.py` は `src/metric/LabelClustering.py`（CircleLoss）を Optuna で最適化します。

```bash
MPLBACKEND=Agg python src/hyper_optimizer/ho_labelclustering.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 30 \
  --min-epochs 10 \
  --max-epochs 60 \
  --hpo-config ./configs/optuna_labelclustering_hpo.json \
  --storage sqlite:///outputs/hpo_labelclustering.db \
  --study-name lc_small_loc1-6 \
  --pruner median \
  --n-jobs 1
```

---

## 4. 実データの前処理（VS13/IDMT）

この工程で、実データを `learn_tool.settings.prepare_dataloader` が読める形式へ変換します。

### 4-1) IDMT_Traffic（車種分類向け）

```bash
MPLBACKEND=Agg python src/real_data_tool/prepare_real_data.py idmt \
  --raw-dir ./data/raw/IDMT_Traffic \
  --out-dir ./data/processed/real/idmt_traffic \
  --csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --duration 6.0 \
  --sampling-rate 16000
```

### 4-2) VS13（速度推定向け）

```bash
MPLBACKEND=Agg python src/real_data_tool/prepare_real_data.py vs13 \
  --raw-dir ./data/raw/VS13 \
  --out-dir ./data/processed/real/vs13 \
  --csv ./data/processed/real/vs13/vs13.csv \
  --duration 6.0 \
  --sampling-rate 16000
```

---

## 5. 速度予測 MLP の学習（VS13）

事前学習済みエンコーダ（CircleLoss/ArcFace/LogRatio いずれか）を利用して、VS13 実データで MLP 回帰モデルを学習します。
全モデル・全事前学習の組み合わせをまとめて実行する場合は以下のシェルスクリプトを使用します:

```bash
bash ./exp4.sh
```

個別に実行する場合の例:

```bash
# 事前学習（Circle Loss: small）からのMLP回帰学習
MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1

# encoder を固定したい場合は --freeze-encoder を追加
MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --freeze-encoder \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1
```

成果物:
- `outputs/speed_prediction/<study_name>/best_speed_regressor_<loss>.pth`
- `outputs/speed_prediction/<study_name>/loss_curve.png`
- `outputs/speed_prediction/<study_name>/train_predictions_<loss>.csv`
- `outputs/speed_prediction/<study_name>/val_predictions_<loss>.csv`

---

## 6. 車種分類評価（IDMT / 実データ）

事前学習済みエンコーダを使って IDMT 実データ上で車種分類性能を評価します。
全モデル・全事前学習の組み合わせをまとめて実行する場合は以下のシェルスクリプトを使用します:

```bash
bash ./exp5.sh
```

個別に実行する場合の例:

```bash
# Circle Loss（small）のシミュレーションデータでの評価
python -m src.eval.deep_metric_eval \
  --encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --model small \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --data-selection loc1-6 \
  --dimension 2 \
  --optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/optuna_best.json \
  --output-dir ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/eval-sim/

# Circle Loss（small）の実データでの評価（車種×方向の4クラス）
python -m src.eval.deep_metric_eval \
  --encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --model small \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --data-selection loc1-6 \
  --dimension 2 \
  --optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/optuna_best.json \
  --output-dir ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/eval-real/

# 実データのモノラル評価（車種の2クラスのみ、方向なし）
python -m src.eval.deep_metric_eval_mono \
  --encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --model small \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --data-selection loc1-6 \
  --dimension 2 \
  --optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/optuna_best.json \
  --output-dir ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/eval-real-mono/
```

---

## 7. 速度推定評価（VS13 / 実データ）

学習済み速度予測モデル（encoder + MLP）を使って VS13 実データ上で速度推定性能を評価します。
全モデル・全事前学習の組み合わせをまとめて実行する場合は以下のシェルスクリプトを使用します:

```bash
bash ./exp6.sh
```

個別に実行する場合の例:

```bash
# Circle - MLP のシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1/eval-sim/

# Circle - MLP の実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1/eval-real/
```

---

## 8. 最小スモークテスト（変換後に「読めるか」だけ確認）

```bash
python src/real_data_tool/smoke_real_pipeline.py \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --data-selection loc1 \
  --representation spectrogram \
  --mel OFF \
  --batch-size 8
```
