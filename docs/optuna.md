# Optuna（AutoEncoderのハイパーパラメータ最適化）

このリポジトリでは、AutoEncoder 学習（`src/autoencoder/CNN_any.py`）を Optuna で呼び出して、検証損失を最小化する探索を行います。

- 実行スクリプト: `src/hyper_optimizer/ho_autoencoder.py`
- 学習本体: `src/autoencoder/CNN_any.py`

---

## 1. 基本実行

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_autoencoder.py \
	--model small \
	--data-selection loc1-6 \
	--data-csv ./data/processed/datasets/data_1-6.csv \
	--main-data-dir ./data/processed/datasets \
	--n-trials 20 \
	--min-epochs 10 \
	--max-epochs 40 \
	--pruner median
```

補足:
- `--storage sqlite:///outputs/hpo.db --study-name ae_hpo` を付けると途中再開・結果保存ができます
- `--n-jobs` は並列 trial 数です（GPU 1枚の場合は基本 `1` 推奨）
- `--seed` は固定seedです（デフォルト 42）。trial 間でも seed は変わりません。

---

## 2. 探索する項目（デフォルト）

`src/hyper_optimizer/ho_autoencoder.py` の `objective()` 内で `trial.suggest_*` している項目が探索対象です。

- 共通: `batch_size`, `lr`, `epochs`
- スペクトログラム時のみ: `n_fft`, `hop_length`, `mel`

---

## 3. 探索空間を JSON で制御する（おすすめ）

探索項目/探索範囲を「コードではなく設定ファイル」で管理できるようにしています。

### 3-1) 使い方

テンプレ: `./configs/optuna_autoencoder_hpo.json`

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_autoencoder.py \
	--model small \
	--data-selection loc1-6 \
	--data-csv ./data/processed/datasets/data_1-6.csv \
	--main-data-dir ./data/processed/datasets \
	--n-trials 20 \
	--min-epochs 10 \
	--max-epochs 40 \
	--hpo-config ./configs/optuna_autoencoder_hpo.json \
	--storage sqlite:///outputs/hpo_autoencoder.db \
	--study-name ae_small_loc1-6
```

`--hpo-config` を付けない場合は、従来どおり「コード内のデフォルト探索空間」で動作します。

### 3-2) JSON スキーマ（version 1）

トップレベル:
- `version`: 1
- `common`: 常に適用される探索/固定項目
- `spectrogram`: `representation == "spectrogram"` のときだけ適用される探索/固定項目

各パラメータの定義方法:

#### a) 固定（最適化しない）

```json
"batch_size": {"type": "fixed", "value": 32}
```

#### b) categorical（離散候補から選ぶ）

```json
"batch_size": {"type": "categorical", "choices": [16, 32, 48, 64]}
```

#### c) float（連続値）

```json
"lr": {"type": "float", "low": 0.0001, "high": 0.005, "log": true}
```

#### d) int（整数）

```json
"epochs": {"type": "int", "low": 10, "high": 40}
```

`low/high` は CLI 引数（`--min-epochs`, `--max-epochs` など）を参照させることもできます:

```json
"epochs": {
	"type": "int",
	"low": {"from_cli": "min_epochs"},
	"high": {"from_cli": "max_epochs"}
}
```

### 3-3) よくある設定例

#### 例1: `mel` は最適化せず OFF に固定したい

設定ファイル側で固定:

```json
"mel": {"type": "fixed", "value": "OFF"}
```

または CLI で上書き（`--mel` を指定すると、設定ファイルの `mel` 探索は無効化されます）:

```bash
... ho_autoencoder.py ... --mel OFF
```

#### 例2: STFT パラメータは固定して、学習ハイパラだけ探索したい

`spectrogram` セクションを空にする（または `fixed` で埋める）ことで、`n_fft/hop_length/mel` を探索しない構成にできます。

---

## 4. 出力と再現

- 各 trial は `outputs/<日付>/<時刻>/` に成果物を出力します
- Optuna の best trial では、コンソールに `Best params` と `Artifacts in: ...` が表示されます
- Optuna 実行後、study サマリは `outputs/optuna_studies/<study_name>/` に自動保存されます:
	- `optuna_best.json`: best trial の値/params + 実行メタ情報（seed, hpo-config, argv など）
	- `optuna_best_params.json`: best params だけ（再利用しやすい）
	- `optuna_trials.csv`: 全 trial の一覧（state/value/params など）
	- `optuna_optimization_history.html`: Optimization History（最適化履歴）
	- `optuna_parallel_coordinate.html`: Parallel Coordinate Plot（平行座標プロット）
	- `optuna_hyperparameter_importance.html`: Hyperparameter Importance（重要度）
	- `optuna_optimization_history.png`
	- `optuna_parallel_coordinate.png`
	- `optuna_hyperparameter_importance.png`
	- `optuna_plots.log`: 図が出せない場合の理由（plotly未導入、trial不足など）

---

# Optuna（CircleLoss/距離学習のハイパーパラメータ最適化）

AutoEncoder とは別に、CircleLoss を用いた深層距離学習（`src/metric/LabelClustering.py`）も Optuna で最適化できます。

- 実行スクリプト: `src/hyper_optimizer/ho_labelclustering.py`
- 学習本体: `src/metric/LabelClustering.py`

## 1. 基本実行

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_labelclustering.py \
	--model small \
	--data-selection loc1-6 \
	--data-csv ./data/processed/datasets/data_1-6.csv \
	--main-data-dir ./data/processed/datasets \
	--n-trials 20 \
	--min-epochs 10 \
	--max-epochs 60 \
	--pruner median \
	--storage sqlite:///outputs/hpo_labelclustering.db \
	--study-name lc_small_loc1-6
```

補足:
- `--seed` は固定seedです（デフォルト 42）。trial 間でも seed は変わりません。
- `--save-checkpoints` を付けると trial ごとに encoder 重みも保存します（ディスク増えます）。

## 2. 探索する項目（デフォルト）

`src/hyper_optimizer/ho_labelclustering.py` の `objective()` 内で `trial.suggest_*` している項目が探索対象です。

- 共通: `batch_size`, `lr`, `epochs`, `margin`, `gamma`
- スペクトログラム時のみ: `n_fft`, `hop_length`, `mel`

## 3. 探索空間を JSON で制御する（おすすめ）

テンプレ: `./configs/optuna_labelclustering_hpo.json`

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_labelclustering.py \
	--model small \
	--data-selection loc1-6 \
	--data-csv ./data/processed/datasets/data_1-6.csv \
	--main-data-dir ./data/processed/datasets \
	--n-trials 20 \
	--min-epochs 10 \
	--max-epochs 60 \
	--hpo-config ./configs/optuna_labelclustering_hpo.json \
	--storage sqlite:///outputs/hpo_labelclustering.db \
	--study-name lc_small_loc1-6
```

JSON の書き方は AutoEncoder と同じです（`fixed` / `categorical` / `float` / `int` / `from_cli`）。

## 4. 出力（最適化したハイパパラメータのログ）

Optuna 実行後、study サマリは `outputs/optuna_studies/<study_name>/` に保存されます:

- `optuna_best.json`
- `optuna_best_params.json`
- `optuna_trials.csv`

plotly が利用できる場合は以下も出ます:
- `optuna_optimization_history.html`
- `optuna_parallel_coordinate.html`
- `optuna_hyperparameter_importance.html`

さらに `kaleido` が利用できる場合はPNGも出ます:
- `optuna_optimization_history.png`
- `optuna_parallel_coordinate.png`
- `optuna_hyperparameter_importance.png`


---

# HPOのベスト重みを保存・可視化する

## 1. ベスト重みを保存する（HPO実行時）

各HPOスクリプトで `--export-best-weights` を付けると、best trial の重みが
`outputs/optuna_studies/<study_name>/` にコピーされます。

例:

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_labelclustering.py \
	--model vgg11 \
	--data-selection loc1-6 \
	--data-csv ./data/processed/datasets/data_1-6.csv \
	--main-data-dir ./data/processed/datasets \
	--n-trials 100 \
	--min-epochs 10 \
	--max-epochs 200 \
	--hpo-config ./configs/optuna_labelclustering_hpo.json \
	--storage sqlite:///outputs/hpo_labelclustering.db \
	--study-name cl_optuna_vgg11_loc1-6_v1 \
	--export-best-weights
```

## 2. ベスト重みで t-SNE 可視化する

`src/metric/visualize_labelclustering_tsne.py` を使うと、学習済み重み + データCSVから
t-SNE 可視化を実行できます。

### a) 直接パス指定

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/metric/visualize_labelclustering_tsne.py \
	--model vgg11 \
	--encoder-weights ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
	--data-selection loc1-6 \
	--data-csv ./data/processed/datasets/data_1-6.csv \
	--main-data-dir ./data/processed/datasets \
	--visualization t-SNE
```

### b) Optunaのstudyから自動解決

```bash
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/metric/visualize_labelclustering_tsne.py \
	--model vgg11 \
	--study-name cl_optuna_vgg11_loc1-6_v1 \
	--storage sqlite:///outputs/hpo_labelclustering.db \
	--data-selection loc1-6 \
	--data-csv ./data/processed/datasets/data_1-6.csv \
	--main-data-dir ./data/processed/datasets \
	--visualization t-SNE
```

