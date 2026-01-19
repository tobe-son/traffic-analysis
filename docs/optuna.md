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
