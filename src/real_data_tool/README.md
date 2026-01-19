# real_data_tool

`data/raw` に置いた実データを、このリポジトリの学習スクリプト（`learn_tool.settings.prepare_dataloader`）が読める
「`locX_cut/<vehicle_type>/<direction>/...` + メタCSV」に変換するツール群です。

## 変換コマンド

### IDMT_Traffic（車種分類向け / car-vs-cv + left/right）

```bash
python src/real_data_tool/prepare_real_data.py idmt \
  --raw-dir ./data/raw/IDMT_Traffic \
  --out-dir ./data/processed/real/idmt_traffic \
  --csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --duration 6.0 --sampling-rate 16000
```

少量でスモーク確認したい場合は `--limit` を付けます:

```bash
python src/real_data_tool/prepare_real_data.py idmt \
  --raw-dir ./data/raw/IDMT_Traffic \
  --out-dir ./data/processed/real/idmt_traffic_small \
  --csv ./data/processed/real/idmt_traffic_small/idmt_traffic.csv \
  --duration 6.0 --sampling-rate 16000 \
  --limit 200
```

- `--vehicle-scheme car_vs_cv`：IDMT の (B,C,M,T) を `car/cv` に二値化します（C/M→car, B/T→cv）。
- `--unknown-speed drop|zero`：`unknownKmh` を扱う方法。
- `--mic` / `--channel`：必要ならフィルタ（例: `--mic SE --channel CH34`）。

### VS13（速度推定向け）

```bash
python src/real_data_tool/prepare_real_data.py vs13 \
  --raw-dir ./data/raw/VS13 \
  --out-dir ./data/processed/real/vs13 \
  --csv ./data/processed/real/vs13/vs13.csv \
  --duration 6.0 --sampling-rate 16000
```

少量でスモーク確認したい場合は `--limit` を付けます:

```bash
python src/real_data_tool/prepare_real_data.py vs13 \
  --raw-dir ./data/raw/VS13 \
  --out-dir ./data/processed/real/vs13_small \
  --csv ./data/processed/real/vs13_small/vs13.csv \
  --duration 6.0 --sampling-rate 16000 \
  --limit 200
```

## 既存学習スクリプトへ渡す例

### 車種分類（CircleLoss）

```bash
python -m metric.LabelClustering \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --data-selection loc1-6
```

### 速度推定（MLP回帰）

```bash
python -m mlp.speedPrediction \
  --encoder-weights ./encoder/<your_encoder>.pth \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --data-selection loc1
```

## スモークテスト（dataloader + Encoder の forward 1回）

変換後に、既存の `prepare_dataloader` で読めることと、Encoder の forward が通ることを最小確認できます:

```bash
python src/real_data_tool/smoke_real_pipeline.py \
  --data-csv ./data/processed/real/idmt_traffic_small/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic_small \
  --data-selection loc1 \
  --representation spectrogram \
  --mel OFF \
  --batch-size 8
```

## メモ

- 出力音声は、`--duration` 秒にセンタークロップ/ゼロパディングし、`--sampling-rate` に揃えます。
- 追加の列（`location_raw` 等）は学習側では無視されます。`path/speed/vehicle_type/direction` が主に必要です。
