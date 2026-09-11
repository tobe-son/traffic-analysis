# 2026年度 専攻科研究 I 成果物索引

| シラバス評価項目 | 成果物 | 現在の状態 |
| --- | --- | --- |
| 背景・目的（報告書10点） | `plan.md`, `interim_report.md` | 昨年度との差と数値目標を明記 |
| 困難への対応（計30点） | `progress_log.md` | 原因仮説・切り分け・次実験を記録開始 |
| 科学的方法・データ収集（10点） | protocol YAML, baseline CSV/JSON, `domain_gap.py` | 事前登録・再集計・統計処理を実装 |
| 口頭報告（20点） | 発表資料・質疑記録 | 13-15週に作成予定 |
| 報告書（20点） | `interim_report.md` | A4約2枚へ組版する本文素案を作成 |
| 学会論文の文献調査（10点） | `literature_review.md` | 学会・査読会議論文5件を比較 |

## 現時点で主張できる成果

既存9モデルを同一基準で再集計した結果、simulation 内の最高モデル
（LogRatio-resnet18, R2=0.955）は実環境で R2=-1.436 まで低下した。
実環境の最良は ArcFace-small (R2=-0.019, MAE=16.932 km/h) であり、
モデル選択順位が逆転することを定量化した。この結果を受けて、研究目的を
「simulation精度向上」から「未知環境での一般化」へ変更した。

## 再現

```bash
PYTHONPATH=src python -m research.domain_gap \
  docs/research_2026/data/baseline_speed_summary.csv \
  --mode summary \
  --output docs/research_2026/baseline_gap.json

PYTHONPATH=src python -m unittest discover -s tests -v
```

`baseline_gap.json` は生成済み結果であり、同じCSVから再作成できる。今後は
サンプル単位の `model,domain,y_true,y_pred` CSVを入力し、95% bootstrap CIも
同じCLIで算出する。

