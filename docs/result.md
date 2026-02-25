# 実験1-3/3c/3c2 HPO結果まとめ

本資料は、実験1〜3、実験3c/3c2のHPO探索範囲と最適化結果、実験4の速度回帰ハイパパラ、実験5および実験6の評価結果を整理したものです。

- 対象データ: loc1-6
- HPO結果: Optuna best params を整理
- 評価結果: `evaluation_metrics.json` の val 指標 (MAE/RMSE/R2/MAPE)
- 欠損: resnet50 系の eval 結果が存在しないため N/A

## 実験1: Circle Loss (Label Clustering)

### 探索範囲

**small / vgg11 (configs/optuna_labelclustering_hpo.json)**

| パラメータ | 範囲/選択肢 |
| --- | --- |
| batch_size | [8, 16, 32, 48, 64] |
| lr | 1e-4 - 1e-2 (log) |
| epochs | min_epochs - max_epochs |
| margin | 0.1 - 0.4 |
| gamma | 16 - 128 (log) |
| n_fft | [512, 1024, 2048] |
| hop_length | [128, 160, 256, 320, 512] |
| mel | OFF / ON |

**resnet18 / resnet50 (configs/optuna_labelclustering_resnet_hpo.json)**

| パラメータ | 範囲/選択肢 |
| --- | --- |
| batch_size | [4, 8, 16] |
| lr | 1e-4 - 5e-3 (log) |
| epochs | min_epochs - max_epochs |
| margin | 0.1 - 0.4 |
| gamma | 16 - 128 (log) |
| n_fft | [512, 1024] |
| hop_length | [256, 320, 512] |
| mel | ON (fixed) |

### 最適化パラメータ

| モデル | batch_size | lr | epochs | margin | gamma | n_fft | hop_length | mel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small (v1) | 8 | 0.007687 | 646 | 0.399241 | 16.353446 | 512 | 320 | OFF |
| small (v2) | 8 | 0.007687 | 646 | 0.399241 | 16.353446 | 512 | 320 | OFF |
| vgg11 | 8 | 0.000180 | 114 | 0.396403 | 16.012907 | 1024 | 256 | ON |
| resnet18 | 4 | 0.000298 | 110 | 0.399219 | 16.093302 | 1024 | 512 | ON |
| resnet50 | 4 | 0.002334 | 99 | 0.399880 | 19.785728 | 512 | 320 | ON |

### 評価

実験5および実験6の評価結果としてまとめて掲載 (後述)。

## 実験2: ArcFace Loss

### 探索範囲 (configs/optuna_labelclustering_arcface_hpo.json)

| パラメータ | 範囲/選択肢 |
| --- | --- |
| batch_size | [8, 16, 32, 48, 64] |
| lr | 1e-4 - 1e-2 (log) |
| epochs | min_epochs - max_epochs |
| arcface_s | 16 - 128 |
| arcface_m | 0.1 - 0.7 |
| n_fft | [512, 1024, 2048] |
| hop_length | [128, 160, 256, 320, 512] |
| mel | OFF / ON |

### 最適化パラメータ

| モデル | batch_size | lr | epochs | arcface_s | arcface_m | n_fft | hop_length | mel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small (v1) | 32 | 0.009568 | 554 | 95.741514 | 0.680462 | 2048 | 512 | ON |
| small (v2) | 32 | 0.009568 | 554 | 95.741514 | 0.680462 | 2048 | 512 | ON |
| vgg11 | 8 | 0.009086 | 65 | 97.197101 | 0.625470 | 512 | 512 | ON |
| resnet18 | 32 | 0.006992 | 151 | 127.716621 | 0.636817 | 2048 | 512 | ON |
| resnet50 | 8 | 0.004885 | 115 | 122.673827 | 0.673535 | 512 | 160 | ON |

### 評価

実験5および実験6の評価結果としてまとめて掲載 (後述)。

## 実験3: LogRatio Loss

### 探索範囲 (configs/optuna_continuouslearning_hpo.json)

| パラメータ | 範囲/選択肢 |
| --- | --- |
| batch_size | [8, 16, 32, 48, 64] |
| lr | 1e-4 - 1e-2 (log) |
| epochs | min_epochs - max_epochs |
| norm_p | 1.0 - 4.0 |
| eps | 1e-8 - 1e-3 (log) |
| n_fft | [512, 1024, 2048] |
| hop_length | [128, 160, 256, 320, 512] |
| mel | OFF / ON |

### 最適化パラメータ

| モデル | batch_size | lr | epochs | norm_p | eps | n_fft | hop_length | mel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small (v1) | 64 | 0.000334 | 951 | 2.649225 | 0.000986 | 2048 | 320 | ON |
| small (v2) | 64 | 0.000334 | 951 | 2.649225 | 0.000986 | 2048 | 320 | ON |
| vgg11 | 64 | 0.002197 | 145 | 1.570632 | 0.000965 | 512 | 320 | OFF |
| resnet18 | 64 | 0.000186 | 62 | 2.586892 | 0.000470 | 1024 | 512 | OFF |
| resnet50 | 64 | 0.000885 | 142 | 1.873451 | 0.000259 | 2048 | 320 | ON |

### 評価

実験5および実験6の評価結果としてまとめて掲載 (後述)。

## 実験3c: Circle Loss -> LogRatio Loss 継続事前学習

### 探索範囲 (configs/optuna_continuouslearning_hpo.json)

| パラメータ | 範囲/選択肢 |
| --- | --- |
| batch_size | [8, 16, 32, 48, 64] |
| lr | 1e-4 - 1e-2 (log) |
| epochs | min_epochs - max_epochs |
| norm_p | 1.0 - 4.0 |
| eps | 1e-8 - 1e-3 (log) |
| n_fft | [512, 1024, 2048] |
| hop_length | [128, 160, 256, 320, 512] |
| mel | OFF / ON |

### 最適化パラメータ

| モデル | batch_size | lr | epochs | norm_p | eps | n_fft | hop_length | mel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small (v1) | 64 | 0.000382 | 831 | 1.072984 | 0.001000 | 2048 | 512 | ON |
| small (v2) | 64 | 0.000241 | 580 | 2.689649 | 0.000998 | 1024 | 512 | ON |
| vgg11 | 64 | 0.000291 | 77 | 3.088249 | 0.000678 | 1024 | 160 | OFF |
| resnet18 | 64 | 0.001245 | 101 | 2.307348 | 0.000723 | 1024 | 512 | OFF |
| resnet50 | 64 | 0.002245 | 85 | 2.889398 | 0.000373 | 2048 | 512 | ON |

### 評価

実験5および実験6の評価結果としてまとめて掲載 (後述)。

## 実験3c2: ArcFace Loss -> LogRatio Loss 継続事前学習

### 探索範囲 (configs/optuna_continuouslearning_hpo.json)

| パラメータ | 範囲/選択肢 |
| --- | --- |
| batch_size | [8, 16, 32, 48, 64] |
| lr | 1e-4 - 1e-2 (log) |
| epochs | min_epochs - max_epochs |
| norm_p | 1.0 - 4.0 |
| eps | 1e-8 - 1e-3 (log) |
| n_fft | [512, 1024, 2048] |
| hop_length | [128, 160, 256, 320, 512] |
| mel | OFF / ON |

### 最適化パラメータ

| モデル | batch_size | lr | epochs | norm_p | eps | n_fft | hop_length | mel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small (v1) | 64 | 0.001535 | 648 | 3.278772 | 0.001000 | 1024 | 320 | ON |
| small (v2) | 64 | 0.000663 | 885 | 2.886820 | 0.000986 | 2048 | 512 | ON |
| vgg11 | 64 | 0.001604 | 137 | 2.804500 | 0.000999 | 512 | 128 | OFF |
| resnet18 | 64 | 0.000241 | 121 | 2.690279 | 0.000998 | 1024 | 512 | ON |
| resnet50 | 64 | 0.000526 | 166 | 2.676860 | 0.000301 | 512 | 512 | ON |

### 評価

実験5および実験6の評価結果としてまとめて掲載 (後述)。

## 実験1-3 (HPO) の考察（総括）

ここでは実験1〜3の Optuna による最適化結果（best params）をまとめて解釈する。なお本HPOは「検証損失（val loss）の最小化」を目的としており、実験5で掲載する `recall@k` や `knn_acc@k` を直接最適化していない点に注意する（= best params は「損失最小」という意味での最適であり、下流指標の最適とは限らない）。

### batch_size と OOM リトライの影響

- LogRatio 系（実験3）では多くのモデルで `batch_size` が探索範囲の上限（64）を選んでおり、バッチを大きくするほど val loss が下がりやすい傾向が示唆される。メトリック学習では、バッチ内の負例・正例の多様性が増えることや、勾配推定が安定することが効いている可能性がある。
- 一方で Circle（実験1）は small/vgg11 が `batch_size=8`、resnet 系が `batch_size=4` になっており、「常に大きいバッチが有利」ではない。Circle のようにバッチ内のサンプル構成に損失の効き方が依存する場合、(1) 小さめのバッチがノイズとして正則化になる、(2) 大きいバッチで hard negative が増えすぎて最適化が難しくなる、などの理由で小さいバッチが選ばれることがある（仮説）。
- 今回の Optuna 実装では、CUDA OOM が発生した trial は同一 trial 内で `batch_size` を半分にして再実行するよう改造している。そのため、best params に記録される `batch_size`（suggest された値）と、実際に学習に使われた `effective_batch_size` がズレる可能性がある。
  - 学習コード側で `effective_batch_size` と `oom_adjustments` を trial.user_attrs に保存しているため、厳密に「バッチサイズが効いているか」を論じる場合は、best params だけでなく `effective_batch_size` を基準に集計するのが望ましい。
  - 特に resnet50 などメモリ制約が強いモデルでは、探索中の多くの trial が OOM を経験し得るため、探索が「実質的に小さいバッチに偏る」選択バイアスが入りやすい。

### 学習率・エポック数（最適化の安定性）

- LogRatio（実験3）では epochs が very large（例: small で 951）になっており、損失の減少が遅い／より長い学習がそのまま val loss 改善に繋がりやすい可能性がある。逆に言うと、early stopping を入れる設計にすると、同等性能を短時間で得られる余地がある。
- Circle（実験1）は small が 600epoch 超、vgg11/resnet が 100epoch 程度で止まっており、モデル容量と最適化難易度（あるいは過学習の始まり）が異なる可能性がある。特に small が長く回るのは、(1) 計算が軽く探索が回りやすい、(2) 早い段階で過学習しにくい、などが考えられる。

### 損失固有ハイパパラの端点選好

- Circle（実験1）は `margin` がほぼ上限（≈0.4）に張り付いている一方、`gamma` は下限寄り（≈16）になっている。探索範囲の端を選ぶのは「その方向に改善余地がある」サインでもあるため、再実験するなら `margin` の上限拡張、`gamma` の下限拡張（16未満）を検討する価値がある。
- ArcFace（実験2）は `arcface_s` と `arcface_m` が上限寄り（s≈96〜128, m≈0.63〜0.68）に寄るモデルがあり、分類損失（CrossEntropy）の最小化には強いスケール／マージンが効きやすい可能性がある。ただし評価の節で述べた通り、ArcFace head 指標の解釈には注意が必要で、下流の近傍指標（recall/knn）と整合しないケースもあり得る。
- LogRatio（実験3）は `eps` が上限寄り（≈1e-3）に寄るモデルが見られ、数値安定性（ゼロ割/極端比の抑制）が重要だった可能性がある。ここも端点選好なので、必要なら探索範囲を見直す余地がある。

### 前処理（n_fft / hop_length / mel）

- best params は mel ON/OFF や FFT 条件が混在しており、一概に「これが最適」とは言いにくい。特に HPO の目的が val loss であるため、下流タスク（実験5/6）でどの前処理が強いかは別途の検証が必要になる。
- ただし、モデル容量が大きい系（vgg/resnet）で mel ON が選ばれる傾向はあり、入力の周波数表現を圧縮・平滑化することが最適化を助けた可能性はある。

### Optuna の重要度/最適化履歴（history）からの所見

各 study には `outputs/optuna_studies/<study>/optuna_trials.csv` と、Optuna の可視化（`optuna_hyperparameter_importance.*`, `optuna_optimization_history.*`）が出力されている。ここでは `optuna_trials.csv` を集計し、(1) どの程度早く最良値に到達したか、(2) どのハイパラが効いていそうか、(3) OOM リトライによる `effective_batch_size` の乖離がどの程度あるか、を確認した。

`optuna_trials.csv` からの簡易集計（収束/retry/OOM/drift は CSV 集計。`top3` は Optuna が出力した importance 図（`optuna_hyperparameter_importance.html`）から抜粋）：

| loss | model | complete | best | reached95 | OOM/drift | top3 |
| --- | --- | --- | --- | --- | --- | --- |
| ArcFace | resnet18_v1 | 11/100 | 7.11e-06 | 1 | 1/1 | epochs/arcface_s/lr |
| ArcFace | resnet50_v1 | 42/100 | 3.219e-07 | 27 | 3/3 | arcface_s/epochs/mel |
| ArcFace | small_v1 | 61/200 | 0 | 13 | 0/0 | arcface_s/arcface_m/epochs |
| ArcFace | small_v2 | 29/100 | 0 | 13 | 0/0 | arcface_m/arcface_s/epochs |
| ArcFace | vgg11_v1 | 26/100 | 2.701e-06 | 1 | 0/0 | arcface_s/lr/epochs |
| Circle | resnet18_v1 | 47/100 | 3.534 | 27 | 0/0 | margin/gamma/epochs |
| Circle | resnet50_v1 | 25/100 | 4.108 | 35 | 0/0 | margin/gamma/epochs |
| Circle | small_v1 | 57/200 | 5.344 | 24 | 0/0 | gamma/batch_size/n_fft |
| Circle | small_v2 | 33/100 | 5.344 | 24 | 0/0 | gamma/n_fft/hop_length |
| Circle | vgg11_v1 | 56/100 | 10.91 | 79 | 2/2 | gamma/batch_size/epochs |
| LogRatio | resnet18_v1 | 31/100 | 2.014 | 11 | 0/0 | batch_size/n_fft/norm_p |
| LogRatio | resnet50_v1 | 34/100 | 2.495 | 78 | 3/3 | eps/hop_length/mel |
| LogRatio | small_v1 | 35/200 | 1.783 | 11 | 0/0 | batch_size/mel/lr |
| LogRatio | small_v2 | 27/100 | 1.783 | 11 | 0/0 | batch_size/lr/mel |
| LogRatio | vgg11_v1 | 29/100 | 1.815 | 13 | 0/0 | batch_size/n_fft/eps |

- 収束の早さ（「最初の成功 trial」から見て最良値の 95% まで改善した trial 番号の目安）
  - Circle: small/resnet18/resnet50 は trial 24/27/35 付近で 95% に到達する一方、vgg11 は trial 79 付近と遅い。探索空間の感度（前処理や損失パラ）や最適化の難しさがモデルごとに異なる可能性がある。
  - ArcFace: resnet18/vgg11 は trial 1 付近で急速に改善する一方、resnet18 は FAIL が多く（100trial中 63 FAIL）、探索の「成功率」が低い。ArcFace は val loss を 0 近傍まで落としやすいが、設定によっては学習が不安定になりやすい可能性がある（要因例: メモリ制約、前処理条件、学習率、クラス数とバッチ構成）。
  - LogRatio: small/resnet18/vgg11 は trial 11〜13 付近で改善が頭打ちになりやすい一方、resnet50 は trial 78 付近まで改善が続き、探索が難しい（あるいは成功 trial が少なく見かけ上遅い）傾向がある。

- 重要度（`optuna_hyperparameter_importance.html` に出力された Optuna 本体の importance から top3 を抜粋して確認）
  - 注意: importance は基本的に COMPLETE trial を前提に算出されるため、COMPLETE 数が少ない study（例: ArcFace/resnet18）では順位が不安定になり得る。また、非線形性・相互作用が強い場合、`optuna_trials.csv` 由来の単純な proxy（相関/平均との差）とは一致しないことがある。
  - Circle（CircleLoss）: resnet 系は `margin` が支配的で、次点が `gamma` / `epochs` になりやすい。small/vgg11 では `gamma` と `batch_size` が上位に出やすく、さらに `n_fft` / `hop_length` / `epochs` が上位に現れる（＝「損失パラメータ」だけでなく、時間周波数分解能や学習の安定性が効きやすい）。
  - ArcFace: ほぼ一貫して `arcface_s` / `arcface_m` と `epochs` が上位に来ており、ArcFace 固有のスケール/マージン設定と学習の進め方（収束までの十分な更新回数）の感度が大きい。resnet50 では `mel` が top3 に入っており、大きいモデルほど前処理（スペクトル表現の圧縮・平滑化）の差が学習安定性/収束に効きやすい可能性がある。
  - LogRatio: モデルにより傾向が分かれ、resnet50 では `eps` が突出して大きく（次いで `hop_length` / `mel`）、損失の数値安定性が探索結果を強く規定している。一方で resnet18/small/vgg11 では `batch_size` が最上位になりやすく、次いで `n_fft` / `lr` / `mel` / `eps` / `norm_p` が続く（＝表現の分解能と最適化の安定性の両方が効いている）。

- OOM リトライによる `batch_size` 解釈の注意（`effective_batch_size` と `oom_adjustments` を user_attrs から確認）
  - OOM によるバッチ半減が COMPLETE trial の中でも発生している study があり、例として Circle(vgg11) は 2 trial、ArcFace(resnet18/resnet50) は 1/3 trial、LogRatio(resnet50) は 3 trial で「suggestされた `batch_size` と実際の `effective_batch_size` が不一致」だった。
  - このため、重要度図で `batch_size` が上位に見えても、それが純粋な最適化効果なのか、OOM による成功/失敗の選択バイアスなのかは切り分けが必要である。厳密に議論する場合は、(1) `effective_batch_size` 基準での再集計、(2) gradient accumulation による疑似大バッチ化、(3) モデルごとに探索範囲を現実的なメモリ上限に合わせて制約、などが有効になる。

## 実験4: 速度回帰 (MLP) ハイパパラメータ

実験4の学習は `src/mlp/speedPrediction.py` を使用し、`scripts/exp4.sh` の実行設定とスクリプト既定値をまとめる。

| 項目 | 値 |
| --- | --- |
| optimizer | Adam |
| lr | 0.001 (既定値) |
| encoder_lr | lr と同一 (既定値) |
| epochs | 100 (scripts/exp4.sh) |
| batch_size | 32 (scripts/exp4.sh) |
| loss | mse (scripts/exp4.sh) |
| hidden_dim | 128 (既定値) |
| dropout | 0.0 (既定値) |
| freeze_encoder | false (既定値) |
| test_split | 0.2 (既定値) |
| seed | 42 (既定値) |
| representation | model依存 (small: spectrogram / vgg11/resnet: spectrogram) |
| n_fft | 1024 (既定値) |
| hop_length | small: 512 / それ以外: 160 (既定値) |
| mel | OFF (既定値) |

## 実験5: クラスラベル分類 評価結果

`outputs/optuna_studies/**/eval-*` の `metrics.json` を集約。

指標は `silhouette` / `dbi` / `nmi` と、top-1/2 の評価として `recall@1/2` と `knn_acc@1/2` を掲載。ArcFace は head 出力の Cosine (`head_top1_cos/head_top2_cos`) と ArcFace (`head_top1_arcface/head_top2_arcface`) を併記。`knn_acc@2` が未出力の場合は N/A。実データの mono では `knn_top1` も併記。

### 指標の解釈 (特に ArcFace head)

- `recall@k` / `knn_acc@k` は、埋め込み空間上の近傍検索/分類として解釈でき、メトリック学習の「表現の良さ」を直接反映しやすい。
- `silhouette` / `dbi` / `nmi` はクラスタ品質の補助的な指標で、クラス数・分布・距離尺度の影響も受けるため、単独で結論は出しにくい。
- ArcFace head について:
  - `head_top*_cos`: 正規化 embedding と正規化 class weight の単純な cosine logits による top-k。ラベルに依存せず、分類器としての精度とみなせる。
  - `head_top*_arcface`: ArcFace forward (margin + scale) の logits による top-k。forward 内で「正解ラベルの logit のみ」にマージンが適用されるため、評価時にラベルを与えると上振れしやすく、一般的な分類精度としては公平ではない（1.000 になり得る）。
  - そのため、ArcFace の性能比較は基本的に `recall@k` / `knn_acc@k` を主に見て、`head_top*_cos` は「weight と embedding の整合性確認」程度の扱いが安全。

### eval-sim / eval-real (vehicle_direction: 4分類)

| 損失 | モデル | 版 | データ | silhouette | dbi | nmi | recall@1 | recall@2 | knn_acc@1 | knn_acc@2 | head_top1_cos | head_top2_cos | head_top1_arcface | head_top2_arcface |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Circle | small | v1 | sim | 0.653 | 0.876 | 0.845 | 0.943 | 0.969 | 0.936 | N/A | N/A | N/A | N/A | N/A |
| Circle | small | v2 | sim | 0.653 | 0.876 | 0.845 | 0.943 | 0.969 | 0.936 | N/A | N/A | N/A | N/A | N/A |
| Circle | vgg11 | v1 | sim | 0.310 | 1.778 | 0.655 | 0.809 | 0.917 | 0.819 | N/A | N/A | N/A | N/A | N/A |
| Circle | resnet18 | v1 | sim | -0.137 | 4.446 | 0.109 | 0.568 | 0.733 | 0.536 | N/A | N/A | N/A | N/A | N/A |
| Circle | resnet50 | v1 | sim | -0.109 | 5.249 | 0.083 | 0.535 | 0.708 | 0.553 | N/A | N/A | N/A | N/A | N/A |
| Circle | small | v1 | real | -0.081 | 28.671 | 0.006 | 0.429 | 0.637 | 0.419 | N/A | N/A | N/A | N/A | N/A |
| Circle | small | v2 | real | -0.081 | 28.671 | 0.006 | 0.429 | 0.637 | 0.419 | N/A | N/A | N/A | N/A | N/A |
| Circle | vgg11 | v1 | real | -0.068 | 11.247 | 0.013 | 0.488 | 0.692 | 0.494 | N/A | N/A | N/A | N/A | N/A |
| Circle | resnet18 | v1 | real | -0.405 | 13.548 | 0.003 | 0.584 | 0.760 | 0.519 | N/A | N/A | N/A | N/A | N/A |
| Circle | resnet50 | v1 | real | -0.207 | 17.854 | 0.016 | 0.453 | 0.656 | 0.448 | N/A | N/A | N/A | N/A | N/A |
| ArcFace | small | v1 | sim | -0.157 | 3.030 | 0.114 | 0.329 | 0.517 | 0.364 | N/A | 0.252 | 0.509 | 1.000 | 1.000 |
| ArcFace | small | v2 | sim | -0.158 | 3.026 | 0.114 | 0.348 | 0.533 | 0.356 | N/A | 0.252 | 0.516 | N/A | N/A |
| ArcFace | vgg11 | v1 | sim | -0.039 | 2.401 | 0.193 | 0.575 | 0.738 | 0.556 | N/A | 0.333 | 0.574 | 1.000 | 1.000 |
| ArcFace | resnet18 | v1 | sim | -0.048 | 2.539 | 0.130 | 0.515 | 0.690 | 0.436 | N/A | 0.290 | 0.523 | 1.000 | 1.000 |
| ArcFace | resnet50 | v1 | sim | -0.268 | 25.978 | 0.150 | 0.484 | 0.692 | 0.431 | N/A | 0.138 | 0.372 | 1.000 | 1.000 |
| ArcFace | small | v1 | real | -0.056 | 13.763 | 0.013 | 0.414 | 0.637 | 0.348 | N/A | 0.415 | 0.867 | 1.000 | 1.000 |
| ArcFace | small | v2 | real | -0.056 | 13.744 | 0.013 | 0.416 | 0.634 | 0.370 | N/A | 0.415 | 0.868 | N/A | N/A |
| ArcFace | vgg11 | v1 | real | -0.166 | 13.845 | 0.010 | 0.504 | 0.710 | 0.431 | N/A | 0.071 | 0.180 | 1.000 | 1.000 |
| ArcFace | resnet18 | v1 | real | -0.090 | 12.162 | 0.013 | 0.507 | 0.703 | 0.454 | N/A | 0.398 | 0.582 | 1.000 | 1.000 |
| ArcFace | resnet50 | v1 | real | -0.038 | 29.692 | 0.002 | 0.500 | 0.712 | 0.426 | N/A | 0.090 | 0.142 | 1.000 | 1.000 |

#### 考察: vehicle_direction (4分類)

- sim では Circle (small) がクラスタ品質・検索性能ともに最も高い (silhouette 0.653 / nmi 0.845 / recall@1 0.943 / knn_acc@1 0.936)。同じ Circle でもモデルが大きくなるほど指標が悪化しており、loc1-6 / sim 条件では small が表現・最適化ともに相性が良い可能性がある。
- ArcFace は `head_top*_arcface` が 1.000 だが、これは前述のとおり「ラベル条件付き」で上振れするため、埋め込みの汎用性を保証しない。実際、ArcFace(small) の sim は recall@1 0.329 / knn_acc@1 0.364 と低く、近傍構造としては弱い。
- ArcFace の `head_top1_cos` が sim で 0.252 (ほぼチャンス) になる一方で recall@1 は 0.329 とわずかに上回っており、「class weight による線形分離」と「近傍構造の良さ」が一致していない。
- real では全体にクラスタ指標が崩れ、Circle(small) は dbi 28.671 / nmi 0.006 とほぼ無構造に近い。一方で近傍指標（recall@1, knn_acc@1）は 0.4〜0.5 程度を維持しており、完全に情報が消失したというより「クラス境界が重なりやすい」状況が示唆される。
- real の ArcFace(small) は `head_top1_cos` 0.415 / `head_top2_cos` 0.867 と相対的に高いが、recall@1 は 0.414、knn_acc@1 は 0.348 なので、cos head が高いこと自体は必ずしも埋め込みの近傍性能に直結しない。

### eval-sim / eval-real (location)

| 損失 | モデル | 版 | データ | silhouette | dbi | nmi | recall@1 | recall@2 | knn_acc@1 | knn_acc@2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Circle | small | v1 | sim | -0.063 | 15.121 | 0.075 | 0.613 | 0.765 | 0.592 | N/A |
| Circle | small | v2 | sim | -0.063 | 15.121 | 0.075 | 0.613 | 0.765 | 0.592 | N/A |
| Circle | vgg11 | v1 | sim | -0.038 | 11.192 | 0.128 | 0.464 | 0.634 | 0.439 | N/A |
| Circle | resnet18 | v1 | sim | -0.267 | 5.329 | 0.182 | 0.382 | 0.562 | 0.344 | N/A |
| Circle | resnet50 | v1 | sim | -0.089 | 3.670 | 0.241 | 0.395 | 0.562 | 0.358 | N/A |
| Circle | small | v1 | real | -0.093 | 6.090 | 0.051 | 0.486 | 0.705 | 0.487 | N/A |
| Circle | small | v2 | real | -0.093 | 6.090 | 0.051 | 0.486 | 0.705 | 0.487 | N/A |
| Circle | vgg11 | v1 | real | -0.085 | 12.001 | 0.041 | 0.570 | 0.768 | 0.563 | N/A |
| Circle | resnet18 | v1 | real | -0.129 | 6.451 | 0.007 | 0.615 | 0.810 | 0.549 | N/A |
| Circle | resnet50 | v1 | real | -0.055 | 12.018 | 0.002 | 0.515 | 0.723 | 0.504 | N/A |
| ArcFace | small | v1 | sim | -0.281 | 15.710 | 0.082 | 0.208 | 0.386 | 0.228 | N/A |
| ArcFace | small | v2 | sim | -0.286 | 15.755 | 0.081 | 0.210 | 0.363 | 0.228 | N/A |
| ArcFace | vgg11 | v1 | sim | -0.199 | 8.881 | 0.116 | 0.309 | 0.447 | 0.292 | N/A |
| ArcFace | resnet18 | v1 | sim | -0.100 | 8.842 | 0.128 | 0.359 | 0.485 | 0.333 | N/A |
| ArcFace | resnet50 | v1 | sim | -0.182 | 103.257 | 0.021 | 0.311 | 0.487 | 0.225 | N/A |
| ArcFace | small | v1 | real | -0.091 | 8.309 | 0.050 | 0.448 | 0.670 | 0.463 | N/A |
| ArcFace | small | v2 | real | -0.091 | 8.301 | 0.050 | 0.456 | 0.680 | 0.448 | N/A |
| ArcFace | vgg11 | v1 | real | -0.115 | 8.646 | 0.063 | 0.585 | 0.795 | 0.553 | N/A |
| ArcFace | resnet18 | v1 | real | -0.115 | 13.148 | 0.011 | 0.598 | 0.785 | 0.566 | N/A |
| ArcFace | resnet50 | v1 | real | -0.141 | 4.840 | 0.077 | 0.583 | 0.789 | 0.550 | N/A |

#### 考察: location

- 本研究の設計意図として、学習時に location を誤差（教師）として導入していない（= location を「環境条件/ヌイザンス」とみなし、そこに頑健な潜在表現を狙う）。従って、location の高い分離性能は必ずしも望ましいとは限らず、むしろ location 情報を強く保持している（環境条件を一緒に符号化している）可能性もある。
- location は全体に nmi が低く、silhouette も負の行が多い。車種/方向に比べて音響的な差分が小さい、あるいは同一 location 内のばらつきが大きい可能性がある。
- sim では Circle(small) が recall@1 0.613 / knn_acc@1 0.592 と相対的に良い。一方で ArcFace(small) は recall@1 0.208 / knn_acc@1 0.228 と低い。ここは「ArcFace が vehicle_direction に寄って location を捨てた」とも、「Circle が location も一緒に拾ってしまった」とも解釈できるため、目的（環境不変性 vs location 識別）を明確にしたうえで評価軸を決める必要がある。
- sim で location まで高く出る場合、ラベルそのものよりも収録条件・伝搬・背景雑音などの「環境特徴」が近傍を支配している（ショートカット学習）可能性がある。これは狭い sim 条件の範囲では性能が良く見えても、条件が変わる real では崩れやすい。
- real では Circle と ArcFace の差が縮まり、vgg11/resnet18 では ArcFace の recall@1・knn_acc@1 が Circle と同等かやや上回るケースがある。sim→real のドメインギャップ下では、損失関数よりもモデル構造や前処理条件（mel/FFT/hop など）の影響が支配的になっている可能性がある。

### eval-real-mono (vehicle: 2分類)

real-mono は元のラベル設計として「車種×進行方向」の4ラベルを想定しているが、明らかに進行方向の分類が悪く、車種に特化した評価も行うべき、という狙いで vehicle (2分類) を別途評価している。

| 損失 | モデル | 版 | silhouette | dbi | nmi | recall@1 | recall@2 | knn_acc@1 | knn_acc@2 | knn_top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Circle | small | v1 | -0.016 | 26.942 | 0.000 | 0.798 | 0.899 | 0.793 | 0.871 | 0.871 |
| Circle | small | v2 | -0.016 | 26.942 | 0.000 | 0.798 | 0.899 | 0.793 | 0.871 | 0.871 |
| Circle | vgg11 | v1 | 0.042 | 7.061 | 0.006 | 0.800 | 0.901 | 0.807 | 0.875 | 0.875 |
| Circle | resnet18 | v1 | -0.500 | 9.930 | 0.001 | 0.840 | 0.921 | 0.793 | 0.875 | 0.875 |
| Circle | resnet50 | v1 | -0.074 | 45.275 | 0.000 | 0.801 | 0.900 | 0.798 | 0.875 | 0.875 |
| ArcFace | small | v1 | 0.087 | 6.607 | 0.005 | 0.795 | 0.899 | 0.881 | 0.884 | 0.884 |
| ArcFace | small | v2 | 0.088 | 6.599 | 0.005 | 0.799 | 0.901 | 0.867 | 0.883 | 0.883 |
| ArcFace | vgg11 | v1 | 0.051 | 5.694 | 0.006 | 0.819 | 0.907 | 0.799 | 0.876 | 0.876 |
| ArcFace | resnet18 | v1 | -0.003 | 8.387 | 0.002 | 0.825 | 0.919 | 0.815 | 0.877 | 0.877 |
| ArcFace | resnet50 | v1 | 0.067 | 13.148 | 0.001 | 0.822 | 0.914 | 0.810 | 0.876 | 0.876 |

#### 考察: eval-real-mono (vehicle: 2分類)

- 2分類になると、Circle/ArcFace ともに recall@1 は 0.79〜0.84 と高い。これは「車種」については、mono 化後でも識別に効くスペクトル差分が残っていることを示唆する。
- 一方で、進行方向の性能が悪い主要因として、データ処理段階でステレオ音声をモノクロ（モノラル）に潰している点が大きい可能性が高い。進行方向は両耳間時間差 (ITD)・両耳間レベル差 (ILD) など空間手掛かりに依存しやすく、モノラル化で情報が落ちるためである。
- ArcFace(small) は knn_acc@1 が 0.881 / 0.867 と Circle(small) の 0.793 を上回っており、「大域的なクラスタリング（vehicle_direction/location）」よりも「二値の判別（車種のみ）」に強く寄る傾向が見える。

### eval-real-mono (location)

| 損失 | モデル | 版 | silhouette | dbi | nmi | recall@1 | recall@2 | knn_acc@1 | knn_acc@2 | knn_top1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Circle | small | v1 | -0.093 | 6.090 | 0.051 | 0.486 | 0.705 | 0.487 | 0.500 | 0.492 |
| Circle | small | v2 | -0.093 | 6.090 | 0.051 | 0.486 | 0.705 | 0.487 | 0.500 | 0.492 |
| Circle | vgg11 | v1 | -0.085 | 12.001 | 0.041 | 0.570 | 0.768 | 0.563 | 0.577 | 0.583 |
| Circle | resnet18 | v1 | -0.129 | 6.451 | 0.007 | 0.615 | 0.810 | 0.549 | 0.579 | 0.566 |
| Circle | resnet50 | v1 | -0.055 | 12.018 | 0.002 | 0.515 | 0.723 | 0.504 | 0.534 | 0.515 |
| ArcFace | small | v1 | -0.091 | 8.309 | 0.050 | 0.448 | 0.670 | 0.463 | 0.434 | 0.436 |
| ArcFace | small | v2 | -0.091 | 8.301 | 0.050 | 0.456 | 0.680 | 0.448 | 0.378 | 0.415 |
| ArcFace | vgg11 | v1 | -0.115 | 13.148 | 0.011 | 0.598 | 0.785 | 0.566 | 0.588 | 0.580 |
| ArcFace | resnet18 | v1 | -0.115 | 8.646 | 0.063 | 0.585 | 0.795 | 0.553 | 0.563 | 0.573 |
| ArcFace | resnet50 | v1 | -0.141 | 4.840 | 0.077 | 0.583 | 0.789 | 0.550 | 0.591 | 0.578 |

#### 考察: eval-real-mono (location)

- mono の location でも、recall@1 は 0.45〜0.62 程度で頭打ちになっており、vehicle に比べて難易度が高い。
- モデル/損失の優劣は一貫しないが、vgg11/resnet18 では knn 指標が比較的高く、small はやや不利に見える。location 系は局所パターンの表現力（モデル容量）が効きやすい可能性がある。

#### 実験5まとめ

- sim の vehicle_direction では Circle(small) が明確に優位。
- real はクラスタ指標が崩れやすく、近傍指標（recall/knn）を主指標として評価するのが妥当。
- ArcFace head の `*_arcface` は解釈上の上限指標に近いので、性能比較の根拠にするのは避ける。
- location は学習で誤差として導入していない（環境不変性を狙う）ため、「location が分離できない」こと自体は必ずしも失敗ではない。むしろ downstream の目的に応じて、vehicle/direction の性能と location 不変性（= location 指標の低さ）をトレードオフとして解釈するのが自然。

## 実験6: 速度回帰 評価結果 (val)

`outputs/speed_prediction` 以下の全結果を網羅。

| 事前学習/損失 | モデル | 版 | データ | MAE | RMSE | R2 | MAPE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Circle | small | v1 | real | 32.312 | 37.440 | -2.498 | 43.673 |
| Circle | small | v1 | sim | 3.817 | 5.137 | 0.933 | 8.273 |
| Circle | small | v2 | real | 32.312 | 37.440 | -2.498 | 43.673 |
| Circle | small | v2 | sim | 3.817 | 5.137 | 0.933 | 8.273 |
| Circle | vgg11 | v1 | real | 17.944 | 21.128 | -0.114 | 30.872 |
| Circle | vgg11 | v1 | sim | 5.244 | 6.610 | 0.889 | 11.112 |
| Circle | resnet18 | v1 | real | 17.062 | 20.338 | -0.032 | 28.246 |
| Circle | resnet18 | v1 | sim | 5.000 | 6.816 | 0.882 | 10.180 |
| Circle | resnet50 | v1 | real | N/A | N/A | N/A | N/A |
| Circle | resnet50 | v1 | sim | N/A | N/A | N/A | N/A |
| ArcFace | small | v1 | real | 16.932 | 20.208 | -0.019 | 24.125 |
| ArcFace | small | v1 | sim | 10.587 | 14.117 | 0.493 | 21.415 |
| ArcFace | small | v2 | real | 17.276 | 20.616 | -0.061 | 24.310 |
| ArcFace | small | v2 | sim | 10.687 | 14.216 | 0.486 | 21.555 |
| ArcFace | vgg11 | v1 | real | 18.588 | 22.623 | -0.277 | 34.132 |
| ArcFace | vgg11 | v1 | sim | 4.460 | 6.106 | 0.905 | 9.398 |
| ArcFace | resnet18 | v1 | real | 25.394 | 30.278 | -1.288 | 32.342 |
| ArcFace | resnet18 | v1 | sim | 3.285 | 4.355 | 0.952 | 6.836 |
| ArcFace | resnet50 | v1 | real | N/A | N/A | N/A | N/A |
| ArcFace | resnet50 | v1 | sim | N/A | N/A | N/A | N/A |
| LogRatio | small | v1 | real | 20.090 | 24.666 | -0.518 | 31.191 |
| LogRatio | small | v1 | sim | 7.314 | 9.314 | 0.779 | 15.352 |
| LogRatio | small | v2 | real | 20.090 | 24.666 | -0.518 | 31.191 |
| LogRatio | small | v2 | sim | 7.314 | 9.314 | 0.779 | 15.352 |
| LogRatio | vgg11 | v1 | real | 21.252 | 25.075 | -0.569 | 38.184 |
| LogRatio | vgg11 | v1 | sim | 4.330 | 5.774 | 0.915 | 8.817 |
| LogRatio | resnet18 | v1 | real | 26.298 | 31.247 | -1.436 | 33.427 |
| LogRatio | resnet18 | v1 | sim | 3.349 | 4.219 | 0.955 | 7.086 |
| LogRatio | resnet50 | v1 | real | N/A | N/A | N/A | N/A |
| LogRatio | resnet50 | v1 | sim | N/A | N/A | N/A | N/A |
| Circle->LogRatio | small | v1 | real | 57.251 | 60.884 | -8.250 | 78.073 |
| Circle->LogRatio | small | v1 | sim | 4.076 | 5.391 | 0.926 | 8.593 |
| Circle->LogRatio | small | v2 | real | 41.625 | 47.634 | -4.662 | 53.605 |
| Circle->LogRatio | small | v2 | sim | 3.806 | 5.140 | 0.933 | 7.602 |
| Circle->LogRatio | vgg11 | v1 | real | 17.933 | 21.028 | -0.103 | 31.715 |
| Circle->LogRatio | vgg11 | v1 | sim | 4.874 | 6.172 | 0.903 | 10.443 |
| Circle->LogRatio | resnet18 | v1 | real | 18.547 | 22.373 | -0.249 | 28.689 |
| Circle->LogRatio | resnet18 | v1 | sim | 3.422 | 4.553 | 0.947 | 7.262 |
| Circle->LogRatio | resnet50 | v1 | real | N/A | N/A | N/A | N/A |
| Circle->LogRatio | resnet50 | v1 | sim | N/A | N/A | N/A | N/A |
| ArcFace->LogRatio | small | v1 | real | 30.197 | 36.031 | -2.239 | 37.744 |
| ArcFace->LogRatio | small | v1 | sim | 5.797 | 7.499 | 0.857 | 11.665 |
| ArcFace->LogRatio | small | v2 | real | 42.789 | 47.173 | -4.553 | 55.829 |
| ArcFace->LogRatio | small | v2 | sim | 6.925 | 9.034 | 0.792 | 13.944 |
| ArcFace->LogRatio | vgg11 | v1 | real | 19.184 | 23.691 | -0.401 | 29.066 |
| ArcFace->LogRatio | vgg11 | v1 | sim | 3.839 | 4.816 | 0.941 | 8.089 |
| ArcFace->LogRatio | resnet18 | v1 | real | 21.330 | 25.515 | -0.624 | 28.327 |
| ArcFace->LogRatio | resnet18 | v1 | sim | 3.879 | 5.004 | 0.936 | 8.016 |
| ArcFace->LogRatio | resnet50 | v1 | real | N/A | N/A | N/A | N/A |
| ArcFace->LogRatio | resnet50 | v1 | sim | N/A | N/A | N/A | N/A |

### 考察

- sim では多くの条件で R2 が高く (例: Circle(small) 0.933, ArcFace(resnet18) 0.952, LogRatio(resnet18) 0.955)、速度回帰としては成立している。一方で ArcFace(small) は sim でも R2 が 0.49 程度と低く、事前学習表現が回帰タスクに十分転移していない可能性がある。
- real では今回の再学習後も全条件で R2 は負のままで、sim→real ギャップは依然として大きい。ただし一部条件（例: ArcFace(small v1) R2=-0.019、Circle(resnet18) R2=-0.032）は 0 に近づいており、以前のような極端な崩壊値は大幅に緩和された。
- 継続事前学習（Circle->LogRatio / ArcFace->LogRatio）は、vgg11/resnet18 では real の誤差が比較的抑えられる一方、small では R2 がより悪化する行（例: Circle->LogRatio(small v1/v2), ArcFace->LogRatio(small v2)）が残る。現状設定では「継続事前学習が常に有利」とは言えず、モデル容量ごとの学習率・凍結方針・エポック設計の再調整が必要である。
