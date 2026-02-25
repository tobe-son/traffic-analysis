# 交通騒音からの環境条件に頑健な潜在表現学習による車両認識と速度推定

このリポジトリは、今年度の卒業研究「交通騒音からの環境条件に頑健な潜在表現学習による車両認識と速度推定」の研究で使用したプログラムを公開するためのものです。
本研究は、昨年度の卒業研究「環境音の特徴を捉える潜在空間表現の設計」の研究を引き継いでいます。

本コードは、論文に記載した実験結果の再現性を担保する目的で公開されていますが、研究プロトタイプとしての提供であり、今後のメンテナンスやサポートは行いません。  

## プロジェクトの目的

本プロジェクトは、従来の環境音解析手法が抱える録音条件依存やノイズの影響を克服し、環境条件に左右されない安定した特徴抽出を実現するための深層学習手法の検証を目的としています。具体的には、深層距離学習を用いて、環境音から低次元の潜在空間表現を獲得します。さらに、実環境で録音された音声を潜在空間に投射してシステムの有用性を評価します。

## 背景

従来の環境音解析では、ノイズや録音環境の違いが大きな障壁となっており、直接的な特徴抽出が難しいという課題がありました。昨年度の研究では、CNN を用いた特徴抽出と深層距離学習を組み合わせることで、環境条件に依存しない特徴空間を取得することを試みました。本研究では、昨年度の研究で使用したアーキテクチャを改良し、ハイパーパラメータの調整も試みました。さらにモニタリングシステムとしての有用性の評価を試みました。

## 研究概要

本プロジェクトは、第一段階として、環境音から交通監視に有用な特徴を抽出し、深層距離学習を用いて潜在空間を取得と速度予測モデルの構築を行いました。実験は以下の4種類を行い、プログラム実装面での詳細な検証を行いました。

- **実験１： Circle Loss による潜在空間の取得**  
   車種（car, cv）および進行方向（right, left）のラベルを用い、4種のCNN系エンコーダで低次元表現を学習しました。Adamを用いて300エポックで学習し、t-SNEによる可視化で4クラスタ（car_left, car_right, cv_left, cv_right）が明確に分離されることを確認しました。

- **実験２： Arcface Loss による潜在空間の取得**  
   実験1と同様にArcface Lossを用いて低次元表現を学習しました。Adamを用いて学習し、t-SNEによる可視化で4クラスタ（car_left, car_right, cv_left, cv_right）が明確に分離されることを確認しました。

- **実験３： Log-ratio Loss による潜在空間の取得**
    Circle LossとArcface Lossで事前学習したモデルに対し、連続値の速度ラベルを反映させるためLog-ratio Lossを適用。Adamで学習し、t-SNEによる可視化で速度情報が保持された潜在空間が形成されることを確認しました。

- **実験４： 速度予測モデルの構築**  
   Log-ratio Lossで得られた潜在空間の特徴を活用し、CNNエンコーダの出力を全結合層で64次元に圧縮後、DNN（構成：64→128→1）を用いて速度を回帰予測するモデルを構築しました。Adam（学習率1e-3、バッチサイズ32）で100エポック学習した結果、事前学習済みCNNの有用性が示され、速度予測が可能であることが確認されました。

第二段階として、作成したモデルに実データを適用して、システムの有用性を検証しました。実験は以下の2種類を行いました。
- **実験５： IDMTデータセットを用いた車種分類評価**  
   事前学習済みエンコーダ（CircleLoss / ArcFace）を IDMT 実データに適用し、車種分類性能（Silhouette係数・Recall@k・k-NN精度等）をシミュレーションデータと比較評価しました。シミュレーションで学習した潜在空間が実データにも有効であることを確認しました。

- **実験６： vs13データセットを用いた速度推定評価**  
   事前学習済みエンコーダに MLP 回帰ヘッドを組み合わせた速度予測モデルを VS13 実データで評価し、各事前学習手法（CircleLoss / ArcFace / LogRatio / 継続学習）の速度推定精度を比較しました。

## 主な機能とモジュール

本プログラムは、以下の主要なモジュールで構成されています。

- **src/**
  学習に関連する主要なプログラム群を収めたフォルダ
  - **autoencoder/**
    - *auto-encoder.py*：実験0：1次元オートエンコーダを学習させるためのプログラム（論文には記載せず、前段階実験用として実施）  
    - *CNN_xxx.py*：モデルの構造を定義するクラスを提供  
      - *CNN_o*：DCASEコンペティションのベースラインに近いCNNエンコーダモデルを学習させるプログラム（前段階実験用）  
      - *CNN_s*：CNNオートエンコーダモデル（CNN_oより大規模）の学習プログラム（前段階実験用）
      - *CNN_resnet*：CNNオートエンコーダResnetを定義
      - *CNN_VGG11*：CNNオートエンコーダVGG11を定義
      - *CNN_any*：実験で用いるCNNオートエンコーダすべてを定義、引数で使用するモデルを可変
  - **encoder/**
    実験1以降で使用するモデルを定義
    - *base_model.py*：1次元畳み込みエンコーダ、CNN_s、CNN_oを定義
    - *new_model.py*：その他実験で用いるエンコーダを定義
  - **hyper_optimizer/**
    実験2, 3, 4のハイパパラメータの最適化用の探索スクリプト
    - 詳細は[資料](./docs/optuna.md)参照
  - **learn_tool/**
    深層距離学習の実行を補助するプログラム
  - **loss/**
    実験1, 2, 3で使用する深層距離学習用の損失関数を配置するフォルダ
  - **metric/**
    実験1, 2, 3の深層距離学習の学習スクリプト
    - *LabelClustering.py*：実験1：Circle Lossを用いた深層距離学習スクリプト
    - *LabelClustering_arcface.py*：実験2：Arcface Lossを用いた深層距離学習スクリプト
    - *ContinuousLearning.py*：実験3：連続値ラベルに対応した誤差関数を用いた深層距離学習スクリプト
  - **mlp/**
    - *speedPrediction.py*：実験4：速度予測モデルを学習させるためスクリプト
  - **real_data_tool/**
    実データを処理するためのスクリプト
    - 詳細は[資料](./src/real_data_tool/README.md)参照
  - **sim_data_tool/**
    シミュレーションデータを処理するためのスクリプト
  - **eval/**
    実験5・6で使用する評価スクリプト
    - *deep_metric_eval.py*：実験5：エンコーダの車種分類性能を評価するスクリプト（4クラス: 車種×方向）
    - *deep_metric_eval_mono.py*：実験5：実データでの車種2クラス評価スクリプト（方向なし）
    - *deep_metric_eval_sim2real_tsne.py*：シミュレーション→実データの潜在空間可視化スクリプト
    - *speed_prediction_eval.py*：実験6：速度予測モデルを評価するスクリプト
- **data/**
  学習に使用するデータを格納するフォルダ
- **configs/**
  HPOで探索するハイパパラメータの範囲の指定するjsonファイルを収めるフォルダ

## 環境構築と実行方法

以下の手順で環境をセットアップし、プログラムを実行してください。

> 2026-01 更新: 本リポジトリの最新の再現手順（simulation→Optuna→実データ推論/評価）は
> `docs/workflow_commands.md` を参照してください。
> Optuna の探索空間を JSON で管理する方法（AutoEncoder / CircleLoss）は `docs/optuna.md` を参照してください。
> テンプレ: `configs/optuna_autoencoder_hpo.json`, `configs/optuna_labelclustering_hpo.json`

1. **リポジトリのクローン**  
    ```bash
    git clone https://github.com/tobe-son/traffic-analysis.git
    cd traffic-analysis
    ```

2. **依存パッケージのインストール**  
   **実行環境:**  
   - CUDA 12.1
   - cuDNN 9.5.1  
   - Python 3.10
   - PyTorch 2.5.1

   **環境構築手順:**  
   ```bash
   conda env create -f environment.yml
   conda activate traf_ana
   ```
   ただし、Condaの環境が導入され、CUDAのバージョンは12.1以上であることが前提です。

3. **プログラムと学習データのダウンロード**  
   - 深層距離学習の損失関数（CircleLoss）は、[Githubページ](https://github.com/TinyZeaMays/CircleLoss)からダウンロードし、`src/loss/`ディレクトリに`circle_loss.py`として配置してください。  
   - 深層距離学習の損失関数（LogRatioLoss）は、[Githubページ](https://github.com/sung-yeon-kim/Beyond-Binary-Supervision-CVPR19)からダウンロードし、`main.py`、`utils.py`、`LogRatioLoss.py`を`src/loss/`ディレクトリに配置してください。  
   - 学習データは、[Zenodo](https://zenodo.org/records/10700792)から`simulation.zip`をダウンロードして解凍し、`loc1`～`loc6`のフォルダを`data/raw/simulation/`に配置してください。
   - 実データ（IDMT Traffic）は[IDMTデータセット](https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html)からダウンロードし、`data/raw/IDMT_Traffic/`に配置してください。
   - 実データ（VS13）は[Zenodo](https://zenodo.org/records/7551553)からダウンロードして解凍し、`data/raw/VS13/`に配置してください。

4. **シミュレーションデータの準備**  
   - 以下のコマンドを実行して、走行音が最も大きい6秒間のデータをトリミングします。  
     ```bash
     python src/sim_data_tool/cut.py
     ```  
   - 次に、以下のコマンドを実行して、`loc1`～`loc6`のメタデータを統合したメタファイルを作成します。  
     ```bash
     python src/sim_data_tool/combain.py
     ```

5. **実データの準備**
   ※詳細はsrc/real_data_tool/README.mdを参照してください。
   - まず、IDMTデータセットを `learn_tool.settings.prepare_dataloader` が読める形式へ変換します。
     ```bash
     MPLBACKEND=Agg python src/real_data_tool/prepare_real_data.py idmt \
     --raw-dir ./data/raw/IDMT_Traffic \
     --out-dir ./data/processed/real/idmt_traffic \
     --csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
     --duration 6.0 \
     --sampling-rate 16000
     ```  
   - 次に、VS13データセットを `learn_tool.settings.prepare_dataloader` が読める形式へ変換します。
     ```bash
     MPLBACKEND=Agg python src/real_data_tool/prepare_real_data.py vs13 \
     --raw-dir ./data/raw/VS13 \
     --out-dir ./data/processed/real/vs13 \
     --csv ./data/processed/real/vs13/vs13.csv \
     --duration 6.0 \
     --sampling-rate 16000
     ```  

6. **プログラムの実行（執筆中）**  
   - **実験０（多様体取得）**  
     各実験前に、対象データ、可視化手法、ハイパーパラメータ等のパラメータ定義を必要に応じて変更してください。  
     - 学習実行例
       - 1次元オートエンコーダの学習:  
         ```bash
          MPLBACKEND=Agg python src/autoencoder/CNN_any.py \
          --model wave1d \
          --data-selection loc1-6 \
          --data-csv ./data/processed/datasets/data_1-6.csv \
          --main-data-dir ./data/processed/datasets \
          --epochs 30 \
          --batch-size 32 \
          --lr 1e-3 \
          --visualization PCA
         ```  
       - DCASE2024のベースラインモデルに近いCNNエンコーダモデルの学習:  
         ```bash
         MPLBACKEND=Agg python src/autoencoder/CNN_any.py \
         --model original \
         --data-selection loc1-6 \
         --data-csv ./data/processed/datasets/data_1-6.csv \
         --main-data-dir ./data/processed/datasets \
         --epochs 30 \
         --batch-size 32 \
         --lr 1e-3 \
         --visualization PCA
         ```  
       - 昨年度研究で主に使用したCNNエンコーダモデルの学習:  
         ```bash
         MPLBACKEND=Agg python src/autoencoder/CNN_any.py \
         --model small \
         --data-selection loc1-6 \
         --data-csv ./data/processed/datasets/data_1-6.csv \
         --main-data-dir ./data/processed/datasets \
         --epochs 30 \
         --batch-size 32 \
         --lr 1e-3 \
         --visualization t-sne
         ```
      - HPO探索実行
        - 全モデルで探索（bashにまとめられている）
          ```bash
          bash ./scripts/exp0_hpo.sh
          ```

   - **実験１（CircleLossによる深層距離学習）**  
     車種と進行方向のラベルを用いてデータの分離を行います。
      - 学習実行例
        - 昨年度研究で使用したCNNモデルの学習:  
          ```bash
          MPLBACKEND=Agg python src/metric/LabelClustering.py \
          --model small \
          --data-selection loc1-6 \
          --data-csv ./data/processed/datasets/data_1-6.csv \
          --main-data-dir ./data/processed/datasets \
          --epochs 100 \
          --batch-size 64 \
          --visualization t-SNE
          ```  
      - HPO探索実行
        - 全モデルで探索（bashにまとめられている）
          ```bash
          bash ./scripts/exp1_hpo.sh
          ```

   - **実験２（ArcfaceLossによる深層距離学習）**
     車種と進行方向のラベルを用いてデータの分離を行います。
      - 学習実行例
        - 昨年度研究で使用したCNNモデルの学習:  
          ```bash
          MPLBACKEND=Agg python src/metric/LabelClustering_arcface.py \
          --model small \
          --data-selection loc1-6 \
          --data-csv ./data/processed/datasets/data_1-6.csv \
          --main-data-dir ./data/processed/datasets \
          --epochs 100 \
          --batch-size 64 \
          --visualization t-SNE
          ```  
      - HPO探索実行
        - 全モデルで探索（bashにまとめられている）
          ```bash
          bash ./scripts/exp2_hpo.sh
          ```

   - **実験３（LogRatioLossによる連続情報を保持した潜在空間の取得）**  
     速度ラベルに基づく連続情報を反映させる深層距離学習を行います。
      - HPO探索実行
        - 新規にLogRatioLossで全モデルを探索（bashにまとめられている）
          ```bash
          bash ./scripts/exp3_hpo.sh
          ```
        - CircleLoss事前学習からの継続学習パラメータを全モデルで探索
          ```bash
          bash ./scripts/exp3c_hpo.sh
          ```
        - ArcFaceLoss事前学習からの継続学習パラメータを全モデルで探索
          ```bash
          bash ./scripts/exp3c2_hpo.sh
          ```

   - **実験４（速度予測モデルの学習）**  
     事前学習済みの潜在空間を利用して、速度予測モデルの学習を行います。全モデル・全事前学習の組み合わせをまとめて実行できます。
     ```bash
     bash ./scripts/exp4.sh
     ```
     個別に実行する場合は `src/mlp/speedPrediction.py` を使用します。詳細は[コマンド集](./docs/workflow_commands.md)を参照してください。

   - **実験５（IDMTデータセットを用いた車種分類評価）**  
     事前学習済みエンコーダを IDMT 実データに適用して、車種分類性能を評価します。
     ```bash
     bash ./scripts/exp5.sh
     ```

   - **実験６（vs13データセットを用いた速度推定評価）**  
     学習済み速度予測モデルを VS13 実データに適用して、速度推定精度を評価します。
     ```bash
     bash ./scripts/exp6.sh
     ```

7. **その他のプログラム**
   - t-SNEによる可視化
      学習済みエンコーダの重みと対応するデータCSVを用いて、`src/metric/visualize_labelclustering_tsne.py` でt-SNEによる可視化が行えます。
      詳細は[Optunaドキュメント](./docs/optuna.md)を参照してください。
     ```bash
     MPLBACKEND=Agg python src/metric/visualize_labelclustering_tsne.py \
       --model small \
       --encoder-weights ./outputs/optuna_studies/<study_name>/best_encoder_small.pth \
       --data-selection loc1-6 \
       --data-csv ./data/processed/datasets/data_1-6.csv \
       --main-data-dir ./data/processed/datasets \
       --visualization t-SNE
     ```

## 注意事項

本プログラムは研究目的で公開しているものであり、実運用を前提としたものではありません。動作保証やサポートは行っておりませんので、あらかじめご了承ください。

## 関連研究

本研究では、環境音の研究コミュニティーであるDCASEが2024年に主催した DCASE 2024 Task10 を参考にしています。以下に示すGitHubリポジトリとは互換性があり、１つのフォルダに統合してプログラムを動作させることができます。
- [ホームページ](https://dcase.community/challenge2024/task-acoustic-based-traffic-monitoring)
- [GitHub](https://github.com/boschresearch/acoustic-traffic-simulation-counting)

- [IDMTデータセット](https://www.idmt.fraunhofer.de/en/publications/datasets/traffic.html)

## ライセンス

本リポジトリ内のプログラムの著作権はlossを除きすべて tobe-son に帰属します。ただし、外部からダウンロードしたファイルは、各公式ページに記載のライセンス条件に従ってご利用ください。

プログラムの実行に必要なコードのダウンロード先とライセンスを示します。
- [circle_loss.py](https://github.com/TinyZeaMays/CircleLoss)（非公式実装・ライセンス未記載）
- [LogRatioLoss.py](https://github.com/sung-yeon-kim/Beyond-Binary-Supervision-CVPR19)（MITライセンス）
- DCASE 2024 Challenge Task 10 Development Dataset [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10700792.svg)](https://doi.org/10.5281/zenodo.10700792)
（CC BY-NC-SA 4.0）

## 参考文献
[1] 平野篤:「画像認識型交通量観測装置(AI カメラ)を活用した初の一般交通量調査の実施について」, 九州技報 第72号, トピックス, 2023, https://www.qsr.mlit.go.jp/site_files/file/n-shiryo/r4kenkyu/4-01.pdf.

[2] M. Crocco, et al., "Audio Surveillance: A Systematic Review," ACM Computing Surveys, 2016, Art. no. 52.

[3] Y. Sun, et al., "Circle Loss: A Unified Perspective of Pair Similarity Optimization," in Proc. CVPR, 2020, pp. 6398-6407.

[4] J. Deng, et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in Proc. CVPR, 2019, pp. 4690-4699.

[5] S. Kim, et al., "Deep Metric Learning Beyond Binary Supervision," in Proc. CVPR, 2019, pp. 2283-2292.

[6] T. Akiba, et al., "Optuna: A Next-generation Hyperparameter Optimization Framework," in Proc. KDD, 2019, pp. 2623-2631.

[7] DCASE community: "Acoustic-Based Traffic Monitoring", DCASEHP, https://dcase.community/challenge2024/task-acoustic-based-traffic-monitoring, 参照日: 2025-11-17.

[8] J. Abeßer, et al., "IDMT-Traffic: An Open Benchmark Dataset for Acoustic Traffic Monitoring Research," in Proc. EUSIPCO, 2021, pp. 551-555.

[9] S. Djukanović, et al., "A dataset for audio-video based vehicle speed estimation," in Proc. 2022 30th Telecommunications Forum (TELFOR), 2022, pp. 1-4.
