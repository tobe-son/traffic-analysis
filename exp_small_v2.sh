#個別実行などやること

# 実験1: Circle Loss による潜在空間の取得HPO
# CNN-Smallのoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_labelclustering.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 1000 \
  --hpo-config ./configs/optuna_labelclustering_hpo.json \
  --storage sqlite:///outputs/hpo_labelclusteringv2.db \
  --study-name cl_optuna_small_loc1-6_v2 \
  --export-best-weights  \
  --reset-study

# 実験2: ArcFace Loss による潜在空間の取得HPO
# CNN-Smallのoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_labelclustering_arcface.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 1000 \
  --hpo-config ./configs/optuna_labelclustering_arcface_hpo.json \
  --storage sqlite:///outputs/hpo_labelclustering_arcfacev2.db \
  --study-name cl_arcface_optuna_small_loc1-6_v2  \
  --export-best-weights \
  --reset-study \
  --n-jobs 1

# 実験3: LogRatio Loss による潜在空間の取得HPO
# CNN-Smallのoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_continuouslearning.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 1000 \
  --hpo-config ./configs/optuna_continuouslearning_hpo.json \
  --storage sqlite:///outputs/hpo_logratiolearningv2.db \
  --study-name log_optuna_small_loc1-6_v2  \
  --export-best-weights \
  --reset-study

# 実験3-C: LogRatio Loss による継続事前学習による潜在空間の取得HPO
# Circle Lossの事前学習からの継続事前学習
# CNN-Smallのoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_continuouslearning.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 1000 \
  --hpo-config ./configs/optuna_continuouslearning_hpo.json \
  --storage sqlite:///outputs/hpo_continuouslearningv2.db \
  --study-name cont_optuna_small_loc1-6_v2  \
  --init-encoder-state ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/best_encoder_small.pth \
  --export-best-weights \
  --reset-study

# 実験3-C2: LogRatio Loss による継続事前学習による潜在空間の取得HPO

# Arcface Lossの事前学習からの継続事前学習
# CNN-Smallのoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_continuouslearning.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 1000 \
  --hpo-config ./configs/optuna_continuouslearning_hpo.json \
  --storage sqlite:///outputs/hpo_continuouslearning2v2.db \
  --study-name cont2_optuna_small_loc1-6_v2  \
  --init-encoder-state ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/best_encoder_small.pth \
  --export-best-weights \
  --reset-study

# 実験4:速度推定 (MLP回帰)
# 事前学習（Circle Loss）からのMLP回帰学習
MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_optuna_small_loc1-6_v2

# 事前学習（Arcface Loss）からのMLP回帰学習
MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_small_loc1-6_v2

# 事前学習（LogRatio Loss）からのMLP回帰学習
MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/log_optuna_small_loc1-6_v2/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize \
  --output-dir ./outputs/speed_prediction/log_optuna_small_loc1-6_v2

# 事前学習（Circle - LogRatio）からのMLP回帰学習
MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/cont_optuna_small_loc1-6_v2/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize \
  --output-dir ./outputs/speed_prediction/cont_optuna_small_loc1-6_v2

# 事前学習（Arcface - LogRatio）からのMLP回帰学習
MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/cont2_optuna_small_loc1-6_v2/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize \
  --output-dir ./outputs/speed_prediction/cont2_optuna_small_loc1-6_v2

# 実験5: 車種分類評価
# 学習に用いたのと同じ環境条件のシミュレーションデータと実データを用いて、車種分類モデルの評価を行う。
# Circle Lossでの事前学習のシミュレーションデータでの評価
python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/datasets/data_1-6.csv \
--main-data-dir ./data/processed/datasets \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/eval-sim/

# Arcface Lossでの事前学習のシミュレーションデータでの評価
python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/datasets/data_1-6.csv \
--main-data-dir ./data/processed/datasets \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/best_metric_fc_small.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/eval-sim/

# Circle Lossでの事前学習の実データでの評価
python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/eval-real/

# Arcface Lossでの事前学習の実データでの評価
python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/best_metric_fc_small.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/eval-real/

# 実データのモノラル評価：車種の2分類のみ。方向は識別しない。
#Circle Lossでの事前学習の実データでの評価（モノラル）
python -m src.eval.deep_metric_eval_mono \
--encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/eval-real-mono/

# Arcface Lossでの事前学習の実データでの評価（モノラル）
python -m src.eval.deep_metric_eval_mono \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/best_metric_fc_small.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/eval-real-mono/

# 実験6: 車速推定評価
# 学習に用いたのと同じ環境条件のシミュレーションデータと実データを用いて、車速推定モデルの評価を行う。

# Circle - mlpのシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_optuna_small_loc1-6_v2/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_optuna_small_loc1-6_v2/eval-sim/

# arcface - mlpのシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_arcface_optuna_small_loc1-6_v2/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_small_loc1-6_v2/eval-sim/

# logratio - mlpのシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/log_optuna_small_loc1-6_v2/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/log_optuna_small_loc1-6_v2/eval-sim/

# Circle - LogRatio - mlpのシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cont_optuna_small_loc1-6_v2/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cont_optuna_small_loc1-6_v2/eval-sim/

# arcface - LogRatio - mlpのシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cont2_optuna_small_loc1-6_v2/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cont2_optuna_small_loc1-6_v2/eval-sim/

# Circle - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_optuna_small_loc1-6_v2/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v2/optuna_best.json \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cl_optuna_small_loc1-6_v2/eval-real/

# arcface - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_arcface_optuna_small_loc1-6_v2/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --optuna-params ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v2/optuna_best.json \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_small_loc1-6_v2/eval-real/

# logratio - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/log_optuna_small_loc1-6_v2/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --optuna-params ./outputs/optuna_studies/log_optuna_small_loc1-6_v2/optuna_best.json \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/log_optuna_small_loc1-6_v2/eval-real/

# Circle - LogRatio - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cont_optuna_small_loc1-6_v2/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --optuna-params ./outputs/optuna_studies/cont_optuna_small_loc1-6_v2/optuna_best.json \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cont_optuna_small_loc1-6_v2/eval-real/

# arcface - LogRatio - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cont2_optuna_small_loc1-6_v2/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --optuna-params ./outputs/optuna_studies/cont2_optuna_small_loc1-6_v2/optuna_best.json \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cont2_optuna_small_loc1-6_v2/eval-real/