# 実験6: 車速推定評価
# 学習に用いたのと同じ環境条件のシミュレーションデータと実データを用いて、車速推定モデルの評価を行う。

# Circle - mlpのシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1/eval-sim/

# arcface - mlpのシミュレーションデータでの評価

# logratio - mlpのシミュレーションデータでの評価

# Circle - LogRatio - mlpのシミュレーションデータでの評価

# arcface - LogRatio - mlpのシミュレーションデータでの評価

# Circle - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/optuna_best.json \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1/eval-real/

# arcface - mlpの実データでの評価

# logratio - mlpの実データでの評価

# Circle - LogRatio - mlpの実データでの評価

# arcface - LogRatio - mlpの実データでの評価

