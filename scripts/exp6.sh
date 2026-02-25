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

python src/eval/speed_prediction_eval.py \
  --model vgg11 \
  --model-weights ./outputs/speed_prediction/cl_optuna_vgg11_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_optuna_vgg11_loc1-6_v1/eval-sim/

python src/eval/speed_prediction_eval.py \
  --model resnet18 \
  --model-weights ./outputs/speed_prediction/cl_optuna_resnet18_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_optuna_resnet18_loc1-6_v1/eval-sim/

#python src/eval/speed_prediction_eval.py \
#  --model resnet50 \
#  --model-weights ./outputs/speed_prediction/cl_optuna_resnet50_loc1-6_v1/best_speed_regressor_mse.pth \
#  --loss mse \
#  --batch-size 32 \
#  --visualize \
#  --output-dir ./outputs/speed_prediction/cl_optuna_resnet50_loc1-6_v1/eval-sim/

# arcface - mlpのシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_arcface_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_small_loc1-6_v1/eval-sim/

python src/eval/speed_prediction_eval.py \
  --model vgg11 \
  --model-weights ./outputs/speed_prediction/cl_arcface_optuna_vgg11_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_vgg11_loc1-6_v1/eval-sim/

python src/eval/speed_prediction_eval.py \
  --model resnet18 \
  --model-weights ./outputs/speed_prediction/cl_arcface_optuna_resnet18_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_resnet18_loc1-6_v1/eval-sim/

#python src/eval/speed_prediction_eval.py \
#  --model resnet50 \
#  --model-weights ./outputs/speed_prediction/cl_arcface_optuna_resnet50_loc1-6_v1/best_speed_regressor_mse.pth \
#  --loss mse \
#  --batch-size 32 \
#  --visualize \
#  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_resnet50_loc1-6_v1/eval-sim/

# logratio - mlpのシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/log_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/log_optuna_small_loc1-6_v1/eval-sim/

python src/eval/speed_prediction_eval.py \
  --model vgg11 \
  --model-weights ./outputs/speed_prediction/log_optuna_vgg11_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/log_optuna_vgg11_loc1-6_v1/eval-sim/

python src/eval/speed_prediction_eval.py \
  --model resnet18 \
  --model-weights ./outputs/speed_prediction/log_optuna_resnet18_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/log_optuna_resnet18_loc1-6_v1/eval-sim/

#python src/eval/speed_prediction_eval.py \
#  --model resnet50 \
#  --model-weights ./outputs/speed_prediction/log_optuna_resnet50_loc1-6_v1/best_speed_regressor_mse.pth \
#  --loss mse \
#  --batch-size 32 \
#  --visualize \
#  --output-dir ./outputs/speed_prediction/log_optuna_resnet50_loc1-6_v1/eval-sim/

# Circle - LogRatio - mlpのシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cont_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cont_optuna_small_loc1-6_v1/eval-sim/

python src/eval/speed_prediction_eval.py \
  --model vgg11 \
  --model-weights ./outputs/speed_prediction/cont_optuna_vgg11_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cont_optuna_vgg11_loc1-6_v1/eval-sim/

python src/eval/speed_prediction_eval.py \
  --model resnet18 \
  --model-weights ./outputs/speed_prediction/cont_optuna_resnet18_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cont_optuna_resnet18_loc1-6_v1/eval-sim/

#python src/eval/speed_prediction_eval.py \
#  --model resnet50 \
#  --model-weights ./outputs/speed_prediction/cont_optuna_resnet50_loc1-6_v1/best_speed_regressor_mse.pth \
#  --loss mse \
#  --batch-size 32 \
#  --visualize \
#  --output-dir ./outputs/speed_prediction/cont_optuna_resnet50_loc1-6_v1/eval-sim/

# arcface - LogRatio - mlpのシミュレーションデータでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cont2_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cont2_optuna_small_loc1-6_v1/eval-sim/

python src/eval/speed_prediction_eval.py \
  --model vgg11 \
  --model-weights ./outputs/speed_prediction/cont2_optuna_vgg11_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cont2_optuna_vgg11_loc1-6_v1/eval-sim/

python src/eval/speed_prediction_eval.py \
  --model resnet18 \
  --model-weights ./outputs/speed_prediction/cont2_optuna_resnet18_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --output-dir ./outputs/speed_prediction/cont2_optuna_resnet18_loc1-6_v1/eval-sim/

#python src/eval/speed_prediction_eval.py \
#  --model resnet50 \
#  --model-weights ./outputs/speed_prediction/cont2_optuna_resnet50_loc1-6_v1/best_speed_regressor_mse.pth \
#  --loss mse \
#  --batch-size 32 \
#  --visualize \
#  --output-dir ./outputs/speed_prediction/cont2_optuna_resnet50_loc1-6_v1/eval-sim/

# Circle - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cl_optuna_small_loc1-6_v1/eval-real/

python src/eval/speed_prediction_eval.py \
  --model vgg11 \
  --model-weights ./outputs/speed_prediction/cl_optuna_vgg11_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cl_optuna_vgg11_loc1-6_v1/eval-real/

python src/eval/speed_prediction_eval.py \
  --model resnet18 \
  --model-weights ./outputs/speed_prediction/cl_optuna_resnet18_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cl_optuna_resnet18_loc1-6_v1/eval-real/

#python src/eval/speed_prediction_eval.py \
#  --model resnet50 \
#  --model-weights ./outputs/speed_prediction/cl_optuna_resnet50_loc1-6_v1/best_speed_regressor_mse.pth \
#  --loss mse \
#  --batch-size 32 \
#  --visualize \
#  --optuna-params ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/optuna_best.json \
#  --data-csv ./data/processed/real/vs13/vs13.csv \
#  --main-data-dir ./data/processed/real/vs13 \
#  --output-dir ./outputs/speed_prediction/cl_optuna_resnet50_loc1-6_v1/eval-real/

# arcface - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cl_arcface_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_small_loc1-6_v1/eval-real/

python src/eval/speed_prediction_eval.py \
  --model vgg11 \
  --model-weights ./outputs/speed_prediction/cl_arcface_optuna_vgg11_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_vgg11_loc1-6_v1/eval-real/

python src/eval/speed_prediction_eval.py \
  --model resnet18 \
  --model-weights ./outputs/speed_prediction/cl_arcface_optuna_resnet18_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_resnet18_loc1-6_v1/eval-real/

#python src/eval/speed_prediction_eval.py \
#  --model resnet50 \
#  --model-weights ./outputs/speed_prediction/cl_arcface_optuna_resnet50_loc1-6_v1/best_speed_regressor_mse.pth \
#  --loss mse \
#  --batch-size 32 \
#  --visualize \
#  --optuna-params ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/optuna_best.json \
#  --data-csv ./data/processed/real/vs13/vs13.csv \
#  --main-data-dir ./data/processed/real/vs13 \
#  --output-dir ./outputs/speed_prediction/cl_arcface_optuna_resnet50_loc1-6_v1/eval-real/

# logratio - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/log_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/log_optuna_small_loc1-6_v1/eval-real/

python src/eval/speed_prediction_eval.py \
  --model vgg11 \
  --model-weights ./outputs/speed_prediction/log_optuna_vgg11_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/log_optuna_vgg11_loc1-6_v1/eval-real/

python src/eval/speed_prediction_eval.py \
  --model resnet18 \
  --model-weights ./outputs/speed_prediction/log_optuna_resnet18_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/log_optuna_resnet18_loc1-6_v1/eval-real/

#python src/eval/speed_prediction_eval.py \
#  --model resnet50 \
#  --model-weights ./outputs/speed_prediction/log_optuna_resnet50_loc1-6_v1/best_speed_regressor_mse.pth \
#  --loss mse \
#  --batch-size 32 \
#  --visualize \
#  --optuna-params ./outputs/optuna_studies/log_optuna_resnet50_loc1-6_v1/optuna_best.json \
#  --data-csv ./data/processed/real/vs13/vs13.csv \
#  --main-data-dir ./data/processed/real/vs13 \
#  --output-dir ./outputs/speed_prediction/log_optuna_resnet50_loc1-6_v1/eval-real/

# Circle - LogRatio - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cont_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cont_optuna_small_loc1-6_v1/eval-real/

python src/eval/speed_prediction_eval.py \
  --model vgg11 \
  --model-weights ./outputs/speed_prediction/cont_optuna_vgg11_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cont_optuna_vgg11_loc1-6_v1/eval-real/

python src/eval/speed_prediction_eval.py \
  --model resnet18 \
  --model-weights ./outputs/speed_prediction/cont_optuna_resnet18_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cont_optuna_resnet18_loc1-6_v1/eval-real/

#python src/eval/speed_prediction_eval.py \
#  --model resnet50 \
#  --model-weights ./outputs/speed_prediction/cont_optuna_resnet50_loc1-6_v1/best_speed_regressor_mse.pth \
#  --loss mse \
#  --batch-size 32 \
#  --visualize \
#  --optuna-params ./outputs/optuna_studies/cont_optuna_resnet50_loc1-6_v1/optuna_best.json \
#  --data-csv ./data/processed/real/vs13/vs13.csv \
#  --main-data-dir ./data/processed/real/vs13 \
#  --output-dir ./outputs/speed_prediction/cont_optuna_resnet50_loc1-6_v1/eval-real/

# arcface - LogRatio - mlpの実データでの評価
python src/eval/speed_prediction_eval.py \
  --model small \
  --model-weights ./outputs/speed_prediction/cont2_optuna_small_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cont2_optuna_small_loc1-6_v1/eval-real/

python src/eval/speed_prediction_eval.py \
  --model vgg11 \
  --model-weights ./outputs/speed_prediction/cont2_optuna_vgg11_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cont2_optuna_vgg11_loc1-6_v1/eval-real/

python src/eval/speed_prediction_eval.py \
  --model resnet18 \
  --model-weights ./outputs/speed_prediction/cont2_optuna_resnet18_loc1-6_v1/best_speed_regressor_mse.pth \
  --loss mse \
  --batch-size 32 \
  --visualize \
  --data-csv ./data/processed/real/vs13/vs13.csv \
  --main-data-dir ./data/processed/real/vs13 \
  --output-dir ./outputs/speed_prediction/cont2_optuna_resnet18_loc1-6_v1/eval-real/

#python src/eval/speed_prediction_eval.py \
#  --model resnet50 \
#  --model-weights ./outputs/speed_prediction/cont2_optuna_resnet50_loc1-6_v1/best_speed_regressor_mse.pth \
#  --loss mse \
#  --batch-size 32 \
#  --visualize \
#  --optuna-params ./outputs/optuna_studies/cont2_optuna_resnet50_loc1-6_v1/optuna_best.json \
#  --data-csv ./data/processed/real/vs13/vs13.csv \
#  --main-data-dir ./data/processed/real/vs13 \
#  --output-dir ./outputs/speed_prediction/cont2_optuna_resnet50_loc1-6_v1/eval-real/

