# 実験4: 速度推定 (MLP回帰)
# 事前学習済み encoder weights を指定して学習

# encoder を微調整する場合
#MPLBACKEND=Agg python src/mlp/speedPrediction.py \
#  --model small \
#  --encoder-weights ./outputs/<日付>/<時刻>/best_encoder_small.pth \
#  --data-csv ./data/processed/real/vs13/vs13.csv \
#  --main-data-dir ./data/processed/real/vs13 \
#  --data-selection loc1 \
#  --epochs 100 \
#  --batch-size 32 \
#  --loss mse

# encoder を固定する場合
#MPLBACKEND=Agg python src/mlp/speedPrediction.py \
#  --model small \
#  --encoder-weights ./outputs/<日付>/<時刻>/best_encoder_small.pth \
#  --data-csv ./data/processed/real/vs13/vs13.csv \
#  --main-data-dir ./data/processed/real/vs13 \
#  --data-selection loc1 \
#  --epochs 100 \
#  --batch-size 32 \
#  --freeze-encoder \
#  --loss mse

# 事前学習（Circle Loss）からのMLP回帰学習

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model vgg11 \
  --encoder-weights ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model resnet18 \
  --encoder-weights ./outputs/optuna_studies/cl_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model resnet50 \
  --encoder-weights ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

# 事前学習（Arcface Loss）からのMLP回帰学習

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model vgg11 \
  --encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model resnet18 \
  --encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model resnet50 \
  --encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

# 事前学習（LogRatio Loss）からのMLP回帰学習

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/log_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model vgg11 \
  --encoder-weights ./outputs/optuna_studies/log_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model resnet18 \
  --encoder-weights ./outputs/optuna_studies/log_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model resnet50 \
  --encoder-weights ./outputs/optuna_studies/log_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

# 事前学習（Circle - LogRatio）からのMLP回帰学習

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/cont_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model vgg11 \
  --encoder-weights ./outputs/optuna_studies/cont_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model resnet18 \
  --encoder-weights ./outputs/optuna_studies/cont_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model resnet50 \
  --encoder-weights ./outputs/optuna_studies/cont_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

# 事前学習（Arcface - LogRatio）からのMLP回帰学習

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model small \
  --encoder-weights ./outputs/optuna_studies/cont2_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model vgg11 \
  --encoder-weights ./outputs/optuna_studies/cont2_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model resnet18 \
  --encoder-weights ./outputs/optuna_studies/cont2_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize

MPLBACKEND=Agg python src/mlp/speedPrediction.py \
  --model resnet50 \
  --encoder-weights ./outputs/optuna_studies/cont2_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
  --epochs 100 \
  --batch-size 32 \
  --loss mse \
  --visualize