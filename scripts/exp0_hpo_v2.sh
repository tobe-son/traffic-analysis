# 実験0: 多様体の取得HPO

# CNN-Smallのoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_autoencoder.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --n-trials 200 \
  --hpo-config ./configs/optuna_autoencoder_hpo.json \
  --storage sqlite:///outputs/hpo_autoencoder.db \
  --study-name ae_optuna_small_loc1-6v2 \
  --pruner median \
  --reset-study \
  --export-best-weights \
  --n-jobs 1

# VGG11のoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_autoencoder.py \
  --model vgg11 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --n-trials 200 \
  --hpo-config ./configs/optuna_autoencoder_hpo.json \
  --storage sqlite:///outputs/hpo_autoencoder.db \
  --study-name ae_optuna_vgg11_loc1-6v2 \
  --pruner median \
  --reset-study \
  --export-best-weights \
  --n-jobs 1

# resnet18のoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_autoencoder.py \
  --model resnet \
  --data-selection loc1-6 \
  --data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
  --main-data-dir ./data/processed/real/idmt_traffic \
  --n-trials 200 \
  --hpo-config ./configs/optuna_autoencoder_hpo.json \
  --storage sqlite:///outputs/hpo_autoencoder.db \
  --study-name ae_optuna_resnet_loc1-6v2 \
  --pruner median \
  --reset-study \
  --export-best-weights \
  --n-jobs 1