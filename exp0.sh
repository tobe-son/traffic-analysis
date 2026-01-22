# 実験0: 多様体の取得

# CNN-Smallのoptuna最適化
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_autoencoder.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 200 \
  --hpo-config ./configs/optuna_autoencoder_hpo.json \
  --storage sqlite:///outputs/hpo_autoencoder.db \
  --study-name optuna_small_loc1-6 \
  --pruner median \
  --n-jobs 1

# VGG11のoptuna最適化
  MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_autoencoder.py \
  --model vgg11 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 200 \
  --hpo-config ./configs/optuna_autoencoder_hpo.json \
  --storage sqlite:///outputs/hpo_autoencoder.db \
  --study-name optuna_vgg11_loc1-6 \
  --pruner median \
  --n-jobs 1

# resnetのoptuna最適化
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_autoencoder.py \
  --model resnet \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 200 \
  --hpo-config ./configs/optuna_autoencoder_hpo.json \
  --storage sqlite:///outputs/hpo_autoencoder.db \
  --study-name optuna_resnet_loc1-6 \
  --pruner median \
  --n-jobs 1