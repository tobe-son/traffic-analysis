# 実験1: Circle Loss による潜在空間の取得

# CNN-Smallのoptuna最適化
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_labelclustering.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 200 \
  --min-epochs 10 \
  --max-epochs 1000 \
  --hpo-config ./configs/optuna_labelclustering_hpo.json \
  --storage sqlite:///outputs/hpo_labelclustering.db \
  --study-name cl_optuna_small_loc1-6_v1 \
  --reset-study

# NOTE: 以前の study を同じ名前で再開したい場合、探索空間(JSON)を変更すると Optuna が
# "CategoricalDistribution does not support dynamic value space" で停止します。
# その場合は (A) study 名を変える (推奨) / (B) `--reset-study` を付けて削除してやり直してください。

# VGG11のoptuna最適化
  MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_labelclustering.py \
  --model vgg11 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 200 \
  --hpo-config ./configs/optuna_labelclustering_hpo.json \
  --storage sqlite:///outputs/hpo_labelclustering.db \
  --study-name cl_optuna_vgg11_loc1-6_v1 \
  --reset-study

# resnet(legacy)のoptuna最適化
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_labelclustering.py \
  --model resnet \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 200 \
  --hpo-config ./configs/optuna_labelclustering_hpo.json \
  --storage sqlite:///outputs/hpo_labelclustering.db \
  --study-name cl_optuna_resnet_loc1-6_v1 \
  --reset-study

  # resnet18のoptuna最適化
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_labelclustering.py \
  --model resnet18 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 200 \
  --hpo-config ./configs/optuna_labelclustering_resnet18_hpo.json \
  --storage sqlite:///outputs/hpo_labelclustering.db \
  --study-name cl_optuna_resnet18_loc1-6_v1 \
  --reset-study \
  --oom-retry-max 5 --oom-min-batch-size 4 \
  --amp

  # resnet50のoptuna最適化
MPLBACKEND=Agg /home/tobeson/miniconda3/envs/traf_ana/bin/python src/hyper_optimizer/ho_labelclustering.py \
  --model resnet50 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 200 \
  --hpo-config ./configs/optuna_labelclustering_hpo.json \
  --storage sqlite:///outputs/hpo_labelclustering.db \
  --study-name cl_optuna_resnet50_loc1-6_v1 \
  --reset-study