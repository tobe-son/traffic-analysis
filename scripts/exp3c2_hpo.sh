# 実験3-C: LogRatio Loss による継続事前学習による潜在空間の取得HPO

# Arcface Lossの事前学習からの継続事前学習
# CNN-Smallのoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_continuouslearning.py \
  --model small \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 200 \
  --min-epochs 10 \
  --max-epochs 1000 \
  --hpo-config ./configs/optuna_continuouslearning_hpo.json \
  --storage sqlite:///outputs/hpo_continuouslearning2.db \
  --study-name cont2_optuna_small_loc1-6_v1  \
  --init-encoder-state ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/best_encoder_small.pth \
  --export-best-weights \
  --reset-study

# VGG11のoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_continuouslearning.py \
  --model vgg11 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 200 \
  --hpo-config ./configs/optuna_continuouslearning_hpo.json \
  --storage sqlite:///outputs/hpo_continuouslearning2_vgg11.db \
  --study-name cont2_optuna_vgg11_loc1-6_v1  \
  --init-encoder-state ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
  --export-best-weights \
  --reset-study

# renet(legacy)のoptuna最適化
#MPLBACKEND=Agg python src/hyper_optimizer/ho_continuouslearning.py \
#  --model resnet \
#  --data-selection loc1-6 \
#  --data-csv ./data/processed/datasets/data_1-6.csv \
#  --main-data-dir ./data/processed/datasets \
#  --n-trials 200 \
#  --min-epochs 10 \
#  --max-epochs 1000 \
#  --hpo-config ./configs/optuna_continuouslearning_hpo.json \
#  --storage sqlite:///outputs/hpo_continuouslearning2_resnet.db \
#  --study-name cont2_optuna_resnet_loc1-6_v1  \
#  --init-encoder-state ./outputs/optuna_studies/cl_arcface_optuna_resnet_loc1-6_v1/best_encoder_resnet.pth \
#  --export-best-weights \
#  --reset-study    

# resnet18のoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_continuouslearning.py \
  --model resnet18 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 200 \
  --hpo-config ./configs/optuna_continuouslearning_hpo.json \
  --storage sqlite:///outputs/hpo_continuouslearning2_resnet18.db \
  --study-name cont2_optuna_resnet18_loc1-6_v1  \
  --init-encoder-state ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
  --export-best-weights \
  --reset-study

# resnet50のoptuna最適化
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPLBACKEND=Agg python src/hyper_optimizer/ho_continuouslearning.py \
  --model resnet50 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 200 \
  --hpo-config ./configs/optuna_continuouslearning_hpo.json \
  --storage sqlite:///outputs/hpo_continuouslearning2_resnet50.db \
  --study-name cont2_optuna_resnet50_loc1-6_v1  \
  --init-encoder-state ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
  --export-best-weights \
  --reset-study