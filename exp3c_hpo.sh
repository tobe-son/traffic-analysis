# 実験3-C: LogRatio Loss による継続事前学習による潜在空間の取得HPO

# Circle Lossの事前学習からの継続事前学習
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
  --storage sqlite:///outputs/hpo_continuouslearning.db \
  --study-name cont_optuna_small_loc1-6_v1  \
  --init-encoder-state ./output/optuna_studies/cl_optuna_small_loc1-6_v1/best_encoder_small.pth \
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
  --storage sqlite:///outputs/hpo_continuouslearning_vgg11.db \
  --study-name cont_optuna_vgg11_loc1-6_v1  \
  --init-encoder-state ./output/optuna_studies/cl_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
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
#  --storage sqlite:///outputs/hpo_continuouslearning_resnet.db \
#  --study-name cont_optuna_resnet_loc1-6_v1  \
#  --init-encoder-state ./output/optuna_studies/cl_optuna_resnet_loc1-6_v1/best_encoder_resnet.pth \
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
  --storage sqlite:///outputs/hpo_continuouslearning_resnet18.db \
  --study-name cont_optuna_resnet18_loc1-6_v1  \
  --init-encoder-state ./output/optuna_studies/cl_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
  --export-best-weights \
  --reset-study

# resnet50のoptuna最適化
MPLBACKEND=Agg python src/hyper_optimizer/ho_continuouslearning.py \
  --model resnet50 \
  --data-selection loc1-6 \
  --data-csv ./data/processed/datasets/data_1-6.csv \
  --main-data-dir ./data/processed/datasets \
  --n-trials 100 \
  --min-epochs 10 \
  --max-epochs 200 \
  --hpo-config ./configs/optuna_continuouslearning_hpo.json \
  --storage sqlite:///outputs/hpo_continuouslearning_resnet50.db \
  --study-name cont_optuna_resnet50_loc1-6_v1  \
  --init-encoder-state ./output/optuna_studies/cl_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
  --export-best-weights \
  --reset-study