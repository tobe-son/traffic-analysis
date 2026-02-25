# 実験5: 車種分類評価
# 学習に用いたのと同じ環境条件のシミュレーションデータと実データを用いて、車種分類モデルの評価を行う。

# Circle Lossでの事前学習のシミュレーションデータでの評価

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/datasets/data_1-6.csv \
--main-data-dir ./data/processed/datasets \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/eval-sim/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
--model vgg11 \
--data-csv ./data/processed/datasets/data_1-6.csv \
--main-data-dir ./data/processed/datasets \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/eval-sim/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
--model resnet18 \
--data-csv ./data/processed/datasets/data_1-6.csv \
--main-data-dir ./data/processed/datasets \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_resnet18_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_resnet18_loc1-6_v1/eval-sim/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
--model resnet50 \
--data-csv ./data/processed/datasets/data_1-6.csv \
--main-data-dir ./data/processed/datasets \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/eval-sim/

# Arcface Lossでの事前学習のシミュレーションデータでの評価

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/datasets/data_1-6.csv \
--main-data-dir ./data/processed/datasets \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/best_metric_fc_small.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/eval-sim/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
--model vgg11 \
--data-csv ./data/processed/datasets/data_1-6.csv \
--main-data-dir ./data/processed/datasets \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/best_metric_fc_vgg11.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/eval-sim/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
--model resnet18 \
--data-csv ./data/processed/datasets/data_1-6.csv \
--main-data-dir ./data/processed/datasets \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/best_metric_fc_resnet18.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/eval-sim/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
--model resnet50 \
--data-csv ./data/processed/datasets/data_1-6.csv \
--main-data-dir ./data/processed/datasets \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/best_metric_fc_resnet50.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/eval-sim/

# Circle Lossでの事前学習の実データでの評価

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/eval-real/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
--model vgg11 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/eval-real/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
--model resnet18 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_resnet18_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_resnet18_loc1-6_v1/eval-real/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
--model resnet50 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/eval-real/

# Arcface Lossでの事前学習の実データでの評価

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/best_metric_fc_small.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/eval-real/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
--model vgg11 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/best_metric_fc_vgg11.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/eval-real/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
--model resnet18 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/best_metric_fc_resnet18.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/eval-real/

python -m src.eval.deep_metric_eval \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
--model resnet50 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/best_metric_fc_resnet50.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/eval-real/

# 実データのモノラル評価：車種の2分類のみ。方向は識別しない。
#Circle Lossでの事前学習の実データでの評価（モノラル）
python -m src.eval.deep_metric_eval_mono \
--encoder-weights ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_small_loc1-6_v1/eval-real-mono/

python -m src.eval.deep_metric_eval_mono \
--encoder-weights ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
--model vgg11 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_vgg11_loc1-6_v1/eval-real-mono/

python -m src.eval.deep_metric_eval_mono \
--encoder-weights ./outputs/optuna_studies/cl_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
--model resnet18 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_resnet18_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_resnet18_loc1-6_v1/eval-real-mono/

python -m src.eval.deep_metric_eval_mono \
--encoder-weights ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
--model resnet50 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--dimension 2 \
--optuna-params ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_optuna_resnet50_loc1-6_v1/eval-real-mono/

# Arcface Lossでの事前学習の実データでの評価（モノラル）
python -m src.eval.deep_metric_eval_mono \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/best_encoder_small.pth \
--model small \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/best_metric_fc_small.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_small_loc1-6_v1/eval-real-mono/

python -m src.eval.deep_metric_eval_mono \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/best_encoder_vgg11.pth \
--model vgg11 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/best_metric_fc_vgg11.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_vgg11_loc1-6_v1/eval-real-mono/

python -m src.eval.deep_metric_eval_mono \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/best_encoder_resnet18.pth \
--model resnet18 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/best_metric_fc_resnet18.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_resnet18_loc1-6_v1/eval-real-mono/

python -m src.eval.deep_metric_eval_mono \
--encoder-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/best_encoder_resnet50.pth \
--model resnet50 \
--data-csv ./data/processed/real/idmt_traffic/idmt_traffic.csv \
--main-data-dir ./data/processed/real/idmt_traffic \
--data-selection loc1-6 \
--loss-type arcface \
--dimension 2 \
--metric-fc-weights ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/best_metric_fc_resnet50.pth \
--optuna-params ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/optuna_best.json \
--output-dir ./outputs/optuna_studies/cl_arcface_optuna_resnet50_loc1-6_v1/eval-real-mono/
