python [ho_autoencoder.py](http://_vscodecontentref_/0) --model small --n-trials 20 --min-epochs 10 --max-epochs 40 --pruner median

python src/hyper_optimizer/ho_autoencoder.py: Optuna を使ってオートエンコーダのハイパラ探索を実行するスクリプト。
--model small: CNN_any.py の MODEL_REGISTRY から small モデルを選択。
--n-trials 20: 試行を 20 回まわす。
--min-epochs 10 --max-epochs 40: 各試行で学習エポック数をこの範囲からサンプリング。
--pruner median: Optuna の MedianPruner で早期打ち切りを有効化（初期数試行・ステップはデフォルト設定）。
補足オプション（必要なら）:

--storage sqlite:///hpo.db --study-name ae_hpo: 途中再開や結果保存に便利。
--representation / --mel: 入力表現やメル使用を固定したいときに上書き。
--n-jobs: 並列試行数（GPU/CPUリソースに合わせて）。
実行後、コンソールに最良 trial の損失とパラメータ、出力ディレクトリが表示されます。