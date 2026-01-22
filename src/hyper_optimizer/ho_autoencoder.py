"""Hyperparameter search for CNN autoencoders using Optuna.

This script reuses the training pipeline in ``src/autoencoder/CNN_any.py`` and
runs Optuna to minimise validation loss. Each trial saves its own outputs under
``./outputs`` via ``output_settings``.
"""

import argparse
import sys
from pathlib import Path

import optuna

# Allow imports from the src directory
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from autoencoder import CNN_any
from hyper_optimizer.hpo_space import apply_hpo_space, load_hpo_space
from hyper_optimizer.optuna_artifacts import default_fallback_output_dir, export_study


def parse_hpo_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna-based HPO for CNN autoencoders")
    parser.add_argument("--model", choices=CNN_any.MODEL_REGISTRY.keys(), default="small")
    parser.add_argument("--representation", choices=["waveform", "spectrogram"], default=None,
                        help="Force input representation. Defaults to model-specific setting.")
    parser.add_argument("--data-selection", default="loc1-6")
    parser.add_argument("--data-csv", default="./data/processed/datasets/data_1-6.csv")
    parser.add_argument("--main-data-dir", default="./data/processed/datasets")
    parser.add_argument("--mel", choices=["ON", "OFF"], default=None,
                        help="Override mel setting when representation is spectrogram.")

    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Global time limit in seconds")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None, help="Optuna storage URL (e.g., sqlite:///hpo.db)")
    parser.add_argument("--pruner", choices=["none", "median"], default="median")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel Optuna workers")

    parser.add_argument(
        "--hpo-config",
        default=None,
        help=(
            "Optional JSON file describing which hyperparameters to optimize/fix. "
            "See ./configs/optuna_autoencoder_hpo.json for a template. "
            "If omitted, uses the built-in search space."
        ),
    )
    return parser.parse_args()


def build_pruner(name: str) -> optuna.pruners.BasePruner:
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2)
    return optuna.pruners.NopPruner()


def objective(trial: optuna.Trial, cli_args: argparse.Namespace) -> float:
    # Start from CNN_any defaults and override per trial
    args = CNN_any.parse_args([])
    args.model = cli_args.model
    args.representation = cli_args.representation
    args.data_selection = cli_args.data_selection
    args.data_csv = cli_args.data_csv
    args.main_data_dir = cli_args.main_data_dir

    # Keep seed fixed for strict reproducibility across trials
    args.seed = cli_args.seed

    representation = CNN_any.resolve_representation(args.model, args.representation)
    hpo_space = load_hpo_space(cli_args.hpo_config)
    resolved = apply_hpo_space(
        trial=trial,
        cli_args=cli_args,
        space=hpo_space,
        representation=representation,
    )

    if resolved:
        args.batch_size = int(resolved.get("batch_size", args.batch_size))
        args.lr = float(resolved.get("lr", args.lr))
        args.epochs = int(resolved.get("epochs", args.epochs))
    else:
        args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 48, 64])
        args.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        args.epochs = trial.suggest_int("epochs", cli_args.min_epochs, cli_args.max_epochs)

    if representation == "spectrogram":
        if resolved:
            if "n_fft" in resolved:
                args.n_fft = int(resolved["n_fft"])
            else:
                args.n_fft = trial.suggest_categorical("n_fft", [512, 1024, 2048])

            if "hop_length" in resolved:
                args.hop_length = int(resolved["hop_length"])
            else:
                args.hop_length = trial.suggest_categorical("hop_length", [128, 256, 512])

            if cli_args.mel is not None:
                args.mel = cli_args.mel
            elif "mel" in resolved:
                args.mel = str(resolved["mel"])
            else:
                args.mel = trial.suggest_categorical("mel", ["OFF", "ON"])
        else:
            args.n_fft = trial.suggest_categorical("n_fft", [512, 1024, 2048])
            args.hop_length = trial.suggest_categorical("hop_length", [128, 256, 512])
            args.mel = cli_args.mel or trial.suggest_categorical("mel", ["OFF", "ON"])
    else:
        # Waveform models ignore these; keep consistent types
        args.n_fft = 1024
        args.hop_length = 512
        args.mel = "OFF"

    # Reduce overhead for HPO runs
    args.save_latent_space = False
    args.skip_latent = True

    best_loss, output_dir = CNN_any.train_autoencoder(
        args,
        enable_latent=False,
        trial=trial,
    )
    trial.set_user_attr("output_dir", output_dir)
    return best_loss


def main() -> None:
    cli_args = parse_hpo_args()

    sampler = optuna.samplers.TPESampler(seed=cli_args.seed)
    pruner = build_pruner(cli_args.pruner)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=cli_args.storage,
        study_name=cli_args.study_name,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, cli_args),
        n_trials=cli_args.n_trials,
        timeout=cli_args.timeout,
        n_jobs=cli_args.n_jobs,
        show_progress_bar=False,
    )

    best = study.best_trial
    print("Best validation loss:", best.value)
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
        output_dir = best.user_attrs.get("output_dir")
        out_path = Path(output_dir) if output_dir else default_fallback_output_dir(study_name=cli_args.study_name)

        meta = {
            "model": cli_args.model,
            "representation": cli_args.representation,
            "data_selection": cli_args.data_selection,
            "data_csv": cli_args.data_csv,
            "main_data_dir": cli_args.main_data_dir,
            "mel": cli_args.mel,
            "min_epochs": cli_args.min_epochs,
            "max_epochs": cli_args.max_epochs,
            "seed": cli_args.seed,
            "hpo_config": cli_args.hpo_config,
            "pruner": cli_args.pruner,
            "n_jobs": cli_args.n_jobs,
            "argv": sys.argv,
        }

        best_json_path = export_study(
            study=study,
            output_dir=out_path,
            storage=cli_args.storage,
            meta=meta,
        )

        print("Artifacts in:", str(out_path))
        print("Optuna summary:", str(best_json_path))


if __name__ == "__main__":
    main()
