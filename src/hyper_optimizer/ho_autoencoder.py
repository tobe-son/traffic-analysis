"""Hyperparameter search for CNN autoencoders using Optuna.

This script reuses the training pipeline in ``src/autoencoder/CNN_any.py`` and
runs Optuna to minimise validation loss. Each trial saves its own outputs under
``./outputs`` via ``output_settings``.
"""

import argparse
import gc
import shutil
import sys
from pathlib import Path

import optuna
from optuna import distributions as optuna_distributions

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

    parser.add_argument(
        "--reset-study",
        action="store_true",
        help=(
            "Delete an existing study (same --study-name/--storage) before starting. "
            "Useful when you changed the HPO search space and Optuna refuses to resume."
        ),
    )

    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel Optuna workers")

    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Keep best/optimizer checkpoint files in each trial output_dir (can consume disk).",
    )

    parser.add_argument(
        "--export-best-weights",
        action="store_true",
        help=(
            "Copy the best trial's autoencoder weights into outputs/optuna_studies/<study_name>/ after optimization. "
            "This implicitly enables --save-checkpoints."
        ),
    )

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


def _build_expected_distributions(
    *,
    cli_args: argparse.Namespace,
    representation: str,
) -> dict[str, optuna_distributions.BaseDistribution]:
    """Build expected Optuna distributions for this run.

    Used to detect incompatible changes when resuming an existing study.
    """

    expected: dict[str, optuna_distributions.BaseDistribution] = {}

    def _resolve_from_cli(value: object) -> object:
        if isinstance(value, dict) and "from_cli" in value:
            attr = value["from_cli"]
            if not hasattr(cli_args, attr):
                raise ValueError(f"HPO config requested from_cli='{attr}', but CLI args has no such attribute")
            return getattr(cli_args, attr)
        return value

    hpo_space = load_hpo_space(cli_args.hpo_config)
    if hpo_space is not None:
        for name, spec in hpo_space.common.items():
            spec_type = spec.get("type")
            if spec_type == "fixed":
                continue
            if spec_type == "categorical":
                expected[name] = optuna_distributions.CategoricalDistribution(choices=spec.get("choices"))
            elif spec_type == "float":
                low = float(_resolve_from_cli(spec.get("low")))
                high = float(_resolve_from_cli(spec.get("high")))
                expected[name] = optuna_distributions.FloatDistribution(
                    low=low,
                    high=high,
                    log=bool(spec.get("log", False)),
                )
            elif spec_type == "int":
                low = int(_resolve_from_cli(spec.get("low")))
                high = int(_resolve_from_cli(spec.get("high")))
                expected[name] = optuna_distributions.IntDistribution(low=low, high=high)
            else:
                raise ValueError(f"Unsupported HPO config spec type for '{name}': {spec_type}")

        if representation == "spectrogram":
            for name, spec in hpo_space.spectrogram.items():
                spec_type = spec.get("type")
                if spec_type == "fixed":
                    continue
                if spec_type == "categorical":
                    expected[name] = optuna_distributions.CategoricalDistribution(choices=spec.get("choices"))
                elif spec_type == "float":
                    low = float(_resolve_from_cli(spec.get("low")))
                    high = float(_resolve_from_cli(spec.get("high")))
                    expected[name] = optuna_distributions.FloatDistribution(
                        low=low,
                        high=high,
                        log=bool(spec.get("log", False)),
                    )
                elif spec_type == "int":
                    low = int(_resolve_from_cli(spec.get("low")))
                    high = int(_resolve_from_cli(spec.get("high")))
                    expected[name] = optuna_distributions.IntDistribution(low=low, high=high)
                else:
                    raise ValueError(f"Unsupported HPO config spec type for 'spectrogram.{name}': {spec_type}")

        return expected

    expected["batch_size"] = optuna_distributions.CategoricalDistribution(choices=[16, 32, 48, 64])
    expected["lr"] = optuna_distributions.FloatDistribution(low=1e-4, high=5e-3, log=True)
    expected["epochs"] = optuna_distributions.IntDistribution(low=int(cli_args.min_epochs), high=int(cli_args.max_epochs))

    if representation == "spectrogram":
        expected["n_fft"] = optuna_distributions.CategoricalDistribution(choices=[512, 1024, 2048])
        expected["hop_length"] = optuna_distributions.CategoricalDistribution(choices=[128, 256, 512])
        if cli_args.mel is None:
            expected["mel"] = optuna_distributions.CategoricalDistribution(choices=["OFF", "ON"])

    return expected


def _ensure_study_compatible(
    *,
    study: optuna.Study,
    expected: dict[str, optuna_distributions.BaseDistribution],
) -> None:
    previous: dict[str, optuna_distributions.BaseDistribution] = {}
    for t in study.trials:
        for name, dist in t.distributions.items():
            if name not in previous:
                previous[name] = dist

    for name, exp_dist in expected.items():
        prev_dist = previous.get(name)
        if prev_dist is None:
            continue
        try:
            optuna.distributions.check_distribution_compatibility(prev_dist, exp_dist)
        except ValueError as exc:
            raise RuntimeError(
                "既存の Optuna study を再開できません（探索空間が過去と互換ではありません）。\n"
                f"- study: {study.study_name}\n"
                f"- param: {name}\n"
                f"- previous: {prev_dist}\n"
                f"- current: {exp_dist}\n"
                "対処: (A) --study-name を変える / (B) --reset-study を付けて既存 study を削除してやり直す\n"
                "例: python -c \"import optuna; optuna.delete_study(study_name='NAME', storage='sqlite:///PATH.db')\"\n"
                f"detail: {exc}"
            )


def build_pruner(name: str) -> optuna.pruners.BasePruner:
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2)
    return optuna.pruners.NopPruner()


def objective(trial: optuna.Trial, cli_args: argparse.Namespace) -> float:
    # Best-effort: start the trial with more free GPU memory.
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

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

    try:
        best_loss, output_dir = CNN_any.train_autoencoder(
            args,
            enable_latent=False,
            trial=trial,
        )
        trial.set_user_attr("output_dir", output_dir)

        if not cli_args.save_checkpoints:
            out = Path(output_dir)
            for name in (f"best_model_{args.model}.pth", f"optimizer_{args.model}.pth"):
                path = out / name
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass

        return best_loss
    except RuntimeError as exc:
        msg = str(exc).lower()
        is_cuda_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
        if is_cuda_oom:
            try:
                trial.set_user_attr("failed_reason", "cuda_oom")
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return float("inf")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def main() -> None:
    cli_args = parse_hpo_args()

    if cli_args.export_best_weights and not cli_args.save_checkpoints:
        cli_args.save_checkpoints = True

    sampler = optuna.samplers.TPESampler(seed=cli_args.seed)
    pruner = build_pruner(cli_args.pruner)

    if cli_args.reset_study:
        if not cli_args.storage or not cli_args.study_name:
            raise ValueError("--reset-study には --storage と --study-name の両方が必要です")
        try:
            optuna.delete_study(study_name=cli_args.study_name, storage=cli_args.storage)
        except KeyError:
            pass

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=cli_args.storage,
        study_name=cli_args.study_name,
        load_if_exists=True,
    )

    representation = CNN_any.resolve_representation(cli_args.model, cli_args.representation)
    expected = _build_expected_distributions(cli_args=cli_args, representation=representation)
    _ensure_study_compatible(study=study, expected=expected)

    study.optimize(
        lambda trial: objective(trial, cli_args),
        n_trials=cli_args.n_trials,
        timeout=cli_args.timeout,
        n_jobs=cli_args.n_jobs,
        catch=(RuntimeError,),
        show_progress_bar=False,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = study.best_trial
        print("Best validation loss:", float(best.value))
        print("Best params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
    else:
        print("No successful (COMPLETE) trials. All trials failed/pruned.")

    out_path = default_fallback_output_dir(study_name=study.study_name)

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
        "reset_study": cli_args.reset_study,
        "save_checkpoints": cli_args.save_checkpoints,
        "export_best_weights": cli_args.export_best_weights,
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

    if cli_args.export_best_weights and completed:
        best_trial = study.best_trial
        best_dir = Path(str(best_trial.user_attrs.get("output_dir", "")))
        best_model = best_dir / f"best_model_{cli_args.model}.pth"
        best_opt = best_dir / f"optimizer_{cli_args.model}.pth"

        copied_any = False
        if best_dir and best_model.exists():
            dest = out_path / best_model.name
            shutil.copy2(best_model, dest)
            print("Best model copied to:", str(dest))
            copied_any = True
        if best_dir and best_opt.exists():
            dest = out_path / best_opt.name
            shutil.copy2(best_opt, dest)
            print("Best optimizer state copied to:", str(dest))
            copied_any = True

        if not copied_any:
            print(
                "Warning: best checkpoint files not found in best trial output_dir. "
                "Ensure the trial completed successfully and checkpoint files were not removed."
            )


if __name__ == "__main__":
    main()
