"""Optuna-based HPO for LogRatioLoss continuous metric learning (ContinuousLearning).

This script runs continuous metric learning using the training logic in
``src/metric/ContinuousLearning.py`` and optimizes the validation loss.

Notes
-----
- By default this script does *not* run latent extraction/visualization to keep
  trials fast.
- It keeps the random seed fixed (default: 42) for reproducibility across trials.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path
from typing import Optional

import optuna
from optuna import distributions as optuna_distributions
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow imports from the src directory
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from learn_tool.settings import output_settings, prepare_dataloader
from hyper_optimizer.hpo_space import apply_hpo_space, load_hpo_space
from hyper_optimizer.optuna_artifacts import default_fallback_output_dir, export_study
from loss.LogRatioLoss import LogRatioLoss
from metric.ContinuousLearning import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_REPRESENTATION,
    MODEL_REGISTRY,
    evaluate,
    log_hyperparameters,
    log_ratio_loss_for_batch,
    resolve_hop_length,
    resolve_representation,
    set_global_seed,
    train_one_epoch,
)


def _train_one_epoch_amp(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: torch.nn.Module,
    epoch_label: str,
    scaler: GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    usable_batches = 0

    for batch in tqdm(loader, desc=epoch_label):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            loss = log_ratio_loss_for_batch(model, batch, device, criterion)
            if loss is None or not torch.isfinite(loss):
                continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        usable_batches += 1

    if usable_batches == 0:
        raise RuntimeError("No valid batches produced a LogRatioLoss signal. Increase batch size or ensure diverse speeds.")

    return total_loss / usable_batches


def _evaluate_amp(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    usable_batches = 0
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                loss = log_ratio_loss_for_batch(model, batch, device, criterion)
            if loss is None or not torch.isfinite(loss):
                continue
            total_loss += float(loss.item())
            usable_batches += 1

    if usable_batches == 0:
        return float("nan")

    return total_loss / usable_batches


def parse_hpo_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna-based HPO for LogRatioLoss continuous metric learning")

    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="small")
    parser.add_argument("--representation", choices=["waveform", "spectrogram"], default=None)
    parser.add_argument("--data-selection", default="loc1-6")
    parser.add_argument("--data-csv", default="./data/processed/datasets/data_1-6.csv")
    parser.add_argument("--main-data-dir", default="./data/processed/datasets")
    parser.add_argument("--mel", choices=["ON", "OFF"], default=None)

    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument(
        "--hop-length",
        type=int,
        default=None,
        help="If provided, fixes hop_length. If omitted and representation is spectrogram, Optuna may tune it.",
    )

    parser.add_argument("--norm-p", type=float, default=None, help="Fix LogRatioLoss p (otherwise tuned)")
    parser.add_argument("--eps", type=float, default=None, help="Fix LogRatioLoss eps (otherwise tuned)")

    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None)
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
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--test-split", type=float, default=0.2)

    parser.add_argument("--n-jobs", type=int, default=1)

    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save best/last encoder weights per trial (can consume disk).",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA AMP (fp16 autocast) to reduce GPU memory usage.",
    )

    parser.add_argument(
        "--oom-retry-max",
        type=int,
        default=3,
        help=(
            "On CUDA OOM, retry the same trial by reducing batch_size. "
            "Set 0 to disable retries (trial will be marked as failed)."
        ),
    )
    parser.add_argument(
        "--oom-min-batch-size",
        type=int,
        default=4,
        help="Minimum batch_size allowed when retrying after CUDA OOM.",
    )

    parser.add_argument(
        "--hpo-config",
        default=None,
        help=(
            "Optional JSON file describing which hyperparameters to optimize/fix. "
            "See ./configs/optuna_continuouslearning_hpo.json for a template. "
            "If omitted, uses the built-in search space."
        ),
    )

    return parser.parse_args()


def _build_expected_distributions(
    *,
    cli_args: argparse.Namespace,
    representation: str,
) -> dict[str, optuna_distributions.BaseDistribution]:
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
                expected[name] = optuna_distributions.FloatDistribution(low=low, high=high, log=bool(spec.get("log", False)))
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
                    expected[name] = optuna_distributions.FloatDistribution(low=low, high=high, log=bool(spec.get("log", False)))
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

    if cli_args.norm_p is None:
        expected["norm_p"] = optuna_distributions.FloatDistribution(low=1.0, high=4.0, log=False)
    if cli_args.eps is None:
        expected["eps"] = optuna_distributions.FloatDistribution(low=1e-8, high=1e-3, log=True)

    if representation == "spectrogram":
        expected["n_fft"] = optuna_distributions.CategoricalDistribution(choices=[512, 1024, 2048])
        if cli_args.hop_length is None:
            base = DEFAULT_HOP_LENGTH.get(cli_args.model, 512)
            candidates = sorted({base, 128, 160, 256, 320, 512})
            expected["hop_length"] = optuna_distributions.CategoricalDistribution(choices=candidates)
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


def _resolve_trial_hop_length(
    *,
    trial: optuna.Trial,
    model_key: str,
    representation: str,
    cli_hop_length: Optional[int],
) -> int:
    if cli_hop_length is not None:
        return int(cli_hop_length)

    base = DEFAULT_HOP_LENGTH.get(model_key, 512)
    if representation != "spectrogram":
        return int(base)

    candidates = sorted({base, 128, 160, 256, 320, 512})
    return int(trial.suggest_categorical("hop_length", candidates))


def objective(trial: optuna.Trial, cli_args: argparse.Namespace) -> float:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    model_key = cli_args.model
    description, model_cls = MODEL_REGISTRY[model_key]

    representation = resolve_representation(model_key, cli_args.representation or DEFAULT_REPRESENTATION.get(model_key))

    hpo_space = load_hpo_space(cli_args.hpo_config)
    resolved = apply_hpo_space(
        trial=trial,
        cli_args=cli_args,
        space=hpo_space,
        representation=representation,
    )

    model = None
    optimizer = None
    criterion = None
    train_loader = None
    val_loader = None
    logger = None

    try:
        set_global_seed(cli_args.seed)

        if resolved:
            if "batch_size" in resolved:
                batch_size = int(resolved["batch_size"])
            else:
                batch_size = int(trial.suggest_categorical("batch_size", [16, 32, 48, 64]))

            if "lr" in resolved:
                lr = float(resolved["lr"])
            else:
                lr = float(trial.suggest_float("lr", 1e-4, 5e-3, log=True))

            if "epochs" in resolved:
                epochs = int(resolved["epochs"])
            else:
                epochs = int(trial.suggest_int("epochs", cli_args.min_epochs, cli_args.max_epochs))
        else:
            batch_size = int(trial.suggest_categorical("batch_size", [16, 32, 48, 64]))
            lr = float(trial.suggest_float("lr", 1e-4, 5e-3, log=True))
            epochs = int(trial.suggest_int("epochs", cli_args.min_epochs, cli_args.max_epochs))

        if cli_args.norm_p is not None:
            norm_p = float(cli_args.norm_p)
        elif resolved and "norm_p" in resolved:
            norm_p = float(resolved["norm_p"])
        else:
            norm_p = float(trial.suggest_float("norm_p", 1.0, 4.0))

        if cli_args.eps is not None:
            eps = float(cli_args.eps)
        elif resolved and "eps" in resolved:
            eps = float(resolved["eps"])
        else:
            eps = float(trial.suggest_float("eps", 1e-8, 1e-3, log=True))

        mel = cli_args.mel

        if representation == "spectrogram":
            if resolved and "n_fft" in resolved:
                n_fft = int(resolved["n_fft"])
            else:
                n_fft = int(trial.suggest_categorical("n_fft", [512, 1024, 2048]))

            if cli_args.hop_length is not None:
                hop_length = int(cli_args.hop_length)
            elif resolved and "hop_length" in resolved:
                hop_length = int(resolved["hop_length"])
            else:
                hop_length = _resolve_trial_hop_length(
                    trial=trial,
                    model_key=model_key,
                    representation=representation,
                    cli_hop_length=cli_args.hop_length,
                )

            if mel is None:
                if resolved and "mel" in resolved:
                    mel = str(resolved["mel"])
                else:
                    mel = str(trial.suggest_categorical("mel", ["OFF", "ON"]))
        else:
            n_fft = int(cli_args.n_fft)
            hop_length = resolve_hop_length(model_key, cli_args.hop_length)
            mel = "OFF"

        logger = output_settings()

        base_hyperparameters = {
            "MODEL": model_key,
            "MODEL_DESCRIPTION": description,
            "DATA_SELECTION": cli_args.data_selection,
            "DATA_CSV_PATH": cli_args.data_csv,
            "MAIN_DATA_DIR": cli_args.main_data_dir,
            "MEL": mel,
            "SAMPLING_RATE": cli_args.sampling_rate,
            "N_FFT": n_fft,
            "HOP_LENGTH": hop_length,
            "TEST_DATASET_PERCENTAGE": cli_args.test_split,
            "EPOCHS": epochs,
            "LEARNING_RATE": lr,
            "NORM_P": norm_p,
            "EPS": eps,
            "REPRESENTATION": representation,
            "SEED": cli_args.seed,
            "HPO_CONFIG": cli_args.hpo_config or "(none)",
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        effective_batch_size = int(batch_size)
        max_attempts = 1 + max(0, int(cli_args.oom_retry_max))
        oom_adjustments: list[dict[str, int]] = []

        for attempt_idx in range(max_attempts):
            hyperparameters = dict(base_hyperparameters)
            hyperparameters["BATCH_SIZE"] = effective_batch_size
            log_hyperparameters(logger, hyperparameters)

            train_loader, val_loader, feature_min, feature_max = prepare_dataloader(
                logger=logger,
                hop_length=hop_length,
                batch_size=effective_batch_size,
                data_csv_path=cli_args.data_csv,
                main_data_dir=cli_args.main_data_dir,
                n_fft=n_fft,
                data_num=cli_args.data_selection,
                mel=mel,
                sampling_rate=cli_args.sampling_rate,
                test_split=cli_args.test_split,
                seed=cli_args.seed,
                representation=representation,
            )
            logger.info(
                "データの準備が完了しました。feature_min=%.6f feature_max=%.6f",
                feature_min,
                feature_max,
            )

            model = model_cls().to(device)
            criterion = LogRatioLoss(p=norm_p, eps=eps).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scaler = GradScaler(enabled=(cli_args.amp and device.type == "cuda"))

            best_val_loss = float("inf")
            best_epoch = -1

            try:
                for epoch in range(epochs):
                    epoch_label = f"Train Epoch {epoch + 1}/{epochs}"
                    if cli_args.amp and device.type == "cuda":
                        _train_loss = _train_one_epoch_amp(model, train_loader, optimizer, device, criterion, epoch_label, scaler)
                    else:
                        _train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch_label)

                    if cli_args.amp and device.type == "cuda":
                        val_loss = _evaluate_amp(model, val_loader, device, criterion)
                    else:
                        val_loss = evaluate(model, val_loader, device, criterion)

                    if math.isfinite(val_loss):
                        trial.report(val_loss, epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_epoch = epoch + 1
                            if cli_args.save_checkpoints:
                                torch.save(
                                    model.state_dict(),
                                    Path(logger.output_dir) / f"best_encoder_{model_key}.pth",
                                )

                if cli_args.save_checkpoints:
                    torch.save(model.state_dict(), Path(logger.output_dir) / f"last_encoder_{model_key}.pth")

                trial.set_user_attr("output_dir", logger.output_dir)
                trial.set_user_attr("best_epoch", best_epoch)
                trial.set_user_attr("effective_batch_size", effective_batch_size)
                if oom_adjustments:
                    trial.set_user_attr("oom_adjustments", oom_adjustments)

                return float(best_val_loss)

            except RuntimeError as exc:
                msg = str(exc).lower()
                is_cuda_oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
                if not is_cuda_oom:
                    logger.error("Trial failed during training: %s", exc)
                    if "no valid batches produced" in msg:
                        trial.set_user_attr("failed_reason", "no_valid_batches")
                    else:
                        trial.set_user_attr("failed_reason", "runtime_error")
                    raise

                logger.error("CUDA OOM (attempt %d/%d): %s", attempt_idx + 1, max_attempts, exc)
                if effective_batch_size <= int(cli_args.oom_min_batch_size):
                    trial.set_user_attr("failed_reason", "cuda_oom")
                    raise

                try:
                    del model, optimizer, criterion, train_loader, val_loader, scaler
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                prev_bs = effective_batch_size
                effective_batch_size = max(int(cli_args.oom_min_batch_size), prev_bs // 2)
                oom_adjustments.append({"from": prev_bs, "to": effective_batch_size})
                logger.info("Retrying trial with smaller batch_size: %d -> %d", prev_bs, effective_batch_size)

                continue

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
            raise
        raise
    finally:
        try:
            del model, optimizer, criterion, train_loader, val_loader, logger
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def main() -> None:
    cli_args = parse_hpo_args()

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

    try:
        model_key = cli_args.model
        representation = resolve_representation(model_key, cli_args.representation or DEFAULT_REPRESENTATION.get(model_key))
        expected = _build_expected_distributions(cli_args=cli_args, representation=representation)
        _ensure_study_compatible(study=study, expected=expected)
    except Exception:
        raise

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
        out_path = default_fallback_output_dir(study_name=study.study_name)
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
        "sampling_rate": cli_args.sampling_rate,
        "n_fft": cli_args.n_fft,
        "hop_length": cli_args.hop_length,
        "norm_p": cli_args.norm_p,
        "eps": cli_args.eps,
        "min_epochs": cli_args.min_epochs,
        "max_epochs": cli_args.max_epochs,
        "test_split": cli_args.test_split,
        "seed": cli_args.seed,
        "hpo_config": cli_args.hpo_config,
        "pruner": cli_args.pruner,
        "n_jobs": cli_args.n_jobs,
        "save_checkpoints": cli_args.save_checkpoints,
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
