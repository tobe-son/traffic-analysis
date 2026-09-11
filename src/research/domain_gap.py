"""Quantify simulation-to-real generalization gaps.

The command accepts either sample-level regression predictions or an aggregated
metric table.  Keeping the analysis separate from model training makes the
research claim reproducible from exported results alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """Return MAE, RMSE and R2 without external dependencies."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    errors = [prediction - target for target, prediction in zip(y_true, y_pred)]
    mae = mean(abs(error) for error in errors)
    mse = mean(error * error for error in errors)
    target_mean = mean(y_true)
    total = sum((target - target_mean) ** 2 for target in y_true)
    residual = sum(error * error for error in errors)
    r2 = 1.0 - residual / total if total else math.nan
    return {"mae": mae, "rmse": math.sqrt(mse), "r2": r2}


def bootstrap_metric(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    metric: str,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percentile 95% CI using paired resampling."""
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    rng = random.Random(seed)
    size = len(y_true)
    values: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(size) for _ in range(size)]
        sampled_true = [y_true[index] for index in indices]
        sampled_pred = [y_pred[index] for index in indices]
        value = regression_metrics(sampled_true, sampled_pred)[metric]
        if math.isfinite(value):
            values.append(value)
    if not values:
        return math.nan, math.nan
    values.sort()
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    return lower, upper


def analyse_predictions(
    rows: Iterable[dict[str, str]], *, iterations: int = 2000, seed: int = 42
) -> list[dict[str, object]]:
    """Aggregate sample predictions by model and domain."""
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["domain"])].append(
            (float(row["y_true"]), float(row["y_pred"]))
        )
    result: list[dict[str, object]] = []
    for (model, domain), pairs in sorted(groups.items()):
        targets, predictions = zip(*pairs)
        metrics = regression_metrics(targets, predictions)
        record: dict[str, object] = {"model": model, "domain": domain, "n": len(pairs)}
        for offset, metric in enumerate(("mae", "rmse", "r2")):
            low, high = bootstrap_metric(
                targets,
                predictions,
                metric,
                iterations=iterations,
                seed=seed + offset,
            )
            record[metric] = metrics[metric]
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        result.append(record)
    return result


def analyse_summary(rows: Iterable[dict[str, str]]) -> dict[str, object]:
    """Rank models using documented simulation and real-domain R2 values."""
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        value = row.get("r2", "").strip()
        if value.lower() in {"", "n/a", "nan"}:
            continue
        model = row["model"]
        grouped[model][row["domain"]] = {
            "r2": float(value),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
        }
    models: list[dict[str, object]] = []
    for model, domains in grouped.items():
        if not {"sim", "real"}.issubset(domains):
            continue
        sim = domains["sim"]
        real = domains["real"]
        models.append(
            {
                "model": model,
                "sim_r2": sim["r2"],
                "real_r2": real["r2"],
                "r2_gap": float(sim["r2"]) - float(real["r2"]),
                "real_mae": real["mae"],
                "real_rmse": real["rmse"],
            }
        )
    models.sort(key=lambda item: (-float(item["real_r2"]), float(item["r2_gap"])))
    return {
        "criterion": "highest real R2, then smallest simulation-to-real R2 gap",
        "n_models": len(models),
        "best_existing_baseline": models[0] if models else None,
        "models": models,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--mode", choices=("predictions", "summary"), default="predictions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = _read_csv(args.input)
    if args.mode == "predictions":
        payload: object = analyse_predictions(
            rows, iterations=args.bootstrap_iterations, seed=args.seed
        )
    else:
        payload = analyse_summary(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

