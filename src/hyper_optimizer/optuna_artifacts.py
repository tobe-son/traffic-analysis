"""Utilities to persist Optuna HPO results as human-readable artifacts.

We already store trials in Optuna storage (e.g., sqlite). These helpers export
"best params" and per-trial summaries to files so that experiments can be
reviewed without opening the DB.

In addition, if Plotly is available, we export common Optuna visualization
charts as self-contained HTML.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import optuna


@dataclass(frozen=True)
class StudyExport:
    exported_at_utc: str
    study_name: str
    direction: str
    storage: Optional[str]
    n_trials_total: int
    n_trials_complete: int
    best_trial_number: int
    best_value: float
    best_params: Mapping[str, Any]
    best_user_attrs: Mapping[str, Any]
    meta: Mapping[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable objects."""

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    return str(obj)


def _trial_state_name(state: optuna.trial.TrialState) -> str:
    try:
        return state.name
    except Exception:
        return str(state)


def _collect_param_names(trials: Iterable[optuna.trial.FrozenTrial]) -> list[str]:
    names: set[str] = set()
    for t in trials:
        names.update(t.params.keys())
    return sorted(names)


def _export_optuna_plots(*, study: optuna.Study, output_dir: Path) -> None:
    """Export Optuna visualization charts as HTML.

    This is best-effort and never raises; failures are written to a log file so
    that study exports still succeed on headless / minimal environments.
    """

    log_path = output_dir / "optuna_plots.log"

    def _log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

    # HTML (Plotly)
    if importlib.util.find_spec("plotly") is None:
        _log("plotly is not available; skipping HTML plots")
    else:
            try:
                from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances

                plotly_plots: list[tuple[str, Any, str]] = [
                    ("optuna_optimization_history", plot_optimization_history, "Optimization History"),
                    ("optuna_parallel_coordinate", plot_parallel_coordinate, "Parallel Coordinate Plot"),
                    ("optuna_hyperparameter_importance", plot_param_importances, "Hyperparameter Importance"),
                ]

                for stem, plot_fn, title in plotly_plots:
                    try:
                        fig = plot_fn(study)
                        fig.write_html(str(output_dir / f"{stem}.html"), include_plotlyjs=True, full_html=True)
                    except Exception as exc:
                        _log(f"failed to export {title} HTML ({type(exc).__name__}: {exc})")
            except Exception as exc:
                _log(f"optuna.visualization (plotly) import failed; skipping HTML plots ({type(exc).__name__}: {exc})")

    # PNG (Matplotlib backend, preferred)
    if importlib.util.find_spec("optuna.visualization.matplotlib") is not None:
        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history as mpl_plot_optimization_history,
                plot_parallel_coordinate as mpl_plot_parallel_coordinate,
                plot_param_importances as mpl_plot_param_importances,
            )

            import matplotlib

            try:
                matplotlib.use("Agg", force=True)
            except Exception:
                pass

            import matplotlib.pyplot as plt

            mpl_plots: list[tuple[str, Any, str]] = [
                ("optuna_optimization_history", mpl_plot_optimization_history, "Optimization History"),
                ("optuna_parallel_coordinate", mpl_plot_parallel_coordinate, "Parallel Coordinate Plot"),
                ("optuna_hyperparameter_importance", mpl_plot_param_importances, "Hyperparameter Importance"),
            ]

            for stem, plot_fn, title in mpl_plots:
                try:
                    ax_or_fig = plot_fn(study)
                    fig = getattr(ax_or_fig, "figure", ax_or_fig)
                    fig.savefig(str(output_dir / f"{stem}.png"), dpi=200, bbox_inches="tight")
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                except Exception as exc:
                    _log(f"failed to export {title} PNG ({type(exc).__name__}: {exc})")
        except Exception as exc:
            _log(f"optuna.visualization.matplotlib import failed; skipping PNG plots ({type(exc).__name__}: {exc})")
    else:
        _log("optuna.visualization.matplotlib is not available; skipping PNG plots")


def export_study(
    *,
    study: optuna.Study,
    output_dir: Path,
    storage: Optional[str],
    meta: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write study exports into output_dir.

    Creates:
      - optuna_best.json: best trial summary + metadata
      - optuna_best_params.json: only best params (easy to reuse)
      - optuna_trials.csv: one row per trial (incl. params columns)

    Returns:
      Path to optuna_best.json
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    trials = list(study.trials)
    n_complete = sum(1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE)

    best_trial = None
    best_value = float("nan")
    best_trial_number = -1
    best_params: Mapping[str, Any] = {}
    best_user_attrs: Mapping[str, Any] = {}
    if n_complete > 0:
        best_trial = study.best_trial
        best_value = float(best_trial.value) if best_trial.value is not None else float("nan")
        best_trial_number = int(best_trial.number)
        best_params = _safe_json(best_trial.params)
        best_user_attrs = _safe_json(best_trial.user_attrs)

    export = StudyExport(
        exported_at_utc=_utc_now_iso(),
        study_name=study.study_name,
        direction=str(study.direction),
        storage=storage,
        n_trials_total=len(trials),
        n_trials_complete=n_complete,
        best_trial_number=best_trial_number,
        best_value=best_value,
        best_params=best_params,
        best_user_attrs=best_user_attrs,
        meta=_safe_json(dict(meta or {})),
    )

    best_json_path = output_dir / "optuna_best.json"
    with best_json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(export), f, ensure_ascii=False, indent=2, sort_keys=True)

    best_params_path = output_dir / "optuna_best_params.json"
    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(_safe_json(best_params), f, ensure_ascii=False, indent=2, sort_keys=True)

    # Trials CSV
    param_names = _collect_param_names(trials)

    fieldnames = [
        "number",
        "state",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration_sec",
        "user_attrs_json",
    ] + [f"param_{n}" for n in param_names]

    trials_csv_path = output_dir / "optuna_trials.csv"
    with trials_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            duration = None
            if t.datetime_start and t.datetime_complete:
                duration = (t.datetime_complete - t.datetime_start).total_seconds()

            row: dict[str, Any] = {
                "number": t.number,
                "state": _trial_state_name(t.state),
                "value": t.value,
                "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
                "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
                "duration_sec": duration,
                "user_attrs_json": json.dumps(_safe_json(t.user_attrs), ensure_ascii=False, sort_keys=True),
            }
            for n in param_names:
                row[f"param_{n}"] = t.params.get(n)

            writer.writerow(row)

    _export_optuna_plots(study=study, output_dir=output_dir)

    return best_json_path


def default_fallback_output_dir(*, study_name: Optional[str]) -> Path:
    safe_name = study_name or "unnamed_study"
    return Path("outputs") / "optuna_studies" / safe_name
