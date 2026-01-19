from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class HpoSpace:
    version: int
    common: Dict[str, dict]
    spectrogram: Dict[str, dict]


def _resolve_bound(value: Any, cli_args: Any) -> Any:
    if isinstance(value, dict) and "from_cli" in value:
        attr = value["from_cli"]
        if not hasattr(cli_args, attr):
            raise ValueError(f"HPO config requested from_cli='{attr}', but CLI args has no such attribute")
        return getattr(cli_args, attr)
    return value


def load_hpo_space(path: Optional[str]) -> Optional[HpoSpace]:
    if not path:
        return None

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"HPO config not found: {cfg_path}")

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    version = int(data.get("version", 1))
    common = data.get("common", {})
    spectrogram = data.get("spectrogram", {})

    if not isinstance(common, dict) or not isinstance(spectrogram, dict):
        raise ValueError("HPO config must contain 'common' and 'spectrogram' objects")

    return HpoSpace(version=version, common=dict(common), spectrogram=dict(spectrogram))


def _suggest_from_spec(trial: Any, name: str, spec: Mapping[str, Any], cli_args: Any) -> Any:
    spec_type = spec.get("type")

    if spec_type == "fixed":
        return spec.get("value")

    if spec_type == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise ValueError(f"HPO config: '{name}' categorical must have non-empty choices")
        return trial.suggest_categorical(name, choices)

    if spec_type == "float":
        low = _resolve_bound(spec.get("low"), cli_args)
        high = _resolve_bound(spec.get("high"), cli_args)
        log = bool(spec.get("log", False))
        if low is None or high is None:
            raise ValueError(f"HPO config: '{name}' float must define low/high")
        return trial.suggest_float(name, float(low), float(high), log=log)

    if spec_type == "int":
        low = _resolve_bound(spec.get("low"), cli_args)
        high = _resolve_bound(spec.get("high"), cli_args)
        if low is None or high is None:
            raise ValueError(f"HPO config: '{name}' int must define low/high")
        return trial.suggest_int(name, int(low), int(high))

    raise ValueError(f"HPO config: '{name}' has unsupported type: {spec_type}")


def apply_hpo_space(
    *,
    trial: Any,
    cli_args: Any,
    space: Optional[HpoSpace],
    representation: str,
) -> Dict[str, Any]:
    """Return resolved hyperparameter values for this trial.

    The caller decides how to apply these to downstream args.
    """

    if space is None:
        return {}

    if space.version != 1:
        raise ValueError(f"Unsupported HPO config version: {space.version}")

    resolved: Dict[str, Any] = {}

    for name, spec in space.common.items():
        if not isinstance(spec, dict):
            raise ValueError(f"HPO config: common.{name} must be an object")
        resolved[name] = _suggest_from_spec(trial, name, spec, cli_args)

    if representation == "spectrogram":
        for name, spec in space.spectrogram.items():
            if not isinstance(spec, dict):
                raise ValueError(f"HPO config: spectrogram.{name} must be an object")
            resolved[name] = _suggest_from_spec(trial, name, spec, cli_args)

    return resolved
