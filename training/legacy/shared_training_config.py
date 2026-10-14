from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_SHARED_CONFIG = ROOT / "revision" / "shared_training.yaml"
DEFAULT_SMOKE_CONFIG = ROOT / "revision" / "shared_training_smoke.yaml"
_AUGMENTATION_PATH = ROOT / "cxas" / "augmentation.py"

def _load_augmentation_symbols() -> tuple[Any, Any]:
    augmentation_spec = importlib.util.spec_from_file_location("anychest_augmentation", _AUGMENTATION_PATH)
    if augmentation_spec is None or augmentation_spec.loader is None:
        raise ImportError(f"Unable to load augmentation module from {_AUGMENTATION_PATH}")
    augmentation_module = importlib.util.module_from_spec(augmentation_spec)
    augmentation_spec.loader.exec_module(augmentation_module)
    return augmentation_module.AugmentationConfig, augmentation_module.AugmentationType


def detect_runtime_layout(config: dict[str, Any]) -> dict[str, Path]:
    runtime_cfg = dict(config.get("runtime", {}))
    profiles = dict(runtime_cfg.get("profiles", {}))
    if not profiles:
        raise ValueError("Training config must define runtime.profiles.")

    required_keys = {"repo_root", "data_dir", "compare_root"}
    candidates: list[tuple[str, dict[str, Path]]] = []
    for profile_name, raw_layout in profiles.items():
        if not isinstance(raw_layout, dict):
            raise ValueError(f"runtime.profiles.{profile_name} must be a mapping.")
        missing = required_keys - set(raw_layout.keys())
        if missing:
            raise ValueError(f"runtime.profiles.{profile_name} is missing keys: {sorted(missing)}")
        layout = {key: Path(str(raw_layout[key])) for key in required_keys}
        candidates.append((str(profile_name), layout))

    for profile_name, layout in candidates:
        if layout["repo_root"].exists():
            detected = dict(layout)
            detected["profile_name"] = str(profile_name)
            return detected

    raise FileNotFoundError(
        "Unable to detect runtime layout from runtime.profiles. Checked repo roots: "
        + ", ".join(str(layout["repo_root"]) for _, layout in candidates)
    )


def _resolve_auto_value(value: Any, layout: dict[str, Path]) -> Any:
    if isinstance(value, str):
        replacements = {
            "{auto_repo_root}": str(layout["repo_root"]),
            "{auto_data_dir}": str(layout["data_dir"]),
            "{auto_compare_root}": str(layout["compare_root"]),
        }
        resolved = value
        for token, replacement in replacements.items():
            resolved = resolved.replace(token, replacement)
        return resolved
    if isinstance(value, list):
        return [_resolve_auto_value(item, layout) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_auto_value(item, layout) for key, item in value.items()}
    return value


def load_training_config(path: Optional[str | Path]) -> dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_SHARED_CONFIG
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Training config at {config_path} must be a mapping.")
    layout = detect_runtime_layout(loaded)
    loaded = _resolve_auto_value(loaded, layout)
    loaded["_config_path"] = str(config_path)
    loaded["_runtime_layout"] = {key: str(value) for key, value in layout.items()}
    return loaded


def resolve_model_config(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    common = dict(config.get("common", {}))
    car = dict(config.get("car", {}))
    models = dict(config.get("models", {}))
    model_cfg = dict(models.get(model_key, {}))
    resolved = {**common, **model_cfg}
    resolved["car"] = car
    resolved["config_path"] = config.get("_config_path")
    return resolved


def parse_inf(value: Any, *, default: float = math.inf) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return math.inf
        return float(lowered)
    raise TypeError(f"Unsupported value for parse_inf: {value!r}")


def as_path(value: Any) -> Optional[Path]:
    if value in (None, "", "null"):
        return None
    return Path(str(value))


def build_augmentation_configs(profile: str) -> list[AugmentationConfig]:
    AugmentationConfig, AugmentationType = _load_augmentation_symbols()
    normalized = profile.strip().lower()
    if normalized == "light":
        return [
            AugmentationConfig(type=AugmentationType.FLIP_HORIZONTAL, probability=0.5),
            AugmentationConfig(type=AugmentationType.FLIP_VERTICAL, probability=0.5),
            AugmentationConfig(type=AugmentationType.ROTATE, probability=0.8, params={"max_angle": 30}),
            AugmentationConfig(type=AugmentationType.NOISE, probability=0.5, params={"max_std": 0.1}),
        ]
    if normalized == "full":
        return [
            AugmentationConfig(type=AugmentationType.FLIP_HORIZONTAL, probability=0.5),
            AugmentationConfig(type=AugmentationType.FLIP_VERTICAL, probability=0.5),
            AugmentationConfig(type=AugmentationType.ROTATE, probability=0.8, params={"max_angle": 30}),
            AugmentationConfig(type=AugmentationType.NOISE, probability=0.5, params={"max_std": 0.1}),
            AugmentationConfig(
                type=AugmentationType.CONTROL_POINT,
                probability=0.7,
                params={"base_x_shift": 0.15, "base_y_shift": 0.10, "shift_range": 0.1},
            ),
            AugmentationConfig(
                type=AugmentationType.RandomScatterArtifact,
                probability=0.5,
                params={"intensity_range": [5, 30], "count_range": [10, 50]},
            ),
            AugmentationConfig(
                type=AugmentationType.RandomGlobalContrastStretch,
                probability=0.5,
                params={"contrast_range": [0.5, 2.0]},
            ),
            AugmentationConfig(type=AugmentationType.BRIGHTNESS, probability=0.7, params={"range": 0.3}),
            AugmentationConfig(type=AugmentationType.CONTRAST, probability=0.7, params={"range": 0.3}),
            AugmentationConfig(type=AugmentationType.GAMMA, probability=0.6, params={"range": 0.05}),
        ]
    raise ValueError(f"Unknown augmentation profile: {profile}")
