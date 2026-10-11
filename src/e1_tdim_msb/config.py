from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "E1_TDIM_MSB_PILOT",
        "profile": "pilot_q2",
        "seeds": [123456789, 234567891],
        "deterministic_seed": 123456789,
    },
    "dataset": {
        "path": "__AUTO__",
        "label_column": "__AUTO__",
        "timestamp_column": "__AUTO__",
        "class_names": None,
        "row_start": 0,
        "max_rows": 2_000_000,
        "require_timestamp": True,
        "validate_chronology": True,
        "chronology_tolerance": 0,
        "feature_include": None,
        "feature_exclude": [],
        "numeric_features_only": True,
        "drop_ip_addresses": True,
        "drop_identifier_columns": True,
        "batch_size": 50_000,
        "use_threads": True,
        "memory_map": True,
        "pre_buffer": True,
    },
    "evaluation": {
        "block_size": 50_000,
        "rolling_windows": [250_000, 500_000, 1_000_000],
        "checkpoint_every": 500_000,
        "resource_log_every": 50_000,
        "store_classwise_metrics": True,
        "store_confusion_matrices": True,
    },
    "models": {
        "algorithms": ["HT", "OB", "MOOB_L20", "MUOB"],
        "ensemble_size": 5,
        "class_size_theta": 0.9,
        "class_size_floor": 1e-12,
        "moob_lambda_cap": 20.0,
        "ht": {
            "grace_period": 200,
            "max_depth": 20,
            "split_criterion": "info_gain",
            "delta": 1e-7,
            "tau": 0.05,
            "leaf_prediction": "nba",
            "nb_threshold": 0,
            "binary_split": False,
            "max_size": 64.0,
            "memory_estimate_period": 250_000,
            "stop_mem_management": False,
            "remove_poor_attrs": True,
            "merit_preprune": True,
        },
    },
    "runtime": {
        "output_root": "__AUTO__",
        "resume": True,
        "overwrite": False,
        "fail_on_nonfinite": False,
        "prediction_fallback": "benign_then_majority",
        "gc_every_blocks": 5,
        "log_level": "INFO",
    },
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)
    cfg["_config_path"] = str(path)
    validate_config(cfg)
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)


def validate_config(cfg: Mapping[str, Any]) -> None:
    ds = cfg["dataset"]
    ev = cfg["evaluation"]
    models = cfg["models"]

    if int(ds["batch_size"]) <= 0:
        raise ValueError("dataset.batch_size must be positive")
    if int(ev["block_size"]) <= 0:
        raise ValueError("evaluation.block_size must be positive")
    if int(ev["checkpoint_every"]) % int(ev["block_size"]) != 0:
        raise ValueError("checkpoint_every must be a multiple of block_size")
    for window in ev["rolling_windows"]:
        if int(window) % int(ev["block_size"]) != 0:
            raise ValueError("Every rolling window must be a multiple of block_size")
    if int(models["ensemble_size"]) < 1:
        raise ValueError("models.ensemble_size must be at least 1")
    theta = float(models["class_size_theta"])
    if not 0.0 < theta < 1.0:
        raise ValueError("models.class_size_theta must be in (0, 1)")
    supported = {"HT", "OB", "MOOB", "MOOB_L20", "MUOB"}
    unknown = set(models["algorithms"]) - supported
    if unknown:
        raise ValueError(f"Unsupported algorithms: {sorted(unknown)}")
    if "MOOB" in models["algorithms"] and "MOOB_L20" in models["algorithms"]:
        raise ValueError("MOOB and MOOB_L20 cannot share one class-size tracker in the same run")
    if int(ds.get("row_start", 0)) < 0:
        raise ValueError("dataset.row_start cannot be negative")
    if ds.get("max_rows") is not None and int(ds["max_rows"]) <= 0:
        raise ValueError("dataset.max_rows must be positive or null")


def apply_overrides(cfg: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        cursor = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    validate_config(result)
    return result


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
