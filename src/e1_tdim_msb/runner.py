from __future__ import annotations

import gc
import json
import logging
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .config import save_config
from .io_utils import (
    CsvAppender,
    atomic_json_dump,
    atomic_pickle_dump,
    fast_file_fingerprint,
    pickle_load,
    truncate_csv_after,
)
from .metrics import ModelMetricState, confusion_metrics, js_divergence
from .models import DynamicClassSizeTracker, StreamClassifier, build_models
from .platforms import environment_manifest
from .schema import inspect_dataset
from .stream import ParquetTemporalStream, discover_classes


LOGGER = logging.getLogger("e1_tdim_msb")


def _iso_from_ns(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat()


def _find_benign_index(class_names: list[str]) -> int | None:
    for i, label in enumerate(class_names):
        normalized = label.strip().lower().replace("_", " ").replace("-", " ")
        if normalized in {"benign", "normal", "normal traffic", "0"}:
            return i
    return None


def _fallback_label(
    class_names: list[str],
    benign_index: int | None,
    observed_counts: np.ndarray,
) -> str:
    if benign_index is not None:
        return class_names[benign_index]
    if observed_counts.sum() > 0:
        return class_names[int(np.argmax(observed_counts))]
    return class_names[0]


def _metric_fields() -> list[str]:
    return [
        "experiment", "profile", "dataset", "seed", "model", "scope", "window_size",
        "instance_end", "absolute_instance_end", "timestamp_start", "timestamp_end", "n",
        "accuracy", "macro_f1", "balanced_accuracy", "gmean", "mcc",
        "dynamic_minority_recall", "dynamic_minority_attack_recall",
        "worst_class_recall", "worst_attack_recall", "benign_fpr", "active_classes",
    ]


def _class_metric_fields() -> list[str]:
    return [
        "experiment", "profile", "dataset", "seed", "model", "scope", "window_size",
        "instance_end", "absolute_instance_end", "class", "support", "predicted", "precision",
        "recall", "f1", "prior", "is_dynamic_minority",
    ]


def _imbalance_fields() -> list[str]:
    return [
        "experiment", "profile", "dataset", "seed", "instance_end", "absolute_instance_end",
        "timestamp_start", "timestamp_end", "class", "support", "prior", "local_ir",
        "class_present", "is_dynamic_minority", "jsd_from_previous_block",
    ]


def _sampling_fields() -> list[str]:
    return [
        "experiment", "profile", "dataset", "seed", "instance_end", "absolute_instance_end",
        "moob_lambda_mean", "moob_lambda_max", "moob_cap", "moob_cap_hits", "moob_cap_hit_rate",
        "muob_lambda_mean", "muob_lambda_min",
    ]


def _resource_fields() -> list[str]:
    return [
        "experiment", "profile", "dataset", "seed", "instance_end", "absolute_instance_end",
        "elapsed_seconds", "throughput_instances_per_second", "rss_mb", "cpu_percent",
    ]


def _confusion_fields() -> list[str]:
    return [
        "experiment", "profile", "dataset", "seed", "model", "scope", "window_size",
        "instance_end", "absolute_instance_end", "true_class", "predicted_class", "count",
    ]


def _write_metric_scope(
    writers: dict[str, CsvAppender],
    cfg: dict[str, Any],
    dataset_name: str,
    seed: int,
    model_name: str,
    scope: str,
    window_size: int | None,
    cm: np.ndarray,
    class_names: list[str],
    benign_index: int | None,
    instance_end: int,
    absolute_instance_end: int,
    timestamp_start: int | None,
    timestamp_end: int | None,
) -> None:
    summary, class_rows = confusion_metrics(cm, class_names, benign_index)
    base = {
        "experiment": cfg["experiment"]["name"],
        "profile": cfg["experiment"]["profile"],
        "dataset": dataset_name,
        "seed": seed,
        "model": model_name,
        "scope": scope,
        "window_size": window_size,
        "instance_end": instance_end,
        "absolute_instance_end": absolute_instance_end,
        "timestamp_start": _iso_from_ns(timestamp_start),
        "timestamp_end": _iso_from_ns(timestamp_end),
    }
    writers["metrics"].write({**base, **summary})

    if cfg["evaluation"]["store_classwise_metrics"]:
        writers["class_metrics"].write_many({**base, **row} for row in class_rows)

    if cfg["evaluation"]["store_confusion_matrices"]:
        rows = []
        for i, true_label in enumerate(class_names):
            for j, predicted_label in enumerate(class_names):
                count = int(cm[i, j])
                if count:
                    rows.append(
                        {
                            **base,
                            "true_class": true_label,
                            "predicted_class": predicted_label,
                            "count": count,
                        }
                    )
        writers["confusion"].write_many(rows)


def run_seed(
    cfg: dict[str, Any],
    dataset_path: str | Path,
    output_root: str | Path,
    seed: int,
) -> Path:
    logging.basicConfig(
        level=getattr(logging, str(cfg["runtime"]["log_level"]).upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    inspection = inspect_dataset(dataset_path, cfg)
    label_column = inspection["label_column"]
    timestamp_column = inspection["timestamp_column"]
    feature_columns = inspection["feature_columns"]

    ds_cfg = cfg["dataset"]
    configured_classes = ds_cfg.get("class_names")
    if configured_classes:
        class_names = [str(label) for label in configured_classes]
    else:
        class_names = discover_classes(
            path=dataset_path,
            label_column=label_column,
            batch_size=int(ds_cfg["batch_size"]),
            row_start=int(ds_cfg["row_start"]),
            max_rows=ds_cfg["max_rows"],
            use_threads=bool(ds_cfg["use_threads"]),
            memory_map=bool(ds_cfg["memory_map"]),
            pre_buffer=bool(ds_cfg["pre_buffer"]),
        )
    if len(class_names) < 2:
        raise ValueError(f"At least two classes are required, found: {class_names}")

    class_to_index = {label: i for i, label in enumerate(class_names)}
    benign_index = _find_benign_index(class_names)
    dataset_name = dataset_path.stem if dataset_path.is_file() else dataset_path.name
    run_dir = output_root / dataset_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint_latest.pkl"
    complete_path = run_dir / "COMPLETED.json"

    if complete_path.exists() and not cfg["runtime"]["overwrite"]:
        LOGGER.info("Run already completed: %s", run_dir)
        return run_dir
    if cfg["runtime"]["overwrite"] and run_dir.exists():
        for filename in (
            "block_metrics.csv", "class_metrics.csv", "imbalance_metrics.csv",
            "resource_metrics.csv", "sampling_metrics.csv", "confusion_matrices.csv",
            "checkpoint_latest.pkl", "COMPLETED.json",
        ):
            target = run_dir / filename
            if target.exists():
                target.unlink()

    resolved_cfg = json.loads(json.dumps({k: v for k, v in cfg.items() if not str(k).startswith("_")}))
    resolved_cfg["dataset"]["path"] = str(dataset_path)
    resolved_cfg["dataset"]["label_column"] = label_column
    resolved_cfg["dataset"]["timestamp_column"] = timestamp_column
    resolved_cfg["dataset"]["feature_include"] = feature_columns
    resolved_cfg["dataset"]["class_names"] = class_names
    resolved_cfg["runtime"]["output_root"] = str(output_root)
    save_config(resolved_cfg, run_dir / "config_resolved.yaml")

    manifest = {
        **environment_manifest(),
        "numpy": np.__version__,
        "dataset_inspection": inspection,
        "class_names": class_names,
        "benign_class": None if benign_index is None else class_names[benign_index],
        "dataset_fingerprints": [fast_file_fingerprint(file) for file in inspection["files"]],
    }
    try:
        import river
        manifest["river"] = river.__version__
    except Exception:
        manifest["river"] = "unknown"
    try:
        import pyarrow
        manifest["pyarrow"] = pyarrow.__version__
    except Exception:
        manifest["pyarrow"] = "unknown"
    atomic_json_dump(manifest, run_dir / "manifest.json")

    models: list[StreamClassifier]
    tracker: DynamicClassSizeTracker | None
    metric_states: dict[str, ModelMetricState]
    processed = 0
    block_index = 0
    previous_block_counts: np.ndarray | None = None
    observed_counts = np.zeros(len(class_names), dtype=np.int64)
    block_class_counts = np.zeros(len(class_names), dtype=np.int64)
    block_timestamp_start: int | None = None
    block_timestamp_end: int | None = None
    cumulative_timestamp_start: int | None = None
    timestamp_windows = {
        int(window): [] for window in cfg["evaluation"]["rolling_windows"]
    }
    moob_lambda_sum = 0.0
    moob_lambda_max = 0.0
    moob_cap_hits = 0
    muob_lambda_sum = 0.0
    muob_lambda_min = float("inf")
    rate_count = 0
    start_time = time.perf_counter()

    if checkpoint_path.exists() and cfg["runtime"]["resume"]:
        state = pickle_load(checkpoint_path)
        if state["seed"] != seed or state["class_names"] != class_names:
            raise ValueError("Checkpoint does not match this run")
        models = state["models"]
        tracker = state["tracker"]
        metric_states = state["metric_states"]
        processed = int(state["processed"])
        block_index = int(state["block_index"])
        previous_block_counts = state["previous_block_counts"]
        observed_counts = state["observed_counts"]
        cumulative_timestamp_start = state.get("cumulative_timestamp_start")
        timestamp_windows = state.get("timestamp_windows", timestamp_windows)
        start_time = time.perf_counter() - float(state.get("elapsed_seconds", 0.0))
        for filename in (
            "block_metrics.csv", "class_metrics.csv", "imbalance_metrics.csv",
            "resource_metrics.csv", "sampling_metrics.csv", "confusion_matrices.csv",
        ):
            truncate_csv_after(run_dir / filename, "instance_end", processed)
        LOGGER.info("Resuming seed %s at instance %s", seed, processed)
    else:
        models, tracker = build_models(cfg, class_names, seed)
        metric_states = {
            model.name: ModelMetricState(
                n_classes=len(class_names),
                block_size=int(cfg["evaluation"]["block_size"]),
                rolling_windows=cfg["evaluation"]["rolling_windows"],
            )
            for model in models
        }

    if not models:
        LOGGER.info("No models assigned to seed %s", seed)
        atomic_json_dump({"seed": seed, "status": "NO_MODELS"}, complete_path)
        return run_dir

    writers = {
        "metrics": CsvAppender(run_dir / "block_metrics.csv", _metric_fields()),
        "class_metrics": CsvAppender(run_dir / "class_metrics.csv", _class_metric_fields()),
        "imbalance": CsvAppender(run_dir / "imbalance_metrics.csv", _imbalance_fields()),
        "resource": CsvAppender(run_dir / "resource_metrics.csv", _resource_fields()),
        "sampling": CsvAppender(run_dir / "sampling_metrics.csv", _sampling_fields()),
        "confusion": CsvAppender(run_dir / "confusion_matrices.csv", _confusion_fields()),
    }

    block_size = int(cfg["evaluation"]["block_size"])
    checkpoint_every = int(cfg["evaluation"]["checkpoint_every"])
    process = psutil.Process(os.getpid())
    stream = ParquetTemporalStream(
        path=dataset_path,
        feature_columns=feature_columns,
        label_column=label_column,
        timestamp_column=timestamp_column,
        batch_size=int(ds_cfg["batch_size"]),
        row_start=int(ds_cfg["row_start"]) + processed,
        max_rows=None if ds_cfg["max_rows"] is None else int(ds_cfg["max_rows"]) - processed,
        use_threads=bool(ds_cfg["use_threads"]),
        memory_map=bool(ds_cfg["memory_map"]),
        pre_buffer=bool(ds_cfg["pre_buffer"]),
        validate_chronology=bool(ds_cfg["validate_chronology"]),
        chronology_tolerance=int(ds_cfg["chronology_tolerance"]),
        fail_on_nonfinite=bool(cfg["runtime"]["fail_on_nonfinite"]),
    )

    model_runtime = []
    for model in models:
        if model.name.startswith("MOOB"):
            mode = "moob"
        elif model.name == "MUOB":
            mode = "muob"
        else:
            mode = "base"
        model_runtime.append((model, metric_states[model.name], mode))

    try:
        for batch in stream:
            for row_idx in range(batch.size):
                x = dict(zip(feature_columns, batch.X[row_idx], strict=True))
                y = batch.y[row_idx]
                y_index = class_to_index[y]
                timestamp_ns = None if batch.timestamps_ns is None else int(batch.timestamps_ns[row_idx])
                if block_timestamp_start is None:
                    block_timestamp_start = timestamp_ns
                if cumulative_timestamp_start is None:
                    cumulative_timestamp_start = timestamp_ns
                block_timestamp_end = timestamp_ns

                fallback = _fallback_label(class_names, benign_index, observed_counts)
                for model, metric_state, _ in model_runtime:
                    prediction = model.predict_one(x, fallback)
                    prediction_index = class_to_index.get(prediction)
                    if prediction_index is None:
                        prediction_index = class_to_index[fallback]
                    metric_state.update(y_index, prediction_index)

                rates = tracker.update_and_get_rates(y) if tracker is not None else None
                if rates is not None:
                    moob_lambda_sum += rates.moob
                    moob_lambda_max = max(moob_lambda_max, rates.moob)
                    configured_cap = cfg["models"].get("moob_lambda_cap")
                    if configured_cap is not None and rates.moob >= float(configured_cap):
                        moob_cap_hits += 1
                    muob_lambda_sum += rates.muob
                    muob_lambda_min = min(muob_lambda_min, rates.muob)
                    rate_count += 1
                for model, _, mode in model_runtime:
                    if mode == "moob":
                        sampling_lambda = rates.moob if rates is not None else 1.0
                    elif mode == "muob":
                        sampling_lambda = rates.muob if rates is not None else 1.0
                    else:
                        sampling_lambda = 1.0
                    model.learn_one(x, y, sampling_lambda)

                observed_counts[y_index] += 1
                block_class_counts[y_index] += 1
                processed += 1

                if processed % block_size != 0:
                    continue

                block_index += 1
                absolute_end = int(ds_cfg["row_start"]) + processed
                jsd = float("nan") if previous_block_counts is None else js_divergence(block_class_counts, previous_block_counts)
                total_block = int(block_class_counts.sum())
                max_count = int(block_class_counts.max()) if total_block else 0
                for class_index, label in enumerate(class_names):
                    support = int(block_class_counts[class_index])
                    prior = support / total_block if total_block else float("nan")
                    local_ir = max_count / support if support > 0 else float("nan")
                    writers["imbalance"].write(
                        {
                            "experiment": cfg["experiment"]["name"],
                            "profile": cfg["experiment"]["profile"],
                            "dataset": dataset_name,
                            "seed": seed,
                            "instance_end": processed,
                            "absolute_instance_end": absolute_end,
                            "timestamp_start": _iso_from_ns(block_timestamp_start),
                            "timestamp_end": _iso_from_ns(block_timestamp_end),
                            "class": label,
                            "support": support,
                            "prior": prior,
                            "local_ir": local_ir,
                            "class_present": support > 0,
                            "is_dynamic_minority": support > 0 and prior < 1.0 / len(class_names),
                            "jsd_from_previous_block": jsd,
                        }
                    )
                previous_block_counts = block_class_counts.copy()
                block_class_counts.fill(0)

                for window_size in timestamp_windows:
                    ranges = timestamp_windows[window_size]
                    ranges.append((block_timestamp_start, block_timestamp_end))
                    max_blocks = int(window_size) // block_size
                    if len(ranges) > max_blocks:
                        del ranges[0]

                for model, state, _ in model_runtime:
                    state.close_block()
                    _write_metric_scope(
                        writers, cfg, dataset_name, seed, model.name,
                        "cumulative", None, state.cumulative, class_names, benign_index,
                        processed, absolute_end, cumulative_timestamp_start, block_timestamp_end,
                    )
                    for window_size, accumulator in state.windows.items():
                        if accumulator.ready:
                            ranges = timestamp_windows[window_size]
                            rolling_start = ranges[0][0] if ranges else None
                            rolling_end = ranges[-1][1] if ranges else block_timestamp_end
                            _write_metric_scope(
                                writers, cfg, dataset_name, seed, model.name,
                                "rolling", window_size, accumulator.total, class_names, benign_index,
                                processed, absolute_end, rolling_start, rolling_end,
                            )

                if rate_count:
                    configured_cap = cfg["models"].get("moob_lambda_cap")
                    writers["sampling"].write(
                        {
                            "experiment": cfg["experiment"]["name"],
                            "profile": cfg["experiment"]["profile"],
                            "dataset": dataset_name,
                            "seed": seed,
                            "instance_end": processed,
                            "absolute_instance_end": absolute_end,
                            "moob_lambda_mean": moob_lambda_sum / rate_count,
                            "moob_lambda_max": moob_lambda_max,
                            "moob_cap": configured_cap,
                            "moob_cap_hits": moob_cap_hits,
                            "moob_cap_hit_rate": moob_cap_hits / rate_count,
                            "muob_lambda_mean": muob_lambda_sum / rate_count,
                            "muob_lambda_min": muob_lambda_min,
                        }
                    )
                moob_lambda_sum = 0.0
                moob_lambda_max = 0.0
                moob_cap_hits = 0
                muob_lambda_sum = 0.0
                muob_lambda_min = float("inf")
                rate_count = 0

                elapsed = time.perf_counter() - start_time
                writers["resource"].write(
                    {
                        "experiment": cfg["experiment"]["name"],
                        "profile": cfg["experiment"]["profile"],
                        "dataset": dataset_name,
                        "seed": seed,
                        "instance_end": processed,
                        "absolute_instance_end": absolute_end,
                        "elapsed_seconds": elapsed,
                        "throughput_instances_per_second": processed / elapsed if elapsed > 0 else float("nan"),
                        "rss_mb": process.memory_info().rss / (1024 * 1024),
                        "cpu_percent": process.cpu_percent(interval=None),
                    }
                )
                block_timestamp_start = None
                block_timestamp_end = None

                if processed % checkpoint_every == 0:
                    for writer in writers.values():
                        writer.flush()
                    atomic_pickle_dump(
                        {
                            "seed": seed,
                            "class_names": class_names,
                            "models": models,
                            "tracker": tracker,
                            "metric_states": metric_states,
                            "processed": processed,
                            "block_index": block_index,
                            "previous_block_counts": previous_block_counts,
                            "observed_counts": observed_counts,
                            "cumulative_timestamp_start": cumulative_timestamp_start,
                            "timestamp_windows": timestamp_windows,
                            "elapsed_seconds": elapsed,
                        },
                        checkpoint_path,
                    )
                    LOGGER.info(
                        "seed=%s processed=%s throughput=%.1f/s rss=%.1f MB",
                        seed, processed, processed / elapsed, process.memory_info().rss / (1024 * 1024),
                    )

                if block_index % int(cfg["runtime"]["gc_every_blocks"]) == 0:
                    gc.collect()

        if processed % block_size != 0 and any(state.block.sum() for state in metric_states.values()):
            absolute_end = int(ds_cfg["row_start"]) + processed
            for model, state, _ in model_runtime:
                state.close_block()
                _write_metric_scope(
                    writers, cfg, dataset_name, seed, model.name,
                    "cumulative", None, state.cumulative, class_names, benign_index,
                    processed, absolute_end, cumulative_timestamp_start, block_timestamp_end,
                )

        for writer in writers.values():
            writer.flush()
        elapsed = time.perf_counter() - start_time
        atomic_json_dump(
            {
                "status": "COMPLETED",
                "seed": seed,
                "processed": processed,
                "elapsed_seconds": elapsed,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "models": [model.name for model in models],
            },
            complete_path,
        )
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        LOGGER.info("Completed seed %s: %s instances", seed, processed)
        return run_dir
    finally:
        for writer in writers.values():
            writer.close()


def run_experiment(
    cfg: dict[str, Any],
    dataset_path: str | Path,
    output_root: str | Path,
    seeds: list[int] | None = None,
) -> list[Path]:
    selected = [int(seed) for seed in (seeds or cfg["experiment"]["seeds"])]
    return [run_seed(cfg, dataset_path, output_root, seed) for seed in selected]
