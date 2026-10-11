#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


PROJECT_ROOT_DEFAULT = Path(
    os.environ.get("E1_PROJECT_ROOT", ".")
).resolve()

ANALYSIS_READY_DEFAULT = (
    PROJECT_ROOT_DEFAULT
    / "outputs"
    / "E1_FINAL_CORE_FULL_v1_1"
    / "_analysis_ready"
).resolve()

OUTPUT_DEFAULT = (
    PROJECT_ROOT_DEFAULT
    / "outputs"
    / "E1_FINAL_ANALYSIS_v1_0"
).resolve()

MODEL_ORDER = ["HT", "OB", "MOOB", "MUOB"]
STOCHASTIC_MODELS = {"OB", "MOOB", "MUOB"}

PRIMARY_METRICS = [
    "macro_f1",
    "balanced_accuracy",
    "gmean",
    "dynamic_minority_attack_recall",
    "worst_attack_recall",
    "benign_fpr",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "balanced_accuracy": "Balanced Accuracy",
    "gmean": "Multiclass G-Mean",
    "mcc": "MCC",
    "dynamic_minority_recall": "Dynamic Minority Recall",
    "dynamic_minority_attack_recall": "Dynamic Minority Attack Recall",
    "worst_class_recall": "Worst-Class Recall",
    "worst_attack_recall": "Worst-Attack Recall",
    "benign_fpr": "Benign FPR",
}

HIGHER_IS_BETTER = {
    "accuracy": True,
    "macro_f1": True,
    "balanced_accuracy": True,
    "gmean": True,
    "mcc": True,
    "dynamic_minority_recall": True,
    "dynamic_minority_attack_recall": True,
    "worst_class_recall": True,
    "worst_attack_recall": True,
    "benign_fpr": False,
}

EXPECTED_ARTIFACT_COLUMNS = {
    "source_protocol",
    "canonical_dataset",
    "run_dataset",
    "algorithm",
    "seed",
    "complete",
    "processed",
    "expected_rows",
    "processed_ok",
    "run_dir",
    "n_features",
    "block_metrics_csv",
    "class_metrics_csv",
    "confusion_matrices_csv",
    "imbalance_metrics_csv",
    "sampling_metrics_csv",
    "resource_metrics_csv",
    "manifest_json",
    "config_resolved_yaml",
    "COMPLETED_json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_or_nan(x: Any) -> float:
    try:
        x = float(x)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def multiclass_mcc(cm: np.ndarray) -> float:
    cm = cm.astype(np.float64, copy=False)
    total = cm.sum()
    if total <= 0:
        return float("nan")
    true_sum = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)
    trace = np.trace(cm)
    numerator = trace * total - np.dot(true_sum, pred_sum)
    denominator = math.sqrt(
        max(total * total - np.dot(pred_sum, pred_sum), 0.0)
        * max(total * total - np.dot(true_sum, true_sum), 0.0)
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def confusion_metrics(
    cm: np.ndarray,
    class_names: list[str],
    benign_index: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cm = cm.astype(np.float64, copy=False)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    recall = safe_divide(tp, support)
    precision = safe_divide(tp, predicted)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    active = support > 0
    total = support.sum()

    if active.any():
        active_recall = recall[active]
        active_f1 = f1[active]
        balanced_accuracy = float(active_recall.mean())
        macro_f1 = float(active_f1.mean())
        worst_class_recall = float(active_recall.min())
        gmean = (
            0.0
            if np.any(active_recall <= 0)
            else float(np.exp(np.log(active_recall).mean()))
        )
    else:
        balanced_accuracy = macro_f1 = worst_class_recall = gmean = float("nan")

    if total > 0:
        priors = safe_divide(support, np.full_like(support, total))
    else:
        priors = np.zeros_like(support)

    minority = active & (priors < 1.0 / len(class_names))
    dynamic_minority_recall = (
        float(recall[minority].mean()) if minority.any() else float("nan")
    )

    attack_mask = active.copy()
    if benign_index is not None:
        attack_mask[benign_index] = False

    minority_attack = minority & attack_mask
    dynamic_minority_attack_recall = (
        float(recall[minority_attack].mean())
        if minority_attack.any()
        else float("nan")
    )
    worst_attack_recall = (
        float(recall[attack_mask].min()) if attack_mask.any() else float("nan")
    )

    benign_fpr = float("nan")
    if benign_index is not None and support[benign_index] > 0:
        benign_fpr = float(
            (support[benign_index] - cm[benign_index, benign_index])
            / support[benign_index]
        )

    summary = {
        "n": int(total),
        "accuracy": float(tp.sum() / total) if total > 0 else float("nan"),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "gmean": gmean,
        "mcc": multiclass_mcc(cm),
        "dynamic_minority_recall": dynamic_minority_recall,
        "dynamic_minority_attack_recall": dynamic_minority_attack_recall,
        "worst_class_recall": worst_class_recall,
        "worst_attack_recall": worst_attack_recall,
        "benign_fpr": benign_fpr,
        "active_classes": int(active.sum()),
    }

    class_rows = []
    for i, label in enumerate(class_names):
        class_rows.append(
            {
                "class": label,
                "support": int(support[i]),
                "predicted": int(predicted[i]),
                "precision": (
                    float(precision[i]) if predicted[i] > 0 else float("nan")
                ),
                "recall": float(recall[i]) if support[i] > 0 else float("nan"),
                "f1": float(f1[i]) if support[i] > 0 else float("nan"),
                "prior": float(priors[i]) if total > 0 else float("nan"),
                "is_dynamic_minority": bool(minority[i]),
            }
        )
    return summary, class_rows


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64, copy=False)
    q = q.astype(np.float64, copy=False)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def normalize_benign(label: str) -> bool:
    x = str(label).strip().lower().replace("_", " ").replace("-", " ")
    return x in {"benign", "normal", "normal traffic", "0"}


def find_benign_index(class_names: list[str]) -> int | None:
    for i, label in enumerate(class_names):
        if normalize_benign(label):
            return i
    return None


def read_config_classes(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    classes = cfg.get("dataset", {}).get("class_names", []) or []
    return [str(x) for x in classes]


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "audit": root / "00_audit",
        "master": root / "01_master",
        "tables": root / "02_tables",
        "figures": root / "03_figures",
        "stats": root / "04_statistics",
        "report": root / "05_report",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


@dataclass
class AnalysisConfig:
    project_root: Path
    analysis_ready: Path
    output_root: Path
    block_size: int = 50_000
    primary_window: int = 500_000
    rolling_windows: tuple[int, ...] = (250_000, 500_000, 1_000_000)
    bootstrap_reps: int = 5000
    bootstrap_block_length: int = 20
    bootstrap_seed: int = 20260805
    alpha: float = 0.05
    support_thresholds: tuple[int, ...] = (10, 50, 100)
    recall_collapse_thresholds: tuple[float, ...] = (0.5, 0.2)
    dpi: int = 180


def load_analysis_config(path: Path | None, args: argparse.Namespace) -> AnalysisConfig:
    raw: dict[str, Any] = {}
    if path is not None:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    project_root = Path(
        args.project_root
        or raw.get("project_root")
        or PROJECT_ROOT_DEFAULT
    ).expanduser().resolve()

    analysis_ready = Path(
        args.analysis_ready
        or raw.get("analysis_ready")
        or (
            project_root
            / "outputs"
            / "E1_FINAL_CORE_FULL_v1_1"
            / "_analysis_ready"
        )
    ).expanduser().resolve()

    output_root = Path(
        args.output_root
        or raw.get("output_root")
        or (
            project_root
            / "outputs"
            / "E1_FINAL_ANALYSIS_v1_0"
        )
    ).expanduser().resolve()

    return AnalysisConfig(
        project_root=project_root,
        analysis_ready=analysis_ready,
        output_root=output_root,
        block_size=int(raw.get("block_size", 50_000)),
        primary_window=int(raw.get("primary_window", 500_000)),
        rolling_windows=tuple(
            int(x) for x in raw.get("rolling_windows", [250_000, 500_000, 1_000_000])
        ),
        bootstrap_reps=int(raw.get("bootstrap_reps", 5000)),
        bootstrap_block_length=int(raw.get("bootstrap_block_length", 20)),
        bootstrap_seed=int(raw.get("bootstrap_seed", 20260805)),
        alpha=float(raw.get("alpha", 0.05)),
        support_thresholds=tuple(
            int(x) for x in raw.get("support_thresholds", [10, 50, 100])
        ),
        recall_collapse_thresholds=tuple(
            float(x) for x in raw.get("recall_collapse_thresholds", [0.5, 0.2])
        ),
        dpi=int(raw.get("dpi", 180)),
    )


def audit_analysis_ready(cfg: AnalysisConfig, dirs: dict[str, Path]) -> pd.DataFrame:
    artifact_path = cfg.analysis_ready / "artifact_index.csv"
    manifest_path = cfg.analysis_ready / "final_manifest.json"

    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact_index.csv not found: {artifact_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"final_manifest.json not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(
            f"E1 final manifest status is {manifest.get('status')!r}, expected COMPLETE."
        )

    artifacts = pd.read_csv(artifact_path)
    missing_cols = sorted(EXPECTED_ARTIFACT_COLUMNS.difference(artifacts.columns))
    if missing_cols:
        raise RuntimeError(
            "artifact_index.csv is missing required columns: " + ", ".join(missing_cols)
        )

    if len(artifacts) != 32:
        raise RuntimeError(f"Expected 32 final E1 runs, found {len(artifacts)}")

    artifacts["complete"] = artifacts["complete"].map(safe_bool)
    artifacts["processed_ok"] = artifacts["processed_ok"].map(safe_bool)

    errors = []
    checks = []
    for _, row in artifacts.iterrows():
        rec = {
            "canonical_dataset": row["canonical_dataset"],
            "algorithm": row["algorithm"],
            "seed": int(row["seed"]),
            "complete": bool(row["complete"]),
            "processed_ok": bool(row["processed_ok"]),
            "n_features": int(row["n_features"]) if pd.notna(row["n_features"]) else None,
        }

        for col in [
            "block_metrics_csv",
            "class_metrics_csv",
            "confusion_matrices_csv",
            "resource_metrics_csv",
            "manifest_json",
            "config_resolved_yaml",
            "COMPLETED_json",
        ]:
            value = row.get(col)
            exists = bool(pd.notna(value)) and Path(str(value)).exists()
            rec[col + "_exists"] = exists
            if not exists:
                errors.append(
                    f"{row['canonical_dataset']} {row['algorithm']} seed={row['seed']}: "
                    f"missing {col} -> {value}"
                )
        checks.append(rec)

    check_df = pd.DataFrame(checks)
    check_df.to_csv(dirs["audit"] / "input_artifact_audit.csv", index=False)

    if errors:
        raise RuntimeError(
            "Required final artifacts are missing:\n - " + "\n - ".join(errors[:30])
        )

    if not artifacts["complete"].all():
        raise RuntimeError("Not all artifact-index runs are COMPLETE.")
    if not artifacts["processed_ok"].all():
        raise RuntimeError("Not all artifact-index runs passed processed-row audit.")
    if not (artifacts["n_features"] == 49).all():
        raise RuntimeError("Not all artifact-index runs have 49 predictive features.")

    # Expected model/seed design.
    expected = []
    seeds = [123456789, 234567891, 345678912, 456789123, 567891234]
    for dataset in sorted(artifacts["canonical_dataset"].unique()):
        expected.append((dataset, "HT", 123456789))
        for model in ["OB", "MOOB", "MUOB"]:
            for seed in seeds:
                expected.append((dataset, model, seed))
    actual = {
        (str(r.canonical_dataset), str(r.algorithm), int(r.seed))
        for r in artifacts.itertuples()
    }
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"Final 32-run design mismatch. missing={missing}, extra={extra}"
        )

    audit_manifest = {
        "audited_at_utc": utc_now(),
        "analysis_ready": str(cfg.analysis_ready),
        "artifact_index_sha256": sha256_file(artifact_path),
        "final_manifest_sha256": sha256_file(manifest_path),
        "runs": len(artifacts),
        "datasets": sorted(artifacts["canonical_dataset"].unique().tolist()),
        "models": MODEL_ORDER,
        "status": "PASSED",
    }
    json_dump(dirs["audit"] / "analysis_input_audit.json", audit_manifest)
    return artifacts


def read_metric_file(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["seed", "instance_end", "absolute_instance_end", "window_size", "n", "support", "predicted"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_master_metrics(
    artifacts: pd.DataFrame,
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    class_frames = []
    resource_frames = []

    for row in artifacts.itertuples(index=False):
        base = {
            "source_protocol": row.source_protocol,
            "canonical_dataset": row.canonical_dataset,
            "run_dataset": row.run_dataset,
            "algorithm": row.algorithm,
            "seed": int(row.seed),
        }

        m = read_metric_file(row.block_metrics_csv)
        for k, v in base.items():
            m[k] = v
        metric_frames.append(m)

        c = read_metric_file(row.class_metrics_csv)
        for k, v in base.items():
            c[k] = v
        class_frames.append(c)

        r = read_metric_file(row.resource_metrics_csv)
        for k, v in base.items():
            r[k] = v
        resource_frames.append(r)

    metrics = pd.concat(metric_frames, ignore_index=True)
    classes = pd.concat(class_frames, ignore_index=True)
    resources = pd.concat(resource_frames, ignore_index=True)

    metrics["window_size"] = pd.to_numeric(metrics["window_size"], errors="coerce")
    classes["window_size"] = pd.to_numeric(classes["window_size"], errors="coerce")

    metrics.to_parquet(
        dirs["master"] / "master_metrics.parquet",
        index=False,
        compression="snappy",
    )
    classes.to_parquet(
        dirs["master"] / "master_class_metrics.parquet",
        index=False,
        compression="snappy",
    )
    resources.to_parquet(
        dirs["master"] / "master_resource_metrics.parquet",
        index=False,
        compression="snappy",
    )
    return metrics, classes, resources


def read_cumulative_confusion(
    path: Path,
    class_names: list[str],
    chunksize: int = 500_000,
) -> dict[int, np.ndarray]:
    idx = {label: i for i, label in enumerate(class_names)}
    frames = []

    usecols = [
        "scope",
        "instance_end",
        "true_class",
        "predicted_class",
        "count",
    ]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        x = chunk[chunk["scope"] == "cumulative"].copy()
        if not x.empty:
            frames.append(x)

    if not frames:
        raise RuntimeError(f"No cumulative confusion rows found in {path}")

    df = pd.concat(frames, ignore_index=True)
    df["instance_end"] = pd.to_numeric(df["instance_end"], errors="raise").astype(int)
    df["count"] = pd.to_numeric(df["count"], errors="raise").astype(np.int64)

    result = {}
    for end, g in df.groupby("instance_end", sort=True):
        cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        ti = g["true_class"].map(idx)
        pi = g["predicted_class"].map(idx)
        if ti.isna().any() or pi.isna().any():
            unknown = sorted(
                set(g.loc[ti.isna(), "true_class"].astype(str))
                | set(g.loc[pi.isna(), "predicted_class"].astype(str))
            )
            raise RuntimeError(f"Unknown classes in confusion matrix: {unknown}")
        np.add.at(
            cm,
            (ti.astype(int).to_numpy(), pi.astype(int).to_numpy()),
            g["count"].to_numpy(dtype=np.int64),
        )
        result[int(end)] = cm
    return result


def derive_nonoverlap_blocks(
    artifacts: pd.DataFrame,
    dirs: dict[str, Path],
    block_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    class_rows_all: list[dict[str, Any]] = []
    data_support_by_run: list[dict[str, Any]] = []

    for row in artifacts.itertuples(index=False):
        class_names = read_config_classes(Path(row.config_resolved_yaml))
        benign_index = find_benign_index(class_names)
        cumulative = read_cumulative_confusion(
            Path(row.confusion_matrices_csv),
            class_names,
        )

        prev_end = 0
        prev_cm = np.zeros(
            (len(class_names), len(class_names)),
            dtype=np.int64,
        )

        for block_index, end in enumerate(sorted(cumulative), start=1):
            current = cumulative[end]
            delta = current - prev_cm
            if np.any(delta < 0):
                raise RuntimeError(
                    f"Cumulative confusion decreased: "
                    f"{row.canonical_dataset} {row.algorithm} seed={row.seed} end={end}"
                )

            n_expected = end - prev_end
            n_observed = int(delta.sum())
            if n_observed != n_expected:
                raise RuntimeError(
                    f"Derived block count mismatch: "
                    f"{row.canonical_dataset} {row.algorithm} seed={row.seed} "
                    f"end={end}, expected={n_expected}, observed={n_observed}"
                )

            summary, class_rows = confusion_metrics(
                delta,
                class_names,
                benign_index,
            )
            base = {
                "source_protocol": row.source_protocol,
                "canonical_dataset": row.canonical_dataset,
                "run_dataset": row.run_dataset,
                "algorithm": row.algorithm,
                "seed": int(row.seed),
                "block_index": block_index,
                "instance_start": prev_end + 1,
                "instance_end": end,
                "block_n": n_observed,
                "is_full_50k_block": n_observed == block_size,
            }
            metric_rows.append({**base, **summary})
            for cr in class_rows:
                class_rows_all.append({**base, **cr})

            # True-label support is prediction-independent. Keep this for
            # cross-run integrity checking and later select one canonical run.
            support_vec = delta.sum(axis=1)
            for i, label in enumerate(class_names):
                data_support_by_run.append(
                    {
                        **base,
                        "class": label,
                        "support": int(support_vec[i]),
                    }
                )

            prev_end = end
            prev_cm = current

    block_metrics = pd.DataFrame(metric_rows)
    block_classes = pd.DataFrame(class_rows_all)
    support_runs = pd.DataFrame(data_support_by_run)

    block_metrics.to_parquet(
        dirs["master"] / "nonoverlap_block_metrics_derived.parquet",
        index=False,
        compression="snappy",
    )
    block_metrics.to_csv(
        dirs["master"] / "nonoverlap_block_metrics_derived.csv",
        index=False,
    )
    block_classes.to_parquet(
        dirs["master"] / "nonoverlap_block_class_metrics_derived.parquet",
        index=False,
        compression="snappy",
    )

    # Integrity: true-label support must be identical across every algorithm/seed.
    support_check = (
        support_runs
        .groupby(
            ["canonical_dataset", "instance_end", "class"],
            as_index=False,
        )["support"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    support_check["consistent"] = support_check["min"] == support_check["max"]
    support_check.to_csv(
        dirs["audit"] / "true_label_support_consistency.csv",
        index=False,
    )
    if not bool(support_check["consistent"].all()):
        bad = support_check[~support_check["consistent"]].head(20)
        raise RuntimeError(
            "True-label support differs across runs. Example:\n"
            + bad.to_string(index=False)
        )

    return block_metrics, block_classes, support_runs


def metric_consistency_audit(
    master_metrics: pd.DataFrame,
    block_metrics: pd.DataFrame,
    dirs: dict[str, Path],
    tolerance: float = 1e-10,
) -> None:
    # Reconstruct final cumulative metric from cumulative confusion via sum of
    # derived block confusions indirectly using the last cumulative row already
    # encoded in block_metrics derivation. For a strict consistency check, compare
    # final cumulative n and the final raw cumulative endpoint for every run.
    raw = master_metrics[master_metrics["scope"] == "cumulative"].copy()
    raw = (
        raw.sort_values("instance_end")
        .groupby(
            ["canonical_dataset", "algorithm", "seed"],
            as_index=False,
        )
        .tail(1)
    )
    derived_n = (
        block_metrics
        .groupby(
            ["canonical_dataset", "algorithm", "seed"],
            as_index=False,
        )["block_n"]
        .sum()
        .rename(columns={"block_n": "derived_n"})
    )
    check = raw.merge(
        derived_n,
        on=["canonical_dataset", "algorithm", "seed"],
        how="left",
    )
    check["n_match"] = (
        pd.to_numeric(check["n"], errors="coerce").astype("Int64")
        == check["derived_n"].astype("Int64")
    )
    check.to_csv(dirs["audit"] / "metric_endpoint_consistency.csv", index=False)
    if not bool(check["n_match"].all()):
        raise RuntimeError("Final cumulative n differs from derived non-overlap blocks.")


def canonical_data_evidence(
    support_runs: pd.DataFrame,
    artifacts: pd.DataFrame,
    cfg: AnalysisConfig,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    rows = []
    for dataset in sorted(artifacts["canonical_dataset"].unique()):
        candidate = artifacts[
            (artifacts["canonical_dataset"] == dataset)
            & (artifacts["algorithm"] == "HT")
        ]
        if candidate.empty:
            candidate = artifacts[artifacts["canonical_dataset"] == dataset].head(1)
        chosen = candidate.iloc[0]
        x = support_runs[
            (support_runs["canonical_dataset"] == dataset)
            & (support_runs["algorithm"] == chosen["algorithm"])
            & (support_runs["seed"] == int(chosen["seed"]))
        ].copy()
        rows.append(x)

    evidence = pd.concat(rows, ignore_index=True)
    k_by_dataset = evidence.groupby("canonical_dataset")["class"].nunique().to_dict()

    totals = (
        evidence.groupby(
            ["canonical_dataset", "block_index", "instance_start", "instance_end", "block_n"],
            as_index=False,
        )["support"]
        .sum()
        .rename(columns={"support": "block_total"})
    )
    evidence = evidence.merge(
        totals,
        on=[
            "canonical_dataset",
            "block_index",
            "instance_start",
            "instance_end",
            "block_n",
        ],
        how="left",
    )
    evidence["prior"] = evidence["support"] / evidence["block_total"]
    evidence["class_present"] = evidence["support"] > 0
    evidence["is_dynamic_minority"] = False

    local_ir = []
    for (dataset, end), g in evidence.groupby(
        ["canonical_dataset", "instance_end"],
        sort=False,
    ):
        max_count = int(g["support"].max())
        k = int(k_by_dataset[dataset])
        idx = g.index
        present = g["support"] > 0
        evidence.loc[idx, "is_dynamic_minority"] = (
            present & (g["prior"] < 1.0 / k)
        ).to_numpy()
        vals = np.where(
            present.to_numpy(),
            max_count / g["support"].replace(0, np.nan).to_numpy(dtype=float),
            np.nan,
        )
        local_ir.extend(zip(idx.tolist(), vals.tolist()))
    for idx, value in local_ir:
        evidence.loc[idx, "local_ir"] = value

    # JSD at block level.
    jsd_rows = []
    for dataset, g in evidence.groupby("canonical_dataset"):
        classes = sorted(g["class"].unique().tolist())
        prev = None
        for end, b in g.groupby("instance_end", sort=True):
            vec = (
                b.set_index("class")
                .reindex(classes)["support"]
                .fillna(0)
                .to_numpy(dtype=np.float64)
            )
            jsd = float("nan") if prev is None else js_divergence(vec, prev)
            jsd_rows.append(
                {
                    "canonical_dataset": dataset,
                    "instance_end": int(end),
                    "jsd_from_previous_block": jsd,
                }
            )
            prev = vec
    jsd_df = pd.DataFrame(jsd_rows)
    evidence = evidence.merge(
        jsd_df,
        on=["canonical_dataset", "instance_end"],
        how="left",
    )

    evidence.to_parquet(
        dirs["master"] / "dynamic_imbalance_evidence.parquet",
        index=False,
        compression="snappy",
    )
    evidence.to_csv(
        dirs["master"] / "dynamic_imbalance_evidence.csv",
        index=False,
    )
    return evidence


def class_lifecycle_tables(
    evidence: pd.DataFrame,
    cfg: AnalysisConfig,
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lifecycle_rows = []
    event_rows = []
    threshold_rows = []

    for (dataset, cls), g in evidence.groupby(["canonical_dataset", "class"]):
        g = g.sort_values("block_index").copy()
        present = g["class_present"].to_numpy(dtype=bool)
        minority = g["is_dynamic_minority"].to_numpy(dtype=bool)
        supports = g["support"].to_numpy(dtype=int)
        ends = g["instance_end"].to_numpy(dtype=int)

        present_idx = np.flatnonzero(present)
        first_seen = int(ends[present_idx[0]]) if present_idx.size else None
        last_seen = int(ends[present_idx[-1]]) if present_idx.size else None

        disappearance = 0
        recurrence = 0
        minority_switches = 0
        ever_seen = False
        last_present_minority = None
        absence_run = 0
        longest_absence = 0

        for i in range(len(g)):
            if present[i]:
                if not ever_seen:
                    ever_seen = True
                    event_rows.append(
                        {
                            "canonical_dataset": dataset,
                            "class": cls,
                            "event": "first_appearance",
                            "block_index": int(g.iloc[i]["block_index"]),
                            "instance_end": int(ends[i]),
                        }
                    )
                elif i > 0 and not present[i - 1]:
                    recurrence += 1
                    event_rows.append(
                        {
                            "canonical_dataset": dataset,
                            "class": cls,
                            "event": "recurrence",
                            "block_index": int(g.iloc[i]["block_index"]),
                            "instance_end": int(ends[i]),
                        }
                    )

                if last_present_minority is not None and minority[i] != last_present_minority:
                    minority_switches += 1
                    event_rows.append(
                        {
                            "canonical_dataset": dataset,
                            "class": cls,
                            "event": "minority_switch",
                            "block_index": int(g.iloc[i]["block_index"]),
                            "instance_end": int(ends[i]),
                            "to_minority": bool(minority[i]),
                        }
                    )
                last_present_minority = bool(minority[i])
                longest_absence = max(longest_absence, absence_run)
                absence_run = 0
            else:
                absence_run += 1
                if i > 0 and present[i - 1]:
                    disappearance += 1
                    event_rows.append(
                        {
                            "canonical_dataset": dataset,
                            "class": cls,
                            "event": "disappearance",
                            "block_index": int(g.iloc[i]["block_index"]),
                            "instance_end": int(ends[i]),
                        }
                    )
        longest_absence = max(longest_absence, absence_run)

        lifecycle_rows.append(
            {
                "canonical_dataset": dataset,
                "class": cls,
                "is_benign": normalize_benign(cls),
                "first_seen_instance_end": first_seen,
                "last_seen_instance_end": last_seen,
                "present_blocks": int(present.sum()),
                "absent_blocks": int((~present).sum()),
                "presence_fraction": float(present.mean()),
                "minority_blocks": int((present & minority).sum()),
                "minority_fraction_when_present": (
                    float(minority[present].mean()) if present.any() else float("nan")
                ),
                "minority_switches": minority_switches,
                "disappearance_events": disappearance,
                "recurrence_events": recurrence,
                "longest_absence_run_blocks": int(longest_absence),
                "min_positive_support": (
                    int(supports[present].min()) if present.any() else 0
                ),
                "median_positive_support": (
                    float(np.median(supports[present])) if present.any() else float("nan")
                ),
                "max_support": int(supports.max()),
                "min_positive_prior": (
                    float(g.loc[g["support"] > 0, "prior"].min())
                    if present.any()
                    else float("nan")
                ),
                "max_prior": float(g["prior"].max()),
                "max_local_ir": (
                    float(g["local_ir"].max())
                    if g["local_ir"].notna().any()
                    else float("nan")
                ),
            }
        )

        for threshold in cfg.support_thresholds:
            threshold_rows.append(
                {
                    "canonical_dataset": dataset,
                    "class": cls,
                    "support_threshold": threshold,
                    "present_blocks": int(present.sum()),
                    "blocks_below_threshold_when_present": int(
                        ((supports > 0) & (supports < threshold)).sum()
                    ),
                    "fraction_below_threshold_when_present": (
                        float(((supports[present] < threshold).mean()))
                        if present.any()
                        else float("nan")
                    ),
                }
            )

    lifecycle = pd.DataFrame(lifecycle_rows)
    events = pd.DataFrame(event_rows)
    thresholds = pd.DataFrame(threshold_rows)

    lifecycle.to_csv(dirs["tables"] / "table_class_lifecycle.csv", index=False)
    events.to_csv(dirs["tables"] / "table_lifecycle_events.csv", index=False)
    thresholds.to_csv(
        dirs["tables"] / "table_support_threshold_sensitivity.csv",
        index=False,
    )

    summary = []
    for dataset, g in evidence.groupby("canonical_dataset"):
        by_block = g.groupby("instance_end")
        summary.append(
            {
                "canonical_dataset": dataset,
                "blocks_total": int(g["block_index"].nunique()),
                "classes_total": int(g["class"].nunique()),
                "mean_active_classes_per_block": float(
                    by_block["class_present"].sum().mean()
                ),
                "min_active_classes_per_block": int(
                    by_block["class_present"].sum().min()
                ),
                "max_active_classes_per_block": int(
                    by_block["class_present"].sum().max()
                ),
                "max_local_ir_observed": float(g["local_ir"].max()),
                "mean_jsd_adjacent_blocks": float(
                    g.drop_duplicates("instance_end")["jsd_from_previous_block"].mean()
                ),
                "max_jsd_adjacent_blocks": float(
                    g.drop_duplicates("instance_end")["jsd_from_previous_block"].max()
                ),
                "disappearance_events": int(
                    (events["canonical_dataset"].eq(dataset) & events["event"].eq("disappearance")).sum()
                ) if not events.empty else 0,
                "recurrence_events": int(
                    (events["canonical_dataset"].eq(dataset) & events["event"].eq("recurrence")).sum()
                ) if not events.empty else 0,
                "minority_switch_events": int(
                    (events["canonical_dataset"].eq(dataset) & events["event"].eq("minority_switch")).sum()
                ) if not events.empty else 0,
            }
        )
    pd.DataFrame(summary).to_csv(
        dirs["tables"] / "table_dynamic_imbalance_summary.csv",
        index=False,
    )
    return lifecycle, events, thresholds


def aggregate_final_cumulative(
    metrics: pd.DataFrame,
    classes: pd.DataFrame,
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = metrics[metrics["scope"] == "cumulative"].copy()
    final = (
        final.sort_values("instance_end")
        .groupby(
            ["canonical_dataset", "algorithm", "seed"],
            as_index=False,
        )
        .tail(1)
    )

    rows = []
    metric_cols = [
        c for c in METRIC_LABELS
        if c in final.columns
    ]
    for (dataset, model), g in final.groupby(["canonical_dataset", "algorithm"]):
        rec = {
            "canonical_dataset": dataset,
            "algorithm": model,
            "n_seeds": int(g["seed"].nunique()),
            "processed": int(g["instance_end"].max()),
        }
        for metric in metric_cols:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna()
            rec[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            rec[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            rec[f"{metric}_min"] = float(vals.min()) if len(vals) else np.nan
            rec[f"{metric}_max"] = float(vals.max()) if len(vals) else np.nan
        rows.append(rec)
    final_summary = pd.DataFrame(rows)
    final_summary["model_order"] = final_summary["algorithm"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    )
    final_summary = final_summary.sort_values(
        ["canonical_dataset", "model_order"]
    ).drop(columns=["model_order"])
    final_summary.to_csv(
        dirs["tables"] / "table_final_cumulative_model_summary.csv",
        index=False,
    )

    final_class = classes[classes["scope"] == "cumulative"].copy()
    final_class = (
        final_class.sort_values("instance_end")
        .groupby(
            ["canonical_dataset", "algorithm", "seed", "class"],
            as_index=False,
        )
        .tail(1)
    )
    class_rows = []
    for (dataset, model, cls), g in final_class.groupby(
        ["canonical_dataset", "algorithm", "class"]
    ):
        class_rows.append(
            {
                "canonical_dataset": dataset,
                "algorithm": model,
                "class": cls,
                "is_benign": normalize_benign(cls),
                "n_seeds": int(g["seed"].nunique()),
                "support": int(pd.to_numeric(g["support"], errors="coerce").max()),
                "prior": float(pd.to_numeric(g["prior"], errors="coerce").mean()),
                "recall_mean": float(pd.to_numeric(g["recall"], errors="coerce").mean()),
                "recall_std": (
                    float(pd.to_numeric(g["recall"], errors="coerce").std(ddof=1))
                    if len(g) > 1
                    else 0.0
                ),
                "f1_mean": float(pd.to_numeric(g["f1"], errors="coerce").mean()),
                "f1_std": (
                    float(pd.to_numeric(g["f1"], errors="coerce").std(ddof=1))
                    if len(g) > 1
                    else 0.0
                ),
                "precision_mean": float(
                    pd.to_numeric(g["precision"], errors="coerce").mean()
                ),
            }
        )
    final_class_summary = pd.DataFrame(class_rows)
    final_class_summary.to_csv(
        dirs["tables"] / "table_final_per_class_performance.csv",
        index=False,
    )
    return final_summary, final_class_summary


def rolling_summary(
    metrics: pd.DataFrame,
    classes: pd.DataFrame,
    cfg: AnalysisConfig,
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rolling = metrics[metrics["scope"] == "rolling"].copy()
    rolling["window_size"] = pd.to_numeric(rolling["window_size"], errors="coerce")

    # First average stochastic seeds at each temporal endpoint. HT naturally has n=1.
    agg_rows = []
    metric_cols = [m for m in METRIC_LABELS if m in rolling.columns]
    group_cols = [
        "canonical_dataset",
        "algorithm",
        "window_size",
        "instance_end",
    ]
    for key, g in rolling.groupby(group_cols):
        rec = dict(zip(group_cols, key))
        rec["n_seeds"] = int(g["seed"].nunique())
        for metric in metric_cols:
            vals = pd.to_numeric(g[metric], errors="coerce")
            rec[f"{metric}_mean_seed"] = float(vals.mean())
            rec[f"{metric}_std_seed"] = (
                float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
            )
        agg_rows.append(rec)
    trajectory = pd.DataFrame(agg_rows)
    trajectory.to_parquet(
        dirs["master"] / "rolling_metrics_seed_aggregated.parquet",
        index=False,
        compression="snappy",
    )
    trajectory.to_csv(
        dirs["master"] / "rolling_metrics_seed_aggregated.csv",
        index=False,
    )

    # Window sensitivity summarizes temporal behavior after seed aggregation.
    sensitivity_rows = []
    for (dataset, model, window), g in trajectory.groupby(
        ["canonical_dataset", "algorithm", "window_size"]
    ):
        rec = {
            "canonical_dataset": dataset,
            "algorithm": model,
            "window_size": int(window),
            "temporal_points": len(g),
        }
        for metric in PRIMARY_METRICS:
            col = f"{metric}_mean_seed"
            if col not in g:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            rec[f"{metric}_temporal_mean"] = float(vals.mean()) if len(vals) else np.nan
            rec[f"{metric}_temporal_std"] = (
                float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            )
            rec[f"{metric}_temporal_min"] = float(vals.min()) if len(vals) else np.nan
            rec[f"{metric}_temporal_p05"] = (
                float(vals.quantile(0.05)) if len(vals) else np.nan
            )
            rec[f"{metric}_temporal_p95"] = (
                float(vals.quantile(0.95)) if len(vals) else np.nan
            )
            rec[f"{metric}_temporal_max"] = float(vals.max()) if len(vals) else np.nan
        sensitivity_rows.append(rec)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        dirs["tables"] / "table_window_sensitivity.csv",
        index=False,
    )

    # Primary-window per-class temporal vulnerability.
    rc = classes[
        (classes["scope"] == "rolling")
        & (
            pd.to_numeric(classes["window_size"], errors="coerce")
            == cfg.primary_window
        )
    ].copy()

    if rc.empty:
        raise RuntimeError(
            f"No rolling class metrics found for primary window {cfg.primary_window}"
        )

    seed_agg = (
        rc.groupby(
            ["canonical_dataset", "algorithm", "instance_end", "class"],
            as_index=False,
        )
        .agg(
            support=("support", "max"),
            prior=("prior", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            precision=("precision", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )

    vuln_rows = []
    for (dataset, model, cls), g in seed_agg.groupby(
        ["canonical_dataset", "algorithm", "class"]
    ):
        present = g[pd.to_numeric(g["support"], errors="coerce") > 0].copy()
        recalls = pd.to_numeric(present["recall"], errors="coerce").dropna()
        rec = {
            "canonical_dataset": dataset,
            "algorithm": model,
            "class": cls,
            "is_benign": normalize_benign(cls),
            "rolling_window": cfg.primary_window,
            "present_windows": len(present),
            "mean_prior_when_present": float(
                pd.to_numeric(present["prior"], errors="coerce").mean()
            ) if len(present) else np.nan,
            "min_prior_when_present": float(
                pd.to_numeric(present["prior"], errors="coerce").min()
            ) if len(present) else np.nan,
            "recall_mean": float(recalls.mean()) if len(recalls) else np.nan,
            "recall_median": float(recalls.median()) if len(recalls) else np.nan,
            "recall_min": float(recalls.min()) if len(recalls) else np.nan,
            "recall_p05": float(recalls.quantile(0.05)) if len(recalls) else np.nan,
            "zero_recall_windows": int((recalls == 0).sum()) if len(recalls) else 0,
        }
        for threshold in cfg.recall_collapse_thresholds:
            name = str(threshold).replace(".", "p")
            rec[f"windows_recall_below_{name}"] = (
                int((recalls < threshold).sum()) if len(recalls) else 0
            )
            rec[f"fraction_recall_below_{name}"] = (
                float((recalls < threshold).mean()) if len(recalls) else np.nan
            )
        vuln_rows.append(rec)

    vulnerability = pd.DataFrame(vuln_rows)
    vulnerability.to_csv(
        dirs["tables"] / "table_primary_window_class_vulnerability.csv",
        index=False,
    )
    return trajectory, sensitivity, vulnerability


def efficiency_tables(
    artifacts: pd.DataFrame,
    resources: pd.DataFrame,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    rows = []
    for row in artifacts.itertuples(index=False):
        completed = json.loads(Path(row.COMPLETED_json).read_text(encoding="utf-8"))
        r = resources[
            (resources["canonical_dataset"] == row.canonical_dataset)
            & (resources["algorithm"] == row.algorithm)
            & (resources["seed"] == int(row.seed))
        ]
        elapsed = float(completed["elapsed_seconds"])
        processed = int(completed["processed"])
        rows.append(
            {
                "canonical_dataset": row.canonical_dataset,
                "algorithm": row.algorithm,
                "seed": int(row.seed),
                "processed": processed,
                "elapsed_seconds": elapsed,
                "elapsed_hours": elapsed / 3600.0,
                "final_throughput_instances_per_second": (
                    processed / elapsed if elapsed > 0 else np.nan
                ),
                "peak_rss_mb_sampled": float(
                    pd.to_numeric(r["rss_mb"], errors="coerce").max()
                ),
                "median_rss_mb_sampled": float(
                    pd.to_numeric(r["rss_mb"], errors="coerce").median()
                ),
            }
        )
    runs = pd.DataFrame(rows)
    runs.to_csv(dirs["master"] / "efficiency_per_run.csv", index=False)

    summary_rows = []
    for (dataset, model), g in runs.groupby(["canonical_dataset", "algorithm"]):
        summary_rows.append(
            {
                "canonical_dataset": dataset,
                "algorithm": model,
                "n_seeds": int(g["seed"].nunique()),
                "elapsed_hours_mean": float(g["elapsed_hours"].mean()),
                "elapsed_hours_std": (
                    float(g["elapsed_hours"].std(ddof=1)) if len(g) > 1 else 0.0
                ),
                "throughput_mean": float(
                    g["final_throughput_instances_per_second"].mean()
                ),
                "throughput_std": (
                    float(g["final_throughput_instances_per_second"].std(ddof=1))
                    if len(g) > 1
                    else 0.0
                ),
                "peak_rss_mb_mean": float(g["peak_rss_mb_sampled"].mean()),
                "peak_rss_mb_max": float(g["peak_rss_mb_sampled"].max()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        dirs["tables"] / "table_efficiency_summary.csv",
        index=False,
    )
    return summary


def sampling_summary(
    artifacts: pd.DataFrame,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    frames = []
    for row in artifacts.itertuples(index=False):
        p = getattr(row, "sampling_metrics_csv")
        if not isinstance(p, str) or not Path(p).exists():
            continue
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["canonical_dataset"] = row.canonical_dataset
        df["algorithm"] = row.algorithm
        df["seed"] = int(row.seed)
        frames.append(df)

    if not frames:
        out = pd.DataFrame()
        out.to_csv(dirs["tables"] / "table_sampling_summary.csv", index=False)
        return out

    all_s = pd.concat(frames, ignore_index=True)
    all_s.to_parquet(
        dirs["master"] / "master_sampling_metrics.parquet",
        index=False,
        compression="snappy",
    )

    numeric = [
        c for c in [
            "moob_lambda_mean",
            "moob_lambda_max",
            "moob_cap_hits",
            "moob_cap_hit_rate",
            "muob_lambda_mean",
            "muob_lambda_min",
        ] if c in all_s.columns
    ]
    rows = []
    for (dataset, model), g in all_s.groupby(["canonical_dataset", "algorithm"]):
        rec = {
            "canonical_dataset": dataset,
            "algorithm": model,
            "rows": len(g),
            "n_seeds": int(g["seed"].nunique()),
        }
        for c in numeric:
            vals = pd.to_numeric(g[c], errors="coerce")
            rec[f"{c}_mean"] = float(vals.mean())
            rec[f"{c}_max"] = float(vals.max())
            rec[f"{c}_min"] = float(vals.min())
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(dirs["tables"] / "table_sampling_summary.csv", index=False)
    return out


def aggregate_block_seed_means(block_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in METRIC_LABELS if m in block_metrics.columns]
    aggregations = {m: "mean" for m in metrics}
    aggregations["block_n"] = "first"
    aggregations["is_full_50k_block"] = "first"
    out = (
        block_metrics
        .groupby(
            ["canonical_dataset", "algorithm", "block_index", "instance_end"],
            as_index=False,
        )
        .agg(aggregations)
    )
    return out


def moving_block_resample_mean(
    series: np.ndarray,
    block_length: int,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    L = max(1, min(int(block_length), n))
    starts = np.arange(0, n - L + 1)
    if len(starts) == 0:
        return np.repeat(np.mean(x), reps)

    out = np.empty(reps, dtype=float)
    blocks_needed = int(math.ceil(n / L))
    for b in range(reps):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([x[s:s+L] for s in chosen])[:n]
        out[b] = float(np.mean(sample))
    return out


def holm_adjust(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    if m == 0:
        return []
    order = np.argsort(pvalues)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def statistical_comparison(
    block_metrics: pd.DataFrame,
    cfg: AnalysisConfig,
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agg = aggregate_block_seed_means(block_metrics)
    # Uniform non-overlapping 50k blocks only. Exclude each dataset's final partial block.
    agg = agg[agg["is_full_50k_block"].astype(bool)].copy()

    pair_rows = []
    rng = np.random.default_rng(cfg.bootstrap_seed)

    for dataset, d in agg.groupby("canonical_dataset"):
        for metric in PRIMARY_METRICS:
            if metric not in d.columns:
                continue
            higher = HIGHER_IS_BETTER[metric]
            metric_pair_rows = []
            for a, b in combinations(MODEL_ORDER, 2):
                xa = d[d["algorithm"] == a][
                    ["instance_end", metric]
                ].rename(columns={metric: "a"})
                xb = d[d["algorithm"] == b][
                    ["instance_end", metric]
                ].rename(columns={metric: "b"})
                z = xa.merge(xb, on="instance_end", how="inner").dropna()
                if z.empty:
                    continue

                raw_diff = z["a"].to_numpy(float) - z["b"].to_numpy(float)
                advantage = raw_diff if higher else -raw_diff
                observed = float(np.mean(advantage))

                boot = moving_block_resample_mean(
                    advantage,
                    cfg.bootstrap_block_length,
                    cfg.bootstrap_reps,
                    rng,
                )
                lo = (
                    float(np.quantile(boot, cfg.alpha / 2))
                    if len(boot)
                    else np.nan
                )
                hi = (
                    float(np.quantile(boot, 1 - cfg.alpha / 2))
                    if len(boot)
                    else np.nan
                )

                centered = advantage - np.mean(advantage)
                null_boot = moving_block_resample_mean(
                    centered,
                    cfg.bootstrap_block_length,
                    cfg.bootstrap_reps,
                    rng,
                )
                p = (
                    float(
                        (np.sum(np.abs(null_boot) >= abs(observed)) + 1)
                        / (len(null_boot) + 1)
                    )
                    if len(null_boot)
                    else np.nan
                )

                metric_pair_rows.append(
                    {
                        "canonical_dataset": dataset,
                        "metric": metric,
                        "metric_direction": "higher_better" if higher else "lower_better",
                        "model_a": a,
                        "model_b": b,
                        "n_nonoverlap_blocks": len(z),
                        "block_size": cfg.block_size,
                        "bootstrap_block_length_blocks": cfg.bootstrap_block_length,
                        "bootstrap_block_length_instances": (
                            cfg.bootstrap_block_length * cfg.block_size
                        ),
                        "bootstrap_reps": cfg.bootstrap_reps,
                        "mean_raw_difference_a_minus_b": float(np.mean(raw_diff)),
                        "median_raw_difference_a_minus_b": float(np.median(raw_diff)),
                        "mean_advantage_a_over_b": observed,
                        "bootstrap_ci_low_advantage": lo,
                        "bootstrap_ci_high_advantage": hi,
                        "bootstrap_p_centered": p,
                        "fraction_blocks_a_better": float(np.mean(advantage > 0)),
                        "fraction_blocks_tied": float(np.mean(advantage == 0)),
                        "ci_excludes_zero": bool(
                            np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
                        ),
                    }
                )

            pvals = [
                float(x["bootstrap_p_centered"])
                for x in metric_pair_rows
            ]
            adjusted = holm_adjust(pvals)
            for rec, adj in zip(metric_pair_rows, adjusted):
                rec["holm_adjusted_p"] = adj
                rec["significant_holm_alpha"] = bool(adj < cfg.alpha)
                pair_rows.append(rec)

    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(
        dirs["stats"] / "pairwise_moving_block_bootstrap.csv",
        index=False,
    )

    # Average temporal rank per non-overlap block after seed aggregation.
    rank_rows = []
    for dataset, d in agg.groupby("canonical_dataset"):
        for metric in PRIMARY_METRICS:
            if metric not in d.columns:
                continue
            pivot = d.pivot_table(
                index="instance_end",
                columns="algorithm",
                values=metric,
                aggfunc="mean",
            ).dropna(subset=MODEL_ORDER, how="any")
            if pivot.empty:
                continue
            ascending = not HIGHER_IS_BETTER[metric]
            ranks = pivot[MODEL_ORDER].rank(
                axis=1,
                ascending=ascending,
                method="average",
            )
            for model in MODEL_ORDER:
                rank_rows.append(
                    {
                        "canonical_dataset": dataset,
                        "metric": metric,
                        "algorithm": model,
                        "n_nonoverlap_blocks": len(ranks),
                        "mean_rank": float(ranks[model].mean()),
                        "median_rank": float(ranks[model].median()),
                        "rank_1_fraction": float((ranks[model] == 1).mean()),
                    }
                )
    ranks_df = pd.DataFrame(rank_rows)
    ranks_df.to_csv(
        dirs["stats"] / "temporal_model_ranks.csv",
        index=False,
    )

    # Supplementary Friedman only. Explicitly not primary because temporal blocks
    # may remain autocorrelated even though they do not overlap.
    friedman_rows = []
    for dataset, d in agg.groupby("canonical_dataset"):
        for metric in PRIMARY_METRICS:
            if metric not in d.columns:
                continue
            pivot = d.pivot_table(
                index="instance_end",
                columns="algorithm",
                values=metric,
                aggfunc="mean",
            ).dropna(subset=MODEL_ORDER, how="any")
            if pivot.empty:
                continue
            if scipy_stats is not None:
                stat, p = scipy_stats.friedmanchisquare(
                    *[pivot[m].to_numpy(float) for m in MODEL_ORDER]
                )
            else:
                stat, p = np.nan, np.nan
            friedman_rows.append(
                {
                    "canonical_dataset": dataset,
                    "metric": metric,
                    "n_nonoverlap_blocks": len(pivot),
                    "friedman_statistic": float(stat) if np.isfinite(stat) else np.nan,
                    "friedman_p": float(p) if np.isfinite(p) else np.nan,
                    "status": (
                        "SUPPLEMENTARY_ONLY_TEMPORAL_AUTOCORRELATION_NOT_MODELED"
                    ),
                    "scipy_available": scipy_stats is not None,
                }
            )
    friedman = pd.DataFrame(friedman_rows)
    friedman.to_csv(
        dirs["stats"] / "supplementary_friedman.csv",
        index=False,
    )

    return pairwise, ranks_df, friedman


def support_performance_relationship(
    block_classes: pd.DataFrame,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    # First average stochastic seed performance at each class/block.
    agg = (
        block_classes
        .groupby(
            [
                "canonical_dataset",
                "algorithm",
                "block_index",
                "instance_end",
                "class",
            ],
            as_index=False,
        )
        .agg(
            support=("support", "max"),
            prior=("prior", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
        )
    )
    agg = agg[
        (agg["support"] > 0)
        & (~agg["class"].map(normalize_benign))
    ].copy()

    rows = []
    for (dataset, model), g in agg.groupby(["canonical_dataset", "algorithm"]):
        for xcol in ["support", "prior"]:
            for ycol in ["recall", "f1"]:
                x = pd.to_numeric(g[xcol], errors="coerce")
                y = pd.to_numeric(g[ycol], errors="coerce")
                mask = x.notna() & y.notna()
                if mask.sum() < 3:
                    rho = p = np.nan
                elif scipy_stats is not None:
                    rho, p = scipy_stats.spearmanr(x[mask], y[mask])
                else:
                    rho = x[mask].rank().corr(y[mask].rank())
                    p = np.nan
                rows.append(
                    {
                        "canonical_dataset": dataset,
                        "algorithm": model,
                        "x": xcol,
                        "y": ycol,
                        "n_class_blocks": int(mask.sum()),
                        "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                        "p_unadjusted_descriptive": float(p) if np.isfinite(p) else np.nan,
                        "interpretation_status": (
                            "DESCRIPTIVE_TEMPORAL_DEPENDENCE_NOT_MODELED"
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(
        dirs["tables"] / "table_support_performance_relationship.csv",
        index=False,
    )
    return out


def save_line_figure(
    data: pd.DataFrame,
    dataset: str,
    metric: str,
    y_col_suffix: str,
    out_path: Path,
    title: str,
    ylabel: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for model in MODEL_ORDER:
        g = data[
            (data["canonical_dataset"] == dataset)
            & (data["algorithm"] == model)
        ].sort_values("instance_end")
        if g.empty:
            continue
        mean_col = f"{metric}_{y_col_suffix}"
        std_col = f"{metric}_std_seed"
        ax.plot(
            g["instance_end"].to_numpy(),
            g[mean_col].to_numpy(),
            label=model,
        )
        if std_col in g.columns and model in STOCHASTIC_MODELS:
            mean = g[mean_col].to_numpy(float)
            std = g[std_col].fillna(0).to_numpy(float)
            x = g["instance_end"].to_numpy()
            ax.fill_between(x, mean - std, mean + std, alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("Processed flow instances")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    evidence: pd.DataFrame,
    rolling_trajectory: pd.DataFrame,
    block_classes: pd.DataFrame,
    efficiency: pd.DataFrame,
    cfg: AnalysisConfig,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    figure_rows = []
    datasets = sorted(evidence["canonical_dataset"].unique())

    for dataset in datasets:
        safe_ds = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)

        # Prevalence trajectories.
        fig, ax = plt.subplots(figsize=(11, 6))
        g = evidence[evidence["canonical_dataset"] == dataset]
        for cls, x in g.groupby("class"):
            ax.plot(
                x.sort_values("instance_end")["instance_end"],
                x.sort_values("instance_end")["prior"],
                label=cls,
            )
        ax.set_title(f"Class prevalence trajectories: {dataset}")
        ax.set_xlabel("Processed flow instances")
        ax.set_ylabel("Class prevalence")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        p = dirs["figures"] / f"fig_prevalence_{safe_ds}.png"
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)
        figure_rows.append(
            {"figure": p.name, "dataset": dataset, "type": "class_prevalence"}
        )

        # Maximum local IR trajectory.
        ir = (
            g.groupby("instance_end", as_index=False)["local_ir"].max()
            .sort_values("instance_end")
        )
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(ir["instance_end"], ir["local_ir"])
        ax.set_yscale("log")
        ax.set_title(f"Maximum local imbalance ratio: {dataset}")
        ax.set_xlabel("Processed flow instances")
        ax.set_ylabel("Maximum local IR among present classes")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        p = dirs["figures"] / f"fig_max_local_ir_{safe_ds}.png"
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)
        figure_rows.append(
            {"figure": p.name, "dataset": dataset, "type": "max_local_ir"}
        )

        # JSD trajectory.
        jsd = (
            g.drop_duplicates("instance_end")[
                ["instance_end", "jsd_from_previous_block"]
            ]
            .sort_values("instance_end")
        )
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(jsd["instance_end"], jsd["jsd_from_previous_block"])
        ax.set_title(f"Adjacent-block class-prior shift magnitude: {dataset}")
        ax.set_xlabel("Processed flow instances")
        ax.set_ylabel("Jensen-Shannon divergence")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        p = dirs["figures"] / f"fig_jsd_{safe_ds}.png"
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)
        figure_rows.append(
            {"figure": p.name, "dataset": dataset, "type": "jsd"}
        )

        primary = rolling_trajectory[
            rolling_trajectory["window_size"] == cfg.primary_window
        ]
        for metric in [
            "macro_f1",
            "gmean",
            "dynamic_minority_attack_recall",
            "worst_attack_recall",
            "benign_fpr",
        ]:
            mean_col = f"{metric}_mean_seed"
            if mean_col not in primary.columns:
                continue
            p = dirs["figures"] / f"fig_rolling_{metric}_{cfg.primary_window}_{safe_ds}.png"
            save_line_figure(
                primary,
                dataset,
                metric,
                "mean_seed",
                p,
                (
                    f"{METRIC_LABELS.get(metric, metric)} "
                    f"({cfg.primary_window:,}-flow rolling window): {dataset}"
                ),
                METRIC_LABELS.get(metric, metric),
                cfg.dpi,
            )
            figure_rows.append(
                {
                    "figure": p.name,
                    "dataset": dataset,
                    "type": f"rolling_{metric}",
                }
            )

        # Primary-window attack-class recall heatmap per model.
        # Use non-overlap 50k block class recall to avoid obscuring short collapses.
        dclass = block_classes[
            (block_classes["canonical_dataset"] == dataset)
            & (~block_classes["class"].map(normalize_benign))
        ].copy()
        dclass = (
            dclass.groupby(
                ["algorithm", "block_index", "class"],
                as_index=False,
            )["recall"]
            .mean()
        )
        for model in MODEL_ORDER:
            z = dclass[dclass["algorithm"] == model]
            if z.empty:
                continue
            pivot = z.pivot(
                index="class",
                columns="block_index",
                values="recall",
            ).sort_index()
            fig, ax = plt.subplots(figsize=(12, max(4.5, 0.35 * len(pivot))))
            im = ax.imshow(
                pivot.to_numpy(),
                aspect="auto",
                vmin=0,
                vmax=1,
            )
            ax.set_title(f"Non-overlap block attack-class recall: {dataset} / {model}")
            ax.set_xlabel("50k block index")
            ax.set_ylabel("Attack class")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Recall")
            fig.tight_layout()
            p = dirs["figures"] / f"fig_block_class_recall_{safe_ds}_{model}.png"
            fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
            plt.close(fig)
            figure_rows.append(
                {
                    "figure": p.name,
                    "dataset": dataset,
                    "type": "block_attack_class_recall",
                    "algorithm": model,
                }
            )

    # Efficiency, one figure per dataset.
    for dataset, g in efficiency.groupby("canonical_dataset"):
        g = g.set_index("algorithm").reindex(MODEL_ORDER).dropna(how="all").reset_index()
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.bar(
            g["algorithm"],
            g["throughput_mean"],
            yerr=g["throughput_std"].fillna(0),
            capsize=4,
        )
        ax.set_title(f"Processing throughput: {dataset}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Instances / second")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        safe_ds = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)
        p = dirs["figures"] / f"fig_efficiency_throughput_{safe_ds}.png"
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)
        figure_rows.append(
            {"figure": p.name, "dataset": dataset, "type": "throughput"}
        )

    index = pd.DataFrame(figure_rows)
    index.to_csv(dirs["figures"] / "figure_index.csv", index=False)
    return index


def model_leader(final_summary: pd.DataFrame, dataset: str, metric: str) -> tuple[str, float]:
    g = final_summary[final_summary["canonical_dataset"] == dataset].copy()
    col = f"{metric}_mean"
    g = g.dropna(subset=[col])
    if g.empty:
        return "NA", np.nan
    if HIGHER_IS_BETTER.get(metric, True):
        row = g.loc[g[col].idxmax()]
    else:
        row = g.loc[g[col].idxmin()]
    return str(row["algorithm"]), float(row[col])


def write_report(
    cfg: AnalysisConfig,
    artifacts: pd.DataFrame,
    final_summary: pd.DataFrame,
    lifecycle: pd.DataFrame,
    pairwise: pd.DataFrame,
    efficiency: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    lines = []
    lines.append("# E1-TDIM-MSB Final Analysis Summary")
    lines.append("")
    lines.append(f"Generated: {utc_now()}")
    lines.append("")
    lines.append("## Analysis integrity")
    lines.append("")
    lines.append(
        f"- Final experimental runs consumed: **{len(artifacts)}/32**."
    )
    lines.append(
        "- NF-CSE evidence is the temporally corrected v1.1 full rerun; "
        "NF-ToN evidence is the validated v1.0 run reused by the v1.1 manifest."
    )
    lines.append(
        f"- Non-overlapping analysis block: **{cfg.block_size:,} flows**."
    )
    lines.append(
        f"- Primary temporal visualization window: **{cfg.primary_window:,} flows**."
    )
    lines.append(
        "- Overlapping rolling windows are used for temporal visualization and "
        "vulnerability summaries, not as independent observations for significance tests."
    )
    lines.append(
        f"- Primary pairwise inference uses paired moving-block bootstrap on "
        f"non-overlapping 50k blocks, block length={cfg.bootstrap_block_length} "
        f"blocks ({cfg.bootstrap_block_length * cfg.block_size:,} flows), "
        f"{cfg.bootstrap_reps:,} replicates, followed by Holm adjustment."
    )
    lines.append("")

    for dataset in sorted(artifacts["canonical_dataset"].unique()):
        lines.append(f"## {dataset}")
        lines.append("")
        for metric in [
            "macro_f1",
            "balanced_accuracy",
            "gmean",
            "dynamic_minority_attack_recall",
            "worst_attack_recall",
            "benign_fpr",
        ]:
            model, value = model_leader(final_summary, dataset, metric)
            if np.isfinite(value):
                lines.append(
                    f"- Best final cumulative {METRIC_LABELS[metric]}: "
                    f"**{model} ({value:.6f})**."
                )

        lc = lifecycle[
            (lifecycle["canonical_dataset"] == dataset)
            & (~lifecycle["is_benign"])
        ]
        if not lc.empty:
            max_dis = int(lc["disappearance_events"].sum())
            max_rec = int(lc["recurrence_events"].sum())
            switches = int(lc["minority_switches"].sum())
            lines.append(
                f"- Attack-family lifecycle events: disappearance={max_dis}, "
                f"recurrence={max_rec}, minority switching={switches}."
            )

        eff = efficiency[efficiency["canonical_dataset"] == dataset]
        if not eff.empty:
            fastest = eff.loc[eff["throughput_mean"].idxmax()]
            lines.append(
                f"- Highest mean throughput: **{fastest['algorithm']} "
                f"({fastest['throughput_mean']:.1f} instances/s)**."
            )
        lines.append("")

    lines.append("## Statistical interpretation")
    lines.append("")
    if pairwise.empty:
        lines.append("- Pairwise statistical table was not generated.")
    else:
        sig = pairwise[pairwise["significant_holm_alpha"] == True]
        lines.append(
            f"- Holm-significant moving-block bootstrap comparisons: "
            f"**{len(sig)}/{len(pairwise)}** pairwise tests."
        )
        lines.append(
            "- A positive `mean_advantage_a_over_b` means model A is better after "
            "respecting metric direction. For benign FPR, lower raw values are better."
        )
        lines.append(
            "- `supplementary_friedman.csv` is retained only as supplementary evidence "
            "because temporal autocorrelation is not modeled by the classical Friedman test."
        )
    lines.append("")

    lines.append("## Generated outputs")
    lines.append("")
    lines.append("- `02_tables/`: final model, per-class, lifecycle, window-sensitivity, and efficiency tables.")
    lines.append("- `03_figures/`: prevalence, imbalance, prior-shift, rolling-performance, class-recall, and throughput figures.")
    lines.append("- `04_statistics/`: moving-block bootstrap pairwise tests, temporal ranks, and supplementary Friedman results.")
    lines.append("- `01_master/`: Parquet master files for reproducible secondary analysis.")
    lines.append("")
    lines.append(
        "This pipeline generates reproducible evidence and statistical summaries. "
        "Interpret the outputs together with the claim boundaries documented in "
        "`docs/claim_boundaries.md`."
    )

    (dirs["report"] / "E1_FINAL_ANALYSIS_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def checksum_outputs(root: Path, out_file: Path) -> None:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p != out_file:
            files.append(p)
    files.sort()
    with out_file.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(f"{sha256_file(p)}  {p.relative_to(root)}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E1-TDIM-MSB Final Analysis Pipeline v1.0"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--analysis-ready", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_analysis_config(args.config, args)
    if args.bootstrap_reps is not None:
        cfg.bootstrap_reps = int(args.bootstrap_reps)

    dirs = ensure_dirs(cfg.output_root)
    start = time.time()

    print("=" * 110)
    print("E1-TDIM-MSB FINAL ANALYSIS PIPELINE v1.0")
    print("=" * 110)
    print("Analysis-ready :", cfg.analysis_ready)
    print("Output root    :", cfg.output_root)
    print("Block size     :", f"{cfg.block_size:,}")
    print("Primary window :", f"{cfg.primary_window:,}")
    print("Bootstrap reps :", f"{cfg.bootstrap_reps:,}")
    print()

    print("[1/11] Auditing _analysis_ready ...")
    artifacts = audit_analysis_ready(cfg, dirs)
    print("       PASS: 32/32 final run artifacts.")

    print("[2/11] Loading master metric/class/resource tables ...")
    metrics, classes, resources = load_master_metrics(artifacts, dirs)
    print(
        f"       metrics={len(metrics):,}, class_metrics={len(classes):,}, "
        f"resource_rows={len(resources):,}"
    )

    print("[3/11] Deriving exact non-overlapping block confusion metrics ...")
    block_metrics, block_classes, support_runs = derive_nonoverlap_blocks(
        artifacts,
        dirs,
        cfg.block_size,
    )
    print(
        f"       block_metrics={len(block_metrics):,}, "
        f"block_class_metrics={len(block_classes):,}"
    )

    print("[4/11] Running metric/support integrity audits ...")
    metric_consistency_audit(metrics, block_metrics, dirs)
    print("       PASS.")

    print("[5/11] Building dynamic-imbalance and class-lifecycle evidence ...")
    evidence = canonical_data_evidence(
        support_runs,
        artifacts,
        cfg,
        dirs,
    )
    lifecycle, events, thresholds = class_lifecycle_tables(
        evidence,
        cfg,
        dirs,
    )
    print(
        f"       lifecycle classes={len(lifecycle):,}, events={len(events):,}"
    )

    print("[6/11] Aggregating final cumulative and per-class results ...")
    final_summary, final_class = aggregate_final_cumulative(
        metrics,
        classes,
        dirs,
    )
    print("       PASS.")

    print("[7/11] Building rolling-window and sensitivity summaries ...")
    rolling_trajectory, sensitivity, vulnerability = rolling_summary(
        metrics,
        classes,
        cfg,
        dirs,
    )
    print("       PASS.")

    print("[8/11] Building efficiency and sampling summaries ...")
    efficiency = efficiency_tables(artifacts, resources, dirs)
    sampling = sampling_summary(artifacts, dirs)
    print("       PASS.")

    print("[9/11] Running temporal statistical comparison ...")
    pairwise, ranks, friedman = statistical_comparison(
        block_metrics,
        cfg,
        dirs,
    )
    support_perf = support_performance_relationship(
        block_classes,
        dirs,
    )
    print(
        f"       pairwise tests={len(pairwise):,}; "
        f"primary method=moving-block bootstrap + Holm."
    )

    if args.no_plots:
        print("[10/11] Figures skipped (--no-plots).")
        figure_index = pd.DataFrame()
    else:
        print("[10/11] Generating figures ...")
        figure_index = create_figures(
            evidence,
            rolling_trajectory,
            block_classes,
            efficiency,
            cfg,
            dirs,
        )
        print(f"        figures={len(figure_index):,}")

    print("[11/11] Writing report, manifest, and checksums ...")
    write_report(
        cfg,
        artifacts,
        final_summary,
        lifecycle,
        pairwise,
        efficiency,
        dirs,
    )

    manifest = {
        "pipeline": "E1_FINAL_ANALYSIS_PIPELINE_v1.0",
        "generated_at_utc": utc_now(),
        "status": "COMPLETE",
        "input_analysis_ready": str(cfg.analysis_ready),
        "input_artifact_index_sha256": sha256_file(
            cfg.analysis_ready / "artifact_index.csv"
        ),
        "input_final_manifest_sha256": sha256_file(
            cfg.analysis_ready / "final_manifest.json"
        ),
        "analysis_configuration": {
            "block_size": cfg.block_size,
            "primary_window": cfg.primary_window,
            "rolling_windows": list(cfg.rolling_windows),
            "bootstrap_reps": cfg.bootstrap_reps,
            "bootstrap_block_length_blocks": cfg.bootstrap_block_length,
            "bootstrap_seed": cfg.bootstrap_seed,
            "alpha": cfg.alpha,
            "support_thresholds": list(cfg.support_thresholds),
            "recall_collapse_thresholds": list(cfg.recall_collapse_thresholds),
        },
        "methodology": {
            "final_cumulative": "aggregate final endpoint across stochastic seeds",
            "rolling": "seed-aggregated trajectories; overlapping windows not treated as independent",
            "nonoverlap_blocks": "derived exactly from differences of cumulative confusion matrices",
            "primary_pairwise_test": "paired moving-block bootstrap on full 50k blocks with centered-null bootstrap and Holm adjustment",
            "supplementary_test": "Friedman on non-overlap blocks; supplementary only because temporal autocorrelation is not explicitly modeled",
        },
        "counts": {
            "runs": len(artifacts),
            "master_metric_rows": len(metrics),
            "master_class_metric_rows": len(classes),
            "derived_block_metric_rows": len(block_metrics),
            "derived_block_class_rows": len(block_classes),
            "lifecycle_classes": len(lifecycle),
            "lifecycle_events": len(events),
            "pairwise_tests": len(pairwise),
            "figures": len(figure_index),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "scipy": (
                getattr(scipy_stats, "__version__", None)
                if scipy_stats is not None
                else None
            ),
            "hostname": socket.gethostname(),
        },
        "output_root": str(cfg.output_root),
        "elapsed_seconds": round(time.time() - start, 3),
    }
    json_dump(dirs["report"] / "analysis_manifest.json", manifest)
    checksum_outputs(
        cfg.output_root,
        dirs["report"] / "SHA256SUMS.txt",
    )

    print()
    print("=" * 110)
    print("E1 FINAL ANALYSIS: COMPLETE")
    print("=" * 110)
    print("Tables     :", dirs["tables"])
    print("Figures    :", dirs["figures"])
    print("Statistics :", dirs["stats"])
    print("Report     :", dirs["report"] / "E1_FINAL_ANALYSIS_SUMMARY.md")
    print("Manifest   :", dirs["report"] / "analysis_manifest.json")
    print(f"Elapsed    : {(time.time() - start) / 60:.2f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
