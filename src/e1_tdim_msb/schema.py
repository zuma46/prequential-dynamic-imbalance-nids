from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


LABEL_PRIORITY = (
    "Attack",
    "attack",
    "ATTACK",
    "Attack_type",
    "attack_type",
    "attack_cat",
    "Label_multiclass",
    "label_multiclass",
    "Label",
    "label",
)

TIMESTAMP_PRIORITY = (
    "FLOW_START_MILLISECONDS",
    "FLOW_START_MICROSECONDS",
    "FLOW_START_SECONDS",
    "FLOW_START",
    "flow_start",
    "Timestamp",
    "timestamp",
    "TIMESTAMP",
    "ts",
    "time",
    "Time",
)

LEAKAGE_PATTERNS = (
    r"^label$",
    r"^attack$",
    r"attack[_\- ]?type",
    r"attack[_\- ]?cat",
    r"^class$",
    r"^target$",
    r"^dataset$",
    r"scenario",
)

IDENTIFIER_PATTERNS = (
    r"(^|_)flow[_\- ]?id($|_)",
    r"(^|_)record[_\- ]?id($|_)",
    r"(^|_)row[_\- ]?id($|_)",
    r"(^|_)session[_\- ]?id($|_)",
    r"(^|_)uuid($|_)",
)

IP_PATTERNS = (
    r"ipv4[_\- ]?(src|dst)[_\- ]?addr",
    r"ip[_\- ]?(src|dst)",
    r"src[_\- ]?ip",
    r"dst[_\- ]?ip",
    r"source[_\- ]?address",
    r"destination[_\- ]?address",
    r"mac[_\- ]?(src|dst)",
)


def parquet_files(path: str | Path) -> list[Path]:
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a Parquet file: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.rglob("*.parquet") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {path}")
    return files


def read_schema(path: str | Path) -> pa.Schema:
    files = parquet_files(path)
    schema = pq.ParquetFile(files[0]).schema_arrow
    for file in files[1:]:
        other = pq.ParquetFile(file).schema_arrow
        if schema.names != other.names:
            raise ValueError(f"Parquet schema mismatch: {files[0]} versus {file}")
    return schema


def _resolve_priority(names: Iterable[str], priority: Iterable[str]) -> str | None:
    names = list(names)
    exact = {name: name for name in names}
    lower = {name.lower(): name for name in names}
    for candidate in priority:
        if candidate in exact:
            return exact[candidate]
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def resolve_label_column(schema: pa.Schema, configured: str) -> str:
    if configured != "__AUTO__":
        if configured not in schema.names:
            raise KeyError(f"Label column not found: {configured}")
        return configured
    resolved = _resolve_priority(schema.names, LABEL_PRIORITY)
    if resolved is None:
        raise KeyError(f"Unable to auto-detect label column from: {schema.names}")
    return resolved


def resolve_timestamp_column(schema: pa.Schema, configured: str, required: bool) -> str | None:
    if configured not in {"__AUTO__", "", None}:
        if configured not in schema.names:
            raise KeyError(f"Timestamp column not found: {configured}")
        return configured
    resolved = _resolve_priority(schema.names, TIMESTAMP_PRIORITY)
    if resolved is None and required:
        raise KeyError(f"Unable to auto-detect timestamp column from: {schema.names}")
    return resolved


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, name, flags=re.IGNORECASE) for pattern in patterns)


def resolve_feature_columns(
    schema: pa.Schema,
    label_column: str,
    timestamp_column: str | None,
    include: list[str] | None,
    exclude: list[str],
    numeric_only: bool,
    drop_ip_addresses: bool,
    drop_identifier_columns: bool,
) -> list[str]:
    if include:
        missing = sorted(set(include) - set(schema.names))
        if missing:
            raise KeyError(f"Configured feature columns not found: {missing}")
        candidates = list(include)
    else:
        candidates = list(schema.names)

    excluded = set(exclude)
    excluded.add(label_column)
    if timestamp_column:
        excluded.add(timestamp_column)

    features: list[str] = []
    for name in candidates:
        if name in excluded:
            continue
        if _matches_any(name, LEAKAGE_PATTERNS):
            continue
        if drop_ip_addresses and _matches_any(name, IP_PATTERNS):
            continue
        if drop_identifier_columns and _matches_any(name, IDENTIFIER_PATTERNS):
            continue
        field = schema.field(name)
        if numeric_only and not (
            pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_boolean(field.type)
            or pa.types.is_decimal(field.type)
        ):
            continue
        features.append(name)

    if not features:
        raise ValueError("No predictive feature columns remain after leakage filtering")
    return features


def inspect_dataset(path: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    schema = read_schema(path)
    ds = cfg["dataset"]
    label = resolve_label_column(schema, ds["label_column"])
    timestamp = resolve_timestamp_column(schema, ds["timestamp_column"], bool(ds["require_timestamp"]))
    features = resolve_feature_columns(
        schema=schema,
        label_column=label,
        timestamp_column=timestamp,
        include=ds.get("feature_include"),
        exclude=ds.get("feature_exclude", []),
        numeric_only=bool(ds.get("numeric_features_only", True)),
        drop_ip_addresses=bool(ds.get("drop_ip_addresses", True)),
        drop_identifier_columns=bool(ds.get("drop_identifier_columns", True)),
    )
    files = parquet_files(path)
    total_rows = sum(pq.ParquetFile(file).metadata.num_rows for file in files)
    return {
        "files": [str(file) for file in files],
        "total_rows": int(total_rows),
        "schema": {field.name: str(field.type) for field in schema},
        "label_column": label,
        "timestamp_column": timestamp,
        "feature_columns": features,
        "n_features": len(features),
    }
