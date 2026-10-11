from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .schema import parquet_files


@dataclass(slots=True)
class StreamBatch:
    X: np.ndarray
    y: list[str]
    timestamps_ns: np.ndarray | None
    absolute_start: int

    @property
    def size(self) -> int:
        return len(self.y)


def normalize_label(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None:
        return "__MISSING_LABEL__"
    if isinstance(value, float) and math.isnan(value):
        return "__MISSING_LABEL__"
    return str(value).strip()


def _numeric_column_to_float32(array: pa.Array) -> np.ndarray:
    if array.null_count:
        array = pc.fill_null(array, 0)
    try:
        array = pc.cast(array, pa.float32(), safe=False)
        values = array.to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        values = np.asarray(array.to_pylist(), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _infer_numeric_timestamp_ns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind == "f":
        if np.any(~np.isfinite(values)):
            raise ValueError("Timestamp column contains NaN or infinite values")
    magnitude = float(np.nanmedian(np.abs(values))) if values.size else 0.0
    if magnitude < 1e11:
        multiplier = 1_000_000_000  # seconds
    elif magnitude < 1e14:
        multiplier = 1_000_000  # milliseconds
    elif magnitude < 1e17:
        multiplier = 1_000  # microseconds
    else:
        multiplier = 1  # nanoseconds
    return np.rint(values.astype(np.float64) * multiplier).astype(np.int64)


def timestamp_to_ns(array: pa.Array) -> np.ndarray:
    if array.null_count:
        raise ValueError("Timestamp column contains null values")
    dtype = array.type
    if pa.types.is_timestamp(dtype):
        return array.cast(pa.timestamp("ns")).to_numpy(zero_copy_only=False).astype("datetime64[ns]").astype(np.int64)
    if pa.types.is_date(dtype):
        return array.cast(pa.timestamp("ns")).to_numpy(zero_copy_only=False).astype("datetime64[ns]").astype(np.int64)
    if pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype):
        return _infer_numeric_timestamp_ns(array.to_numpy(zero_copy_only=False))

    import pandas as pd

    parsed = pd.to_datetime(array.to_pylist(), utc=True, errors="raise")
    return parsed.astype("int64").to_numpy(dtype=np.int64, copy=False)


class ParquetTemporalStream:
    def __init__(
        self,
        path: str | Path,
        feature_columns: Sequence[str],
        label_column: str,
        timestamp_column: str | None,
        batch_size: int,
        row_start: int = 0,
        max_rows: int | None = None,
        use_threads: bool = True,
        memory_map: bool = True,
        pre_buffer: bool = True,
        validate_chronology: bool = True,
        chronology_tolerance: int = 0,
        fail_on_nonfinite: bool = False,
    ) -> None:
        self.files = parquet_files(path)
        self.feature_columns = list(feature_columns)
        self.label_column = label_column
        self.timestamp_column = timestamp_column
        self.batch_size = int(batch_size)
        self.row_start = int(row_start)
        self.max_rows = None if max_rows is None else int(max_rows)
        self.use_threads = bool(use_threads)
        self.memory_map = bool(memory_map)
        self.pre_buffer = bool(pre_buffer)
        self.validate_chronology = bool(validate_chronology)
        self.chronology_tolerance = int(chronology_tolerance)
        self.fail_on_nonfinite = bool(fail_on_nonfinite)

    @property
    def columns(self) -> list[str]:
        cols = list(self.feature_columns) + [self.label_column]
        if self.timestamp_column:
            cols.append(self.timestamp_column)
        return cols

    def __iter__(self) -> Iterator[StreamBatch]:
        skipped = 0
        yielded = 0
        absolute = 0
        previous_ts: int | None = None

        for file in self.files:
            parquet = pq.ParquetFile(
                file,
                memory_map=self.memory_map,
                pre_buffer=self.pre_buffer,
            )
            for batch in parquet.iter_batches(
                batch_size=self.batch_size,
                columns=self.columns,
                use_threads=self.use_threads,
            ):
                n = batch.num_rows
                batch_absolute_start = absolute
                absolute += n

                if skipped + n <= self.row_start:
                    skipped += n
                    continue
                local_start = max(0, self.row_start - skipped)
                skipped += local_start

                remaining = None if self.max_rows is None else self.max_rows - yielded
                if remaining is not None and remaining <= 0:
                    return
                local_end = n if remaining is None else min(n, local_start + remaining)
                if local_start >= local_end:
                    continue
                if local_start or local_end < n:
                    batch = batch.slice(local_start, local_end - local_start)

                X = np.empty((batch.num_rows, len(self.feature_columns)), dtype=np.float32)
                for j, name in enumerate(self.feature_columns):
                    X[:, j] = _numeric_column_to_float32(batch.column(batch.schema.get_field_index(name)))

                finite_mask = np.isfinite(X)
                if not finite_mask.all():
                    if self.fail_on_nonfinite:
                        bad = int((~finite_mask).sum())
                        raise ValueError(f"Found {bad} non-finite feature values")
                    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                label_array = batch.column(batch.schema.get_field_index(self.label_column))
                y = [normalize_label(value) for value in label_array.to_pylist()]
                if "__MISSING_LABEL__" in y:
                    raise ValueError("Missing class label detected")

                timestamps_ns: np.ndarray | None = None
                if self.timestamp_column:
                    ts_array = batch.column(batch.schema.get_field_index(self.timestamp_column))
                    timestamps_ns = timestamp_to_ns(ts_array)
                    if self.validate_chronology and timestamps_ns.size:
                        tolerance = self.chronology_tolerance
                        if previous_ts is not None and int(timestamps_ns[0]) + tolerance < previous_ts:
                            raise ValueError(
                                f"Non-monotonic timestamp across batches at absolute row "
                                f"{batch_absolute_start + local_start}"
                            )
                        if timestamps_ns.size > 1 and np.any(np.diff(timestamps_ns) < -tolerance):
                            local_bad = int(np.flatnonzero(np.diff(timestamps_ns) < -tolerance)[0] + 1)
                            raise ValueError(
                                f"Non-monotonic timestamp inside batch at absolute row "
                                f"{batch_absolute_start + local_start + local_bad}"
                            )
                        previous_ts = int(timestamps_ns[-1])

                yield StreamBatch(
                    X=X,
                    y=y,
                    timestamps_ns=timestamps_ns,
                    absolute_start=batch_absolute_start + local_start,
                )
                yielded += batch.num_rows
                if self.max_rows is not None and yielded >= self.max_rows:
                    return


def discover_classes(
    path: str | Path,
    label_column: str,
    batch_size: int,
    row_start: int,
    max_rows: int | None,
    use_threads: bool,
    memory_map: bool,
    pre_buffer: bool,
) -> list[str]:
    seen: dict[str, None] = {}
    skipped = 0
    yielded = 0
    for file in parquet_files(path):
        parquet = pq.ParquetFile(file, memory_map=memory_map, pre_buffer=pre_buffer)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=[label_column], use_threads=use_threads):
            n = batch.num_rows
            if skipped + n <= row_start:
                skipped += n
                continue
            local_start = max(0, row_start - skipped)
            skipped += local_start
            remaining = None if max_rows is None else max_rows - yielded
            if remaining is not None and remaining <= 0:
                return list(seen)
            local_end = n if remaining is None else min(n, local_start + remaining)
            if local_start or local_end < n:
                batch = batch.slice(local_start, local_end - local_start)
            for value in batch.column(0).to_pylist():
                label = normalize_label(value)
                if label == "__MISSING_LABEL__":
                    raise ValueError("Missing class label detected during class discovery")
                seen.setdefault(label, None)
            yielded += batch.num_rows
            if max_rows is not None and yielded >= max_rows:
                return list(seen)
    return list(seen)
