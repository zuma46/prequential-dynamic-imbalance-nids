from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .metrics import js_divergence
from .schema import parquet_files
from .stream import normalize_label


def profile_label_blocks(
    path: str | Path,
    label_column: str,
    block_size: int = 50_000,
    use_threads: bool = True,
    memory_map: bool = True,
    pre_buffer: bool = True,
) -> tuple[list[str], np.ndarray]:
    block_counters: list[Counter[str]] = []
    current: Counter[str] = Counter()
    current_size = 0
    class_order: dict[str, None] = {}

    for file in parquet_files(path):
        parquet = pq.ParquetFile(file, memory_map=memory_map, pre_buffer=pre_buffer)
        for batch in parquet.iter_batches(
            batch_size=block_size,
            columns=[label_column],
            use_threads=use_threads,
        ):
            labels = [normalize_label(value) for value in batch.column(0).to_pylist()]
            offset = 0
            while offset < len(labels):
                take = min(block_size - current_size, len(labels) - offset)
                chunk = labels[offset : offset + take]
                chunk_counts = Counter(chunk)
                current.update(chunk_counts)
                for label in chunk_counts:
                    class_order.setdefault(label, None)
                current_size += take
                offset += take
                if current_size == block_size:
                    block_counters.append(current)
                    current = Counter()
                    current_size = 0
    if current_size:
        block_counters.append(current)

    classes = list(class_order)
    class_to_index = {label: i for i, label in enumerate(classes)}
    counts = np.zeros((len(block_counters), len(classes)), dtype=np.int64)
    for block_index, counter in enumerate(block_counters):
        for label, count in counter.items():
            counts[block_index, class_to_index[label]] = int(count)
    return classes, counts


def select_representative_segment(
    path: str | Path,
    label_column: str,
    segment_rows: int = 2_000_000,
    block_size: int = 50_000,
    output_dir: str | Path | None = None,
    use_threads: bool = True,
    memory_map: bool = True,
    pre_buffer: bool = True,
) -> dict[str, Any]:
    if segment_rows % block_size != 0:
        raise ValueError("segment_rows must be a multiple of block_size")
    classes, blocks = profile_label_blocks(
        path=path,
        label_column=label_column,
        block_size=block_size,
        use_threads=use_threads,
        memory_map=memory_map,
        pre_buffer=pre_buffer,
    )
    window_blocks = segment_rows // block_size
    if len(blocks) <= window_blocks:
        selected_start = 0
        selected_end = len(blocks)
        scores = np.array([1.0])
    else:
        global_counts = blocks.sum(axis=0).astype(np.float64)
        rare_weights = 1.0 / np.sqrt(np.maximum(global_counts, 1.0))
        rare_weights /= rare_weights.sum()
        scores_list: list[float] = []

        rolling_counts = blocks[:window_blocks].sum(axis=0).astype(np.float64)
        for start in range(0, len(blocks) - window_blocks + 1):
            if start > 0:
                rolling_counts += blocks[start + window_blocks - 1]
                rolling_counts -= blocks[start - 1]
            present = rolling_counts > 0
            coverage = float(present.mean())
            rare_coverage = float(rare_weights[present].sum())
            total = rolling_counts.sum()
            probabilities = rolling_counts / total if total else np.zeros_like(rolling_counts)
            positive = probabilities > 0
            entropy = (
                float(-np.sum(probabilities[positive] * np.log(probabilities[positive])) / np.log(len(classes)))
                if positive.sum() > 1 and len(classes) > 1
                else 0.0
            )
            local_blocks = blocks[start : start + window_blocks]
            js_values = [
                js_divergence(local_blocks[i], local_blocks[i - 1])
                for i in range(1, len(local_blocks))
            ]
            variation = float(np.nanmean(js_values)) if js_values else 0.0
            variation_scaled = variation / (variation + 0.05) if variation > 0 else 0.0
            mixed_blocks = float(np.mean(np.count_nonzero(local_blocks, axis=1) > 1))
            score = (
                0.30 * coverage
                + 0.25 * rare_coverage
                + 0.20 * entropy
                + 0.15 * variation_scaled
                + 0.10 * mixed_blocks
            )
            scores_list.append(score)
        scores = np.asarray(scores_list, dtype=np.float64)
        selected_start = int(np.argmax(scores))
        selected_end = selected_start + window_blocks

    selected_counts = blocks[selected_start:selected_end].sum(axis=0)
    result = {
        "dataset": str(Path(path).expanduser().resolve()),
        "label_column": label_column,
        "block_size": block_size,
        "segment_rows_requested": segment_rows,
        "row_start": selected_start * block_size,
        "row_end_exclusive": min(selected_end * block_size, int(blocks.sum())),
        "selected_blocks": [selected_start, selected_end],
        "score": float(scores[selected_start] if len(scores) > selected_start else scores[0]),
        "classes": classes,
        "selected_class_counts": {
            label: int(selected_counts[i]) for i, label in enumerate(classes)
        },
        "total_blocks": int(len(blocks)),
        "total_rows": int(blocks.sum()),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem if Path(path).is_file() else Path(path).name
        (output_dir / f"{stem}_segment_selection.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        with (output_dir / f"{stem}_label_blocks.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["block", "row_start", "row_end_exclusive", *classes]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for block_index, row in enumerate(blocks):
                record = {
                    "block": block_index,
                    "row_start": block_index * block_size,
                    "row_end_exclusive": min((block_index + 1) * block_size, int(blocks.sum())),
                }
                record.update({label: int(row[i]) for i, label in enumerate(classes)})
                writer.writerow(record)
    return result
