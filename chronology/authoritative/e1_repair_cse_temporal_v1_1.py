#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(
    os.environ.get("E1_PROJECT_ROOT", ".")
).resolve()
DATASET_ROOT = Path(
    os.environ.get("E1_DATA_ROOT", ".")
).resolve()

SOURCE = DATASET_ROOT / "NF-CSE-CIC-IDS2018-v3.parquet"
CORRECTED = DATASET_ROOT / "NF-CSE-CIC-IDS2018-v3-temporal-corrected.parquet"
TIMESTAMP_COLUMN = "FLOW_START_MILLISECONDS"
LABEL_COLUMN = "Attack"
EXPECTED_ROWS = 20_115_529
EXPECTED_CLASSES = 15
EXPECTED_INVERSION_ZERO_BASED = 20_115_069

V10_LOCK = (
    PROJECT_ROOT / "runtime" / "e1_final_core_v1_0" / "E1_FINAL_PROTOCOL_LOCK.json"
)
V11_RUNTIME = PROJECT_ROOT / "runtime" / "e1_final_core_v1_1"
PROVENANCE = V11_RUNTIME / "temporal_repair_provenance.json"
INCIDENT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "E1_FINAL_CORE_FULL_v1_0"
    / "temporal_integrity_incident"
)
INCIDENT_FILE = INCIDENT_DIR / "NF-CSE-CIC-IDS2018-v3_incident.json"

BATCH_SIZE = 200_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parquet_uniform_compression(pf: pq.ParquetFile) -> str:
    if pf.metadata.num_row_groups == 0 or pf.metadata.num_columns == 0:
        return "zstd"
    compressions = {
        pf.metadata.row_group(0).column(i).compression.lower()
        for i in range(pf.metadata.num_columns)
    }
    supported = {"snappy", "gzip", "brotli", "lz4", "zstd", "none"}
    if len(compressions) == 1:
        value = next(iter(compressions))
        if value in supported:
            return value
    return "zstd"


def read_timestamp_array(path: Path) -> np.ndarray:
    table = pq.read_table(
        path,
        columns=[TIMESTAMP_COLUMN],
        memory_map=True,
        use_threads=True,
    )
    col = table[TIMESTAMP_COLUMN].combine_chunks()
    if col.null_count:
        raise ValueError(
            f"{TIMESTAMP_COLUMN} contains {col.null_count} null values"
        )
    arr = np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.int64)
    return arr


def discover_inversions(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError("Timestamp array must be one-dimensional")
    if len(arr) < 2:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(np.diff(arr) < 0)


def stream_class_counts(path: Path) -> dict[str, int]:
    pf = pq.ParquetFile(path, memory_map=True, pre_buffer=False)
    counts: Counter[str] = Counter()
    for batch in pf.iter_batches(
        batch_size=500_000,
        columns=[LABEL_COLUMN],
        use_threads=True,
    ):
        for value in batch.column(0).to_pylist():
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def read_parquet_slice(
    pf: pq.ParquetFile,
    start: int,
    length: int,
) -> pa.Table:
    if length <= 0:
        return pa.Table.from_batches([], schema=pf.schema_arrow)

    end = start + length
    row_group_start = 0
    parts: list[pa.Table] = []

    for rg in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        row_group_end = row_group_start + n

        if row_group_end <= start:
            row_group_start = row_group_end
            continue
        if row_group_start >= end:
            break

        table = pf.read_row_group(rg, use_threads=True)
        local_start = max(start, row_group_start) - row_group_start
        local_end = min(end, row_group_end) - row_group_start
        take_n = local_end - local_start
        if take_n > 0:
            parts.append(table.slice(local_start, take_n))

        row_group_start = row_group_end

    if not parts:
        raise RuntimeError(
            f"Could not read source slice start={start} length={length}"
        )
    return pa.concat_tables(parts, promote_options="none")


def stable_two_run_repair(
    source: Path,
    destination_tmp: Path,
    split_index: int,
    timestamps: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> dict[str, Any]:
    """Merge two individually monotonic source runs without changing row values.

    Source layout is assumed to be A || B, where A and B are individually sorted
    by timestamp and exactly one A->B descent caused the full-file inversion.
    A stable merge preserves source order for equal timestamps. Because all A rows
    originally precede all B rows, B rows are inserted using searchsorted(...,
    side='right'), i.e. after all equal-timestamp A rows.
    """
    pf = pq.ParquetFile(source, memory_map=True, pre_buffer=False)
    total_rows = pf.metadata.num_rows
    if len(timestamps) != total_rows:
        raise ValueError("Timestamp length differs from Parquet row count")
    if not (0 < split_index < total_rows):
        raise ValueError(f"Invalid split index: {split_index}")

    a_ts = timestamps[:split_index]
    b_ts = timestamps[split_index:]

    if np.any(np.diff(a_ts) < 0):
        raise ValueError("Run A is not monotonic")
    if np.any(np.diff(b_ts) < 0):
        raise ValueError("Run B is not monotonic")

    insertion_positions = np.searchsorted(a_ts, b_ts, side="right").astype(np.int64)
    if np.any(np.diff(insertion_positions) < 0):
        raise RuntimeError("Insertion positions are not monotonic")

    b_table = read_parquet_slice(pf, split_index, total_rows - split_index)
    if b_table.num_rows != len(b_ts):
        raise RuntimeError("Tail table row count mismatch")

    compression = parquet_uniform_compression(pf)
    destination_tmp.parent.mkdir(parents=True, exist_ok=True)
    if destination_tmp.exists():
        destination_tmp.unlink()

    writer_kwargs: dict[str, Any] = {
        "compression": None if compression == "none" else compression,
        "use_dictionary": True,
        "write_statistics": True,
    }
    if compression == "zstd":
        writer_kwargs["compression_level"] = 3

    writer = pq.ParquetWriter(
        destination_tmp,
        pf.schema_arrow,
        **writer_kwargs,
    )

    output_rows = 0
    a_absolute = 0
    b_index = 0
    next_report = 1_000_000
    started = time.time()

    try:
        for raw_batch in pf.iter_batches(
            batch_size=batch_size,
            use_threads=True,
        ):
            if a_absolute >= split_index:
                break

            batch = raw_batch
            remaining_a = split_index - a_absolute
            if batch.num_rows > remaining_a:
                batch = batch.slice(0, remaining_a)

            batch_len = batch.num_rows
            batch_end = a_absolute + batch_len
            batch_table = pa.Table.from_batches([batch], schema=pf.schema_arrow)
            pieces: list[pa.Table] = []
            local_cursor = 0

            while b_index < len(b_ts) and insertion_positions[b_index] <= batch_end:
                position = int(insertion_positions[b_index])
                if position < a_absolute:
                    raise RuntimeError(
                        "Tail insertion position moved behind emitted A rows"
                    )

                rel = position - a_absolute
                if rel > batch_len:
                    break

                if rel > local_cursor:
                    pieces.append(batch_table.slice(local_cursor, rel - local_cursor))

                group_end = b_index + 1
                while (
                    group_end < len(b_ts)
                    and insertion_positions[group_end] == position
                ):
                    group_end += 1

                pieces.append(b_table.slice(b_index, group_end - b_index))
                b_index = group_end
                local_cursor = rel

            if local_cursor < batch_len:
                pieces.append(batch_table.slice(local_cursor, batch_len - local_cursor))

            if pieces:
                merged = (
                    pieces[0]
                    if len(pieces) == 1
                    else pa.concat_tables(pieces, promote_options="none")
                )
                writer.write_table(merged, row_group_size=batch_size)
                output_rows += merged.num_rows

            a_absolute += batch_len

            if output_rows >= next_report:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    f"\rRepair write: {output_rows:,}/{total_rows:,} rows "
                    f"({output_rows/elapsed:,.0f} rows/s)",
                    end="",
                    flush=True,
                )
                next_report += 1_000_000

        if a_absolute != split_index:
            raise RuntimeError(
                f"Only emitted {a_absolute:,}/{split_index:,} Run-A rows"
            )

        if b_index < len(b_ts):
            # Valid only for B rows whose timestamp is after every A timestamp.
            writer.write_table(
                b_table.slice(b_index),
                row_group_size=batch_size,
            )
            output_rows += len(b_ts) - b_index
            b_index = len(b_ts)

    finally:
        writer.close()

    print()

    if output_rows != total_rows:
        raise RuntimeError(
            f"Output row count mismatch during write: {output_rows:,} != {total_rows:,}"
        )

    return {
        "split_index_zero_based": split_index,
        "run_a_rows": split_index,
        "run_b_rows": total_rows - split_index,
        "moved_run_b_rows": int(np.count_nonzero(insertion_positions < split_index)),
        "earliest_insertion_position_zero_based": int(insertion_positions.min()),
        "latest_insertion_position_zero_based": int(insertion_positions.max()),
        "compression": compression,
        "output_rows": output_rows,
        "elapsed_seconds": time.time() - started,
    }


def chronology_audit(path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(path, memory_map=True, pre_buffer=False)
    processed = 0
    previous: int | None = None
    inversions = 0
    first_bad = None
    first_timestamp = None
    last_timestamp = None

    for batch in pf.iter_batches(
        batch_size=1_000_000,
        columns=[TIMESTAMP_COLUMN],
        use_threads=True,
    ):
        arr = np.asarray(
            batch.column(0).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        if arr.size == 0:
            continue

        if first_timestamp is None:
            first_timestamp = int(arr[0])

        if previous is not None and int(arr[0]) < previous:
            inversions += 1
            if first_bad is None:
                first_bad = {
                    "absolute_row_zero_based": processed,
                    "previous": previous,
                    "current": int(arr[0]),
                }

        bad = np.flatnonzero(np.diff(arr) < 0)
        inversions += int(len(bad))
        if len(bad) and first_bad is None:
            i = int(bad[0])
            first_bad = {
                "absolute_row_zero_based": processed + i + 1,
                "previous": int(arr[i]),
                "current": int(arr[i + 1]),
            }

        previous = int(arr[-1])
        last_timestamp = int(arr[-1])
        processed += len(arr)

    return {
        "rows_checked": processed,
        "strict_inversions": inversions,
        "first_bad": first_bad,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
    }


def validate_corrected(
    source: Path,
    corrected: Path,
    source_counts: dict[str, int],
) -> dict[str, Any]:
    src_pf = pq.ParquetFile(source, memory_map=True, pre_buffer=False)
    dst_pf = pq.ParquetFile(corrected, memory_map=True, pre_buffer=False)

    if dst_pf.metadata.num_rows != src_pf.metadata.num_rows:
        raise RuntimeError("Corrected Parquet row count differs from source")
    if dst_pf.metadata.num_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS:,} corrected rows, got {dst_pf.metadata.num_rows:,}"
        )
    if not src_pf.schema_arrow.equals(dst_pf.schema_arrow, check_metadata=True):
        raise RuntimeError("Corrected Parquet schema/metadata differs from source")

    corrected_counts = stream_class_counts(corrected)
    if corrected_counts != source_counts:
        raise RuntimeError("Class counts changed during temporal repair")
    if len(corrected_counts) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} classes, found {len(corrected_counts)}"
        )

    chronology = chronology_audit(corrected)
    if chronology["strict_inversions"] != 0:
        raise RuntimeError(
            f"Corrected dataset still has {chronology['strict_inversions']} inversions"
        )

    return {
        "rows": dst_pf.metadata.num_rows,
        "schema_equal_with_metadata": True,
        "class_counts_equal": True,
        "class_counts": corrected_counts,
        "chronology": chronology,
    }


def source_hash_from_v10_lock() -> str | None:
    lock = load_json(V10_LOCK, {}) or {}
    fp = (
        lock.get("dataset_fingerprints", {})
        .get("NF-CSE-CIC-IDS2018-v3", {})
    )
    return fp.get("sha256")


def build_incident_manifest(
    source_sha: str,
    inversion_row: int,
    previous_ts: int,
    current_ts: int,
    split_index: int,
) -> dict[str, Any]:
    return {
        "incident_id": "E1_CSE_TEMPORAL_INTEGRITY_2026_08_01",
        "status": "INVALIDATED_FOR_FINAL_ANALYSIS",
        "dataset": "NF-CSE-CIC-IDS2018-v3",
        "source_path": str(SOURCE),
        "source_sha256": source_sha,
        "detected_by": "strict full-stream prequential chronology guard",
        "strict_inversions": 1,
        "inversion_absolute_row_zero_based": inversion_row,
        "inversion_human_row_one_based": inversion_row + 1,
        "previous_timestamp_ms": previous_ts,
        "current_timestamp_ms": current_ts,
        "delta_ms": current_ts - previous_ts,
        "split_index_zero_based": split_index,
        "reason_checkpoints_not_reusable": (
            "Run-B timestamps precede timestamps already consumed before the 20M "
            "checkpoint. Corrected chronological order therefore changes the prefix "
            "that produced every v1.0 NF-CSE model state."
        ),
        "affected_v1_0_jobs": 16,
        "required_action": (
            "Preserve v1.0 files as audit evidence. Exclude all v1.0 NF-CSE metrics "
            "and checkpoints from final analysis. Rerun all 16 NF-CSE jobs from row 0 "
            "against the temporal-corrected dataset."
        ),
        "created_at_utc": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--skip-source-hash-verification",
        action="store_true",
        help="Not recommended; only for emergency diagnostics.",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("E1-TDIM-MSB TEMPORAL REPAIR v1.1: NF-CSE-CIC-IDS2018-v3")
    print("=" * 100)
    print("Source    :", SOURCE)
    print("Corrected :", CORRECTED)

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    source_pf = pq.ParquetFile(SOURCE, memory_map=True, pre_buffer=False)
    if source_pf.metadata.num_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"Source row count changed: expected {EXPECTED_ROWS:,}, "
            f"found {source_pf.metadata.num_rows:,}"
        )

    free_bytes = shutil.disk_usage(DATASET_ROOT).free
    required_free = int(SOURCE.stat().st_size * 1.5)
    print(f"Free disk  : {free_bytes / 1024**3:.2f} GB")
    print(f"Source size: {SOURCE.stat().st_size / 1024**3:.2f} GB")
    if not args.audit_only and free_bytes < required_free:
        raise RuntimeError(
            "Insufficient free disk for safe corrected-dataset creation. "
            f"Need at least {required_free / 1024**3:.2f} GB free."
        )

    expected_source_sha = source_hash_from_v10_lock()
    print("\nComputing source SHA-256...")
    started = time.time()
    source_sha = sha256_file(SOURCE)
    print(f"Source SHA : {source_sha}")
    print(f"Hash time  : {time.time() - started:.1f}s")

    if (
        not args.skip_source_hash_verification
        and expected_source_sha
        and source_sha != expected_source_sha
    ):
        raise RuntimeError(
            "Source SHA-256 differs from the v1.0 protocol lock. "
            "Refusing temporal repair."
        )
    if expected_source_sha:
        print("v1.0 source fingerprint: PASS")
    else:
        print("WARNING: v1.0 lock has no source SHA-256; current hash will be provenance source.")

    print("\nLoading timestamp column and locating strict inversions...")
    timestamps = read_timestamp_array(SOURCE)
    bad = discover_inversions(timestamps)
    print("Strict inversions:", len(bad))

    if len(bad) != 1:
        raise RuntimeError(
            f"v1.1 repair is intentionally restricted to exactly one inversion; found {len(bad)}"
        )

    inversion_previous_index = int(bad[0])
    split_index = inversion_previous_index + 1
    previous_ts = int(timestamps[inversion_previous_index])
    current_ts = int(timestamps[split_index])

    print("Inversion row (0-based):", f"{split_index:,}")
    print("Previous timestamp     :", previous_ts)
    print("Current timestamp      :", current_ts)
    print("Delta ms               :", current_ts - previous_ts)

    if split_index != EXPECTED_INVERSION_ZERO_BASED:
        raise RuntimeError(
            "Detected inversion location differs from the audited incident. "
            f"Expected {EXPECTED_INVERSION_ZERO_BASED:,}, found {split_index:,}."
        )

    a_ts = timestamps[:split_index]
    b_ts = timestamps[split_index:]
    if np.any(np.diff(a_ts) < 0) or np.any(np.diff(b_ts) < 0):
        raise RuntimeError("Source is not exactly two individually monotonic runs")

    insertion_positions = np.searchsorted(a_ts, b_ts, side="right")
    tail_repair_safe = bool(np.min(b_ts) >= a_ts[19_999_999])
    print("Run A rows             :", f"{len(a_ts):,}")
    print("Run B rows             :", f"{len(b_ts):,}")
    print("Earliest B insertion   :", f"{int(insertion_positions.min()):,}")
    print("Tail-only resume safe  :", tail_repair_safe)

    incident = build_incident_manifest(
        source_sha,
        split_index,
        previous_ts,
        current_ts,
        split_index,
    )
    save_json(INCIDENT_FILE, incident)
    print("Incident manifest      :", INCIDENT_FILE)

    if args.audit_only:
        print("AUDIT-ONLY: no corrected dataset written.")
        return 0

    if CORRECTED.exists() and not args.force_rebuild:
        existing = load_json(PROVENANCE, {}) or {}
        if existing.get("corrected_sha256"):
            current_corrected_sha = sha256_file(CORRECTED)
            if current_corrected_sha == existing.get("corrected_sha256"):
                print("\nCorrected dataset already exists and matches provenance.")
                print("No rewrite required.")
                return 0
        raise FileExistsError(
            f"Corrected path already exists: {CORRECTED}. "
            "Use --force-rebuild only after verifying it is safe to replace."
        )

    if args.force_rebuild and CORRECTED.exists():
        backup = CORRECTED.with_suffix(
            CORRECTED.suffix + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        CORRECTED.replace(backup)
        print("Existing corrected dataset moved to:", backup)

    print("\nComputing source class counts...")
    source_counts = stream_class_counts(SOURCE)
    print("Classes:", len(source_counts))

    tmp = CORRECTED.with_suffix(CORRECTED.suffix + ".tmp")
    print("\nPerforming stable two-run chronological merge...")
    merge_info = stable_two_run_repair(
        SOURCE,
        tmp,
        split_index=split_index,
        timestamps=timestamps,
        batch_size=BATCH_SIZE,
    )

    print("Validating temporary corrected dataset...")
    validation = validate_corrected(SOURCE, tmp, source_counts)

    # Atomic publication only after all semantic validations pass.
    tmp.replace(CORRECTED)

    print("Computing corrected SHA-256...")
    corrected_sha = sha256_file(CORRECTED)

    provenance = {
        "repair_id": "E1_CSE_TEMPORAL_REPAIR_v1.1",
        "created_at_utc": utc_now(),
        "method": "stable two-way merge of two individually monotonic source runs",
        "timestamp_column": TIMESTAMP_COLUMN,
        "stable_tie_policy": (
            "For equal timestamps, Run-A rows retain precedence over Run-B rows, "
            "and original order is preserved within each run."
        ),
        "source": {
            "path": str(SOURCE),
            "sha256": source_sha,
            "size_bytes": SOURCE.stat().st_size,
            "rows": EXPECTED_ROWS,
        },
        "corrected": {
            "path": str(CORRECTED),
            "sha256": corrected_sha,
            "size_bytes": CORRECTED.stat().st_size,
            "rows": EXPECTED_ROWS,
        },
        "incident": incident,
        "merge": merge_info,
        "validation": validation,
        "invariants": {
            "row_count_unchanged": True,
            "schema_and_metadata_unchanged": True,
            "class_counts_unchanged": True,
            "row_values_transformed": False,
            "only_row_order_changed": True,
            "corrected_strict_inversions": 0,
        },
        "checkpoint_policy": {
            "v1_0_nf_cse_checkpoints_reusable": False,
            "reason": (
                "The moved Run-B rows belong before the 20M checkpoint boundary."
            ),
            "v1_1_nf_cse_rerun_from_row": 0,
        },
    }
    save_json(PROVENANCE, provenance)

    print("\n" + "=" * 100)
    print("TEMPORAL REPAIR: PASSED")
    print("=" * 100)
    print("Corrected dataset :", CORRECTED)
    print("Rows              :", f"{validation['rows']:,}")
    print("Strict inversions :", validation["chronology"]["strict_inversions"])
    print("Classes           :", len(validation["class_counts"]))
    print("Source SHA-256    :", source_sha)
    print("Corrected SHA-256 :", corrected_sha)
    print("Provenance        :", PROVENANCE)
    print("Incident record   :", INCIDENT_FILE)
    print(
    "\nChronology correction complete. "
    "Use the corrected dataset with the frozen configurations under configs/.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
