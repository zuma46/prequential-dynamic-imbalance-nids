from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


class CsvAppender:
    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._handle = self.path.open("a", newline="", encoding="utf-8", buffering=1_048_576)
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames, extrasaction="ignore")
        if self.path.stat().st_size == 0:
            self._writer.writeheader()

    def write(self, row: Mapping[str, Any]) -> None:
        self._writer.writerow(row)

    def write_many(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self._writer.writerows(rows)

    def flush(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if not self._handle.closed:
            self.flush()
            self._handle.close()


def atomic_json_dump(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def atomic_pickle_dump(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def pickle_load(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def truncate_csv_after(path: str | Path, instance_column: str, max_instance: int) -> None:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    temp = path.with_suffix(path.suffix + ".truncate.tmp")
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or instance_column not in reader.fieldnames:
            return
        with temp.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                try:
                    instance = int(float(row[instance_column]))
                except (TypeError, ValueError):
                    continue
                if instance <= max_instance:
                    writer.writerow(row)
    os.replace(temp, path)


def fast_file_fingerprint(path: str | Path, sample_bytes: int = 1_048_576) -> dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        head = handle.read(sample_bytes)
        digest.update(head)
        if stat.st_size > sample_bytes:
            handle.seek(max(0, stat.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sample_sha256": digest.hexdigest(),
    }
