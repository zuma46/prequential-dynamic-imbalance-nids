from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Iterable


def detect_platform() -> str:
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists() and "google.colab" in sys.modules:
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    return "jupyterlab_hpc"


def default_output_root() -> Path:
    env = detect_platform()
    if env == "colab":
        drive = Path("/content/drive/MyDrive")
        if drive.exists():
            return drive / "Eksperimen_Disertasi" / "E1_TDIM_MSB_Pilot" / "outputs"
        return Path("/content/E1_TDIM_MSB_Pilot/outputs")
    if env == "kaggle":
        return Path("/kaggle/working/E1_TDIM_MSB_Pilot/outputs")
    return Path.cwd() / "outputs"


def dataset_search_roots() -> list[Path]:
    env = detect_platform()
    roots: list[Path] = []
    if env == "colab":
        roots.extend([
            Path("/content/drive/MyDrive/Eksperimen_Disertasi/datasets"),
            Path("/content/drive/MyDrive/Eksperimen_Disertasi/E1_TDIM_MSB_Pilot/datasets"),
            Path("/content"),
        ])
    elif env == "kaggle":
        roots.extend([Path("/kaggle/input"), Path("/kaggle/working")])
    else:
        cwd = Path.cwd()
        roots.extend([
            cwd,
            cwd / "dataset",
            cwd / "datasets",
            cwd.parent / "dataset",
            cwd.parent / "datasets",
            Path.home() / "dataset",
            Path.home() / "datasets",
        ])
    return [p for p in roots if p.exists()]


def find_dataset(
    exact_names: Iterable[str],
    roots: Iterable[Path] | None = None,
    max_depth: int = 8,
) -> Path:
    roots = list(roots or dataset_search_roots())
    names = {name.lower() for name in exact_names}
    candidates: list[Path] = []
    for root in roots:
        root = root.resolve()
        for current, dirs, files in os.walk(root):
            rel_depth = len(Path(current).relative_to(root).parts)
            if rel_depth >= max_depth:
                dirs[:] = []
            for filename in files:
                if filename.lower() in names:
                    candidates.append(Path(current) / filename)
    if not candidates:
        searched = ", ".join(str(p) for p in roots)
        raise FileNotFoundError(f"Dataset not found. Searched: {searched}")
    candidates.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return candidates[0].resolve()


def environment_manifest() -> dict[str, str]:
    return {
        "platform_profile": detect_platform(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
