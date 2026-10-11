from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import apply_overrides, load_config
from .platforms import default_output_root, find_dataset
from .runner import run_experiment
from .schema import inspect_dataset
from .segment import select_representative_segment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="e1-tdim-msb")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.add_argument("--dataset", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--output-root", default=None)
    run_parser.add_argument("--seed", action="append", type=int, default=None)
    run_parser.add_argument("--max-rows", type=int, default=None)
    run_parser.add_argument("--row-start", type=int, default=None)
    run_parser.add_argument("--overwrite", action="store_true")

    find_parser = subparsers.add_parser("find-dataset")
    find_parser.add_argument("names", nargs="+")

    segment_parser = subparsers.add_parser("select-segment")
    segment_parser.add_argument("--config", required=True)
    segment_parser.add_argument("--dataset", required=True)
    segment_parser.add_argument("--segment-rows", type=int, default=2_000_000)
    segment_parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "find-dataset":
        print(find_dataset(args.names))
        return

    cfg = load_config(args.config)
    if args.command == "select-segment":
        inspection = inspect_dataset(args.dataset, cfg)
        result = select_representative_segment(
            path=args.dataset,
            label_column=inspection["label_column"],
            segment_rows=args.segment_rows,
            block_size=int(cfg["evaluation"]["block_size"]),
            output_dir=args.output_dir,
            use_threads=bool(cfg["dataset"]["use_threads"]),
            memory_map=bool(cfg["dataset"]["memory_map"]),
            pre_buffer=bool(cfg["dataset"]["pre_buffer"]),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if args.command == "inspect":
        print(json.dumps(inspect_dataset(args.dataset, cfg), indent=2, ensure_ascii=False))
        return

    overrides = {}
    if args.max_rows is not None:
        overrides["dataset.max_rows"] = args.max_rows
    if args.row_start is not None:
        overrides["dataset.row_start"] = args.row_start
    if args.overwrite:
        overrides["runtime.overwrite"] = True
    if overrides:
        cfg = apply_overrides(cfg, overrides)

    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else default_output_root()
    paths = run_experiment(cfg, args.dataset, output_root, seeds=args.seed)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
