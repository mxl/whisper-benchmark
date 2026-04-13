#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import benchmark_whisper


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick benchmark smoke test against a prepared sample."
    )
    parser.add_argument(
        "--audio",
        default="en",
        help=(
            "Audio selector passed through to benchmark_whisper.py. Defaults to the "
            "bundled English sample."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=[
            "faster-whisper",
            "mlx-whisper",
            "mlx-audio",
            "lightning-whisper-mlx",
            "insanely-fast-whisper",
            "openai-whisper",
        ],
        default="mlx-whisper",
        help="Backend to benchmark. Defaults to mlx-whisper.",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help="Model to benchmark. Defaults to tiny for a fast sanity check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/smoke_test_results.json"),
        help="Where to write the JSON benchmark output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    command = [
        "--audio",
        args.audio,
        "--backends",
        args.backend,
        "--models",
        args.model,
        "--runs",
        "1",
        "--output",
        str(args.output),
    ]

    exit_code = benchmark_whisper.main(command)
    if exit_code == 0:
        print(f"\nSmoke test passed. Results written to {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
