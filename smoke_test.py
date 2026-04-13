#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a quick benchmark smoke test against a prepared sample."
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=Path("samples/librispeech_1089_134686.mp3"),
        help="Audio file to benchmark. Defaults to the prepared English sample.",
    )
    parser.add_argument(
        "--reference-transcript",
        type=Path,
        default=Path("samples/librispeech_1089_134686.txt"),
        help="Reference transcript used for WER/CER. Defaults to the prepared English sample transcript.",
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
        "--language",
        default="en",
        help="Language passed to benchmark_whisper.py. Defaults to en.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/smoke_test_results.json"),
        help="Where to write the JSON benchmark output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    command = [
        str(Path(".venv/bin/python3")),
        "benchmark_whisper.py",
        str(args.audio),
        "--backends",
        args.backend,
        "--models",
        args.model,
        "--runs",
        "1",
        "--language",
        args.language,
        "--reference-transcript",
        str(args.reference_transcript),
        "--output",
        str(args.output),
    ]

    completed = subprocess.run(command)
    if completed.returncode == 0:
        print(f"\nSmoke test passed. Results written to {args.output}")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
