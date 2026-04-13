from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def ensure_workspace_root_first() -> None:
    root = str(WORKSPACE_ROOT)
    sys.path[:] = [root, *[entry for entry in sys.path if entry != root]]


ensure_workspace_root_first()

import benchmark_whisper
import download_models
import prepare_samples
import smoke_test


CommandMain = Callable[[list[str] | None], int]


COMMAND_HELP: dict[str, str] = {
    "benchmark": "Run the benchmark across selected speech-to-text engines.",
    "download-models": "Download model files for supported engines.",
    "prepare-samples": "Download and prepare benchmark audio samples.",
    "smoke-test": "Run a quick benchmark smoke test.",
}


def resolve_command_main(command: str) -> CommandMain:
    if command == "benchmark":
        return benchmark_whisper.main
    if command == "download-models":
        return download_models.main
    if command == "prepare-samples":
        return prepare_samples.main
    if command == "smoke-test":
        return smoke_test.main
    raise ValueError(f"Unsupported command: {command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stt-benchmark",
        description="Unified CLI for STT benchmark workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    for command, help_text in COMMAND_HELP.items():
        subparsers.add_parser(command, help=help_text, description=help_text)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return 2

    return resolve_command_main(args.command)(remaining)
