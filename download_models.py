#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys

from huggingface_hub import snapshot_download

from benchmark_whisper import (
    DEFAULT_MODELS,
    INSANELY_FAST_WHISPER_REPOS,
    LIGHTNING_WHISPER_MLX_REPOS,
    MLX_AUDIO_WHISPER_REPOS,
    MLX_WHISPER_REPOS,
)


FASTER_WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}

OPENAI_WHISPER_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "large-v3-turbo": "turbo",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download benchmark model files for each supported engine."
    )
    parser.add_argument(
        "--faster-whisper",
        action="store_true",
        help="Download faster-whisper model repos only.",
    )
    parser.add_argument(
        "--mlx-whisper",
        action="store_true",
        help="Download mlx-whisper model repos only.",
    )
    parser.add_argument(
        "--mlx-audio",
        action="store_true",
        help="Download mlx-audio Whisper model repos only.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all supported model repos.",
    )
    parser.add_argument(
        "--openai",
        "--openai-whisper",
        dest="openai_whisper",
        action="store_true",
        help="Download openai-whisper model repos only.",
    )
    parser.add_argument(
        "--lightning-whisper-mlx",
        action="store_true",
        help="Download lightning-whisper-mlx model repos only.",
    )
    parser.add_argument(
        "--insanely-fast-whisper",
        action="store_true",
        help="Download insanely-fast-whisper model repos only.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to download. Defaults to tiny base small medium large-v3 large-v3-turbo.",
    )
    return parser.parse_args(argv)


def download_faster_whisper(model_name: str) -> None:
    repo_id = FASTER_WHISPER_REPOS[model_name]
    print(f"Downloading faster-whisper model: {repo_id}", flush=True)
    snapshot_download(repo_id=repo_id)
    print(f"\nFinished faster-whisper model: {repo_id}", flush=True)


def download_openai_whisper(model_name: str) -> None:
    import whisper

    whisper_model_name = OPENAI_WHISPER_MODELS[model_name]
    print(f"Downloading openai-whisper model: {whisper_model_name}", flush=True)
    whisper.load_model(whisper_model_name)
    print(f"\nFinished openai-whisper model: {whisper_model_name}", flush=True)


def download_mlx_whisper(model_name: str) -> None:
    repo_name = MLX_WHISPER_REPOS[model_name]
    print(f"Downloading mlx-whisper model: {repo_name}", flush=True)
    snapshot_download(repo_id=repo_name)
    print(f"\nFinished mlx-whisper model: {repo_name}", flush=True)


def download_mlx_audio_whisper(model_name: str) -> None:
    repo_name = MLX_AUDIO_WHISPER_REPOS.get(model_name)
    if repo_name is None:
        return
    print(f"Downloading mlx-audio model: {repo_name}", flush=True)
    snapshot_download(repo_id=repo_name)
    print(f"\nFinished mlx-audio model: {repo_name}", flush=True)


def download_lightning_whisper_mlx(model_name: str) -> None:
    if model_name not in LIGHTNING_WHISPER_MLX_REPOS:
        return
    repo_id = LIGHTNING_WHISPER_MLX_REPOS[model_name]
    print(f"Downloading lightning-whisper-mlx model: {repo_id}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["config.json", "weights.npz"],
    )
    print(f"\nFinished lightning-whisper-mlx model: {repo_id}", flush=True)


def download_insanely_fast_whisper(model_name: str) -> None:
    repo_id = INSANELY_FAST_WHISPER_REPOS[model_name]
    print(f"Downloading insanely-fast-whisper model: {repo_id}", flush=True)
    snapshot_download(repo_id=repo_id)
    print(f"\nFinished insanely-fast-whisper model: {repo_id}", flush=True)


def resolve_engines(args: argparse.Namespace) -> set[str]:
    engines = set()
    if args.all:
        return {
            "faster-whisper",
            "mlx-whisper",
            "mlx-audio",
            "lightning-whisper-mlx",
            "insanely-fast-whisper",
            "openai-whisper",
        }
    if args.faster_whisper:
        engines.add("faster-whisper")
    if args.openai_whisper:
        engines.add("openai-whisper")
    if args.mlx_whisper:
        engines.add("mlx-whisper")
    if args.mlx_audio:
        engines.add("mlx-audio")
    if args.lightning_whisper_mlx:
        engines.add("lightning-whisper-mlx")
    if args.insanely_fast_whisper:
        engines.add("insanely-fast-whisper")
    if not engines:
        return {
            "faster-whisper",
            "mlx-whisper",
            "mlx-audio",
            "lightning-whisper-mlx",
            "insanely-fast-whisper",
            "openai-whisper",
        }
    return engines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    engines = resolve_engines(args)
    failures: list[tuple[str, str]] = []

    for model_name in args.models:
        if "faster-whisper" in engines:
            try:
                download_faster_whisper(model_name)
            except Exception as exc:  # pragma: no cover - downloads should keep going.
                failures.append((f"faster-whisper:{model_name}", str(exc)))

        if "openai-whisper" in engines:
            try:
                download_openai_whisper(model_name)
            except Exception as exc:  # pragma: no cover - downloads should keep going.
                failures.append((f"openai-whisper:{model_name}", str(exc)))

        if "mlx-whisper" in engines:
            try:
                download_mlx_whisper(model_name)
            except Exception as exc:  # pragma: no cover - downloads should keep going.
                failures.append((f"mlx-whisper:{model_name}", str(exc)))

        if "mlx-audio" in engines:
            try:
                download_mlx_audio_whisper(model_name)
            except Exception as exc:  # pragma: no cover - downloads should keep going.
                failures.append((f"mlx-audio:{model_name}", str(exc)))

        if "lightning-whisper-mlx" in engines:
            try:
                download_lightning_whisper_mlx(model_name)
            except Exception as exc:  # pragma: no cover - downloads should keep going.
                failures.append((f"lightning-whisper-mlx:{model_name}", str(exc)))

        if "insanely-fast-whisper" in engines:
            try:
                download_insanely_fast_whisper(model_name)
            except Exception as exc:  # pragma: no cover - downloads should keep going.
                failures.append((f"insanely-fast-whisper:{model_name}", str(exc)))

    if failures:
        print("\nSome model downloads failed:", file=sys.stderr, flush=True)
        for label, error in failures:
            print(f"- {label}: {error}", file=sys.stderr, flush=True)
        return 1

    print("\nSelected models downloaded successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
