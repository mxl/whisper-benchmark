#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import statistics
import sys
import time
import traceback
import unicodedata
from dataclasses import asdict, dataclass
from huggingface_hub import snapshot_download
import jiwer
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
DEFAULT_OUTPUT = "benchmark_results.json"
OPENAI_WHISPER_REPOS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large",
    "large-v3-turbo": "turbo",
}
MLX_AUDIO_WHISPER_REPOS = {
    "tiny": "mlx-community/whisper-tiny-asr-fp16",
    "base": "mlx-community/whisper-base-asr-fp16",
    "small": "mlx-community/whisper-small-asr-fp16",
    "medium": "mlx-community/whisper-medium-asr-fp16",
    "large-v3": "mlx-community/whisper-large-v3-asr-fp16",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo-asr-fp16",
}
LIGHTNING_WHISPER_MLX_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
}
INSANELY_FAST_WHISPER_REPOS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}


@dataclass
class RunResult:
    backend: str
    model: str
    run_index: int
    load_seconds: float | None
    transcribe_seconds: float | None
    total_seconds: float | None
    transcript: str | None
    transcript_chars: int | None
    transcript_words: int | None
    wer: float | None
    cer: float | None
    detected_language: str | None
    detected_language_probability: float | None
    status: str
    error: str | None


@dataclass
class BackendSession:
    backend: str
    model: str
    session: Any
    load_seconds: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark faster-whisper vs mlx-whisper across multiple Whisper model sizes."
    )
    parser.add_argument("audio", type=Path, help="Path to an input audio file.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to benchmark. Defaults to tiny base small medium large-v3 large-v3-turbo.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=[
            "faster-whisper",
            "mlx-whisper",
            "mlx-audio",
            "lightning-whisper-mlx",
            "insanely-fast-whisper",
            "openai-whisper",
        ],
        default=[
            "faster-whisper",
            "mlx-whisper",
            "mlx-audio",
            "lightning-whisper-mlx",
            "insanely-fast-whisper",
            "openai-whisper",
        ],
        help="Backends to benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs per backend/model pair.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code to force transcription language.",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisper task.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size used for both backends when supported.",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="faster-whisper compute type. Example: default, int8, float16, int8_float16.",
    )
    parser.add_argument(
        "--faster-whisper-vad-filter",
        action="store_true",
        default=True,
        help="Enable faster-whisper VAD filter to drop silence and reduce hallucinations.",
    )
    parser.add_argument(
        "--no-faster-whisper-vad-filter",
        dest="faster_whisper_vad_filter",
        action="store_false",
        help="Disable the faster-whisper VAD filter.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        default=True,
        help="Condition each window on previously decoded text (improves long-form context).",
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_false",
        help="Disable conditioning on previously decoded text.",
    )
    parser.add_argument(
        "--openai-whisper-temperature-fallback",
        action="store_true",
        default=True,
        help="Enable openai-whisper temperature fallback (0.0..1.0) on low-confidence segments.",
    )
    parser.add_argument(
        "--no-openai-whisper-temperature-fallback",
        dest="openai_whisper_temperature_fallback",
        action="store_false",
        help="Disable openai-whisper temperature fallback and use temperature=0.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="faster-whisper device. Example: auto, cpu, cuda.",
    )
    parser.add_argument(
        "--mlx-prefix",
        default="mlx-community/whisper-",
        help="Prefix used to map model names to mlx-whisper Hugging Face repos.",
    )
    parser.add_argument(
        "--mlx-suffix",
        default="-mlx",
        help="Suffix used to map model names to mlx-whisper Hugging Face repos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional path to also write flat CSV results.",
    )
    parser.add_argument(
        "--reference-transcript",
        type=Path,
        default=None,
        help="Optional path to a ground-truth transcript for WER/CER scoring.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one untimed transcription warmup per backend/model before timed runs.",
    )
    parser.add_argument(
        "--insanely-fast-whisper-binary",
        default="insanely-fast-whisper",
        help="Path to the insanely-fast-whisper CLI binary.",
    )
    parser.add_argument(
        "--insanely-fast-whisper-device-id",
        default="mps",
        help='Device id passed to insanely-fast-whisper. Use "mps" on Apple Silicon.',
    )
    parser.add_argument(
        "--insanely-fast-whisper-batch-size",
        type=int,
        default=1,
        help="Batch size passed to insanely-fast-whisper.",
    )
    parser.add_argument(
        "--insanely-fast-whisper-flash",
        action="store_true",
        help="Enable insanely-fast-whisper flash attention mode.",
    )
    parser.add_argument(
        "--lightning-whisper-mlx-batch-size",
        type=int,
        default=12,
        help="Batch size passed to lightning-whisper-mlx.",
    )
    parser.add_argument(
        "--lightning-whisper-mlx-quant",
        choices=["none", "4bit", "8bit"],
        default="none",
        help="Quantization mode passed to lightning-whisper-mlx.",
    )
    return parser.parse_args()


def ensure_audio_file(audio_path: Path) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")
    return audio_path.resolve()


def summarize_text(text: str) -> tuple[int, int]:
    return len(text), len(text.split())


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split()).strip()


def load_reference_transcript(reference_path: Path | None) -> str | None:
    if reference_path is None:
        return None
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference transcript does not exist: {reference_path}"
        )
    if not reference_path.is_file():
        raise ValueError(f"Reference transcript is not a file: {reference_path}")
    return normalize_transcript(reference_path.read_text(encoding="utf-8"))


def get_audio_duration_seconds(audio_path: Path) -> float:
    import wave

    if audio_path.suffix.lower() == ".wav":
        with wave.open(str(audio_path), "rb") as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
        if sample_rate <= 0:
            raise ValueError(f"Invalid WAV sample rate in {audio_path}")
        return frames / sample_rate

    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.samplerate <= 0:
            raise ValueError(f"Invalid sample rate in {audio_path}")
        return info.frames / info.samplerate
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is required to compute audio duration for non-WAV files"
        ) from exc


def compute_word_error_rate(reference: str, hypothesis: str) -> float:
    return jiwer.wer(reference, hypothesis)


def compute_character_error_rate(reference: str, hypothesis: str) -> float:
    return jiwer.cer(reference, hypothesis)


def score_transcript(
    transcript: str, reference_transcript: str | None
) -> tuple[float | None, float | None]:
    if reference_transcript is None:
        return None, None
    normalized_transcript = normalize_transcript(transcript)
    return (
        compute_word_error_rate(reference_transcript, normalized_transcript),
        compute_character_error_rate(reference_transcript, normalized_transcript),
    )


def build_run_result(
    *,
    backend: str,
    model_name: str,
    run_index: int,
    load_seconds: float | None,
    transcribe_seconds: float,
    transcript: str,
    detected_language: str | None,
    detected_language_probability: float | None,
    reference_transcript: str | None,
) -> RunResult:
    chars, words = summarize_text(transcript)
    wer, cer = score_transcript(transcript, reference_transcript)
    return RunResult(
        backend=backend,
        model=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        total_seconds=(load_seconds or 0.0) + transcribe_seconds,
        transcript=transcript,
        transcript_chars=chars,
        transcript_words=words,
        wer=wer,
        cer=cer,
        detected_language=detected_language,
        detected_language_probability=detected_language_probability,
        status="ok",
        error=None,
    )


def run_faster_whisper(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    transcribe_started = time.perf_counter()
    segments, info = session.transcribe(
        str(audio_path),
        beam_size=args.beam_size,
        language=args.language,
        task=args.task,
        vad_filter=args.faster_whisper_vad_filter,
        condition_on_previous_text=args.condition_on_previous_text,
    )
    # faster-whisper yields segments lazily, so timing must include iteration.
    transcript = "".join(segment.text for segment in segments).strip()
    transcribe_seconds = time.perf_counter() - transcribe_started
    return build_run_result(
        backend="faster-whisper",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=getattr(info, "language", None),
        detected_language_probability=getattr(info, "language_probability", None),
        reference_transcript=args.reference_transcript_text,
    )


def run_mlx_whisper(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    import mlx_whisper

    transcribe_started = time.perf_counter()
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=session["model_repo"],
        language=args.language,
        task=args.task,
        condition_on_previous_text=args.condition_on_previous_text,
        fp16=True,
        verbose=False,
    )
    transcribe_seconds = time.perf_counter() - transcribe_started
    transcript = (result.get("text") or "").strip()
    return build_run_result(
        backend="mlx-whisper",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=result.get("language"),
        detected_language_probability=result.get("language_probability"),
        reference_transcript=args.reference_transcript_text,
    )


def run_insanely_fast_whisper(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    from insanely_fast_whisper.utils.result import build_result

    transcribe_started = time.perf_counter()
    outputs = session["pipe"](
        str(audio_path),
        chunk_length_s=30,
        batch_size=args.insanely_fast_whisper_batch_size,
        generate_kwargs=session["generate_kwargs"],
        return_timestamps=True,
    )
    transcribe_seconds = time.perf_counter() - transcribe_started

    result = build_result([], outputs)
    transcript = (result.get("text") or "").strip()
    return build_run_result(
        backend="insanely-fast-whisper",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=result.get("language"),
        detected_language_probability=None,
        reference_transcript=args.reference_transcript_text,
    )


def run_mlx_audio(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    from mlx_audio.stt.generate import generate_transcription

    transcribe_started = time.perf_counter()
    result = generate_transcription(
        model=session,
        audio=str(audio_path),
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        condition_on_previous_text=args.condition_on_previous_text,
    )
    transcribe_seconds = time.perf_counter() - transcribe_started
    transcript = (getattr(result, "text", None) or result.get("text") or "").strip()
    detected_language = getattr(result, "language", None)
    if detected_language is None and isinstance(result, dict):
        detected_language = result.get("language")

    return build_run_result(
        backend="mlx-audio",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=detected_language,
        detected_language_probability=None,
        reference_transcript=args.reference_transcript_text,
    )


def run_lightning_whisper_mlx(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    from lightning_whisper_mlx.transcribe import transcribe_audio

    transcribe_started = time.perf_counter()
    result = transcribe_audio(
        str(audio_path),
        path_or_hf_repo=session["model_path"],
        language=args.language,
        task=args.task,
        condition_on_previous_text=args.condition_on_previous_text,
        batch_size=args.lightning_whisper_mlx_batch_size,
        fp16=True,
        verbose=False,
    )
    transcribe_seconds = time.perf_counter() - transcribe_started
    transcript = (result.get("text") or "").strip()
    return build_run_result(
        backend="lightning-whisper-mlx",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=result.get("language"),
        detected_language_probability=None,
        reference_transcript=args.reference_transcript_text,
    )


def lightning_model_repo(model_name: str, quant: str | None) -> str:
    base = {
        "tiny": "mlx-community/whisper-tiny",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
    }
    quantized = {
        "tiny": {
            "4bit": "mlx-community/whisper-tiny-mlx-4bit",
            "8bit": "mlx-community/whisper-tiny-mlx-8bit",
        },
        "base": {
            "4bit": "mlx-community/whisper-base-mlx-4bit",
            "8bit": "mlx-community/whisper-base-mlx-8bit",
        },
        "small": {
            "4bit": "mlx-community/whisper-small-mlx-4bit",
            "8bit": "mlx-community/whisper-small-mlx-8bit",
        },
        "medium": {
            "4bit": "mlx-community/whisper-medium-mlx-4bit",
            "8bit": "mlx-community/whisper-medium-mlx-8bit",
        },
        "large-v3": {
            "4bit": "mlx-community/whisper-large-v3-mlx-4bit",
            "8bit": "mlx-community/whisper-large-v3-mlx-8bit",
        },
    }
    if quant is None:
        return base[model_name]
    return quantized[model_name][quant]


def run_openai_whisper(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    temperature = (
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        if args.openai_whisper_temperature_fallback
        else 0
    )
    transcribe_started = time.perf_counter()
    result = session.transcribe(
        str(audio_path),
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        temperature=temperature,
        best_of=None,
        condition_on_previous_text=args.condition_on_previous_text,
        verbose=False,
    )
    transcribe_seconds = time.perf_counter() - transcribe_started
    transcript = (result.get("text") or "").strip()
    return build_run_result(
        backend="openai-whisper",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=result.get("language"),
        detected_language_probability=None,
        reference_transcript=args.reference_transcript_text,
    )


def load_backend_session(
    backend: str, model_name: str, args: argparse.Namespace
) -> BackendSession:
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel

        load_started = time.perf_counter()
        session = WhisperModel(
            model_name,
            device=args.device,
            compute_type=args.compute_type,
        )
        return BackendSession(
            backend, model_name, session, time.perf_counter() - load_started
        )

    if backend == "mlx-whisper":
        import mlx.core as mx
        from mlx_whisper.load_models import load_model as mlx_whisper_load_model
        from mlx_whisper.transcribe import ModelHolder

        mlx_suffix = "" if model_name == "large-v3-turbo" else args.mlx_suffix
        model_repo = f"{args.mlx_prefix}{model_name}{mlx_suffix}"
        load_started = time.perf_counter()
        ModelHolder.model = mlx_whisper_load_model(model_repo, dtype=mx.float16)
        ModelHolder.model_path = model_repo
        return BackendSession(
            backend,
            model_name,
            {"model_repo": model_repo},
            time.perf_counter() - load_started,
        )

    if backend == "mlx-audio":
        from mlx_audio.stt.utils import load_model

        model_repo = MLX_AUDIO_WHISPER_REPOS.get(model_name)
        if model_repo is None:
            raise ValueError(f"Unsupported mlx-audio model: {model_name}")
        load_started = time.perf_counter()
        session = load_model(model_repo)
        return BackendSession(
            backend, model_name, session, time.perf_counter() - load_started
        )

    if backend == "lightning-whisper-mlx":
        import mlx.core as mx
        from lightning_whisper_mlx.transcribe import ModelHolder

        lightning_model_name = LIGHTNING_WHISPER_MLX_MODELS.get(model_name)
        if lightning_model_name is None:
            raise ValueError(f"Unsupported lightning-whisper-mlx model: {model_name}")
        quant = (
            None
            if args.lightning_whisper_mlx_quant == "none"
            else args.lightning_whisper_mlx_quant
        )
        load_started = time.perf_counter()
        model_path = snapshot_download(
            repo_id=lightning_model_repo(lightning_model_name, quant),
            allow_patterns=["config.json", "weights.npz"],
        )
        ModelHolder.get_model(model_path, dtype=mx.float16)
        return BackendSession(
            backend,
            model_name,
            {"model_path": model_path},
            time.perf_counter() - load_started,
        )

    if backend == "insanely-fast-whisper":
        import torch
        from transformers import pipeline

        model_repo = INSANELY_FAST_WHISPER_REPOS.get(model_name)
        if model_repo is None:
            raise ValueError(f"Unsupported insanely-fast-whisper model: {model_name}")
        device = (
            "mps"
            if args.insanely_fast_whisper_device_id == "mps"
            else f"cuda:{args.insanely_fast_whisper_device_id}"
        )
        attn = "flash_attention_2" if args.insanely_fast_whisper_flash else "sdpa"
        generate_kwargs = {"task": args.task, "language": args.language or None}
        if model_repo.endswith(".en"):
            generate_kwargs.pop("task")

        load_started = time.perf_counter()
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_repo,
            dtype=torch.float16,
            device=device,
            model_kwargs={"attn_implementation": attn},
        )
        if args.insanely_fast_whisper_device_id == "mps":
            torch.mps.empty_cache()
        return BackendSession(
            backend,
            model_name,
            {"pipe": pipe, "generate_kwargs": generate_kwargs},
            time.perf_counter() - load_started,
        )

    if backend == "openai-whisper":
        import whisper

        whisper_model_name = OPENAI_WHISPER_REPOS.get(model_name)
        if whisper_model_name is None:
            raise ValueError(f"Unsupported openai-whisper model: {model_name}")
        device = args.device if args.device != "auto" else None
        load_started = time.perf_counter()
        session = whisper.load_model(whisper_model_name, device=device)
        return BackendSession(
            backend, model_name, session, time.perf_counter() - load_started
        )

    raise ValueError(f"Unsupported backend: {backend}")


def run_single_backend(
    backend: str,
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    try:
        if backend == "faster-whisper":
            return run_faster_whisper(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        if backend == "mlx-whisper":
            return run_mlx_whisper(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        if backend == "mlx-audio":
            return run_mlx_audio(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        if backend == "lightning-whisper-mlx":
            return run_lightning_whisper_mlx(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        if backend == "insanely-fast-whisper":
            return run_insanely_fast_whisper(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        if backend == "openai-whisper":
            return run_openai_whisper(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        raise ValueError(f"Unsupported backend: {backend}")
    except (
        Exception
    ) as exc:  # pragma: no cover - benchmark scripts should continue after failures.
        return RunResult(
            backend=backend,
            model=model_name,
            run_index=run_index,
            load_seconds=None,
            transcribe_seconds=None,
            total_seconds=None,
            transcript=None,
            transcript_chars=None,
            transcript_words=None,
            wer=None,
            cer=None,
            detected_language=None,
            detected_language_probability=None,
            status="error",
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        )


def maybe_warmup(
    backend: str,
    audio_path: Path,
    model_name: str,
    args: argparse.Namespace,
    session: Any,
) -> None:
    if not args.warmup:
        return
    warmup_result = run_single_backend(
        backend, audio_path, model_name, 0, args, session, None
    )
    if warmup_result.status != "ok":
        print(
            f"warmup failed for {backend} {model_name}: {warmup_result.error}",
            file=sys.stderr,
        )


def aggregate_results(
    results: list[RunResult], audio_duration_seconds: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for result in results:
        grouped.setdefault((result.backend, result.model), []).append(result)

    aggregated: list[dict[str, Any]] = []
    for (backend, model), group in sorted(grouped.items()):
        ok_runs = [
            item
            for item in group
            if item.status == "ok" and item.total_seconds is not None
        ]
        total_values = [
            item.total_seconds for item in ok_runs if item.total_seconds is not None
        ]
        load_values = [
            item.load_seconds for item in ok_runs if item.load_seconds is not None
        ]
        transcribe_values = [
            item.transcribe_seconds
            for item in ok_runs
            if item.transcribe_seconds is not None
        ]
        wer_values = [item.wer for item in ok_runs if item.wer is not None]
        cer_values = [item.cer for item in ok_runs if item.cer is not None]

        aggregated.append(
            {
                "backend": backend,
                "model": model,
                "runs": len(group),
                "successful_runs": len(ok_runs),
                "failed_runs": len(group) - len(ok_runs),
                "avg_total_seconds": mean_or_none(total_values),
                "median_total_seconds": median_or_none(total_values),
                "min_total_seconds": min(total_values) if total_values else None,
                "max_total_seconds": max(total_values) if total_values else None,
                "avg_load_seconds": mean_or_none(load_values),
                "avg_transcribe_seconds": mean_or_none(transcribe_values),
                "stddev_transcribe_seconds": stdev_or_none(transcribe_values),
                "avg_rtf": (
                    mean_or_none(transcribe_values) / audio_duration_seconds
                    if transcribe_values and audio_duration_seconds > 0
                    else None
                ),
                "avg_wer": mean_or_none(wer_values),
                "avg_cer": mean_or_none(cer_values),
                "last_detected_language": ok_runs[-1].detected_language
                if ok_runs
                else None,
                "last_detected_language_probability": (
                    ok_runs[-1].detected_language_probability if ok_runs else None
                ),
                "last_transcript_chars": ok_runs[-1].transcript_chars
                if ok_runs
                else None,
                "last_transcript_words": ok_runs[-1].transcript_words
                if ok_runs
                else None,
                "last_wer": ok_runs[-1].wer if ok_runs else None,
                "last_cer": ok_runs[-1].cer if ok_runs else None,
                "errors": [item.error for item in group if item.error],
            }
        )
    return aggregated


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def stdev_or_none(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def print_summary(aggregated: list[dict[str, Any]]) -> None:
    headers = [
        "backend",
        "model",
        "ok",
        "avg_total_s",
        "median_total_s",
        "avg_load_s",
        "avg_transcribe_s",
        "stddev_transcribe_s",
        "avg_rtf",
        "avg_wer",
        "avg_cer",
    ]
    rows = []
    for row in aggregated:
        rows.append(
            [
                row["backend"],
                row["model"],
                f"{row['successful_runs']}/{row['runs']}",
                format_float(row["avg_total_seconds"]),
                format_float(row["median_total_seconds"]),
                format_float(row["avg_load_seconds"]),
                format_float(row["avg_transcribe_seconds"]),
                format_float(row["stddev_transcribe_seconds"]),
                format_float(row["avg_rtf"]),
                format_float(row["avg_wer"]),
                format_float(row["avg_cer"]),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print("\nColumns:")
    print("backend: benchmarked engine")
    print("model: normalized model name requested by the benchmark")
    print("ok: successful runs over total runs")
    print("avg_total_s: average end-to-end runtime in seconds")
    print("median_total_s: median end-to-end runtime in seconds")
    print("avg_load_s: average model load time in seconds when measurable")
    print("avg_transcribe_s: average transcription time in seconds")
    print("stddev_transcribe_s: standard deviation of transcription time in seconds")
    print("avg_rtf: average real-time factor (transcribe_s / audio_duration_s)")
    print("avg_wer: average word error rate against the reference transcript")
    print("avg_cer: average character error rate against the reference transcript")


def format_float(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(output_path: Path, results: list[RunResult]) -> None:
    fieldnames = (
        list(asdict(results[0]).keys())
        if results
        else list(RunResult.__dataclass_fields__.keys())
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def build_metadata(
    args: argparse.Namespace, audio_path: Path, audio_duration_seconds: float
) -> dict[str, Any]:
    return {
        "audio": str(audio_path),
        "audio_duration_seconds": audio_duration_seconds,
        "models": args.models,
        "backends": args.backends,
        "runs": args.runs,
        "language": args.language,
        "task": args.task,
        "reference_transcript": str(args.reference_transcript)
        if args.reference_transcript is not None
        else None,
        "beam_size": args.beam_size,
        "compute_type": args.compute_type,
        "device": args.device,
        "faster_whisper_vad_filter": args.faster_whisper_vad_filter,
        "condition_on_previous_text": args.condition_on_previous_text,
        "openai_whisper_temperature_fallback": args.openai_whisper_temperature_fallback,
        "mlx_prefix": args.mlx_prefix,
        "mlx_suffix": args.mlx_suffix,
        "lightning_whisper_mlx_batch_size": args.lightning_whisper_mlx_batch_size,
        "lightning_whisper_mlx_quant": args.lightning_whisper_mlx_quant,
        "insanely_fast_whisper_binary": args.insanely_fast_whisper_binary,
        "insanely_fast_whisper_device_id": args.insanely_fast_whisper_device_id,
        "insanely_fast_whisper_batch_size": args.insanely_fast_whisper_batch_size,
        "insanely_fast_whisper_flash": args.insanely_fast_whisper_flash,
        "warmup": args.warmup,
        "platform": platform.platform(),
        "python_version": sys.version,
    }


def main() -> int:
    args = parse_args()
    audio_path = ensure_audio_file(args.audio)
    audio_duration_seconds = get_audio_duration_seconds(audio_path)
    args.reference_transcript_text = load_reference_transcript(
        args.reference_transcript
    )

    results: list[RunResult] = []
    for model_name in args.models:
        for backend in args.backends:
            print(f"Benchmarking {backend} on model {model_name}...", file=sys.stderr)
            try:
                backend_session = load_backend_session(backend, model_name, args)
            except Exception as exc:  # pragma: no cover - benchmark scripts should continue after failures.
                error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                print(f"  load error: {error}", file=sys.stderr)
                for run_index in range(1, args.runs + 1):
                    results.append(
                        RunResult(
                            backend=backend,
                            model=model_name,
                            run_index=run_index,
                            load_seconds=None,
                            transcribe_seconds=None,
                            total_seconds=None,
                            transcript=None,
                            transcript_chars=None,
                            transcript_words=None,
                            wer=None,
                            cer=None,
                            detected_language=None,
                            detected_language_probability=None,
                            status="error",
                            error=error,
                        )
                    )
                continue

            maybe_warmup(backend, audio_path, model_name, args, backend_session.session)
            for run_index in range(1, args.runs + 1):
                result = run_single_backend(
                    backend,
                    audio_path,
                    model_name,
                    run_index,
                    args,
                    backend_session.session,
                    backend_session.load_seconds if run_index == 1 else None,
                )
                results.append(result)
                if result.status == "ok":
                    print(
                        f"  run {run_index}: total={format_float(result.total_seconds)}s",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  run {run_index}: error={result.error}",
                        file=sys.stderr,
                    )

    aggregated = aggregate_results(results, audio_duration_seconds)
    payload = {
        "metadata": build_metadata(args, audio_path, audio_duration_seconds),
        "summary": aggregated,
        "runs": [asdict(result) for result in results],
    }
    write_json(args.output, payload)
    if args.csv_output is not None:
        write_csv(args.csv_output, results)
    print_summary(aggregated)
    print(f"\nWrote JSON results to {args.output}")
    if args.csv_output is not None:
        print(f"Wrote CSV results to {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
