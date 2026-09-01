from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit


DEFAULT_VARIANT = "large_ctc"
SUPPORTED_VARIANTS = frozenset({"ctc", "large_ctc"})
DEFAULT_LANGUAGE = "ru"
SUPPORTED_LANGUAGES = frozenset({"ru", "en", "kk", "ky", "uz"})
DEFAULT_CHUNK_SECONDS = 25.0
MIN_CHUNK_SECONDS = 1.0
MAX_CHUNK_SECONDS = 29.0
_SAMPLE_RATE = 16_000
_LONGFORM_ERROR_MESSAGE = "Too long wav file, use 'transcribe_longform' method."


def _error(error_type: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error_type": error_type, "error": message}


def _exception_message(exc: Exception) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _validate_local_path(request: dict[str, Any], field: str, *, directory: bool) -> str:
    value = request.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty local path")

    parsed = urlsplit(value)
    if parsed.scheme or "://" in value:
        raise ValueError(f"{field} must be a local path, not a URI")

    path = Path(value)
    try:
        valid = path.is_dir() if directory else path.is_file()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{field} is not a usable local path") from exc
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{field} must point to an existing local {kind}")
    return value


def _validate_model_files(model_path: str) -> None:
    path = Path(model_path)
    required = ("config.json", "modeling_gigaam.py")
    missing = [name for name in required if not (path / name).is_file()]

    # The official ai-sage/GigaAM-Multilingual snapshots currently contain
    # pytorch_model.bin. Accept the standard safetensors and sharded layouts
    # too, without weakening the local-only requirement.
    weight_patterns = (
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.safetensors",
        "pytorch_model-*.bin",
        "model-*.safetensors",
    )
    has_weights = any(
        (path / pattern).is_file()
        if "*" not in pattern
        else any(path.glob(pattern))
        for pattern in weight_patterns
    )
    if not has_weights:
        missing.append("model weights (pytorch_model.bin or safetensors)")

    if missing:
        raise ValueError(
            f"model_path is missing required model files: {', '.join(missing)}"
        )


def _validate_request(request: Any) -> tuple[str, str, str, str, float]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    model_path = _validate_local_path(request, "model_path", directory=True)
    audio_path = _validate_local_path(request, "audio_path", directory=False)
    _validate_model_files(model_path)

    variant = request.get("variant", DEFAULT_VARIANT)
    if not isinstance(variant, str) or variant not in SUPPORTED_VARIANTS:
        choices = ", ".join(sorted(SUPPORTED_VARIANTS))
        raise ValueError(f"variant must be one of: {choices}")

    language = request.get("language", DEFAULT_LANGUAGE)
    if not isinstance(language, str) or language not in SUPPORTED_LANGUAGES:
        choices = ", ".join(sorted(SUPPORTED_LANGUAGES))
        raise ValueError(f"language must be one of: {choices}")

    chunk_seconds = request.get("chunk_seconds", DEFAULT_CHUNK_SECONDS)
    if isinstance(chunk_seconds, bool) or not isinstance(chunk_seconds, (int, float)):
        raise ValueError(
            f"chunk_seconds must be a number between {MIN_CHUNK_SECONDS:g} and "
            f"{MAX_CHUNK_SECONDS:g}"
        )
    chunk_seconds = float(chunk_seconds)
    if not math.isfinite(chunk_seconds) or not (
        MIN_CHUNK_SECONDS <= chunk_seconds <= MAX_CHUNK_SECONDS
    ):
        raise ValueError(
            f"chunk_seconds must be between {MIN_CHUNK_SECONDS:g} and "
            f"{MAX_CHUNK_SECONDS:g} seconds"
        )

    return model_path, audio_path, variant, language, chunk_seconds


def _load_model(model_path: str) -> Any:
    # Keep the foreign runtime import on the execution path so injected unit-test
    # loaders do not require transformers, torch, or GigaAM dependencies.
    from transformers import AutoModel

    return AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def _normalize_transcript(result: Any) -> str:
    if isinstance(result, str):
        return result

    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    raise TypeError("model returned a result without a string .text")


def _transcribe_offline_fixed_chunks(
    model: Any,
    audio_path: str,
    chunk_seconds: float,
    *,
    soundfile_module: Any | None = None,
) -> str:
    # Import only when the short-form API reports that chunking is required.
    if soundfile_module is None:
        import soundfile as soundfile_module

    audio, sample_rate = soundfile_module.read(
        audio_path, dtype="float32", always_2d=False
    )
    if sample_rate != _SAMPLE_RATE:
        raise ValueError(
            f"unsupported sample rate {sample_rate} Hz; expected {_SAMPLE_RATE} Hz"
        )

    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    chunk_frames = int(round(chunk_seconds * _SAMPLE_RATE))
    transcripts: list[str] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for chunk_index, start in enumerate(range(0, len(audio), chunk_frames)):
            chunk = audio[start : start + chunk_frames]
            if len(chunk) == 0:
                continue

            chunk_path = Path(temp_dir) / f"chunk_{chunk_index:04d}.wav"
            soundfile_module.write(chunk_path, chunk, _SAMPLE_RATE)
            text = _normalize_transcript(model.transcribe(str(chunk_path))).strip()
            if text:
                transcripts.append(text)

    return " ".join(transcripts)


def execute(
    request: Any,
    *,
    model_loader: Callable[[str], Any] | None = None,
    soundfile_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute one offline GigaAM Multilingual transcription request."""
    try:
        model_path, audio_path, variant, language, chunk_seconds = _validate_request(
            request
        )
    except Exception as exc:
        return _error("validation_error", _exception_message(exc))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    loader = _load_model if model_loader is None else model_loader
    load_started = clock()
    try:
        model = loader(model_path)
    except Exception as exc:
        return _error("load_error", _exception_message(exc))
    load_seconds = float(clock() - load_started)

    transcribe_started = clock()
    transcription_mode = "short"
    try:
        raw_transcript = model.transcribe(audio_path)
    except Exception as exc:
        if not (
            isinstance(exc, ValueError)
            and str(exc) == _LONGFORM_ERROR_MESSAGE
        ):
            return _error("transcribe_error", _exception_message(exc))

        transcription_mode = "offline_fixed_chunks"
        try:
            raw_transcript = _transcribe_offline_fixed_chunks(
                model,
                audio_path,
                chunk_seconds,
                soundfile_module=soundfile_module,
            )
        except Exception as chunk_exc:
            return _error("transcribe_error", _exception_message(chunk_exc))
    transcribe_seconds = float(clock() - transcribe_started)

    try:
        transcript = _normalize_transcript(raw_transcript)
    except Exception as exc:
        return _error("transcribe_error", _exception_message(exc))

    return {
        "status": "ok",
        "transcript": transcript,
        "load_seconds": load_seconds,
        "transcribe_seconds": transcribe_seconds,
        "language": language,
        "variant": variant,
        "transcription_mode": transcription_mode,
        "effective_config": {
            "model_path": model_path,
            "audio_path": audio_path,
            "offline": True,
            "local_files_only": True,
            "trust_remote_code": True,
            "transcription_mode": transcription_mode,
            "chunk_seconds": chunk_seconds,
            "chunk_overlap_seconds": 0.0,
            "offline_environment": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        },
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        payload = _error("request_error", _exception_message(exc))
    else:
        # Foreign runtimes may print warnings or progress; keep stdout a
        # single JSON response and route those diagnostics to stderr.
        with redirect_stdout(sys.stderr):
            payload = execute(request)

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
