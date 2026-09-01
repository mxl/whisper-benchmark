from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0
TIMESTAMP_SEMANTICS = "chunk/segment offsets"
_SEGMENT_FIELDS = ("start", "end", "text", "language")


def _error(error_type: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error_type": error_type, "error": message}


def _exception_message(exc: Exception) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _validate_local_path(
    request: dict[str, Any], field: str, *, directory: bool
) -> str:
    value = request.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty local path")

    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or "://" in value:
        raise ValueError(f"{field} must be a local path, not a URI")

    path = Path(value)
    try:
        valid = path.is_dir() if directory else path.is_file()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{field} is not a usable local path") from exc
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"{field} must point to an existing local {kind}")

    if directory:
        try:
            nonempty = next(path.iterdir(), None) is not None
        except (OSError, ValueError) as exc:
            raise ValueError(f"{field} is not a readable local directory") from exc
        if not nonempty:
            raise ValueError(f"{field} must point to a non-empty local directory")

    return value


def _validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    model_path = _validate_local_path(request, "model_path", directory=True)
    audio_path = _validate_local_path(request, "audio_path", directory=False)

    language = request.get("language")
    if "language" in request and (
        not isinstance(language, str) or not language
    ):
        raise ValueError("language must be a non-empty string when provided")
    if language == "auto":
        language = None

    max_tokens = request.get("max_tokens", DEFAULT_MAX_TOKENS)
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")

    temperature = request.get("temperature", DEFAULT_TEMPERATURE)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite number greater than or equal to 0")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a finite number greater than or equal to 0")

    return {
        "model_path": model_path,
        "audio_path": audio_path,
        "language": language,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def _load_model(local_snapshot_path: str) -> Any:
    # Keep the MLX import lazy so validation works without Apple Silicon or
    # mlx-audio installed, and so no repository ID can reach the loader.
    from mlx_audio.stt import load

    return load(local_snapshot_path)


def _field(value: Any, name: str) -> tuple[bool, Any]:
    if isinstance(value, Mapping):
        if name in value:
            return True, value[name]
        return False, None

    try:
        return True, getattr(value, name)
    except AttributeError:
        return False, None


def _json_safe(value: Any, field_name: str) -> Any:
    # MLX/numpy scalar values are common in timestamp records but are not
    # accepted by the standard JSON encoder.  Convert only scalar objects that
    # explicitly expose a scalar item; do not serialize arbitrary model data.
    item = getattr(value, "item", None)
    if item is not None and callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = value
        if scalar is not value:
            value = scalar

    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"segment {field_name} is not JSON-safe") from exc
    return value


def _normalize_segment(segment: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    found_field = False
    for field_name in _SEGMENT_FIELDS:
        present, value = _field(segment, field_name)
        if present:
            found_field = True
            normalized[field_name] = _json_safe(value, field_name)

    if not found_field:
        raise TypeError("model returned an unsupported segment value")
    return normalized


def _segment_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (str, bytes)):
        raise TypeError("model returned unsupported segments")
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _normalize_result(result: Any) -> tuple[str, list[dict[str, Any]], Any]:
    text_present, text = _field(result, "text")
    if not text_present or not isinstance(text, str):
        raise TypeError("model returned an unsupported transcript value")

    segments_present, raw_segments = _field(result, "segments")
    segments = (
        [_normalize_segment(segment) for segment in _segment_items(raw_segments)]
        if segments_present
        else []
    )

    language_present, language = _field(result, "language")
    if not language_present:
        language = None
    language = _json_safe(language, "language")
    return text, segments, language


def execute(
    request: Any,
    *,
    model_loader: Callable[[str], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute one offline local Qwen3-ASR MLX transcription request."""
    try:
        config = _validate_request(request)
    except Exception as exc:
        return _error("validation_error", _exception_message(exc))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    loader = _load_model if model_loader is None else model_loader
    load_started = clock()
    try:
        model = loader(config["model_path"])
    except Exception as exc:
        return _error("load_error", _exception_message(exc))
    load_seconds = float(clock() - load_started)

    transcribe_started = clock()
    try:
        result = model.generate(
            config["audio_path"],
            language=config["language"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            verbose=False,
            stream=False,
        )
    except Exception as exc:
        return _error("transcribe_error", _exception_message(exc))
    transcribe_seconds = float(clock() - transcribe_started)

    try:
        transcript, segments, language = _normalize_result(result)
    except Exception as exc:
        return _error("transcribe_error", _exception_message(exc))

    effective_config = {
        "model_path": config["model_path"],
        "audio_path": config["audio_path"],
        "language": config["language"],
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "backend": "mlx",
        "offline": True,
        "local_files_only": True,
        "network_access": False,
        "verbose": False,
        "stream": False,
        "no_word_timestamps": True,
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    }

    return {
        "status": "ok",
        "transcript": transcript,
        "segments": segments,
        "language": language,
        "load_seconds": load_seconds,
        "transcribe_seconds": transcribe_seconds,
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "effective_config": effective_config,
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        payload = _error("request_error", _exception_message(exc))
    else:
        # Foreign runtimes may print progress or warnings; stdout remains one
        # JSON response while those diagnostics are routed to stderr.
        with redirect_stdout(sys.stderr):
            payload = execute(request)

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
