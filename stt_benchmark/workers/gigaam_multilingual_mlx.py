from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Iterable, Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit


DEFAULT_VARIANT = "fp16"
SUPPORTED_VARIANTS = frozenset({"fp16"})
DEFAULT_LANGUAGE = "auto"
SUPPORTED_LANGUAGES = frozenset({"ru", "en", "kk", "ky", "uz"})
DEFAULT_CHUNK_SECONDS = 20.0
DEFAULT_OVERLAP_SECONDS = 2.0
MIN_CHUNK_SECONDS = 0.1
TIMESTAMP_SEMANTICS = "approximate greedy-CTC word emission times"
_SAMPLE_RATE = 16_000
_MISSING = object()


def _error(error_type: str, message: str) -> dict[str, Any]:
    return {"status": "error", "error_type": error_type, "error": message}


def _exception_message(exc: Exception) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


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
    return value


def _validate_model_files(model_path: str) -> None:
    path = Path(model_path)
    required = ("config.json", "manifest.json", "model.safetensors")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(
            f"model_path is missing required model files: {', '.join(missing)}"
        )


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or (
        minimum is not None and converted < minimum
    ):
        suffix = f" greater than or equal to {minimum:g}" if minimum is not None else ""
        raise ValueError(f"{field} must be a finite number{suffix}")
    return converted


def _validate_request(request: Any) -> dict[str, Any]:
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
    if language is None:
        language = DEFAULT_LANGUAGE
    if not isinstance(language, str) or language not in SUPPORTED_LANGUAGES | {"auto"}:
        choices = ", ".join(sorted(SUPPORTED_LANGUAGES | {"auto"}))
        raise ValueError(f"language must be one of: {choices}")

    chunk_seconds = _number(
        request.get("chunk_seconds", DEFAULT_CHUNK_SECONDS),
        "chunk_seconds",
        minimum=MIN_CHUNK_SECONDS,
    )
    overlap_seconds = _number(
        request.get("overlap_seconds", DEFAULT_OVERLAP_SECONDS),
        "overlap_seconds",
        minimum=0.0,
    )
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be less than chunk_seconds")

    return {
        "model_path": model_path,
        "audio_path": audio_path,
        "variant": variant,
        "language": language,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
    }


def _load_model(model_path: str) -> Any:
    # The package accepts a portable local artifact directory.  Passing the
    # local-files-only flag is intentional even for a local path: it prevents a
    # future package fallback from resolving a Hub repository or revision.
    from gigaam_multilingual_mlx import load_model

    return load_model(model_path, local_files_only=True)


def _load_audio(audio_path: str, sample_rate: int = _SAMPLE_RATE) -> Any:
    # The package's official loader uses ffmpeg to decode, downmix to one
    # channel, resample, and return a float32 NumPy waveform.
    from gigaam_multilingual_mlx.audio import load_audio

    return load_audio(audio_path, sample_rate=sample_rate)


def _import_mlx() -> Any:
    import mlx.core as mx

    return mx


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _json_scalar(value: Any, field: str) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"timestamp {field} is not JSON-safe")


def _finite_timestamp(value: Any, field: str) -> float:
    value = _json_scalar(value, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"timestamp {field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise TypeError(f"timestamp {field} must be finite")
    return converted


def _normalize_timestamp_items(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    items = [value] if isinstance(value, Mapping) else _as_list(value)
    normalized: list[dict[str, Any]] = []
    for item in items:
        text = _field(item, "text")
        start = _field(item, "start")
        end = _field(item, "end")
        if not isinstance(text, str) or start is _MISSING or end is _MISSING:
            raise TypeError("model returned an incomplete timestamp item")
        text = text.strip()
        if not text:
            continue
        normalized.append(
            {
                "text": text,
                "start": _finite_timestamp(start, "start"),
                "end": _finite_timestamp(end, "end"),
            }
        )
    return normalized


def _length_value(value: Any) -> int:
    # MLX arrays do not need to be converted to Python lists by the model API;
    # mirror the package's service implementation and use NumPy for this scalar.
    try:
        import numpy as np

        values = np.asarray(value).reshape(-1)
        if values.size == 0:
            raise TypeError("model returned no encoded lengths")
        value = values[0].item()
    except ImportError:
        values = _as_list(value)
        if not values:
            raise TypeError("model returned no encoded lengths")
        value = _json_scalar(values[0], "length")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("encoded length must be numeric")
    converted = int(value)
    if float(value) != converted or converted < 0:
        raise TypeError("encoded length must be a non-negative integer")
    return converted


def _token_values(value: Any, field: str) -> list[int]:
    values = _as_list(value)
    normalized: list[int] = []
    for item in values:
        item = _json_scalar(item, field)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{field} must contain integers")
        converted = int(item)
        if float(item) != converted or converted < 0:
            raise TypeError(f"{field} must contain non-negative integers")
        normalized.append(converted)
    return normalized


def _derive_word_timestamps(
    decoded: Mapping[str, Any],
    model: Any,
    audio_samples: int,
    encoded_length: int,
) -> list[dict[str, Any]]:
    explicit = decoded.get("words", _MISSING)
    if explicit is _MISSING:
        explicit = decoded.get("timestamps", _MISSING)
    if explicit is not _MISSING:
        return _normalize_timestamp_items(explicit)

    token_ids = decoded.get("token_ids", _MISSING)
    token_frames = decoded.get("token_frames", _MISSING)
    if token_ids is _MISSING or token_frames is _MISSING:
        return []

    ids = _token_values(token_ids, "token_ids")
    frames = _token_values(token_frames, "token_frames")
    if len(ids) != len(frames):
        raise TypeError("token_ids and token_frames must have the same length")
    if encoded_length <= 0:
        return []

    config = getattr(model, "config", None)
    vocabulary = getattr(config, "vocabulary", None)
    if not isinstance(vocabulary, (list, tuple)):
        raise TypeError("model config does not expose a vocabulary")

    blank = len(vocabulary)
    shift = audio_samples / _SAMPLE_RATE / encoded_length
    words: list[dict[str, Any]] = []
    chars: list[str] = []
    word_frames: list[int] = []

    def commit() -> None:
        text = "".join(chars).strip()
        if text and word_frames:
            words.append(
                {
                    "text": text,
                    "start": word_frames[0] * shift,
                    "end": (word_frames[-1] + 1) * shift,
                }
            )
        chars.clear()
        word_frames.clear()

    for token, frame in zip(ids, frames, strict=True):
        if frame >= encoded_length:
            raise TypeError("token frame exceeds encoded length")
        if token == blank:
            continue
        if token >= len(vocabulary):
            raise TypeError("token id exceeds model vocabulary")
        char = vocabulary[token]
        if not isinstance(char, str):
            raise TypeError("model vocabulary contains a non-string token")
        if char == " ":
            commit()
        else:
            chars.append(char)
            word_frames.append(frame)
    commit()
    return words


def _decoded_item(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) == 1 and isinstance(values[0], Mapping):
            return values[0]
    raise TypeError("model returned an unsupported greedy-decode result")


def _monotonic_words(
    words: list[dict[str, Any]], duration: float
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    previous_end = 0.0
    for word in words:
        start = min(duration, max(previous_end, float(word["start"])))
        end = min(duration, max(start, float(word["end"])))
        normalized.append({"text": str(word["text"]), "start": start, "end": end})
        previous_end = end
    return normalized


def _fixed_chunks(audio: Any, chunk_seconds: float, overlap_seconds: float):
    chunk_size = max(1, round(chunk_seconds * _SAMPLE_RATE))
    overlap = round(overlap_seconds * _SAMPLE_RATE)
    step = chunk_size - overlap
    for start in range(0, len(audio), step):
        end = min(start + chunk_size, len(audio))
        yield start, end, audio[start:end]
        if end == len(audio):
            break


def _transcribe_audio(
    model: Any,
    audio_path: str,
    chunk_seconds: float,
    overlap_seconds: float,
    *,
    audio_loader: Callable[[str, int], Any] | None = None,
    mlx_module: Any | None = None,
) -> dict[str, Any]:
    loader = _load_audio if audio_loader is None else audio_loader
    audio = loader(audio_path, _SAMPLE_RATE)
    mx = _import_mlx() if mlx_module is None else mlx_module

    duration = len(audio) / _SAMPLE_RATE
    chunks: list[dict[str, Any]] = []
    kept_words: list[dict[str, Any]] = []
    decoded_texts: list[str] = []
    ranges = list(_fixed_chunks(audio, chunk_seconds, overlap_seconds))

    for index, (start, end, samples) in enumerate(ranges):
        audio_batch = mx.array(samples)[None, :]
        lengths = mx.array([len(samples)])
        logits, encoded_lengths = model(audio_batch, lengths)
        mx.eval(logits, encoded_lengths)
        decoded = _decoded_item(model.greedy_decode(logits, encoded_lengths))
        text = decoded.get("text")
        if not isinstance(text, str):
            raise TypeError("model returned a decoded result without string text")
        text = text.strip()
        decoded_texts.append(text)

        local_words = _derive_word_timestamps(
            decoded,
            model,
            len(samples),
            _length_value(encoded_lengths),
        )
        start_seconds = start / _SAMPLE_RATE
        end_seconds = end / _SAMPLE_RATE
        keep_start = (
            start_seconds
            if index == 0
            else start_seconds + overlap_seconds / 2
        )
        keep_end = (
            end_seconds
            if index == len(ranges) - 1
            else end_seconds - overlap_seconds / 2
        )
        global_words: list[dict[str, Any]] = []
        for word in local_words:
            item = {
                "text": word["text"],
                "start": start_seconds + float(word["start"]),
                "end": start_seconds + float(word["end"]),
            }
            midpoint = (item["start"] + item["end"]) / 2
            kept = keep_start <= midpoint < keep_end or (
                index == len(ranges) - 1 and midpoint <= keep_end
            )
            item["kept"] = kept
            global_words.append(item)
            if kept:
                kept_words.append(
                    {
                        "text": item["text"],
                        "start": item["start"],
                        "end": item["end"],
                    }
                )

        chunks.append(
            {
                "index": index,
                "start": start_seconds,
                "end": end_seconds,
                "text": text,
                "words": global_words,
            }
        )

    timestamps = _monotonic_words(kept_words, duration)
    transcript = (
        " ".join(str(word["text"]) for word in timestamps).strip()
        if timestamps
        else " ".join(decoded_texts).strip()
    )
    return {
        "transcript": transcript,
        "timestamps": timestamps,
        "chunks": chunks,
        "duration_seconds": duration,
        "word_timestamps": bool(timestamps),
    }


def _model_dtype(model: Any) -> str | None:
    config = getattr(model, "config", None)
    dtype = getattr(config, "dtype", None)
    return str(dtype) if dtype is not None else None


def execute(
    request: Any,
    *,
    model_loader: Callable[[str], Any] | None = None,
    audio_loader: Callable[[str, int], Any] | None = None,
    mlx_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute one deterministic, offline GigaAM Multilingual MLX request."""
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

    model_sample_rate = getattr(
        getattr(model, "config", None), "sample_rate", _SAMPLE_RATE
    )
    if model_sample_rate != _SAMPLE_RATE:
        return _error(
            "load_error",
            f"model sample rate {model_sample_rate} Hz is not supported; "
            f"expected {_SAMPLE_RATE} Hz",
        )

    transcribe_started = clock()
    try:
        result = _transcribe_audio(
            model,
            config["audio_path"],
            config["chunk_seconds"],
            config["overlap_seconds"],
            audio_loader=audio_loader,
            mlx_module=mlx_module,
        )
    except Exception as exc:
        return _error("transcribe_error", _exception_message(exc))
    transcribe_seconds = float(clock() - transcribe_started)

    word_timestamps = bool(result["word_timestamps"])
    effective_config = {
        "model_path": config["model_path"],
        "audio_path": config["audio_path"],
        "variant": config["variant"],
        "language": config["language"],
        "language_metadata_only": True,
        "backend": "mlx",
        "package": "gigaam-multilingual-mlx",
        "package_version": "0.2.0",
        "decoder": "greedy_ctc",
        "deterministic": True,
        "model_dtype": _model_dtype(model),
        "sample_rate": _SAMPLE_RATE,
        "audio_dtype": "float32",
        "audio_channels": 1,
        "resample_to_sample_rate": _SAMPLE_RATE,
        "audio_loader": "gigaam_multilingual_mlx.audio.load_audio",
        "chunk_seconds": config["chunk_seconds"],
        "chunk_overlap_seconds": config["overlap_seconds"],
        "offline": True,
        "local_files_only": True,
        "network_access": False,
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "word_timestamps": word_timestamps,
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    }

    return {
        "status": "ok",
        "transcript": result["transcript"],
        "timestamps": result["timestamps"],
        "chunks": result["chunks"],
        "duration_seconds": result["duration_seconds"],
        "language": config["language"],
        "language_source": "request_metadata",
        "load_seconds": load_seconds,
        "transcribe_seconds": transcribe_seconds,
        "variant": config["variant"],
        "decoder": "greedy_ctc",
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "effective_config": effective_config,
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        payload = _error("request_error", _exception_message(exc))
    else:
        # MLX or the model loader may emit diagnostics; stdout remains exactly
        # one JSON response for the subprocess runner.
        with redirect_stdout(sys.stderr):
            payload = execute(request)

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
