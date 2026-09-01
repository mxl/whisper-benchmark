from __future__ import annotations

import json
import math
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


_SAMPLE_RATE = 16_000
_DEFAULT_CHUNK_SECONDS = 20.0
_MAX_CHUNK_SECONDS = 29.0
_DEFAULT_DECODING_METHOD = "modified_beam_search"
_DECODING_METHODS = {"greedy_search", "modified_beam_search"}
_QUANTIZATIONS = {"fp32", "int8"}

# Match the layouts' official local decode defaults: the full model lets the
# runtime choose, while the small model uses four worker threads.
_DEFAULT_NUM_THREADS = {"big": 0, "small": 4}


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


def _model_files(model_path: str, quantization: str) -> tuple[str, dict[str, str]]:
    model_dir = Path(model_path)
    suffix = ".int8.onnx" if quantization == "int8" else ".onnx"
    filenames = {
        "encoder": f"encoder{suffix}",
        "decoder": f"decoder{suffix}",
        "joiner": f"joiner{suffix}",
    }

    # Prefer the explicit full-model layout when both layouts happen to be
    # present.  No filename fallback is allowed between fp32 and int8 files.
    for layout, directory_name in (("big", "am-onnx"), ("small", "am")):
        model_subdir = model_dir / directory_name
        paths = {
            name: str(model_subdir / filename)
            for name, filename in filenames.items()
        }
        if all(Path(path).is_file() for path in paths.values()):
            return layout, paths

    expected = ", ".join(
        f"{directory_name}/{filename}"
        for directory_name in ("am-onnx", "am")
        for filename in filenames.values()
    )
    raise ValueError(
        f"model_path does not contain the exact {quantization} transducer files "
        f"({expected})"
    )


def _validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    model_path = _validate_local_path(request, "model_path", directory=True)
    audio_path = _validate_local_path(request, "audio_path", directory=False)

    decoding_method = request.get("decoding_method", _DEFAULT_DECODING_METHOD)
    if not isinstance(decoding_method, str) or decoding_method not in _DECODING_METHODS:
        raise ValueError(
            "decoding_method must be one of: greedy_search, modified_beam_search"
        )

    quantization = request.get("quantization", "fp32")
    if not isinstance(quantization, str) or quantization not in _QUANTIZATIONS:
        raise ValueError("quantization must be one of: fp32, int8")

    chunk_seconds = request.get("chunk_seconds", _DEFAULT_CHUNK_SECONDS)
    if isinstance(chunk_seconds, bool) or not isinstance(chunk_seconds, (int, float)):
        raise ValueError(
            f"chunk_seconds must be a positive number no greater than {_MAX_CHUNK_SECONDS:g}"
        )
    chunk_seconds = float(chunk_seconds)
    if not math.isfinite(chunk_seconds) or not (
        0 < chunk_seconds <= _MAX_CHUNK_SECONDS
    ):
        raise ValueError(
            f"chunk_seconds must be greater than 0 and no greater than "
            f"{_MAX_CHUNK_SECONDS:g} seconds"
        )

    layout, files = _model_files(model_path, quantization)
    tokens = Path(model_path) / "lang" / "tokens.txt"
    if not tokens.is_file():
        raise ValueError("model_path must contain lang/tokens.txt")
    files["tokens"] = str(tokens)

    if "num_threads" in request:
        num_threads = request["num_threads"]
        if isinstance(num_threads, bool) or not isinstance(num_threads, int):
            raise ValueError("num_threads must be a positive integer")
        if num_threads <= 0:
            raise ValueError("num_threads must be a positive integer")
    else:
        num_threads = _DEFAULT_NUM_THREADS[layout]

    return {
        "model_path": model_path,
        "audio_path": audio_path,
        "layout": layout,
        "files": files,
        "decoding_method": decoding_method,
        "quantization": quantization,
        "num_threads": num_threads,
        "chunk_seconds": chunk_seconds,
    }


def _import_sherpa_onnx() -> Any:
    # Keep the foreign runtime import after all local request/model validation.
    import sherpa_onnx

    return sherpa_onnx


def _import_soundfile() -> Any:
    # soundfile is also intentionally lazy so validation tests need no audio
    # runtime or native dependency.
    import soundfile

    return soundfile


def _as_mono(audio: Any) -> Any:
    ndim = getattr(audio, "ndim", None)
    if ndim is None and isinstance(audio, (list, tuple)) and audio:
        ndim = 2 if isinstance(audio[0], (list, tuple)) else 1

    if ndim is not None and ndim > 1:
        try:
            audio = audio.mean(axis=1)
        except AttributeError:
            audio = [sum(frame) / len(frame) for frame in audio]
    return audio


def _as_float32(audio: Any) -> Any:
    astype = getattr(audio, "astype", None)
    if astype is not None:
        try:
            return astype("float32", copy=False)
        except TypeError:
            return astype("float32")

    # soundfile normally already returns a float32 numpy array.  This fallback
    # keeps injected/simple readers faithful to the worker contract as well.
    try:
        import numpy as np
    except ImportError:
        return audio
    return np.asarray(audio, dtype=np.float32)


def _normalize_timestamps(result: Any) -> list[float]:
    raw_timestamps = getattr(result, "timestamps", None)
    if raw_timestamps is None:
        return []

    if isinstance(raw_timestamps, (str, bytes)):
        raise TypeError("recognizer returned unsupported timestamps")
    try:
        values = list(raw_timestamps)
    except TypeError as exc:
        raise TypeError("recognizer returned unsupported timestamps") from exc

    timestamps: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("recognizer returned non-numeric timestamps")
        try:
            timestamp = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("recognizer returned non-numeric timestamps") from exc
        if not math.isfinite(timestamp):
            raise TypeError("recognizer returned non-finite timestamps")
        timestamps.append(timestamp)
    return timestamps


def _effective_config(config: dict[str, Any]) -> dict[str, Any]:
    files = dict(config["files"])
    return {
        "model_path": config["model_path"],
        "audio_path": config["audio_path"],
        "layout": config["layout"],
        "files": files,
        "encoder": files["encoder"],
        "decoder": files["decoder"],
        "joiner": files["joiner"],
        "tokens": files["tokens"],
        "provider": "cpu",
        "sample_rate": _SAMPLE_RATE,
        "num_threads": config["num_threads"],
        "decoding_method": config["decoding_method"],
        "quantization": config["quantization"],
        "chunk_seconds": config["chunk_seconds"],
        "chunk_overlap_seconds": 0.0,
        "chunked": True,
        "offline": True,
        "local_files_only": True,
        "network_access": False,
        "timestamp_semantics": "token/frame starts",
        "word_timestamps": False,
    }


def _decode_fixed_chunks(
    recognizer: Any, audio: Any, chunk_seconds: float
) -> tuple[str, list[float]]:
    chunk_frames = max(1, int(round(chunk_seconds * _SAMPLE_RATE)))
    transcripts: list[str] = []
    timestamps: list[float] = []

    for start in range(0, len(audio), chunk_frames):
        chunk = audio[start : start + chunk_frames]
        if len(chunk) == 0:
            continue

        stream = recognizer.create_stream()
        stream.accept_waveform(_SAMPLE_RATE, chunk)
        recognizer.decode_stream(stream)
        result = stream.result

        transcript = getattr(result, "text", None)
        if not isinstance(transcript, str):
            raise TypeError("recognizer returned an unsupported transcript value")
        transcript = transcript.strip()
        if transcript:
            transcripts.append(transcript)

        offset = start / _SAMPLE_RATE
        timestamps.extend(
            timestamp + offset for timestamp in _normalize_timestamps(result)
        )

    return " ".join(transcripts), timestamps


def execute(
    request: Any,
    *,
    sherpa_module: Any | None = None,
    soundfile_module: Any | None = None,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    """Execute one local, CPU-only sherpa-onnx Vosk-layout request."""
    try:
        config = _validate_request(request)
    except Exception as exc:
        return _error("validation_error", _exception_message(exc))

    load_started = clock()
    try:
        sherpa = _import_sherpa_onnx() if sherpa_module is None else sherpa_module
        recognizer = sherpa.OfflineRecognizer.from_transducer(
            encoder=config["files"]["encoder"],
            decoder=config["files"]["decoder"],
            joiner=config["files"]["joiner"],
            tokens=config["files"]["tokens"],
            num_threads=config["num_threads"],
            provider="cpu",
            sample_rate=_SAMPLE_RATE,
            decoding_method=config["decoding_method"],
        )
    except Exception as exc:
        return _error("load_error", _exception_message(exc))
    load_seconds = float(clock() - load_started)

    transcribe_started = clock()
    try:
        soundfile = (
            _import_soundfile() if soundfile_module is None else soundfile_module
        )
        audio, sample_rate = soundfile.read(
            config["audio_path"], dtype="float32", always_2d=False
        )
        if sample_rate != _SAMPLE_RATE:
            raise ValueError(
                f"unsupported sample rate {sample_rate} Hz; expected {_SAMPLE_RATE} Hz"
            )

        audio = _as_float32(_as_mono(audio))
        transcript, timestamps = _decode_fixed_chunks(
            recognizer, audio, config["chunk_seconds"]
        )
    except Exception as exc:
        return _error("transcribe_error", _exception_message(exc))
    transcribe_seconds = float(clock() - transcribe_started)

    return {
        "status": "ok",
        "transcript": transcript.strip(),
        "timestamps": timestamps,
        "load_seconds": load_seconds,
        "transcribe_seconds": transcribe_seconds,
        "decoding_method": config["decoding_method"],
        "quantization": config["quantization"],
        "chunk_seconds": config["chunk_seconds"],
        "chunk_overlap_seconds": 0.0,
        "chunked": True,
        "timestamp_semantics": "token/frame starts",
        "effective_config": _effective_config(config),
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        payload = _error("request_error", _exception_message(exc))
    else:
        # Foreign runtimes may print diagnostics; stdout remains one JSON
        # object while those diagnostics are routed to stderr.
        with redirect_stdout(sys.stderr):
            payload = execute(request)

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
