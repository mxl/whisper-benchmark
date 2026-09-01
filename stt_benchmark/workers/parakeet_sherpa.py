from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Iterable, Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit


SAMPLE_RATE = 16_000
DEFAULT_THREADS = 4
MAX_ACTIVE_PATHS = 4
DECODING_METHOD = "greedy_search"
MODEL_TYPE = "nemo_transducer"
TIMESTAMP_SEMANTICS = "token/frame starts"
UNSUPPORTED_TIMESTAMP_SEMANTICS = "unsupported"
SUPPORTED_QUANTIZATIONS = frozenset({"fp32", "int8"})

_MODEL_ARTIFACTS: dict[str, dict[str, str]] = {
    "fp32": {
        "encoder": "encoder.onnx",
        "encoder_weights": "encoder.weights",
        "decoder": "decoder.onnx",
        "joiner": "joiner.onnx",
        "tokens": "tokens.txt",
        "bpe_vocab": "bpe.vocab",
    },
    "int8": {
        "encoder": "encoder.int8.onnx",
        "decoder": "decoder.int8.onnx",
        "joiner": "joiner.int8.onnx",
        "tokens": "tokens.txt",
        "bpe_vocab": "bpe.vocab",
    },
}


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
    return value


def _model_files(model_path: str, quantization: str) -> dict[str, str]:
    model_dir = Path(model_path)
    artifacts = _MODEL_ARTIFACTS[quantization]
    missing = [
        filename
        for filename in artifacts.values()
        if not (model_dir / filename).is_file()
    ]
    if missing:
        raise ValueError(
            f"model_path is missing required {quantization} Parakeet Sherpa "
            f"artifacts: {', '.join(missing)}"
        )

    return {
        name: str(model_dir / filename) for name, filename in artifacts.items()
    }


def _validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    model_path = _validate_local_path(request, "model_path", directory=True)
    audio_path = _validate_local_path(request, "audio_path", directory=False)

    quantization = request.get("quantization", "fp32")
    if not isinstance(quantization, str) or quantization not in SUPPORTED_QUANTIZATIONS:
        if quantization == "fp16":
            raise ValueError(
                "quantization fp16 is unsupported: the Parakeet FP16 export is "
                "invalid; use fp32 or int8"
            )
        raise ValueError("quantization must be one of: fp32, int8")

    threads = request.get("threads", DEFAULT_THREADS)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    files = _model_files(model_path, quantization)
    return {
        "model_path": model_path,
        "audio_path": audio_path,
        "quantization": quantization,
        "threads": threads,
        "files": files,
    }


def _import_sherpa_onnx() -> Any:
    # Keep the foreign runtime import after local request and artifact
    # validation so protocol tests do not need sherpa-onnx installed.
    import sherpa_onnx

    return sherpa_onnx


def _import_soundfile() -> Any:
    import soundfile

    return soundfile


def _import_resample_poly() -> Callable[..., Any]:
    from scipy.signal import resample_poly

    return resample_poly


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
    if callable(astype):
        try:
            return astype("float32", copy=False)
        except TypeError:
            return astype("float32")

    try:
        import numpy as np
    except ImportError:
        return audio
    return np.asarray(audio, dtype=np.float32)


def _resample_audio(
    audio: Any,
    source_rate: Any,
    *,
    resample_fn: Callable[..., Any] | None = None,
) -> tuple[Any, bool]:
    if isinstance(source_rate, bool):
        raise ValueError(f"unsupported sample rate {source_rate!r}; expected {SAMPLE_RATE} Hz")

    try:
        integer_rate = int(source_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unsupported sample rate {source_rate!r}; expected {SAMPLE_RATE} Hz"
        ) from exc
    if integer_rate <= 0 or integer_rate != source_rate:
        raise ValueError(
            f"unsupported sample rate {source_rate!r}; expected {SAMPLE_RATE} Hz"
        )

    if integer_rate == SAMPLE_RATE:
        return _as_float32(audio), False

    if resample_fn is None:
        resample_fn = _import_resample_poly()

    divisor = math.gcd(SAMPLE_RATE, integer_rate)
    audio = resample_fn(
        audio,
        SAMPLE_RATE // divisor,
        integer_rate // divisor,
    )
    return _as_float32(audio), True


def _field(value: Any, name: str) -> tuple[bool, Any]:
    if isinstance(value, Mapping):
        if name in value:
            return True, value[name]
        return False, None

    try:
        return True, getattr(value, name)
    except AttributeError:
        return False, None


def _items(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"recognizer returned unsupported {field_name}")
    if isinstance(value, Iterable):
        return list(value)
    raise TypeError(f"recognizer returned unsupported {field_name}")


def _timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("recognizer returned non-numeric timestamps")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("recognizer returned non-numeric timestamps") from exc
    if not math.isfinite(normalized):
        raise TypeError("recognizer returned non-finite timestamps")
    return normalized


def _normalize_result(result: Any) -> tuple[str, list[float], list[dict[str, Any]], str]:
    text_present, text = _field(result, "text")
    if not text_present or not isinstance(text, str):
        raise TypeError("recognizer returned an unsupported transcript value")

    tokens_present, raw_tokens = _field(result, "tokens")
    timestamps_present, raw_timestamps = _field(result, "timestamps")
    if (
        not tokens_present
        or not timestamps_present
        or raw_tokens is None
        or raw_timestamps is None
    ):
        return text.strip(), [], [], UNSUPPORTED_TIMESTAMP_SEMANTICS

    tokens = _items(raw_tokens, "tokens")
    timestamps = _items(raw_timestamps, "timestamps")
    if len(tokens) != len(timestamps):
        raise TypeError("recognizer returned different token and timestamp counts")

    normalized_timestamps: list[float] = []
    segments: list[dict[str, Any]] = []
    for token, raw_timestamp in zip(tokens, timestamps):
        if not isinstance(token, str):
            raise TypeError("recognizer returned non-string tokens")
        timestamp = _timestamp(raw_timestamp)
        normalized_timestamps.append(timestamp)
        segments.append({"text": token, "start": timestamp})

    return text.strip(), normalized_timestamps, segments, TIMESTAMP_SEMANTICS


def _effective_config(
    config: dict[str, Any], *, resampled: bool, timestamp_semantics: str
) -> dict[str, Any]:
    return {
        "backend": "sherpa-onnx",
        "model_type": MODEL_TYPE,
        "model_path": config["model_path"],
        "audio_path": config["audio_path"],
        "files": dict(config["files"]),
        "quantization": config["quantization"],
        "provider": "cpu",
        "device": "cpu",
        "threads": config["threads"],
        "num_threads": config["threads"],
        "sample_rate": SAMPLE_RATE,
        "decoding_method": DECODING_METHOD,
        "max_active_paths": MAX_ACTIVE_PATHS,
        "offline": True,
        "local_files_only": True,
        "network_access": False,
        "input_format": "float32_mono",
        "resampled_to_sample_rate": resampled,
        "timestamp_semantics": timestamp_semantics,
    }


def execute(
    request: Any,
    *,
    sherpa_module: Any | None = None,
    soundfile_module: Any | None = None,
    resample_fn: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute one local, CPU-only Parakeet Sherpa-ONNX request."""
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
            bpe_vocab=config["files"]["bpe_vocab"],
            provider="cpu",
            num_threads=config["threads"],
            sample_rate=SAMPLE_RATE,
            decoding_method=DECODING_METHOD,
            max_active_paths=MAX_ACTIVE_PATHS,
            model_type=MODEL_TYPE,
        )
    except Exception as exc:
        return _error("load_error", _exception_message(exc))
    load_seconds = float(clock() - load_started)

    transcribe_started = clock()
    try:
        soundfile = (
            _import_soundfile() if soundfile_module is None else soundfile_module
        )
        audio, source_rate = soundfile.read(
            config["audio_path"], dtype="float32", always_2d=False
        )
        audio = _as_float32(_as_mono(audio))
        audio, resampled = _resample_audio(
            audio,
            source_rate,
            resample_fn=resample_fn,
        )

        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        recognizer.decode_stream(stream)
        result = stream.result
        transcript, timestamps, segments, timestamp_semantics = _normalize_result(
            result
        )
    except Exception as exc:
        return _error("transcribe_error", _exception_message(exc))
    transcribe_seconds = float(clock() - transcribe_started)

    return {
        "status": "ok",
        "transcript": transcript,
        "segments": segments,
        "timestamps": timestamps,
        "timestamp_semantics": timestamp_semantics,
        "load_seconds": load_seconds,
        "transcribe_seconds": transcribe_seconds,
        "files": dict(config["files"]),
        "quantization": config["quantization"],
        "threads": config["threads"],
        "provider": "cpu",
        "device": "cpu",
        "effective_config": _effective_config(
            config,
            resampled=resampled,
            timestamp_semantics=timestamp_semantics,
        ),
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        payload = _error("request_error", _exception_message(exc))
    else:
        # Foreign runtimes may print diagnostics; stdout remains one JSON object.
        with redirect_stdout(sys.stderr):
            payload = execute(request)

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
