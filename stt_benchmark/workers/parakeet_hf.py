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


DEFAULT_DEVICE = "auto"
SUPPORTED_DEVICES = frozenset({"auto", "mps", "cpu"})
DEFAULT_SAMPLE_RATE = 16_000
TIMESTAMP_SEMANTICS = "token timestamps"
_MISSING = object()


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


def _has_any_file(
    path: Path, names: tuple[str, ...], patterns: tuple[str, ...] = ()
) -> bool:
    if any((path / name).is_file() for name in names):
        return True
    return any(any(path.glob(pattern)) for pattern in patterns)


def _validate_model_files(model_path: str) -> None:
    path = Path(model_path)
    missing: list[str] = []

    if not (path / "config.json").is_file():
        missing.append("config.json")

    # Accept the standard single-file and sharded PyTorch/safetensors layouts.
    # The model loader remains local-only; this check only catches incomplete
    # snapshots before importing Transformers.
    has_weights = _has_any_file(
        path,
        (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model.bin",
        ),
        (
            "*.safetensors.index.json",
            "*.bin.index.json",
            "model-*.safetensors",
            "pytorch_model-*.safetensors",
            "pytorch_model-*.bin",
        ),
    )
    if not has_weights:
        missing.append(
            "model weights (model.safetensors or a safetensors index)"
        )

    # Parakeet's official snapshot currently contains tokenizer.json. Keep the
    # accepted set broad enough for equivalent local tokenizer exports without
    # treating tokenizer_config.json alone as tokenizer content.
    has_tokenizer = _has_any_file(
        path,
        (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
        ),
    ) or ((path / "vocab.json").is_file() and (path / "merges.txt").is_file())
    if not has_tokenizer:
        missing.append("tokenizer files (tokenizer.json or tokenizer.model)")

    # Transformers 5.9 uses processor_config.json for the official Parakeet
    # snapshot. Accept the older feature-extractor names as well so a local
    # save from an equivalent Transformers version remains usable.
    has_processor_config = _has_any_file(
        path,
        (
            "processor_config.json",
            "preprocessor_config.json",
            "feature_extractor_config.json",
        ),
    )
    if not has_processor_config:
        missing.append("processor config (processor_config.json)")

    if missing:
        raise ValueError(
            f"model_path is missing required model files: {', '.join(missing)}"
        )


def _validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    model_path = _validate_local_path(request, "model_path", directory=True)
    audio_path = _validate_local_path(request, "audio_path", directory=False)
    _validate_model_files(model_path)

    device = request.get("device", DEFAULT_DEVICE)
    if not isinstance(device, str) or device not in SUPPORTED_DEVICES:
        choices = ", ".join(sorted(SUPPORTED_DEVICES))
        raise ValueError(f"device must be one of: {choices}")

    return {
        "model_path": model_path,
        "audio_path": audio_path,
        "device": device,
    }


def _load_model(model_path: str) -> tuple[Any, Any]:
    # Keep the foreign-runtime import lazy so validation and injected tests do
    # not require Transformers. Both loaders receive the exact local path.
    from transformers import AutoModelForTDT, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForTDT.from_pretrained(
        model_path,
        dtype="auto",
        local_files_only=True,
    )
    return processor, model


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _unpack_loaded(value: Any) -> tuple[Any, Any]:
    if isinstance(value, Mapping):
        processor = value.get("processor", _MISSING)
        model = value.get("model", _MISSING)
        if processor is not _MISSING and model is not _MISSING:
            return processor, model

    processor = getattr(value, "processor", _MISSING)
    model = getattr(value, "model", _MISSING)
    if processor is not _MISSING and model is not _MISSING:
        return processor, model

    if isinstance(value, (tuple, list)) and len(value) == 2:
        first, second = value
        # The official loader returns (processor, model), but accepting the
        # inverse here makes the execution seam convenient for small fakes.
        if callable(second) and callable(getattr(first, "generate", None)):
            return second, first
        if callable(getattr(second, "generate", None)) and not callable(first):
            return first, second
        return first, second

    raise TypeError("model loader must return (processor, model)")


def _import_torch() -> Any:
    import torch

    return torch


def _mps_is_available(torch_module: Any) -> bool:
    try:
        return bool(torch_module.backends.mps.is_available())
    except (AttributeError, RuntimeError):
        return False


def _select_device(requested: str, torch_module: Any) -> str:
    mps_available = _mps_is_available(torch_module)
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("requested device 'mps' is unavailable")
        return "mps"
    return "mps" if mps_available else "cpu"


def _processor_sample_rate(processor: Any) -> int:
    feature_extractor = _field(processor, "feature_extractor", None)
    sample_rate = _field(feature_extractor, "sampling_rate", _MISSING)
    if sample_rate is _MISSING:
        sample_rate = _field(processor, "sampling_rate", DEFAULT_SAMPLE_RATE)

    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
        raise ValueError("processor sampling rate must be a positive number")
    sample_rate = float(sample_rate)
    if (
        not math.isfinite(sample_rate)
        or sample_rate <= 0
        or not sample_rate.is_integer()
    ):
        raise ValueError("processor sampling rate must be a positive integer")
    return int(sample_rate)


def _read_audio(
    audio_path: str,
    target_sample_rate: int,
    *,
    soundfile_module: Any | None = None,
    resample_fn: Callable[..., Any] | None = None,
) -> tuple[Any, int, bool]:
    if soundfile_module is None:
        import soundfile as soundfile_module

    audio, source_sample_rate = soundfile_module.read(
        audio_path,
        dtype="float32",
        always_2d=False,
    )
    if isinstance(source_sample_rate, bool) or not isinstance(source_sample_rate, int):
        raise ValueError("audio sample rate must be a positive integer")
    if source_sample_rate <= 0:
        raise ValueError("audio sample rate must be a positive integer")

    ndim = getattr(audio, "ndim", None)
    if ndim is None and isinstance(audio, (list, tuple)) and audio:
        ndim = 2 if isinstance(audio[0], (list, tuple)) else 1
    if ndim is not None and ndim > 1:
        try:
            audio = audio.mean(axis=1)
        except AttributeError:
            audio = [sum(frame) / len(frame) for frame in audio]

    resampled = source_sample_rate != target_sample_rate
    if resampled:
        if resample_fn is None:
            from scipy.signal import resample_poly

            resample_fn = resample_poly
        divisor = math.gcd(source_sample_rate, target_sample_rate)
        audio = resample_fn(
            audio,
            target_sample_rate // divisor,
            source_sample_rate // divisor,
        )

    # soundfile already returns float32. scipy may return another floating
    # dtype, so restore the worker's explicit float32 audio contract when the
    # array implementation exposes astype(). Simple injected list fakes remain
    # valid and are passed through unchanged.
    astype = getattr(audio, "astype", None)
    if callable(astype):
        try:
            audio = astype("float32", copy=False)
        except TypeError:
            audio = astype("float32")

    return audio, target_sample_rate, resampled


def _call_processor(processor: Any, audio: Any, sample_rate: int) -> Any:
    # ParakeetProcessor normalizes a single waveform into a one-item batch.
    return processor(audio, sampling_rate=sample_rate, return_tensors="pt")


def _move_value(value: Any, device: str, dtype: Any) -> Any:
    to = getattr(value, "to", None)
    if callable(to):
        if dtype is not None:
            try:
                moved = to(device, dtype=dtype)
            except TypeError:
                moved = to(device)
        else:
            moved = to(device)
        return value if moved is None else moved

    if isinstance(value, Mapping):
        return {key: _move_value(item, device, dtype) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_value(item, device, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value(item, device, dtype) for item in value)
    return value


def _move_inputs(inputs: Any, device: str, dtype: Any) -> Any:
    return _move_value(inputs, device, dtype)


def _generation_kwargs(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, Mapping):
        return dict(inputs)

    items = getattr(inputs, "items", None)
    if callable(items):
        return dict(items())

    try:
        return dict(vars(inputs))
    except TypeError as exc:
        raise TypeError(
            "processor returned inputs that cannot be passed to generate"
        ) from exc


def _normalize_transcript(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, Mapping):
        text = value.get("text", _MISSING)
        if text is not _MISSING:
            return _normalize_transcript(text)

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        values = list(value)
        if len(values) == 1:
            return _normalize_transcript(values[0])

    raise TypeError("processor returned an unsupported transcript value")


def _json_scalar(value: Any, field_name: str) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            pass

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    raise TypeError(f"segment {field_name} is not JSON-safe")


def _normalize_token_segments(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        items: list[Any] = [value]
    elif isinstance(value, (str, bytes)):
        raise TypeError("processor returned unsupported token timestamps")
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        items = [value]

    # decode() returns one list per input waveform. Unwrap the one-item batch,
    # while also accepting a direct list of token dictionaries.
    if items and _field(items[0], "token", _MISSING) is _MISSING:
        if len(items) == 1 and isinstance(items[0], Iterable) and not isinstance(
            items[0], (str, bytes, Mapping)
        ):
            items = list(items[0])
        else:
            raise TypeError("processor returned unsupported token timestamps")

    normalized: list[dict[str, Any]] = []
    for segment in items:
        token = _field(segment, "token", _MISSING)
        start = _field(segment, "start", _MISSING)
        end = _field(segment, "end", _MISSING)
        if token is _MISSING or start is _MISSING or end is _MISSING:
            raise TypeError("processor returned an incomplete token timestamp")

        token = _json_scalar(token, "token")
        if not isinstance(token, str):
            raise TypeError("token timestamp token must be a string")
        start = _json_scalar(start, "start")
        end = _json_scalar(end, "end")
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise TypeError("token timestamp start must be numeric")
        if isinstance(end, bool) or not isinstance(end, (int, float)):
            raise TypeError("token timestamp end must be numeric")
        start_float = float(start)
        end_float = float(end)
        if not math.isfinite(start_float) or not math.isfinite(end_float):
            raise TypeError("token timestamp offsets must be finite")
        normalized.append({"token": token, "start": start_float, "end": end_float})

    return normalized


def _decode(
    processor: Any, sequences: Any, durations: Any
) -> tuple[str, list[dict[str, Any]]]:
    decoded: Any
    if durations is not _MISSING and durations is not None:
        try:
            decoded = processor.decode(
                sequences,
                durations=durations,
                skip_special_tokens=True,
            )
        except (AttributeError, TypeError, ValueError):
            decoded = processor.decode(sequences, skip_special_tokens=True)
    else:
        decoded = processor.decode(sequences, skip_special_tokens=True)

    if isinstance(decoded, tuple) and len(decoded) == 2:
        transcript_value, timestamp_value = decoded
        return _normalize_transcript(transcript_value), _normalize_token_segments(
            timestamp_value
        )
    return _normalize_transcript(decoded), []


def execute(
    request: Any,
    *,
    model_loader: Callable[[str], Any] | None = None,
    torch_module: Any | None = None,
    soundfile_module: Any | None = None,
    resample_fn: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute one offline official Parakeet Transformers transcription request."""
    try:
        config = _validate_request(request)
    except Exception as exc:
        return _error("validation_error", _exception_message(exc))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    loader = _load_model if model_loader is None else model_loader
    load_started = clock()
    try:
        processor, model = _unpack_loaded(loader(config["model_path"]))
        torch_runtime = _import_torch() if torch_module is None else torch_module
        device = _select_device(config["device"], torch_runtime)
        model.to(device)
        model.eval()
    except Exception as exc:
        return _error("load_error", _exception_message(exc))
    load_seconds = float(clock() - load_started)

    transcribe_started = clock()
    try:
        sample_rate = _processor_sample_rate(processor)
        audio, sample_rate, resampled = _read_audio(
            config["audio_path"],
            sample_rate,
            soundfile_module=soundfile_module,
            resample_fn=resample_fn,
        )
        inputs = _call_processor(processor, audio, sample_rate)
        model_dtype = getattr(model, "dtype", None)
        model_device = getattr(model, "device", None) or device
        inputs = _move_inputs(inputs, model_device, model_dtype)
        generation_kwargs = _generation_kwargs(inputs)
        generation_kwargs["return_dict_in_generate"] = True
        output = model.generate(**generation_kwargs)

        sequences = _field(output, "sequences", _MISSING)
        if sequences is _MISSING:
            raise TypeError("model generation output has no sequences")
        durations = _field(output, "durations", _MISSING)
        transcript, segments = _decode(processor, sequences, durations)
    except Exception as exc:
        return _error("transcribe_error", _exception_message(exc))
    transcribe_seconds = float(clock() - transcribe_started)

    model_class = type(model).__name__
    processor_class = type(processor).__name__
    effective_config = {
        "model_path": config["model_path"],
        "audio_path": config["audio_path"],
        "device_requested": config["device"],
        "device": device,
        "sample_rate": sample_rate,
        "resampled": resampled,
        "backend": "transformers",
        "model_class": model_class,
        "processor_class": processor_class,
        "offline": True,
        "local_files_only": True,
        "network_access": False,
        "torchcodec": False,
        "no_torchcodec": True,
        "audio_backend": "soundfile",
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
        "load_seconds": load_seconds,
        "transcribe_seconds": transcribe_seconds,
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "model_class": model_class,
        "processor_class": processor_class,
        "device": device,
        "effective_config": effective_config,
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        payload = _error("request_error", _exception_message(exc))
    else:
        # Keep stdout as one JSON response while routing library diagnostics to
        # stderr for the subprocess runner.
        with redirect_stdout(sys.stderr):
            payload = execute(request)

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
