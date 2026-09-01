from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DEFAULT_DEVICE = "auto"
SUPPORTED_DEVICES = frozenset({"auto", "mps", "cpu"})
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_CHUNK_SECONDS = 30.0
MAX_CHUNK_SECONDS = 300.0
TIMESTAMP_SEMANTICS = "chunk/segment offsets"
TIMESTAMP_REASON = "Qwen3-ForcedAligner is not run by this worker"
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

    if directory:
        try:
            nonempty = next(path.iterdir(), None) is not None
        except (OSError, ValueError) as exc:
            raise ValueError(f"{field} is not a readable local directory") from exc
        if not nonempty:
            raise ValueError(f"{field} must point to a non-empty local directory")

    return value


def _has_any_file(
    path: Path, names: tuple[str, ...], patterns: tuple[str, ...] = ()
) -> bool:
    if any((path / name).is_file() for name in names):
        return True
    return any(any(path.glob(pattern)) for pattern in patterns)


def _validate_json_file(path: Path, description: str) -> None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")


def _validate_model_files(model_path: str) -> None:
    """Reject incomplete snapshots before importing the Transformers runtime."""
    path = Path(model_path)
    missing: list[str] = []

    required_json = (
        ("config.json", "config.json"),
        ("processor_config.json", "processor config"),
    )
    for filename, description in required_json:
        if not (path / filename).is_file():
            missing.append(filename)

    has_weights = _has_any_file(
        path,
        (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        ),
        ("*.safetensors", "*.safetensors.index.json", "*.bin"),
    )
    if not has_weights:
        missing.append("model weights (model.safetensors or a safetensors index)")

    has_tokenizer = _has_any_file(
        path,
        ("tokenizer.json", "tokenizer.model", "spiece.model"),
        ("vocab.json",),
    )
    if not has_tokenizer:
        missing.append("tokenizer files (tokenizer.json or tokenizer.model)")

    if missing:
        raise ValueError(
            f"model_path is missing required model files: {', '.join(missing)}"
        )

    for filename, description in required_json:
        _validate_json_file(path / filename, description)


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

    language = request.get("language")
    if language is not None and (not isinstance(language, str) or not language):
        raise ValueError("language must be a non-empty string when provided")
    if isinstance(language, str) and language.casefold() == "auto":
        language = None

    max_new_tokens = request.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")

    chunk_seconds = request.get("chunk_seconds", DEFAULT_CHUNK_SECONDS)
    if isinstance(chunk_seconds, bool) or not isinstance(
        chunk_seconds, (int, float)
    ):
        raise ValueError(
            f"chunk_seconds must be greater than 0 and no greater than "
            f"{MAX_CHUNK_SECONDS:g} seconds"
        )
    chunk_seconds = float(chunk_seconds)
    if not math.isfinite(chunk_seconds) or not (
        0 < chunk_seconds <= MAX_CHUNK_SECONDS
    ):
        raise ValueError(
            f"chunk_seconds must be greater than 0 and no greater than "
            f"{MAX_CHUNK_SECONDS:g} seconds"
        )

    return {
        "model_path": model_path,
        "audio_path": audio_path,
        "device": device,
        "language": language,
        "max_new_tokens": max_new_tokens,
        "chunk_seconds": chunk_seconds,
    }


def _load_model(model_path: str) -> tuple[Any, Any]:
    # Keep Transformers lazy so validation and injected tests do not require the
    # isolated Qwen environment. The path is already a validated local snapshot.
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
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
        return value[0], value[1]

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
    numpy_module: Any | None = None,
) -> tuple[Any, int, bool]:
    if soundfile_module is None:
        import soundfile as soundfile_module

    audio, source_sample_rate = soundfile_module.read(
        audio_path,
        dtype="float32",
        always_2d=False,
    )
    if isinstance(source_sample_rate, bool) or not isinstance(
        source_sample_rate, (int, float)
    ):
        raise ValueError("audio sample rate must be a positive number")
    source_sample_rate = float(source_sample_rate)
    if not math.isfinite(source_sample_rate) or not source_sample_rate.is_integer():
        raise ValueError("audio sample rate must be a positive integer")
    source_sample_rate = int(source_sample_rate)
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

    # The processor must receive a real mono float32 NumPy array, not a path or
    # an audio-library object. NumPy is provided by soundfile/scipy in the env.
    if numpy_module is None:
        import numpy as np
    else:
        np = numpy_module

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be a mono waveform")
    if audio.size == 0:
        raise ValueError("audio waveform is empty")

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
        audio = np.asarray(audio, dtype=np.float32)

    return audio, target_sample_rate, resampled


def _move_value(value: Any, device: str, dtype: Any) -> Any:
    to = getattr(value, "to", None)
    if callable(to):
        if dtype is not None:
            try:
                moved = to(device, dtype=dtype)
            except TypeError:
                try:
                    moved = to(device, dtype)
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


def _input_sequence_length(inputs: Any) -> int:
    input_ids = _field(inputs, "input_ids", _MISSING)
    if input_ids is _MISSING:
        raise TypeError("processor inputs have no input_ids")

    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        try:
            if len(shape) >= 2:
                length = shape[1]
                if isinstance(length, int) and length > 0:
                    return length
        except TypeError:
            pass

    if isinstance(input_ids, (list, tuple)) and input_ids:
        first = input_ids[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(input_ids)

    raise TypeError("processor inputs have an unsupported input_ids shape")


def _generated_ids(output_ids: Any, input_length: int) -> Any:
    try:
        return output_ids[:, input_length:]
    except (IndexError, KeyError, TypeError):
        if isinstance(output_ids, (list, tuple)):
            if output_ids and isinstance(output_ids[0], (list, tuple)):
                return [row[input_length:] for row in output_ids]
            return output_ids[input_length:]
        raise TypeError("model generation output has an unsupported shape")


def _unwrap_single(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _normalize_parsed(value: Any) -> tuple[str, Any]:
    value = _unwrap_single(value)
    if not isinstance(value, Mapping):
        raise TypeError("processor returned unsupported parsed transcription")

    transcription = value.get("transcription", _MISSING)
    if transcription is _MISSING:
        raise TypeError("parsed transcription is missing transcription")
    if not isinstance(transcription, str):
        raise TypeError("parsed transcription must contain string transcription")

    language = value.get("language")
    if language is not None and not isinstance(language, str):
        raise TypeError("parsed transcription language must be a string or null")
    return transcription.strip(), language


def _normalize_transcription_only(value: Any) -> str:
    value = _unwrap_single(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        transcription = value.get("transcription", _MISSING)
        if isinstance(transcription, str):
            return transcription.strip()
    raise TypeError("processor returned unsupported transcription-only output")


def _decode(processor: Any, generated_ids: Any) -> tuple[str, Any, str]:
    try:
        parsed = processor.decode(generated_ids, return_format="parsed")
        transcript, language = _normalize_parsed(parsed)
        return transcript, language, "parsed"
    except Exception as parsed_error:
        try:
            plain = processor.decode(
                generated_ids,
                return_format="transcription_only",
            )
            return _normalize_transcription_only(plain), None, "transcription_only"
        except Exception as plain_error:
            raise TypeError(
                "processor decode failed for parsed and transcription_only formats: "
                f"{_exception_message(parsed_error)}; "
                f"{_exception_message(plain_error)}"
            ) from plain_error


def _apply_transcription_request(
    processor: Any, audio: Any, language: str | None
) -> Any:
    kwargs: dict[str, Any] = {"audio": audio}
    # Omitting language is the official auto-detection path. Passing None is
    # observably different for some processor versions and is not needed.
    if language is not None:
        kwargs["language"] = language
    return processor.apply_transcription_request(**kwargs)


def _inference_context(torch_module: Any) -> Any:
    inference_mode = getattr(torch_module, "inference_mode", None)
    if callable(inference_mode):
        return inference_mode()
    no_grad = getattr(torch_module, "no_grad", None)
    return no_grad() if callable(no_grad) else nullcontext()


def _transcribe_fixed_chunks(
    processor: Any,
    model: Any,
    audio: Any,
    *,
    sample_rate: int,
    language: str | None,
    max_new_tokens: int,
    chunk_seconds: float,
    device: str,
    torch_runtime: Any,
) -> tuple[str, list[dict[str, Any]], str | None, str, int]:
    chunk_samples = max(1, int(round(chunk_seconds * sample_rate)))
    transcripts: list[str] = []
    segments: list[dict[str, Any]] = []
    detected_language: str | None = None
    decode_formats: list[str] = []
    chunk_count = 0
    model_dtype = getattr(model, "dtype", None)
    model_device = getattr(model, "device", None) or device
    sample_count = len(audio)

    with _inference_context(torch_runtime):
        for start in range(0, sample_count, chunk_samples):
            end = min(start + chunk_samples, sample_count)
            chunk = audio[start:end]
            if len(chunk) == 0:
                continue
            chunk_count += 1

            inputs = _apply_transcription_request(processor, chunk, language)
            inputs = _move_value(inputs, model_device, model_dtype)
            input_length = _input_sequence_length(inputs)
            generation_kwargs = _generation_kwargs(inputs)
            generation_kwargs["max_new_tokens"] = max_new_tokens
            generation_kwargs["do_sample"] = False

            output_ids = model.generate(**generation_kwargs)
            generated_ids = _generated_ids(output_ids, input_length)
            transcript, chunk_language, decode_format = _decode(
                processor, generated_ids
            )
            if decode_format not in decode_formats:
                decode_formats.append(decode_format)
            if detected_language is None and chunk_language is not None:
                detected_language = chunk_language
            if transcript:
                transcripts.append(transcript)

            segments.append(
                {
                    "start": start / sample_rate,
                    "end": end / sample_rate,
                    "text": transcript,
                }
            )

    if not decode_formats:
        raise ValueError("audio produced no transcription chunks")
    decode_format = decode_formats[0] if len(decode_formats) == 1 else "mixed"
    return (
        " ".join(transcripts).strip(),
        segments,
        detected_language,
        decode_format,
        chunk_count,
    )


def execute(
    request: Any,
    *,
    model_loader: Callable[[str], Any] | None = None,
    torch_module: Any | None = None,
    soundfile_module: Any | None = None,
    resample_fn: Callable[..., Any] | None = None,
    numpy_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute one offline official Qwen3-ASR Transformers transcription request."""
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
        moved_model = model.to(device)
        if moved_model is not None:
            model = moved_model
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
            numpy_module=numpy_module,
        )
        (
            transcript,
            segments,
            language,
            decode_format,
            chunk_count,
        ) = _transcribe_fixed_chunks(
            processor,
            model,
            audio,
            sample_rate=sample_rate,
            language=config["language"],
            max_new_tokens=config["max_new_tokens"],
            chunk_seconds=config["chunk_seconds"],
            device=device,
            torch_runtime=torch_runtime,
        )
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
        "language": config["language"],
        "max_new_tokens": config["max_new_tokens"],
        "chunk_seconds": config["chunk_seconds"],
        "chunk_overlap_seconds": 0.0,
        "chunk_count": chunk_count,
        "chunked": True,
        "decode_format": decode_format,
        "backend": "transformers",
        "model_class": model_class,
        "processor_class": processor_class,
        "deterministic": True,
        "do_sample": False,
        "generate_kwargs": {
            "max_new_tokens": config["max_new_tokens"],
            "do_sample": False,
        },
        "offline": True,
        "local_files_only": True,
        "network_access": False,
        "pipeline": False,
        "no_pipeline": True,
        "torchcodec": False,
        "no_torchcodec": True,
        "audio_backend": "soundfile",
        "sample_rate": sample_rate,
        "resampled": resampled,
        "forced_aligner": False,
        "timestamps_supported": False,
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "timestamp_reason": TIMESTAMP_REASON,
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
        "chunk_seconds": config["chunk_seconds"],
        "chunk_overlap_seconds": 0.0,
        "chunk_count": chunk_count,
        "chunked": True,
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "timestamps_supported": False,
        "timestamp_reason": TIMESTAMP_REASON,
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
