from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit


DEFAULT_DEVICE = "auto"
SUPPORTED_DEVICES = frozenset({"auto", "mps", "cpu"})
DEFAULT_MODE = "transcription_only"
SUPPORTED_MODES = frozenset({"transcription_only", "parsed"})
DEFAULT_SAMPLE_RATE = 24_000
ACOUSTIC_TOKENIZER_HOP_LENGTH = 3_200
TIMESTAMP_SEMANTICS = "speaker segment offsets"
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


def _validate_json_file(path: Path, description: str) -> None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")


def _validate_model_files(model_path: str) -> None:
    path = Path(model_path)
    required_files = (
        ("config.json", "config.json"),
        ("model.safetensors.index.json", "model.safetensors index"),
        ("tokenizer.json", "tokenizer.json"),
        ("processor_config.json", "processor config"),
    )
    missing = [name for name, _ in required_files if not (path / name).is_file()]

    shard_names = tuple(
        f"model-{shard:05d}-of-00008.safetensors" for shard in range(1, 9)
    )
    missing.extend(name for name in shard_names if not (path / name).is_file())
    if missing:
        raise ValueError(
            "model_path is missing required model files: " + ", ".join(missing)
        )

    # Check the small metadata files before importing Transformers.  The
    # tokenizer itself is intentionally checked by presence only: parsing a
    # 750k-line tokenizer JSON here would add work without catching a loader
    # error that Transformers can report more precisely.
    for filename, description in required_files:
        if filename == "tokenizer.json":
            continue
        _validate_json_file(path / filename, description)

    try:
        with (path / "model.safetensors.index.json").open(encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model.safetensors index is not valid JSON") from exc

    weight_map = index.get("weight_map")
    if isinstance(weight_map, Mapping):
        referenced = {value for value in weight_map.values() if isinstance(value, str)}
        missing_references = sorted(
            filename for filename in referenced if not (path / filename).is_file()
        )
        if missing_references:
            raise ValueError(
                "model.safetensors index references missing shards: "
                + ", ".join(missing_references)
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

    mode = request.get("mode", DEFAULT_MODE)
    if not isinstance(mode, str) or mode not in SUPPORTED_MODES:
        choices = ", ".join(sorted(SUPPORTED_MODES))
        raise ValueError(f"mode must be one of: {choices}")

    chunk_size = request.get("acoustic_tokenizer_chunk_size")
    if chunk_size is not None:
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValueError(
                "acoustic_tokenizer_chunk_size must be a positive integer"
            )
        if chunk_size % ACOUSTIC_TOKENIZER_HOP_LENGTH:
            raise ValueError(
                "acoustic_tokenizer_chunk_size must be a multiple of "
                f"{ACOUSTIC_TOKENIZER_HOP_LENGTH}"
            )

    return {
        "model_path": model_path,
        "audio_path": audio_path,
        "device": device,
        "mode": mode,
        "acoustic_tokenizer_chunk_size": chunk_size,
    }


def _load_model(model_path: str) -> tuple[Any, Any]:
    # Keep the Transformers import lazy so injected tests and validation do not
    # require the isolated VibeVoice environment.
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
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

    astype = getattr(audio, "astype", None)
    if callable(astype):
        try:
            audio = astype("float32", copy=False)
        except TypeError:
            audio = astype("float32")
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
            return [row[input_length:] for row in output_ids]
        raise TypeError("model generation output has an unsupported shape")


def _unwrap_single_decoded(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        first = value[0]
        if isinstance(first, (str, list, tuple)):
            return first
    return value


def _normalize_plain_transcript(value: Any) -> str:
    value = _unwrap_single_decoded(value)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping) and isinstance(value.get("text"), str):
        return value["text"].strip()
    raise TypeError("processor returned an unsupported plain transcript")


def _json_safe(value: Any, field_name: str) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = value
        if scalar is not value:
            value = scalar

    if isinstance(value, Mapping):
        return {key: _json_safe(item, field_name) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, field_name) for item in value]

    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"segment {field_name} is not JSON-safe") from exc
    return value


def _normalize_parsed(value: Any) -> tuple[str, list[dict[str, Any]]]:
    value = _unwrap_single_decoded(value)
    if isinstance(value, str):
        return value.strip(), []

    if isinstance(value, Mapping):
        raw_segments: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        raw_segments = list(value)
    else:
        raise TypeError("processor returned unsupported parsed transcription")

    segments: list[dict[str, Any]] = []
    contents: list[str] = []
    for segment in raw_segments:
        if not isinstance(segment, Mapping):
            raise TypeError("processor returned an unsupported parsed segment")
        if "Content" not in segment or not isinstance(segment["Content"], str):
            raise TypeError("parsed segment is missing a string Content field")
        normalized = _json_safe(dict(segment), "parsed segment")
        if not isinstance(normalized, dict):
            raise TypeError("processor returned an unsupported parsed segment")
        segments.append(normalized)
        contents.append(segment["Content"])

    return " ".join(contents).strip(), segments


def _inference_context(torch_module: Any) -> Any:
    no_grad = getattr(torch_module, "no_grad", None)
    return no_grad() if callable(no_grad) else nullcontext()


def execute(
    request: Any,
    *,
    model_loader: Callable[[str], Any] | None = None,
    torch_module: Any | None = None,
    soundfile_module: Any | None = None,
    resample_fn: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute one offline official VibeVoice Transformers transcription request."""
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
        )

        # This is the official VibeVoice API.  The waveform is already at the
        # processor's 24 kHz rate, so no path/URL loader can access the network.
        inputs = processor.apply_transcription_request(audio=audio)
        model_dtype = getattr(model, "dtype", None)
        inputs = _move_inputs(inputs, device, model_dtype)
        generation_kwargs = _generation_kwargs(inputs)
        generation_kwargs["do_sample"] = False
        chunk_size = config["acoustic_tokenizer_chunk_size"]
        if chunk_size is not None:
            generation_kwargs["acoustic_tokenizer_chunk_size"] = chunk_size

        with _inference_context(torch_runtime):
            output_ids = model.generate(**generation_kwargs)
        generated_ids = _generated_ids(
            output_ids,
            _input_sequence_length(inputs),
        )
        decoded = processor.decode(
            generated_ids,
            return_format=config["mode"],
        )
        if config["mode"] == "parsed":
            transcript, segments = _normalize_parsed(decoded)
        else:
            transcript = _normalize_plain_transcript(decoded)
            segments = []
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
        "mode": config["mode"],
        "sample_rate": sample_rate,
        "resampled": resampled,
        "acoustic_tokenizer_chunk_size": config["acoustic_tokenizer_chunk_size"],
        "backend": "transformers",
        "model_class": model_class,
        "processor_class": processor_class,
        "deterministic": True,
        "do_sample": False,
        "offline": True,
        "local_files_only": True,
        "network_access": False,
        "pipeline": False,
        "no_pipeline": True,
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
        "mode": config["mode"],
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
