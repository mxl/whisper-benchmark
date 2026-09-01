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


# Podlodka's Whisper decoder has 448 target positions and uses four prompt
# positions, leaving 444 positions for generated tokens.
DEFAULT_MAX_NEW_TOKENS = 444
DEFAULT_DEVICE = "mps"
DEFAULT_CHUNK_SECONDS = 30.0
_SAMPLE_RATE = 16_000
TIMESTAMP_SEMANTICS = "chunk/segment offsets"
_SUPPORTED_DEVICES = frozenset({"cpu", "mps"})


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


def _model_weight_files(model_path: str) -> list[str]:
    path = Path(model_path)
    names: set[str] = set()

    exact_names = (
        "model.safetensors",
        "pytorch_model.safetensors",
        "pytorch_model.bin",
    )
    for name in exact_names:
        if (path / name).is_file():
            names.add(name)

    # These are the standard Hugging Face sharded filenames.  Do not accept an
    # arbitrary .safetensors file: an adapter or unrelated artifact is not a
    # complete Whisper checkpoint.
    for pattern in (
        "model-*.safetensors",
        "pytorch_model-*.safetensors",
        "model-*.bin",
        "pytorch_model-*.bin",
    ):
        names.update(item.name for item in path.glob(pattern) if item.is_file())

    return sorted(names)


def _validate_model_files(model_path: str) -> list[str]:
    path = Path(model_path)
    missing: list[str] = []
    if not (path / "config.json").is_file():
        missing.append("config.json")

    weights = _model_weight_files(model_path)
    if not weights:
        missing.append(
            "model weights (model.safetensors or sharded safetensors/bin)"
        )

    if missing:
        raise ValueError(
            f"model_path is missing required model files: {', '.join(missing)}"
        )
    return weights


def _validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    model_path = _validate_local_path(request, "model_path", directory=True)
    audio_path = _validate_local_path(request, "audio_path", directory=False)
    weights = _validate_model_files(model_path)

    language = request.get("language")
    if language is not None and (
        not isinstance(language, str) or not language
    ):
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

    device = request.get("device", DEFAULT_DEVICE)
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string when provided")
    if device not in _SUPPORTED_DEVICES:
        choices = ", ".join(sorted(_SUPPORTED_DEVICES))
        raise ValueError(f"device must be one of: {choices}")

    chunk_seconds = request.get("chunk_seconds", DEFAULT_CHUNK_SECONDS)
    if isinstance(chunk_seconds, bool) or not isinstance(
        chunk_seconds, (int, float)
    ):
        raise ValueError(
            "chunk_seconds must be a finite number greater than 0 and at most 30"
        )
    chunk_seconds = float(chunk_seconds)
    if not math.isfinite(chunk_seconds) or not 0 < chunk_seconds <= 30:
        raise ValueError(
            "chunk_seconds must be a finite number greater than 0 and at most 30"
        )

    return {
        "model_path": model_path,
        "audio_path": audio_path,
        "language": language,
        "max_new_tokens": max_new_tokens,
        "requested_device": device,
        "chunk_seconds": chunk_seconds,
        "weights": weights,
    }


def _import_torch() -> Any:
    # Torch and Transformers must stay off the validation path.  This also
    # makes the worker usable by protocol tests without either dependency.
    import torch

    return torch


def _import_soundfile() -> Any:
    # Audio decoding stays lazy so validation and model-loading paths do not
    # require soundfile or its native dependencies.
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
    if callable(astype):
        try:
            return astype("float32", copy=False)
        except TypeError:
            return astype("float32")

    # soundfile normally returns a numpy array.  This fallback keeps injected
    # readers aligned with the worker's raw_float32 input contract as well.
    try:
        import numpy as np
    except ImportError:
        return audio
    return np.asarray(audio, dtype=np.float32)


def _resample_audio(audio: Any, sample_rate: Any) -> Any:
    if sample_rate == _SAMPLE_RATE:
        return audio

    try:
        source_rate = int(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unsupported sample rate {sample_rate!r}; expected {_SAMPLE_RATE} Hz"
        ) from exc
    if source_rate <= 0 or source_rate != sample_rate:
        raise ValueError(
            f"unsupported sample rate {sample_rate!r}; expected {_SAMPLE_RATE} Hz"
        )

    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ValueError(
            f"unsupported sample rate {source_rate} Hz; scipy is required to "
            f"resample audio to {_SAMPLE_RATE} Hz"
        ) from exc

    divisor = math.gcd(_SAMPLE_RATE, source_rate)
    return _as_float32(
        resample_poly(
            audio,
            _SAMPLE_RATE // divisor,
            source_rate // divisor,
        )
    )


def _mps_available(torch_module: Any) -> bool:
    try:
        return bool(torch_module.backends.mps.is_available())
    except Exception:
        return False


def _select_device(requested_device: str, torch_module: Any) -> str:
    if requested_device == "cpu":
        return "cpu"
    return "mps" if _mps_available(torch_module) else "cpu"


def _load_model(local_path: str) -> tuple[Any, Any]:
    """Load the local Transformers processor and Whisper model only."""
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        local_path,
        local_files_only=True,
    )
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            local_path,
            local_files_only=True,
            dtype="auto",
        )
    except TypeError:
        # Transformers versions before the dtype keyword used torch_dtype.
        model = WhisperForConditionalGeneration.from_pretrained(
            local_path,
            local_files_only=True,
            torch_dtype="auto",
        )
    return processor, model


def _unpack_model_bundle(bundle: Any) -> tuple[Any, Any]:
    if isinstance(bundle, Mapping):
        try:
            return bundle["processor"], bundle["model"]
        except KeyError as exc:
            raise TypeError("model loader must return processor and model") from exc

    if isinstance(bundle, (tuple, list)) and len(bundle) == 2:
        return bundle[0], bundle[1]

    processor = getattr(bundle, "processor", None)
    model = getattr(bundle, "model", None)
    if processor is not None and model is not None:
        return processor, model
    raise TypeError("model loader must return a (processor, model) pair")


def _move_to_device(value: Any, device: str) -> Any:
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}

    to = getattr(value, "to", None)
    if not callable(to):
        return value
    moved = to(device)
    return value if moved is None else moved


def _inference_context(torch_module: Any) -> Any:
    inference_mode = getattr(torch_module, "inference_mode", None)
    return inference_mode() if callable(inference_mode) else nullcontext()


def _generate_sequences(generated: Any) -> Any:
    sequences = getattr(generated, "sequences", None)
    if sequences is not None:
        return sequences
    if isinstance(generated, Mapping) and "sequences" in generated:
        return generated["sequences"]
    return generated


def _transcribe_chunks(
    processor: Any,
    model: Any,
    audio: Any,
    *,
    device: str,
    chunk_seconds: float,
    generate_kwargs: Mapping[str, Any],
    torch_module: Any,
) -> tuple[str, list[dict[str, Any]]]:
    chunk_samples = max(1, int(round(chunk_seconds * _SAMPLE_RATE)))
    sample_count = len(audio)
    segments: list[dict[str, Any]] = []

    with _inference_context(torch_module):
        for offset in range(0, sample_count, chunk_samples):
            chunk_end = min(offset + chunk_samples, sample_count)
            chunk = audio[offset:chunk_end]
            inputs = processor(
                chunk,
                sampling_rate=_SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = _move_to_device(inputs, device)
            generated = model.generate(**inputs, **generate_kwargs)
            decoded = processor.batch_decode(
                _generate_sequences(generated),
                skip_special_tokens=True,
            )
            if not isinstance(decoded, (list, tuple)) or not decoded:
                raise TypeError("processor returned no decoded transcription")
            text = decoded[0]
            if not isinstance(text, str):
                raise TypeError("processor returned a non-string transcription")

            segments.append(
                {
                    "start": offset / _SAMPLE_RATE,
                    "end": chunk_end / _SAMPLE_RATE,
                    "text": text,
                }
            )

    transcript = " ".join(segment["text"] for segment in segments).strip()
    return transcript, segments


def execute(
    request: Any,
    *,
    model_loader: Callable[[str], Any] | None = None,
    torch_module: Any | None = None,
    soundfile_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute one offline local Whisper-Podlodka transcription request."""
    try:
        config = _validate_request(request)
    except Exception as exc:
        return _error("validation_error", _exception_message(exc))

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    load_started = clock()
    try:
        torch = _import_torch() if torch_module is None else torch_module
        device = _select_device(config["requested_device"], torch)
        loader = _load_model if model_loader is None else model_loader
        processor, model = _unpack_model_bundle(loader(config["model_path"]))

        to = getattr(model, "to", None)
        if callable(to):
            to(device)
        eval_model = getattr(model, "eval", None)
        if callable(eval_model):
            eval_model()
    except Exception as exc:
        return _error("load_error", _exception_message(exc))
    load_seconds = float(clock() - load_started)

    generate_kwargs: dict[str, Any] = {
        "task": "transcribe",
        "max_new_tokens": config["max_new_tokens"],
    }
    if config["language"] is not None:
        generate_kwargs["language"] = config["language"]

    transcribe_started = clock()
    try:
        soundfile = (
            _import_soundfile() if soundfile_module is None else soundfile_module
        )
        audio, sample_rate = soundfile.read(
            config["audio_path"], dtype="float32", always_2d=False
        )
        audio = _as_float32(_as_mono(audio))
        audio = _resample_audio(audio, sample_rate)

        transcript, segments = _transcribe_chunks(
            processor,
            model,
            audio,
            device=device,
            chunk_seconds=config["chunk_seconds"],
            generate_kwargs=generate_kwargs,
            torch_module=torch,
        )
    except Exception as exc:
        return _error("transcribe_error", _exception_message(exc))
    transcribe_seconds = float(clock() - transcribe_started)

    effective_config = {
        "model_path": config["model_path"],
        "audio_path": config["audio_path"],
        "language": config["language"],
        "max_new_tokens": config["max_new_tokens"],
        "requested_device": config["requested_device"],
        "device": device,
        "backend": "transformers",
        "model_class": "WhisperForConditionalGeneration",
        "processor": "AutoProcessor",
        "direct_generate": True,
        "chunk_seconds": config["chunk_seconds"],
        "input_format": "raw_float32",
        "sampling_rate": _SAMPLE_RATE,
        "no_torchcodec_path": True,
        "weights": config["weights"],
        "offline": True,
        "local_files_only": True,
        "network_access": False,
        "generate_kwargs": generate_kwargs,
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
        "language": config["language"],
        "load_seconds": load_seconds,
        "transcribe_seconds": transcribe_seconds,
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "weights": config["weights"],
        "effective_config": effective_config,
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        payload = _error("request_error", _exception_message(exc))
    else:
        # Third-party model code may print progress or warnings.  Keep stdout
        # as exactly one JSON response and route those diagnostics to stderr.
        with redirect_stdout(sys.stderr):
            payload = execute(request)

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
