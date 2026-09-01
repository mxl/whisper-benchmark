from __future__ import annotations

import json
import os
import sys
import time
from contextlib import redirect_stdout
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit


_SAMPLE_RATE = 8_000
_CHUNK_SAMPLES = 2_400
_DEFAULT_DECODER = "greedy"
_DECODERS = {"greedy": "GREEDY", "beam": "BEAM_SEARCH"}


class _InjectedDecoderType(Enum):
    """Decoder stand-in for tests that inject both foreign dependencies."""

    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"


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


def _validate_request(request: Any) -> tuple[str, str, str, bool]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    model_path = _validate_local_path(request, "model_path", directory=True)
    audio_path = _validate_local_path(request, "audio_path", directory=False)

    decoder = request.get("decoder", _DEFAULT_DECODER)
    if not isinstance(decoder, str) or decoder not in _DECODERS:
        raise ValueError("decoder must be one of: greedy, beam")

    streaming = request.get("streaming", False)
    if not isinstance(streaming, bool):
        raise ValueError("streaming must be a boolean")

    model_dir = Path(model_path)
    if not (model_dir / "model.onnx").is_file():
        raise ValueError("model_path must contain model.onnx")
    if decoder == "beam" and not (model_dir / "kenlm.bin").is_file():
        raise ValueError("model_path must contain kenlm.bin for beam decoding")

    return model_path, audio_path, decoder, streaming


def _import_tone() -> Any:
    # T-one is intentionally imported only after request and local artifact
    # validation. This keeps validation usable in environments without tone.
    import tone

    return tone


def _resolve_dependencies(
    pipeline_factory: Callable[[str, Any], Any] | None,
    read_audio_fn: Callable[[str], Any] | None,
    tone_module: Any | None,
) -> tuple[Callable[[str, Any], Any], Callable[[str], Any], Any]:
    if tone_module is None and (pipeline_factory is None or read_audio_fn is None):
        tone_module = _import_tone()

    decoder_type = (
        getattr(tone_module, "DecoderType", None)
        if tone_module is not None
        else None
    )
    if decoder_type is None:
        if pipeline_factory is None:
            raise RuntimeError("tone.DecoderType is unavailable")
        decoder_type = _InjectedDecoderType

    if pipeline_factory is None:
        if tone_module is None:
            raise RuntimeError("tone.StreamingCTCPipeline is unavailable")
        pipeline_class = tone_module.StreamingCTCPipeline

        def pipeline_factory(model_path: str, selected_decoder: Any) -> Any:
            return pipeline_class.from_local(
                model_path,
                decoder_type=selected_decoder,
            )

    if read_audio_fn is None:
        if tone_module is None:
            raise RuntimeError("tone.read_audio is unavailable")
        read_audio_fn = tone_module.read_audio

    return pipeline_factory, read_audio_fn, decoder_type


def _phrase_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _timestamp(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"phrase {field} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"phrase {field} must be numeric") from exc


def _normalize_phrases(phrases: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for phrase in phrases:
        text = getattr(phrase, "text", None)
        if not isinstance(text, str):
            raise TypeError("pipeline returned an unsupported phrase value")

        text = text.strip()
        if not text:
            continue

        normalized.append(
            {
                "text": text,
                "start_time": _timestamp(getattr(phrase, "start_time", None), "start_time"),
                "end_time": _timestamp(getattr(phrase, "end_time", None), "end_time"),
            }
        )
    return normalized


def _run_streaming(pipeline: Any, audio: Any) -> list[Any]:
    phrases: list[Any] = []
    state = None

    # The official forward() API accepts exactly one 2,400-sample chunk. Do
    # not pad or resample a trailing partial chunk; finalize closes the state.
    for start in range(0, len(audio) - _CHUNK_SAMPLES + 1, _CHUNK_SAMPLES):
        chunk = audio[start : start + _CHUNK_SAMPLES]
        new_phrases, state = pipeline.forward(chunk, state)
        phrases.extend(_phrase_items(new_phrases))

    final_phrases, _ = pipeline.finalize(state)
    phrases.extend(_phrase_items(final_phrases))
    return phrases


def execute(
    request: Any,
    *,
    pipeline_factory: Callable[[str, Any], Any] | None = None,
    read_audio_fn: Callable[[str], Any] | None = None,
    tone_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute one local, CPU-only T-one transcription request."""
    try:
        model_path, audio_path, decoder, streaming = _validate_request(request)
    except Exception as exc:
        return _error("validation_error", _exception_message(exc))

    os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        pipeline_factory, read_audio_fn, decoder_types = _resolve_dependencies(
            pipeline_factory,
            read_audio_fn,
            tone_module,
        )
        selected_decoder = getattr(decoder_types, _DECODERS[decoder])
    except Exception as exc:
        return _error("load_error", _exception_message(exc))

    load_started = clock()
    try:
        pipeline = pipeline_factory(model_path, selected_decoder)
    except Exception as exc:
        return _error("load_error", _exception_message(exc))
    load_seconds = float(clock() - load_started)

    mode = "streaming" if streaming else "offline"
    transcribe_started = clock()
    try:
        audio = read_audio_fn(audio_path)
        if streaming:
            raw_phrases = _run_streaming(pipeline, audio)
        else:
            raw_phrases = _phrase_items(pipeline.forward_offline(audio))
        transcribe_seconds = float(clock() - transcribe_started)
        timestamps = _normalize_phrases(raw_phrases)
    except Exception as exc:
        return _error("transcribe_error", _exception_message(exc))

    return {
        "status": "ok",
        "transcript": " ".join(phrase["text"] for phrase in timestamps),
        "timestamps": timestamps,
        "load_seconds": load_seconds,
        "transcribe_seconds": transcribe_seconds,
        "mode": mode,
        "decoder": decoder,
        "effective_config": {
            "local_files_only": True,
            "offline": True,
            "model_path": model_path,
            "audio_path": audio_path,
            "sample_rate": _SAMPLE_RATE,
            "chunk_samples": _CHUNK_SAMPLES,
            "streaming": streaming,
            "decoder": decoder,
            "offline_environment": {"HF_HUB_OFFLINE": "1"},
        },
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        payload = _error("request_error", _exception_message(exc))
    else:
        # Keep stdout to one JSON response even if the foreign runtime emits
        # warnings or progress messages.
        with redirect_stdout(sys.stderr):
            payload = execute(request)

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
