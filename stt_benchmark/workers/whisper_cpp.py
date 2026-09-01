from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DEFAULT_EXECUTABLE = "/opt/homebrew/bin/whisper-cli"
DEFAULT_BEAM_SIZE = 5
DEFAULT_THREADS = 8
DEFAULT_TIMEOUT_SECONDS = 900.0
TIMESTAMP_SEMANTICS = "segment start/end seconds"

_VERSION_RE = re.compile(
    r"(?P<version>\d+\.\d+(?:\.\d+)?(?:[-+][^\s]+)?)",
    re.IGNORECASE,
)
_VERSION_LABEL_RE = re.compile(
    r"whisper\.cpp\s+version\s*:\s*(?P<version>[^\s]+)", re.IGNORECASE
)
_UNSUPPORTED_FULL_JSON_RE = re.compile(
    r"(?:unknown|unrecognized|invalid|unsupported).*(?:option|argument)|"
    r"(?:option|argument).*(?:unknown|unrecognized|invalid|unsupported)",
    re.IGNORECASE | re.DOTALL,
)


def _error(error_type: str, message: str, **details: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "error",
        "error_type": error_type,
        "error": message,
    }
    payload.update(details)
    return payload


def _exception_message(exc: Exception) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _validate_local_file(request: Mapping[str, Any], field: str) -> str:
    value = request.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty local file path")

    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or "://" in value:
        raise ValueError(f"{field} must be a local path, not a URI or repository ID")

    path = Path(value).expanduser()
    try:
        is_file = path.is_file()
    except (OSError, ValueError) as exc:
        raise ValueError(f"{field} is not a usable local path") from exc
    if not is_file:
        raise ValueError(f"{field} must point to an existing local file")
    # Make relative paths safe across the temporary CLI cwd without resolving
    # symlinks (the configured executable/model path is useful provenance).
    return str(path.absolute())


def _validate_executable(request: Mapping[str, Any]) -> str:
    value = request.get("executable", DEFAULT_EXECUTABLE)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("executable must be a non-empty local executable path")

    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or "://" in value:
        raise ValueError("executable must be a local executable path, not a URI")

    # Bare names are intentionally left for PATH lookup.  Paths containing a
    # directory component must be made absolute because inference runs with a
    # temporary working directory.
    if value.startswith("~") or "/" in value or "\\" in value:
        return str(Path(value).expanduser().absolute())
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_timeout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be a positive finite number")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    return timeout


def _validate_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")

    model_path = _validate_local_file(request, "model_path")
    audio_path = _validate_local_file(request, "audio_path")
    executable = _validate_executable(request)

    language: str | None
    if "language" not in request or request["language"] is None:
        language = None
    else:
        raw_language = request["language"]
        if not isinstance(raw_language, str) or not raw_language.strip():
            raise ValueError("language must be a non-empty string when provided")
        language = raw_language.strip().lower()

    task = request.get("task", "transcribe")
    if not isinstance(task, str) or task not in {"transcribe", "translate"}:
        raise ValueError("task must be one of: transcribe, translate")

    beam_size = _positive_integer(
        request.get("beam_size", DEFAULT_BEAM_SIZE), "beam_size"
    )
    threads = _positive_integer(
        request.get("threads", DEFAULT_THREADS), "threads"
    )
    timeout_seconds = _positive_timeout(
        request.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    )

    model_file = Path(model_path)
    try:
        model_size_bytes = model_file.stat().st_size
    except OSError as exc:
        raise ValueError("model_path could not be inspected") from exc

    return {
        "model_path": model_path,
        "audio_path": audio_path,
        "model_filename": model_file.name,
        "model_size_bytes": model_size_bytes,
        "quantization": _infer_quantization(model_file.name),
        "executable": executable,
        "language": language,
        "task": task,
        "beam_size": beam_size,
        "threads": threads,
        "timeout_seconds": timeout_seconds,
    }


def _infer_quantization(filename: str) -> str | None:
    match = re.search(r"(?<![a-z0-9])q([58])(?:[_-]?([0-9]))?", filename.lower())
    if match is None:
        return None
    quantization = f"q{match.group(1)}"
    if match.group(2) is not None:
        quantization += f"_{match.group(2)}"
    return quantization


def _build_command(
    config: Mapping[str, Any], output_base: Path, json_option: str
) -> list[str]:
    command = [
        str(config["executable"]),
        "--model",
        str(config["model_path"]),
        str(config["audio_path"]),
        "--threads",
        str(config["threads"]),
        "--beam-size",
        str(config["beam_size"]),
        json_option,
        "--output-file",
        str(output_base),
        "--no-prints",
    ]
    language = config.get("language")
    if language is not None:
        command.extend(["--language", str(language)])
    if config["task"] == "translate":
        command.append("--translate")
    return command


def _parse_cli_version(stdout: Any, stderr: Any) -> str | None:
    combined = f"{_as_text(stdout)}\n{_as_text(stderr)}"
    match = _VERSION_LABEL_RE.search(combined)
    if match is None:
        match = _VERSION_RE.search(_as_text(stdout))
    if match is None:
        match = _VERSION_RE.search(_as_text(stderr))
    return match.group("version") if match else None


def _probe_cli_version(
    runner: Callable[..., Any], executable: str, timeout_seconds: float
) -> str | None:
    try:
        completed = runner(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _parse_cli_version(
        getattr(completed, "stdout", None), getattr(completed, "stderr", None)
    )


def _looks_like_unsupported_full_json(completed: Any) -> bool:
    text = f"{_as_text(getattr(completed, 'stdout', None))}\n{_as_text(getattr(completed, 'stderr', None))}"
    return "output-json-full" in text.lower() and bool(
        _UNSUPPORTED_FULL_JSON_RE.search(text)
    )


def _find_output_file(output_base: Path, temp_dir: Path) -> Path | None:
    candidates = [
        output_base.with_name(output_base.name + ".json"),
        output_base.with_suffix(".json"),
        output_base,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Keep this fallback narrow: the temporary directory is fresh, so a JSON
    # file here can only have been produced by this CLI invocation.
    generated = sorted(path for path in temp_dir.rglob("*.json") if path.is_file())
    return generated[0] if generated else None


def _timestamp_from_string(value: str) -> float:
    text = value.strip().strip("[]")
    text = text.replace(",", ".")
    pieces = text.split(":")
    if len(pieces) == 3:
        hours, minutes, seconds = pieces
        result = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    elif len(pieces) == 2:
        minutes, seconds = pieces
        result = float(minutes) * 60 + float(seconds)
    else:
        result = float(text)
    if not math.isfinite(result) or result < 0:
        raise ValueError("timestamp must be a finite non-negative number")
    return result


def _timestamp(value: Any, *, milliseconds: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError("timestamp must be numeric or a timestamp string")
    formatted_string = isinstance(value, str) and ":" in value
    if isinstance(value, str):
        result = _timestamp_from_string(value)
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("timestamp must be numeric or a timestamp string") from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError("timestamp must be a finite non-negative number")
    if milliseconds and not formatted_string:
        result /= 1000.0
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _segment_timestamps(segment: Mapping[str, Any]) -> tuple[float, float]:
    timestamp_object = segment.get("timestamps")
    if timestamp_object is not None:
        timestamps = _mapping(timestamp_object, "timestamps")
        if "from" in timestamps and "to" in timestamps:
            return (
                _timestamp(timestamps["from"]),
                _timestamp(timestamps["to"]),
            )

    offset_object = segment.get("offsets")
    if offset_object is not None:
        offsets = _mapping(offset_object, "offsets")
        if "from" in offsets and "to" in offsets:
            return (
                _timestamp(offsets["from"], milliseconds=True),
                _timestamp(offsets["to"], milliseconds=True),
            )

    if "start" in segment and "end" in segment:
        return _timestamp(segment["start"]), _timestamp(segment["end"])
    if "start_time" in segment and "end_time" in segment:
        return _timestamp(segment["start_time"]), _timestamp(segment["end_time"])
    if "start_ms" in segment and "end_ms" in segment:
        return (
            _timestamp(segment["start_ms"], milliseconds=True),
            _timestamp(segment["end_ms"], milliseconds=True),
        )
    raise ValueError("segment is missing start and end timestamps")


def _normalize_output(document: Any) -> tuple[str, list[dict[str, Any]], str | None]:
    if not isinstance(document, Mapping):
        raise TypeError("CLI JSON output must be an object")

    raw_segments: Any
    if "transcription" in document:
        raw_segments = document["transcription"]
    elif "segments" in document:
        raw_segments = document["segments"]
    elif isinstance(document.get("text"), str):
        raw_segments = []
    else:
        raise ValueError("CLI JSON output has no transcription array")

    if not isinstance(raw_segments, list):
        raise TypeError("CLI transcription must be an array")

    segments: list[dict[str, Any]] = []
    for raw_segment in raw_segments:
        segment = _mapping(raw_segment, "segment")
        text = segment.get("text")
        if not isinstance(text, str):
            raise TypeError("segment text must be a string")
        text = text.strip()
        if not text:
            continue
        start, end = _segment_timestamps(segment)
        if end < start:
            raise ValueError("segment end timestamp precedes start timestamp")
        segments.append({"start": start, "end": end, "text": text})

    if isinstance(document.get("text"), str):
        transcript = document["text"].strip()
    else:
        transcript = " ".join(segment["text"] for segment in segments)

    language: str | None = None
    result = document.get("result")
    if isinstance(result, Mapping) and isinstance(result.get("language"), str):
        language = result["language"]
    elif isinstance(document.get("language"), str):
        language = document["language"]

    return transcript, segments, language


def _read_output(path: Path) -> tuple[str, list[dict[str, Any]], str | None]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"could not read CLI JSON output: {exc}") from exc
    try:
        document = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"CLI output is not valid JSON: {exc}") from exc
    return _normalize_output(document)


def _command_options(
    config: Mapping[str, Any], command: list[str], json_option: str
) -> dict[str, Any]:
    return {
        "model": config["model_path"],
        "audio": config["audio_path"],
        "threads": config["threads"],
        "beam_size": config["beam_size"],
        "json": json_option,
        "output_file": command[command.index("--output-file") + 1],
        "no_prints": True,
        "language": config["language"],
        "translate": config["task"] == "translate",
    }


class _CommandFailure(Exception):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload["error"])
        self.payload = payload


def _run_command(
    runner: Callable[..., Any],
    command: list[str],
    directory: str,
    timeout_seconds: float,
) -> Any:
    try:
        return runner(
            command,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _CommandFailure(
            _error(
                "timeout_error",
                f"whisper-cli timed out after {timeout_seconds} seconds",
                stdout=_as_text(
                    getattr(exc, "stdout", None) or getattr(exc, "output", None)
                ),
                stderr=_as_text(getattr(exc, "stderr", None)),
            )
        ) from exc
    except OSError as exc:
        raise _CommandFailure(_error("spawn_error", _exception_message(exc))) from exc
    except Exception as exc:
        raise _CommandFailure(_error("process_error", _exception_message(exc))) from exc


def _process_failure(
    error_type: str, message: str, completed: Any
) -> dict[str, Any]:
    return _error(
        error_type,
        message,
        returncode=getattr(completed, "returncode", None),
        stdout=_as_text(getattr(completed, "stdout", None)),
        stderr=_as_text(getattr(completed, "stderr", None)),
    )


def execute(
    request: Any,
    *,
    run_fn: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Execute one local request through the official whisper.cpp CLI."""
    try:
        config = _validate_request(request)
    except Exception as exc:
        return _error("validation_error", _exception_message(exc))

    runner = subprocess.run if run_fn is None else run_fn
    cli_version = _probe_cli_version(
        runner, config["executable"], config["timeout_seconds"]
    )

    with tempfile.TemporaryDirectory(prefix="whisper-cpp-") as directory:
        temp_dir = Path(directory)
        output_base = temp_dir / "transcription"
        command = _build_command(config, output_base, "--output-json-full")

        transcribe_started = clock()
        try:
            completed = _run_command(
                runner,
                command,
                directory,
                config["timeout_seconds"],
            )
            json_option = "--output-json-full"
            if getattr(completed, "returncode", None) != 0 and (
                _looks_like_unsupported_full_json(completed)
            ):
                json_option = "--output-json"
                command = _build_command(config, output_base, json_option)
                completed = _run_command(
                    runner,
                    command,
                    directory,
                    config["timeout_seconds"],
                )
        except _CommandFailure as failure:
            failure.payload["transcribe_seconds"] = float(clock() - transcribe_started)
            return failure.payload

        transcribe_seconds = float(clock() - transcribe_started)
        returncode = getattr(completed, "returncode", None)
        if returncode != 0:
            failure = _process_failure(
                "cli_error",
                f"whisper-cli exited with return code {returncode}",
                completed,
            )
            failure["transcribe_seconds"] = transcribe_seconds
            return failure

        output_file = _find_output_file(output_base, temp_dir)
        if output_file is None:
            failure = _process_failure(
                "protocol_error",
                "whisper-cli did not produce a JSON output file",
                completed,
            )
            failure["transcribe_seconds"] = transcribe_seconds
            return failure

        try:
            transcript, segments, language = _read_output(output_file)
        except Exception as exc:
            failure = _process_failure(
                "protocol_error",
                _exception_message(exc),
                completed,
            )
            failure["transcribe_seconds"] = transcribe_seconds
            return failure

        effective_config = {
            "backend": "whisper.cpp",
            "executable": config["executable"],
            "cli_version": cli_version,
            "model_path": config["model_path"],
            "model_filename": config["model_filename"],
            "model_size_bytes": config["model_size_bytes"],
            "quantization": config["quantization"],
            "audio_path": config["audio_path"],
            "language": config["language"],
            "task": config["task"],
            "threads": config["threads"],
            "beam_size": config["beam_size"],
            "command": command,
            "command_options": _command_options(config, command, json_option),
            "offline": True,
            "local_files_only": True,
            "network_access": False,
            "timestamp_semantics": TIMESTAMP_SEMANTICS,
        }
        return {
            "status": "ok",
            "transcript": transcript,
            "segments": segments,
            "language": language,
            "load_seconds": None,
            "transcribe_seconds": transcribe_seconds,
            "cli_version": cli_version,
            "task": config["task"],
            "threads": config["threads"],
            "beam_size": config["beam_size"],
            "quantization": config["quantization"],
            "model_size_bytes": config["model_size_bytes"],
            "timestamp_semantics": TIMESTAMP_SEMANTICS,
            "effective_config": effective_config,
        }


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:
        payload = _error("request_error", _exception_message(exc))
    else:
        # Keep the worker protocol on stdout even if an injected dependency or
        # a future subprocess wrapper writes diagnostics to Python stdout.
        with redirect_stdout(sys.stderr):
            payload = execute(request)

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
