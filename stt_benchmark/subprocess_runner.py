from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class WorkerExecution:
    status: str
    payload: dict[str, Any] | None
    returncode: int | None
    stdout: str
    stderr: str
    wall_seconds: float
    peak_rss_mb: float | None
    error: str | None


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _parse_payload(stdout: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_peak_rss_mb(stderr: str) -> float | None:
    match = re.search(
        r"maximum resident set size\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(KB|KiB|MB|MiB)?",
        stderr,
        re.IGNORECASE,
    )
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "KB").lower()
        if unit in {"mb", "mib"}:
            return value if unit == "mib" else value * (1000**2) / (1024**2)
        return value / 1024

    match = re.search(
        r"\b([0-9]+(?:\.[0-9]+)?)\s+maximum resident set size\b",
        stderr,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1)) / (1024**2)
    return None


def run_json_worker(
    python_executable: str | Path,
    module: str,
    request: dict[str, Any],
    timeout_seconds: float,
    env: Mapping[str, str] | None = None,
) -> WorkerExecution:
    is_macos = platform.system() == "Darwin"
    command = [str(python_executable), "-m", module]
    if is_macos:
        command = ["/usr/bin/time", "-l", *command]

    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)
    request_text = json.dumps(request) + "\n"
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            input=request_text,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        wall_seconds = time.perf_counter() - started
        stdout = getattr(exc, "stdout", None)
        if stdout is None:
            stdout = getattr(exc, "output", None)
        stderr = _as_text(getattr(exc, "stderr", None))
        return WorkerExecution(
            status="timeout",
            payload=None,
            returncode=None,
            stdout=_as_text(stdout),
            stderr=stderr,
            wall_seconds=wall_seconds,
            peak_rss_mb=_parse_peak_rss_mb(stderr) if is_macos else None,
            error=f"worker timed out after {timeout_seconds} seconds",
        )
    except OSError as exc:
        wall_seconds = time.perf_counter() - started
        return WorkerExecution(
            status="spawn_error",
            payload=None,
            returncode=None,
            stdout="",
            stderr="",
            wall_seconds=wall_seconds,
            peak_rss_mb=None,
            error=str(exc),
        )

    wall_seconds = time.perf_counter() - started
    stdout = _as_text(completed.stdout)
    stderr = _as_text(completed.stderr)
    payload = _parse_payload(stdout)
    returncode = completed.returncode

    if returncode != 0:
        return WorkerExecution(
            status="worker_error",
            payload=payload,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            wall_seconds=wall_seconds,
            peak_rss_mb=_parse_peak_rss_mb(stderr) if is_macos else None,
            error=f"worker exited with return code {returncode}",
        )
    if payload is None:
        return WorkerExecution(
            status="protocol_error",
            payload=None,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            wall_seconds=wall_seconds,
            peak_rss_mb=_parse_peak_rss_mb(stderr) if is_macos else None,
            error="worker stdout is not a JSON object",
        )
    return WorkerExecution(
        status="ok",
        payload=payload,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        wall_seconds=wall_seconds,
        peak_rss_mb=_parse_peak_rss_mb(stderr) if is_macos else None,
        error=None,
    )
