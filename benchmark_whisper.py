#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import subprocess
import statistics
import sys
import tempfile
import time
import traceback
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from huggingface_hub import snapshot_download
import jiwer
from pathlib import Path
from typing import Any
from stt_benchmark.subprocess_runner import run_json_worker


DEFAULT_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
DEFAULT_OUTPUT = "benchmark_results.json"
CURRENT_WHISPER_BACKENDS = [
    "faster-whisper",
    "mlx-whisper",
    "mlx-audio",
    "lightning-whisper-mlx",
    "insanely-fast-whisper",
    "openai-whisper",
]
CURRENT_BACKENDS = [
    *CURRENT_WHISPER_BACKENDS,
    "whisper-cpp",
    "gigaam",
    "gigaam-multilingual",
    "gigaam-multilingual-mlx",
    "t-one",
    "vosk",
    "qwen3-asr",
    "qwen3-asr-hf",
    "podlodka",
    "parakeet-hf",
    "parakeet-sherpa",
    "vibevoice",
]
WHISPER_CPP_MODEL_FILES = {
    "whisper-cpp-tiny-fp16": "ggml-tiny.bin",
    "whisper-cpp-tiny-q5_1": "ggml-tiny-q5_1.bin",
    "whisper-cpp-tiny-q8_0": "ggml-tiny-q8_0.bin",
    "whisper-cpp-large-v3-turbo-fp16": "ggml-large-v3-turbo.bin",
    "whisper-cpp-large-v3-turbo-q5_0": "ggml-large-v3-turbo-q5_0.bin",
    "whisper-cpp-large-v3-turbo-q8_0": "ggml-large-v3-turbo-q8_0.bin",
}
WHISPER_CPP_MODEL_VARIANTS = tuple(WHISPER_CPP_MODEL_FILES)
WHISPER_CPP_MODELS = set(WHISPER_CPP_MODEL_FILES)
DEFAULT_WHISPER_CPP_EXECUTABLE = Path("/opt/homebrew/bin/whisper-cli")
# Keep the backend-prefixed spelling alongside the executable option's default.
WHISPER_CPP_DEFAULT_EXECUTABLE = DEFAULT_WHISPER_CPP_EXECUTABLE
WHISPER_CPP_DEFAULT_THREADS = 8
WHISPER_CPP_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--ggerganov--whisper.cpp"
)
WHISPER_CPP_WORKER_MODULE = "stt_benchmark.workers.whisper_cpp"
WHISPER_CPP_BACKEND_DEVICE = "subprocess/Metal"
PARAKEET_HF_MODEL_VARIANT = "parakeet-tdt-0.6b-v3-hf"
PARAKEET_HF_MODELS = {PARAKEET_HF_MODEL_VARIANT}
DEFAULT_PARAKEET_HF_PYTHON = Path(".venvs/parakeet-hf/bin/python")
DEFAULT_PARAKEET_HF_DEVICE = "auto"
PARAKEET_HF_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--nvidia--parakeet-tdt-0.6b-v3"
)
PARAKEET_HF_WORKER_MODULE = "stt_benchmark.workers.parakeet_hf"
PARAKEET_HF_OFFLINE_ENV = {
    "HF_HOME": "/Volumes/512GB/hf",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
PARAKEET_HF_DEVICES = ("auto", "mps", "cpu")
PARAKEET_HF_REQUIRED_MODEL_FILES = (
    Path("config.json"),
    Path("model.safetensors"),
    Path("processor_config.json"),
    Path("tokenizer.json"),
)
PARAKEET_SHERPA_FP32_MODEL_VARIANT = "parakeet-tdt-0.6b-v3-sherpa-fp32"
PARAKEET_SHERPA_INT8_MODEL_VARIANT = "parakeet-tdt-0.6b-v3-sherpa-int8"
PARAKEET_SHERPA_MODELS = {
    PARAKEET_SHERPA_FP32_MODEL_VARIANT,
    PARAKEET_SHERPA_INT8_MODEL_VARIANT,
}
PARAKEET_SHERPA_MODEL_VARIANTS = (
    PARAKEET_SHERPA_FP32_MODEL_VARIANT,
    PARAKEET_SHERPA_INT8_MODEL_VARIANT,
)
PARAKEET_SHERPA_MODEL_QUANTIZATION = {
    PARAKEET_SHERPA_FP32_MODEL_VARIANT: "fp32",
    PARAKEET_SHERPA_INT8_MODEL_VARIANT: "int8",
}
PARAKEET_SHERPA_REQUIRED_MODEL_FILES = {
    "fp32": (
        Path("encoder.onnx"),
        Path("encoder.weights"),
        Path("decoder.onnx"),
        Path("joiner.onnx"),
        Path("tokens.txt"),
        Path("bpe.vocab"),
    ),
    "int8": (
        Path("encoder.int8.onnx"),
        Path("decoder.int8.onnx"),
        Path("joiner.int8.onnx"),
        Path("tokens.txt"),
        Path("bpe.vocab"),
    ),
}
PARAKEET_SHERPA_DEFAULT_MODEL_PATH = Path(
    "/Volumes/512GB/hf/derived/parakeet-tdt-0.6b-v3-sherpa-onnx/spike"
)
DEFAULT_PARAKEET_SHERPA_PYTHON = Path(".venvs/parakeet-sherpa/bin/python")
DEFAULT_PARAKEET_SHERPA_THREADS = 4
PARAKEET_SHERPA_WORKER_MODULE = "stt_benchmark.workers.parakeet_sherpa"
PARAKEET_HF_TIMESTAMP_SEMANTICS = "token timestamps"
PARAKEET_SHERPA_TIMESTAMP_SEMANTICS = "token/frame starts"
PARAKEET_BACKEND_DEVICE = "subprocess"
PROFILE_DEFAULTS = {
    "main": {
        "audios": ["ru"],
        "models": [
            "large-v3-turbo",
            "e2e_rnnt",
            "gigaam-multilingual-large-ctc",
            "t-one-greedy",
            "vosk-ru",
            "qwen3-asr-0.6b-8bit",
        ],
        "backends": [
            "mlx-whisper",
            "gigaam",
            "gigaam-multilingual",
            "t-one",
            "vosk",
            "qwen3-asr",
        ],
    },
    "ru-variants": {
        "audios": ["ru"],
        "models": ["e2e_rnnt", "e2e_ctc", "vosk-ru", "vosk-small-ru"],
        "backends": ["gigaam", "vosk"],
    },
    "whisper": {
        "audios": [],
        "models": DEFAULT_MODELS,
        "backends": CURRENT_WHISPER_BACKENDS,
    },
    "whisper-cpp": {
        "audios": [],
        "models": list(WHISPER_CPP_MODEL_VARIANTS),
        "backends": ["whisper-cpp"],
    },
    "podlodka": {
        "audios": [],
        "models": ["whisper-podlodka-turbo"],
        "backends": ["podlodka"],
    },
    "parakeet": {
        "audios": [],
        "models": [
            PARAKEET_HF_MODEL_VARIANT,
            PARAKEET_SHERPA_FP32_MODEL_VARIANT,
            PARAKEET_SHERPA_INT8_MODEL_VARIANT,
        ],
        "backends": ["parakeet-hf", "parakeet-sherpa"],
    },
    "qwen": {
        "audios": [],
        "models": [
            "qwen3-asr-0.6b-8bit",
            "qwen3-asr-1.7b-8bit",
            "qwen3-asr-1.7b-hf",
        ],
        "backends": ["qwen3-asr", "qwen3-asr-hf"],
    },
    "gigaam-multilingual": {
        "audios": [],
        "models": [
            "gigaam-multilingual-large-ctc",
            "gigaam-multilingual-mlx-fp16",
        ],
        "backends": ["gigaam-multilingual", "gigaam-multilingual-mlx"],
    },
    "vibevoice": {
        "audios": [],
        "models": ["vibevoice-asr-hf"],
        "backends": ["vibevoice"],
    },
}
MAIN_BENCHMARK_PAIRS = [
    ("mlx-whisper", "large-v3-turbo"),
    ("gigaam", "e2e_rnnt"),
    ("gigaam-multilingual", "gigaam-multilingual-large-ctc"),
    ("t-one", "t-one-greedy"),
    ("vosk", "vosk-ru"),
    ("qwen3-asr", "qwen3-asr-0.6b-8bit"),
]
RU_VARIANTS_BENCHMARK_PAIRS = [
    ("gigaam", "e2e_rnnt"),
    ("gigaam", "e2e_ctc"),
    ("vosk", "vosk-ru"),
    ("vosk", "vosk-small-ru"),
]
PARAKEET_BENCHMARK_PAIRS = [
    ("parakeet-hf", PARAKEET_HF_MODEL_VARIANT),
    ("parakeet-sherpa", PARAKEET_SHERPA_FP32_MODEL_VARIANT),
    ("parakeet-sherpa", PARAKEET_SHERPA_INT8_MODEL_VARIANT),
]
GIGAAM_MULTILINGUAL_BENCHMARK_PAIRS = [
    ("gigaam-multilingual", "gigaam-multilingual-large-ctc"),
    ("gigaam-multilingual-mlx", "gigaam-multilingual-mlx-fp16"),
]
VIBEVOICE_BENCHMARK_PAIRS = [("vibevoice", "vibevoice-asr-hf")]
QWEN_BENCHMARK_PAIRS = [
    ("qwen3-asr", "qwen3-asr-0.6b-8bit"),
    ("qwen3-asr", "qwen3-asr-1.7b-8bit"),
    ("qwen3-asr-hf", "qwen3-asr-1.7b-hf"),
]
GIGAAM_MODEL_VARIANT = "e2e_rnnt"
GIGAAM_CTC_MODEL_VARIANT = "e2e_ctc"
GIGAAM_MODEL_VARIANTS = (GIGAAM_MODEL_VARIANT, GIGAAM_CTC_MODEL_VARIANT)
GIGAAM_MODELS = set(GIGAAM_MODEL_VARIANTS)
GIGAAM_RU_ONLY_REASON = "ru-only model"
DEFAULT_GIGAAM_PYTHON = Path(".venvs/gigaam/bin/python")
GIGAAM_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-v3"
)
GIGAAM_WORKER_MODULE = "stt_benchmark.workers.gigaam"
GIGAAM_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
GIGAAM_REQUIRED_MODEL_FILES = (
    Path("config.json"),
    Path("modeling_gigaam.py"),
    Path("pytorch_model.bin"),
    Path("tokenizer.model"),
)
GIGAAM_MULTILINGUAL_MODEL_VARIANT = "large_ctc"
GIGAAM_MULTILINGUAL_MODEL = "gigaam-multilingual-large-ctc"
GIGAAM_MULTILINGUAL_MODELS = {GIGAAM_MULTILINGUAL_MODEL}
GIGAAM_MULTILINGUAL_SUPPORTED_LANGUAGES = ("ru", "en", "kk", "ky", "uz")
DEFAULT_GIGAAM_MULTILINGUAL_PYTHON = Path(".venvs/gigaam/bin/python")
GIGAAM_MULTILINGUAL_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-Multilingual"
)
GIGAAM_MULTILINGUAL_WORKER_MODULE = "stt_benchmark.workers.gigaam_multilingual"
GIGAAM_MULTILINGUAL_REQUIRED_MODEL_FILES = (
    Path("config.json"),
    Path("modeling_gigaam.py"),
    Path("pytorch_model.bin"),
)
GIGAAM_MULTILINGUAL_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
GIGAAM_MULTILINGUAL_MLX_MODEL_VARIANT = "fp16"
GIGAAM_MULTILINGUAL_MLX_MODEL = "gigaam-multilingual-mlx-fp16"
GIGAAM_MULTILINGUAL_MLX_MODELS = {GIGAAM_MULTILINGUAL_MLX_MODEL}
GIGAAM_MULTILINGUAL_MLX_SUPPORTED_LANGUAGES = GIGAAM_MULTILINGUAL_SUPPORTED_LANGUAGES
DEFAULT_GIGAAM_MULTILINGUAL_MLX_PYTHON = Path(
    ".venvs/gigaam-multilingual-mlx/bin/python"
)
DEFAULT_GIGAAM_MULTILINGUAL_MLX_CHUNK_SECONDS = 20.0
DEFAULT_GIGAAM_MULTILINGUAL_MLX_OVERLAP_SECONDS = 2.0
GIGAAM_MULTILINGUAL_MLX_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--ai-babai--gigaam-multilingual-mlx"
)
GIGAAM_MULTILINGUAL_MLX_WORKER_MODULE = (
    "stt_benchmark.workers.gigaam_multilingual_mlx"
)
GIGAAM_MULTILINGUAL_MLX_REQUIRED_MODEL_FILES = (
    Path("config.json"),
    Path("manifest.json"),
    Path("model.safetensors"),
)
GIGAAM_MULTILINGUAL_MLX_TIMESTAMP_SEMANTICS = (
    "approximate greedy-CTC word emission times"
)
GIGAAM_MULTILINGUAL_MLX_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
VIBEVOICE_MODEL_VARIANT = "vibevoice-asr-hf"
VIBEVOICE_MODELS = {VIBEVOICE_MODEL_VARIANT}
VIBEVOICE_DEVICES = ("auto", "mps", "cpu")
VIBEVOICE_MODES = ("transcription_only", "parsed")
DEFAULT_VIBEVOICE_PYTHON = Path(".venvs/vibevoice/bin/python")
DEFAULT_VIBEVOICE_DEVICE = "auto"
DEFAULT_VIBEVOICE_MODE = "transcription_only"
VIBEVOICE_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--microsoft--VibeVoice-ASR-HF"
)
VIBEVOICE_WORKER_MODULE = "stt_benchmark.workers.vibevoice"
VIBEVOICE_MODEL_SHARD_FILES = tuple(
    Path(f"model-{shard:05d}-of-00008.safetensors")
    for shard in range(1, 9)
)
VIBEVOICE_REQUIRED_MODEL_FILES = (
    Path("config.json"),
    Path("model.safetensors.index.json"),
    Path("tokenizer.json"),
    Path("processor_config.json"),
    *VIBEVOICE_MODEL_SHARD_FILES,
)
VIBEVOICE_TIMESTAMP_SEMANTICS = "speaker segment offsets"
VIBEVOICE_OFFLINE_ENV = {
    "HF_HOME": "/Volumes/512GB/hf",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
}
TONE_MODEL_VARIANT = "t-one-greedy"
TONE_MODELS = {TONE_MODEL_VARIANT}
TONE_RU_ONLY_REASON = "ru-only model"
DEFAULT_TONE_PYTHON = Path(".venvs/t-one/bin/python")
TONE_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--t-tech--T-one"
)
TONE_WORKER_MODULE = "stt_benchmark.workers.tone"
TONE_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
VOSK_MODEL_VARIANT = "vosk-ru"
VOSK_SMALL_MODEL_VARIANT = "vosk-small-ru"
VOSK_MODEL_VARIANTS = (VOSK_MODEL_VARIANT, VOSK_SMALL_MODEL_VARIANT)
VOSK_MODELS = set(VOSK_MODEL_VARIANTS)
VOSK_RU_ONLY_REASON = "ru-only model"
DEFAULT_VOSK_PYTHON = Path(".venvs/vosk/bin/python")
VOSK_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--alphacep--vosk-model-ru"
)
VOSK_SMALL_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--alphacep--vosk-model-small-ru"
)
VOSK_MODEL_CACHE_ROOTS = {
    VOSK_MODEL_VARIANT: VOSK_MODEL_CACHE_ROOT,
    VOSK_SMALL_MODEL_VARIANT: VOSK_SMALL_MODEL_CACHE_ROOT,
}
VOSK_MODEL_CACHE_ROOT_BY_VARIANT = VOSK_MODEL_CACHE_ROOTS
_DEFAULT_VOSK_MODEL_CACHE_ROOT = VOSK_MODEL_CACHE_ROOT
_DEFAULT_VOSK_SMALL_MODEL_CACHE_ROOT = VOSK_SMALL_MODEL_CACHE_ROOT
VOSK_WORKER_MODULE = "stt_benchmark.workers.vosk"
VOSK_DECODING_METHODS = ("modified_beam_search", "greedy_search")
VOSK_DEFAULT_DECODING_METHOD = "modified_beam_search"
VOSK_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
VOSK_REQUIRED_MODEL_FILES = (
    Path("am-onnx/encoder.onnx"),
    Path("am-onnx/decoder.onnx"),
    Path("am-onnx/joiner.onnx"),
    Path("lang/tokens.txt"),
)
VOSK_REQUIRED_MODEL_FILES_BY_VARIANT = {
    VOSK_MODEL_VARIANT: VOSK_REQUIRED_MODEL_FILES,
    VOSK_SMALL_MODEL_VARIANT: (
        Path("am/encoder.onnx"),
        Path("am/decoder.onnx"),
        Path("am/joiner.onnx"),
        Path("lang/tokens.txt"),
    ),
}
QWEN3_ASR_MODEL_VARIANT = "qwen3-asr-0.6b-8bit"
QWEN3_ASR_1_7B_MODEL_VARIANT = "qwen3-asr-1.7b-8bit"
QWEN3_ASR_MODEL_VARIANTS = (
    QWEN3_ASR_MODEL_VARIANT,
    QWEN3_ASR_1_7B_MODEL_VARIANT,
)
QWEN3_ASR_MODELS = set(QWEN3_ASR_MODEL_VARIANTS)
# Qwen3-ASR uses the repository's main environment, where mlx-audio is installed.
DEFAULT_QWEN3_ASR_PYTHON = Path(".venv/bin/python")
QWEN3_ASR_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--mlx-community--Qwen3-ASR-0.6B-8bit"
)
QWEN3_ASR_1_7B_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--mlx-community--Qwen3-ASR-1.7B-8bit"
)
_DEFAULT_QWEN3_ASR_MODEL_CACHE_ROOT = QWEN3_ASR_MODEL_CACHE_ROOT
_DEFAULT_QWEN3_ASR_1_7B_MODEL_CACHE_ROOT = QWEN3_ASR_1_7B_MODEL_CACHE_ROOT
QWEN3_ASR_MODEL_CACHE_ROOTS = {
    QWEN3_ASR_MODEL_VARIANT: QWEN3_ASR_MODEL_CACHE_ROOT,
    QWEN3_ASR_1_7B_MODEL_VARIANT: QWEN3_ASR_1_7B_MODEL_CACHE_ROOT,
}
QWEN3_ASR_MODEL_CACHE_ROOT_BY_VARIANT = QWEN3_ASR_MODEL_CACHE_ROOTS
QWEN3_ASR_WORKER_MODULE = "stt_benchmark.workers.qwen3_asr"
QWEN3_ASR_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
QWEN3_ASR_HF_MODEL_VARIANT = "qwen3-asr-1.7b-hf"
QWEN3_ASR_HF_MODELS = {QWEN3_ASR_HF_MODEL_VARIANT}
DEFAULT_QWEN3_ASR_HF_PYTHON = Path(".venvs/qwen3-asr-hf/bin/python")
DEFAULT_QWEN3_ASR_HF_DEVICE = "auto"
QWEN3_ASR_HF_DEFAULT_MAX_TOKENS = 4096
QWEN3_ASR_HF_DEFAULT_MAX_NEW_TOKENS = QWEN3_ASR_HF_DEFAULT_MAX_TOKENS
QWEN3_ASR_HF_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--Qwen--Qwen3-ASR-1.7B-hf"
)
QWEN3_ASR_HF_WORKER_MODULE = "stt_benchmark.workers.qwen3_asr_hf"
QWEN3_ASR_HF_OFFLINE_ENV = {
    "HF_HOME": "/Volumes/512GB/hf",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
}
QWEN3_ASR_HF_DEVICES = ("auto", "mps", "cpu")
QWEN3_ASR_HF_REQUIRED_MODEL_FILES = (
    Path("config.json"),
    Path("model.safetensors"),
    Path("processor_config.json"),
    Path("tokenizer.json"),
)
QWEN3_ASR_HF_TIMESTAMP_SEMANTICS = "unsupported"
QWEN3_ASR_HF_TIMESTAMP_REASON = (
    "Qwen3-ForcedAligner is not run by this worker"
)
QWEN3_ASR_HF_BACKEND_DEVICE = "subprocess"
PODLODKA_MODEL_VARIANT = "whisper-podlodka-turbo"
PODLODKA_MODELS = {PODLODKA_MODEL_VARIANT}
PODLODKA_DEFAULT_MAX_NEW_TOKENS = 444
DEFAULT_PODLODKA_PYTHON = Path(".venv/bin/python")
PODLODKA_MODEL_CACHE_ROOT = Path(
    "/Volumes/512GB/hf/hub/models--bond005--whisper-podlodka-turbo"
)
PODLODKA_WORKER_MODULE = "stt_benchmark.workers.podlodka"
PODLODKA_OFFLINE_ENV = {
    "HF_HOME": "/Volumes/512GB/hf",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
PODLODKA_REQUIRED_MODEL_FILES = (Path("config.json"),)
PODLODKA_MODEL_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
PODLODKA_MODEL_WEIGHT_PATTERNS = (
    "model-*.safetensors",
    "pytorch_model-*.safetensors",
    "model-*.bin",
    "pytorch_model-*.bin",
)
OPENAI_WHISPER_REPOS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large-v3": "large-v3",
    "large-v3-turbo": "turbo",
}
MLX_WHISPER_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}
MLX_AUDIO_WHISPER_REPOS = {
    "tiny": "mlx-community/whisper-tiny-asr-fp16",
    "base": "mlx-community/whisper-base-asr-fp16",
    "small": "mlx-community/whisper-small-asr-fp16",
    "medium": "mlx-community/whisper-medium-asr-fp16",
    "large-v3": "mlx-community/whisper-large-v3-asr-fp16",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo-asr-fp16",
}
LIGHTNING_WHISPER_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-turbo",
}
INSANELY_FAST_WHISPER_REPOS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}
DEFAULT_SAMPLES = {
    "en": {
        "audio": Path("samples/librispeech_1089_134686.mp3"),
        "reference_transcript": Path("samples/librispeech_1089_134686.txt"),
        "attribution": Path("samples/librispeech_1089_134686.attribution.txt"),
        "sample_label": "librispeech_1089_134686",
    },
    "ru": {
        "audio": Path("samples/ruls_sample_8169_13240.mp3"),
        "reference_transcript": Path("samples/ruls_sample_8169_13240.txt"),
        "attribution": Path("samples/ruls_sample_8169_13240.attribution.txt"),
        "sample_label": "ruls_sample_8169_13240",
    },
}
DEFAULT_SAMPLE_LANGUAGES = tuple(sorted(DEFAULT_SAMPLES))


@dataclass
class RunResult:
    audio: str
    sample_label: str
    audio_duration_seconds: float
    forced_language: str | None
    backend: str
    model: str
    backend_device: str | None
    run_index: int
    load_seconds: float | None
    transcribe_seconds: float | None
    total_seconds: float | None
    transcript: str | None
    transcript_chars: int | None
    transcript_words: int | None
    wer: float | None
    cer: float | None
    detected_language: str | None
    detected_language_probability: float | None
    status: str
    error: str | None
    peak_rss_mb: float | None = None


@dataclass
class BackendSession:
    backend: str
    model: str
    device: str | None
    session: Any
    load_seconds: float | None


@dataclass
class SkippedBenchmark:
    audio: str
    sample_label: str
    forced_language: str | None
    backend: str
    model: str
    reason: str


@dataclass(frozen=True)
class ResolvedAudioInput:
    audio_path: Path
    reference_transcript_path: Path | None
    reference_transcript_text: str | None
    forced_language: str | None
    selector_language: str
    sample_label: str
    source: str
    audio_duration_seconds: float


@dataclass(frozen=True)
class BackendCapabilities:
    supported_models: set[str] | None
    supports_hallucination_silence_threshold: bool
    multilingual: bool = False
    supported_tasks: tuple[str, ...] = ("transcribe",)
    supports_segment_timestamps: bool = False
    supports_condition_on_previous_text: bool = True
    device: str | None = None


BACKEND_CAPABILITIES: dict[str, BackendCapabilities] = {
    "faster-whisper": BackendCapabilities(
        supported_models=None,
        supports_hallucination_silence_threshold=True,
    ),
    "mlx-whisper": BackendCapabilities(
        supported_models=set(MLX_WHISPER_REPOS),
        supports_hallucination_silence_threshold=True,
    ),
    "mlx-audio": BackendCapabilities(
        supported_models=set(MLX_AUDIO_WHISPER_REPOS),
        supports_hallucination_silence_threshold=False,
    ),
    "lightning-whisper-mlx": BackendCapabilities(
        supported_models=set(LIGHTNING_WHISPER_MLX_REPOS),
        supports_hallucination_silence_threshold=True,
    ),
    "insanely-fast-whisper": BackendCapabilities(
        supported_models=set(INSANELY_FAST_WHISPER_REPOS),
        supports_hallucination_silence_threshold=False,
    ),
    "openai-whisper": BackendCapabilities(
        supported_models=set(OPENAI_WHISPER_REPOS),
        supports_hallucination_silence_threshold=True,
    ),
    "whisper-cpp": BackendCapabilities(
        supported_models=WHISPER_CPP_MODELS,
        supports_hallucination_silence_threshold=False,
        multilingual=True,
        supported_tasks=("transcribe", "translate"),
        supports_segment_timestamps=True,
        supports_condition_on_previous_text=False,
        device=WHISPER_CPP_BACKEND_DEVICE,
    ),
    "gigaam": BackendCapabilities(
        supported_models=GIGAAM_MODELS,
        supports_hallucination_silence_threshold=False,
    ),
    "gigaam-multilingual": BackendCapabilities(
        supported_models=GIGAAM_MULTILINGUAL_MODELS,
        supports_hallucination_silence_threshold=False,
    ),
    "gigaam-multilingual-mlx": BackendCapabilities(
        supported_models=GIGAAM_MULTILINGUAL_MLX_MODELS,
        supports_hallucination_silence_threshold=False,
        multilingual=True,
        supported_tasks=("transcribe",),
        supports_segment_timestamps=True,
        supports_condition_on_previous_text=False,
        device="subprocess",
    ),
    "t-one": BackendCapabilities(
        supported_models=TONE_MODELS,
        supports_hallucination_silence_threshold=False,
    ),
    "vosk": BackendCapabilities(
        supported_models=VOSK_MODELS,
        supports_hallucination_silence_threshold=False,
    ),
    "qwen3-asr": BackendCapabilities(
        supported_models=QWEN3_ASR_MODELS,
        supports_hallucination_silence_threshold=False,
    ),
    "qwen3-asr-hf": BackendCapabilities(
        supported_models=QWEN3_ASR_HF_MODELS,
        supports_hallucination_silence_threshold=False,
        multilingual=True,
        supported_tasks=("transcribe",),
        supports_segment_timestamps=False,
        supports_condition_on_previous_text=False,
        device=QWEN3_ASR_HF_BACKEND_DEVICE,
    ),
    "podlodka": BackendCapabilities(
        supported_models=PODLODKA_MODELS,
        supports_hallucination_silence_threshold=False,
    ),
    "parakeet-hf": BackendCapabilities(
        supported_models=PARAKEET_HF_MODELS,
        supports_hallucination_silence_threshold=False,
        multilingual=True,
        supported_tasks=("transcribe",),
        supports_segment_timestamps=True,
        supports_condition_on_previous_text=False,
        device=PARAKEET_BACKEND_DEVICE,
    ),
    "parakeet-sherpa": BackendCapabilities(
        supported_models=PARAKEET_SHERPA_MODELS,
        supports_hallucination_silence_threshold=False,
        multilingual=True,
        supported_tasks=("transcribe",),
        supports_segment_timestamps=True,
        supports_condition_on_previous_text=False,
        device=PARAKEET_BACKEND_DEVICE,
    ),
    "vibevoice": BackendCapabilities(
        supported_models=VIBEVOICE_MODELS,
        supports_hallucination_silence_threshold=False,
        multilingual=True,
        supported_tasks=("transcribe",),
        supports_segment_timestamps=True,
        supports_condition_on_previous_text=False,
        device="subprocess",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple Whisper backends across multiple model sizes."
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_DEFAULTS),
        default="main",
        help=(
            "Benchmark profile. main runs the Russian sample with the mlx-whisper "
            "GigaAM, GigaAM Multilingual, T-one, Vosk, and Qwen3-ASR baselines; "
            "ru-variants compares the GigaAM and Vosk Russian model variants; "
            "whisper preserves the all-runtime Whisper benchmark; podlodka runs "
            "the Podlodka model on all bundled samples; whisper-cpp runs the "
            "six official whisper.cpp model variants on all bundled samples; "
            "parakeet runs the official HF and Sherpa-ONNX variants on all bundled "
            "samples; qwen runs the Qwen3-ASR MLX 0.6B and 1.7B variants plus the "
            "official HF 1.7B variant on all bundled "
            "samples; gigaam-multilingual compares the official and MLX multilingual "
            "models; vibevoice runs the official VibeVoice ASR model."
        ),
    )
    parser.add_argument(
        "--audio",
        dest="audios",
        action="append",
        default=None,
        help=(
            "Audio selector. Use <language> for a bundled sample, auto for all bundled "
            "samples with language autodetection, or <language>:<audio path>:<reference "
            "transcript path> for a custom sample. Repeatable."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to benchmark. Defaults depend on --profile.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=CURRENT_BACKENDS,
        default=None,
        help="Backends to benchmark. Defaults depend on --profile.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs per backend/model pair.",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisper task.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size used for both backends when supported.",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="faster-whisper compute type. Example: default, int8, float16, int8_float16.",
    )
    parser.add_argument(
        "--faster-whisper-vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable faster-whisper VAD filter to drop silence and reduce hallucinations.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Condition each window on previously decoded text (improves long-form context).",
    )
    parser.add_argument(
        "--openai-whisper-temperature-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable openai-whisper temperature fallback (0.0..1.0) on low-confidence segments.",
    )
    parser.add_argument(
        "--hallucination-silence-threshold",
        type=float,
        default=2.0,
        help="Skip silent periods longer than this (seconds) when a possible hallucination is detected. "
        "Supported by faster-whisper, mlx-whisper, openai-whisper, and lightning-whisper-mlx. "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="faster-whisper device. Example: auto, cpu, cuda.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON results. Defaults to a timestamped filename.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one untimed transcription warmup per backend/model before timed runs.",
    )
    parser.add_argument(
        "--show-full-table",
        action="store_true",
        help="Print the full per-run results table before the aggregated summary.",
    )
    parser.add_argument(
        "--insanely-fast-whisper-device-id",
        default="mps",
        help='Device id passed to insanely-fast-whisper. Defaults to "mps" with CPU fallback when MPS is unavailable.',
    )
    parser.add_argument(
        "--insanely-fast-whisper-batch-size",
        type=int,
        default=1,
        help="Batch size passed to insanely-fast-whisper.",
    )
    parser.add_argument(
        "--insanely-fast-whisper-flash",
        action="store_true",
        help="Enable insanely-fast-whisper flash attention mode.",
    )
    parser.add_argument(
        "--lightning-whisper-mlx-batch-size",
        type=int,
        default=12,
        help="Batch size passed to lightning-whisper-mlx.",
    )
    parser.add_argument(
        "--gigaam-python",
        type=Path,
        default=Path(os.environ.get("GIGAAM_PYTHON") or DEFAULT_GIGAAM_PYTHON),
        help="Python executable for the isolated GigaAM worker (default: GIGAAM_PYTHON or .venvs/gigaam/bin/python).",
    )
    parser.add_argument(
        "--gigaam-model-path",
        type=Path,
        default=(
            Path(os.environ["GIGAAM_MODEL_PATH"])
            if os.environ.get("GIGAAM_MODEL_PATH")
            else None
        ),
        help="Exact local GigaAM snapshot directory. Defaults to the cached e2e_rnnt ref.",
    )
    parser.add_argument(
        "--gigaam-multilingual-python",
        type=Path,
        default=Path(
            os.environ.get("GIGAAM_MULTILINGUAL_PYTHON")
            or DEFAULT_GIGAAM_MULTILINGUAL_PYTHON
        ),
        help=(
            "Python executable for the isolated GigaAM Multilingual worker "
            "(default: GIGAAM_MULTILINGUAL_PYTHON or .venvs/gigaam/bin/python)."
        ),
    )
    parser.add_argument(
        "--gigaam-multilingual-model-path",
        type=Path,
        default=(
            Path(os.environ["GIGAAM_MULTILINGUAL_MODEL_PATH"])
            if os.environ.get("GIGAAM_MULTILINGUAL_MODEL_PATH")
            else None
        ),
        help=(
            "Exact local GigaAM Multilingual snapshot directory. Defaults to the "
            "cached large_ctc ref."
        ),
    )
    parser.add_argument(
        "--gigaam-multilingual-language",
        choices=GIGAAM_MULTILINGUAL_SUPPORTED_LANGUAGES,
        default=None,
        help=(
            "Optional GigaAM Multilingual language override. If unset, use the "
            "resolved audio language when forced and omit it for autodetection."
        ),
    )
    parser.add_argument(
        "--gigaam-multilingual-mlx-python",
        type=Path,
        default=Path(
            os.environ.get("GIGAAM_MULTILINGUAL_MLX_PYTHON")
            or DEFAULT_GIGAAM_MULTILINGUAL_MLX_PYTHON
        ),
        help=(
            "Python executable for the isolated GigaAM Multilingual MLX worker "
            "(default: GIGAAM_MULTILINGUAL_MLX_PYTHON or "
            ".venvs/gigaam-multilingual-mlx/bin/python)."
        ),
    )
    parser.add_argument(
        "--gigaam-multilingual-mlx-model-path",
        type=Path,
        default=(
            Path(os.environ["GIGAAM_MULTILINGUAL_MLX_MODEL_PATH"])
            if os.environ.get("GIGAAM_MULTILINGUAL_MLX_MODEL_PATH")
            else None
        ),
        help=(
            "Exact local GigaAM Multilingual MLX snapshot directory. Defaults to "
            "the cached main ref."
        ),
    )
    parser.add_argument(
        "--gigaam-multilingual-mlx-chunk-seconds",
        type=float,
        default=float(
            os.environ.get("GIGAAM_MULTILINGUAL_MLX_CHUNK_SECONDS")
            or DEFAULT_GIGAAM_MULTILINGUAL_MLX_CHUNK_SECONDS
        ),
        help="Audio chunk length in seconds passed to GigaAM Multilingual MLX.",
    )
    parser.add_argument(
        "--gigaam-multilingual-mlx-overlap-seconds",
        type=float,
        default=float(
            os.environ.get("GIGAAM_MULTILINGUAL_MLX_OVERLAP_SECONDS")
            or DEFAULT_GIGAAM_MULTILINGUAL_MLX_OVERLAP_SECONDS
        ),
        help="Audio chunk overlap in seconds passed to GigaAM Multilingual MLX.",
    )
    parser.add_argument(
        "--tone-python",
        type=Path,
        default=Path(os.environ.get("TONE_PYTHON") or DEFAULT_TONE_PYTHON),
        help="Python executable for the isolated T-one worker (default: TONE_PYTHON or .venvs/t-one/bin/python).",
    )
    parser.add_argument(
        "--tone-model-path",
        type=Path,
        default=(
            Path(os.environ["TONE_MODEL_PATH"])
            if os.environ.get("TONE_MODEL_PATH")
            else None
        ),
        help="Exact local T-one snapshot directory. Defaults to the cached main ref.",
    )
    parser.add_argument(
        "--tone-decoder",
        choices=["greedy", "beam"],
        default="greedy",
        help="T-one decoder to use for the worker invocation.",
    )
    parser.add_argument(
        "--vosk-python",
        type=Path,
        default=Path(os.environ.get("VOSK_PYTHON") or DEFAULT_VOSK_PYTHON),
        help="Python executable for the isolated Vosk worker (default: VOSK_PYTHON or .venvs/vosk/bin/python).",
    )
    parser.add_argument(
        "--vosk-model-path",
        type=Path,
        default=(
            Path(os.environ["VOSK_MODEL_PATH"])
            if os.environ.get("VOSK_MODEL_PATH")
            else None
        ),
        help="Exact local Vosk snapshot directory. Defaults to the cached vosk-model-ru main ref.",
    )
    parser.add_argument(
        "--vosk-decoding-method",
        choices=VOSK_DECODING_METHODS,
        default=VOSK_DEFAULT_DECODING_METHOD,
        help="Sherpa-ONNX Vosk decoding method for the worker invocation.",
    )
    parser.add_argument(
        "--qwen3-asr-python",
        type=Path,
        default=Path(
            os.environ.get("QWEN3_ASR_PYTHON") or DEFAULT_QWEN3_ASR_PYTHON
        ),
        help=(
            "Python executable for the isolated Qwen3-ASR worker "
            "(default: QWEN3_ASR_PYTHON or the repo .venv/bin/python, "
            "where mlx-audio is installed)."
        ),
    )
    parser.add_argument(
        "--qwen3-asr-model-path",
        type=Path,
        default=(
            Path(os.environ["QWEN3_ASR_MODEL_PATH"])
            if os.environ.get("QWEN3_ASR_MODEL_PATH")
            else None
        ),
        help="Exact local Qwen3-ASR snapshot directory. Defaults to the cached main ref.",
    )
    parser.add_argument(
        "--qwen3-asr-language",
        default=None,
        help=(
            "Optional Qwen3-ASR language override. If unset, use the resolved "
            "audio language when forced and omit it for autodetection."
        ),
    )
    parser.add_argument(
        "--qwen3-asr-max-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens passed to Qwen3-ASR.",
    )
    parser.add_argument(
        "--qwen3-asr-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature passed to Qwen3-ASR.",
    )
    parser.add_argument(
        "--qwen3-asr-hf-python",
        type=Path,
        default=Path(
            os.environ.get("QWEN3_ASR_HF_PYTHON")
            or DEFAULT_QWEN3_ASR_HF_PYTHON
        ),
        help=(
            "Python executable for the isolated official Qwen3-ASR HF worker "
            "(default: QWEN3_ASR_HF_PYTHON or .venvs/qwen3-asr-hf/bin/python)."
        ),
    )
    parser.add_argument(
        "--qwen3-asr-hf-model-path",
        type=Path,
        default=(
            Path(os.environ["QWEN3_ASR_HF_MODEL_PATH"])
            if os.environ.get("QWEN3_ASR_HF_MODEL_PATH")
            else None
        ),
        help=(
            "Exact local official Qwen3-ASR HF snapshot directory. Defaults to "
            "the cached main ref."
        ),
    )
    parser.add_argument(
        "--qwen3-asr-hf-device",
        choices=QWEN3_ASR_HF_DEVICES,
        default=os.environ.get("QWEN3_ASR_HF_DEVICE")
        or DEFAULT_QWEN3_ASR_HF_DEVICE,
        help="Official Qwen3-ASR HF worker device: auto, mps, or cpu.",
    )
    parser.add_argument(
        "--qwen3-asr-hf-max-tokens",
        "--qwen3-asr-hf-max-new-tokens",
        dest="qwen3_asr_hf_max_tokens",
        type=int,
        default=int(
            os.environ.get("QWEN3_ASR_HF_MAX_TOKENS")
            or os.environ.get("QWEN3_ASR_HF_MAX_NEW_TOKENS")
            or QWEN3_ASR_HF_DEFAULT_MAX_TOKENS
        ),
        help="Maximum number of new tokens passed to official Qwen3-ASR HF.",
    )
    parser.add_argument(
        "--whisper-cpp-executable",
        type=Path,
        default=Path(
            os.environ.get("WHISPER_CPP_EXECUTABLE")
            or DEFAULT_WHISPER_CPP_EXECUTABLE
        ),
        help=(
            "Local whisper.cpp CLI executable (default: WHISPER_CPP_EXECUTABLE "
            "or /opt/homebrew/bin/whisper-cli)."
        ),
    )
    parser.add_argument(
        "--whisper-cpp-threads",
        type=int,
        default=WHISPER_CPP_DEFAULT_THREADS,
        help="Number of CPU threads passed to whisper.cpp.",
    )
    parser.add_argument(
        "--podlodka-python",
        type=Path,
        default=Path(os.environ.get("PODLODKA_PYTHON") or DEFAULT_PODLODKA_PYTHON),
        help=(
            "Python executable for the isolated Podlodka worker "
            "(default: PODLODKA_PYTHON or .venv/bin/python)."
        ),
    )
    parser.add_argument(
        "--podlodka-model-path",
        type=Path,
        default=(
            Path(os.environ["PODLODKA_MODEL_PATH"])
            if os.environ.get("PODLODKA_MODEL_PATH")
            else None
        ),
        help="Exact local Podlodka snapshot directory. Defaults to the cached main ref.",
    )
    parser.add_argument(
        "--podlodka-language",
        default=None,
        help=(
            "Optional Podlodka language override. If unset, use the resolved "
            "audio language when concrete and omit it for autodetection."
        ),
    )
    parser.add_argument(
        "--podlodka-max-new-tokens",
        type=int,
        default=PODLODKA_DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of new tokens passed to Podlodka.",
    )
    parser.add_argument(
        "--parakeet-hf-python",
        type=Path,
        default=Path(
            os.environ.get("PARAKEET_HF_PYTHON") or DEFAULT_PARAKEET_HF_PYTHON
        ),
        help=(
            "Python executable for the isolated official Parakeet HF worker "
            "(default: PARAKEET_HF_PYTHON or .venvs/parakeet-hf/bin/python)."
        ),
    )
    parser.add_argument(
        "--parakeet-hf-model-path",
        type=Path,
        default=(
            Path(os.environ["PARAKEET_HF_MODEL_PATH"])
            if os.environ.get("PARAKEET_HF_MODEL_PATH")
            else None
        ),
        help=(
            "Exact local official Parakeet HF snapshot directory. Defaults to "
            "the cached main ref."
        ),
    )
    parser.add_argument(
        "--parakeet-hf-device",
        choices=PARAKEET_HF_DEVICES,
        default=os.environ.get("PARAKEET_HF_DEVICE") or DEFAULT_PARAKEET_HF_DEVICE,
        help="Parakeet HF worker device: auto, mps, or cpu.",
    )
    parser.add_argument(
        "--parakeet-sherpa-python",
        type=Path,
        default=Path(
            os.environ.get("PARAKEET_SHERPA_PYTHON")
            or DEFAULT_PARAKEET_SHERPA_PYTHON
        ),
        help=(
            "Python executable for the isolated Parakeet Sherpa-ONNX worker "
            "(default: PARAKEET_SHERPA_PYTHON or .venvs/parakeet-sherpa/bin/python)."
        ),
    )
    parser.add_argument(
        "--parakeet-sherpa-model-path",
        type=Path,
        default=(
            Path(os.environ["PARAKEET_SHERPA_MODEL_PATH"])
            if os.environ.get("PARAKEET_SHERPA_MODEL_PATH")
            else None
        ),
        help=(
            "Exact local Parakeet Sherpa-ONNX artifact directory. Defaults to "
            f"{PARAKEET_SHERPA_DEFAULT_MODEL_PATH}."
        ),
    )
    parser.add_argument(
        "--parakeet-sherpa-threads",
        type=int,
        default=int(
            os.environ.get("PARAKEET_SHERPA_THREADS")
            or DEFAULT_PARAKEET_SHERPA_THREADS
        ),
        help="Number of CPU threads passed to Parakeet Sherpa-ONNX.",
    )
    parser.add_argument(
        "--vibevoice-python",
        type=Path,
        default=Path(os.environ.get("VIBEVOICE_PYTHON") or DEFAULT_VIBEVOICE_PYTHON),
        help=(
            "Python executable for the isolated VibeVoice worker "
            "(default: VIBEVOICE_PYTHON or .venvs/vibevoice/bin/python)."
        ),
    )
    parser.add_argument(
        "--vibevoice-model-path",
        type=Path,
        default=(
            Path(os.environ["VIBEVOICE_MODEL_PATH"])
            if os.environ.get("VIBEVOICE_MODEL_PATH")
            else None
        ),
        help="Exact local VibeVoice ASR snapshot directory. Defaults to the cached main ref.",
    )
    parser.add_argument(
        "--vibevoice-device",
        choices=VIBEVOICE_DEVICES,
        default=os.environ.get("VIBEVOICE_DEVICE") or DEFAULT_VIBEVOICE_DEVICE,
        help="VibeVoice worker device: auto, mps, or cpu.",
    )
    parser.add_argument(
        "--vibevoice-mode",
        choices=VIBEVOICE_MODES,
        default=os.environ.get("VIBEVOICE_MODE") or DEFAULT_VIBEVOICE_MODE,
        help="VibeVoice decode mode: transcription_only or parsed.",
    )
    parser.add_argument(
        "--vibevoice-acoustic-tokenizer-chunk-size",
        type=int,
        default=(
            int(os.environ["VIBEVOICE_ACOUSTIC_TOKENIZER_CHUNK_SIZE"])
            if os.environ.get("VIBEVOICE_ACOUSTIC_TOKENIZER_CHUNK_SIZE")
            else None
        ),
        help=(
            "Optional VibeVoice acoustic tokenizer chunk size in samples; it must "
            "be a positive multiple of 3200."
        ),
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=900.0,
        help="Timeout for an isolated worker invocation in seconds.",
    )
    args = parser.parse_args(argv)
    args.models_explicit = args.models is not None
    args.backends_explicit = args.backends is not None
    args.qwen3_asr_hf_max_new_tokens = args.qwen3_asr_hf_max_tokens
    profile = PROFILE_DEFAULTS[args.profile]
    if args.audios is None:
        args.audios = list(profile["audios"])
    if args.models is None:
        args.models = list(profile["models"])
    if args.backends is None:
        args.backends = list(profile["backends"])
    return args


def uses_exact_profile_pairs(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "profile", None)
        in {
            "main",
            "parakeet",
            "ru-variants",
            "gigaam-multilingual",
            "vibevoice",
            "qwen",
        }
        and not getattr(args, "models_explicit", False)
        and not getattr(args, "backends_explicit", False)
    )


def iter_benchmark_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if uses_exact_profile_pairs(args):
        if args.profile == "parakeet":
            return list(PARAKEET_BENCHMARK_PAIRS)
        if args.profile == "ru-variants":
            return list(RU_VARIANTS_BENCHMARK_PAIRS)
        if args.profile == "gigaam-multilingual":
            return list(GIGAAM_MULTILINGUAL_BENCHMARK_PAIRS)
        if args.profile == "vibevoice":
            return list(VIBEVOICE_BENCHMARK_PAIRS)
        if args.profile == "qwen":
            return list(QWEN_BENCHMARK_PAIRS)
        return list(MAIN_BENCHMARK_PAIRS)
    return [
        (backend, model)
        for model in args.models
        for backend in args.backends
    ]


def parse_audio_spec(value: str) -> tuple[str, Path | None, Path | None]:
    if value == "auto":
        return "auto", None, None
    if ":" not in value:
        return value, None, None
    language, remainder = value.split(":", 1)
    if not language:
        raise ValueError(f"Invalid audio spec: {value}")
    if remainder.startswith("/"):
        split_token = ":/"
        split_index = remainder.rfind(split_token)
        if split_index == -1:
            raise ValueError(
                f"Custom audio spec must be <language>:<audio path>:<reference transcript path>: {value}"
            )
        audio_part = remainder[:split_index]
        reference_part = remainder[split_index + 1 :]
    else:
        parts = remainder.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Custom audio spec must be <language>:<audio path>:<reference transcript path>: {value}"
            )
        audio_part, reference_part = parts
    if not audio_part or not reference_part:
        raise ValueError(
            f"Custom audio spec must be <language>:<audio path>:<reference transcript path>: {value}"
        )
    return language, Path(audio_part), Path(reference_part)


def validate_audio_selector_language(language: str) -> None:
    if language == "auto":
        return
    if language in DEFAULT_SAMPLES:
        return
    if not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]+)*", language):
        raise ValueError(
            f"Invalid audio selector language '{language}'. Use a language code or 'auto'."
        )


def resolve_audio_inputs(args: argparse.Namespace) -> list[ResolvedAudioInput]:
    def resolve_default_sample(
        language: str, source: str, forced_language: str | None
    ) -> ResolvedAudioInput:
        sample = DEFAULT_SAMPLES[language]
        audio_path = ensure_audio_file(sample["audio"])
        reference_path = sample["reference_transcript"].resolve()
        return ResolvedAudioInput(
            audio_path=audio_path,
            reference_transcript_path=reference_path,
            reference_transcript_text=load_reference_transcript(reference_path),
            forced_language=forced_language,
            selector_language=language,
            sample_label=sample["sample_label"],
            source=source,
            audio_duration_seconds=get_audio_duration_seconds(audio_path),
        )

    def resolve_custom_sample(
        language: str,
        audio_path: Path,
        reference_transcript_path: Path,
    ) -> ResolvedAudioInput:
        resolved_audio = ensure_audio_file(audio_path)
        resolved_reference = reference_transcript_path.resolve()
        return ResolvedAudioInput(
            audio_path=resolved_audio,
            reference_transcript_path=resolved_reference,
            reference_transcript_text=load_reference_transcript(resolved_reference),
            forced_language=None if language == "auto" else language,
            selector_language=language,
            sample_label=resolved_audio.stem,
            source="explicit",
            audio_duration_seconds=get_audio_duration_seconds(resolved_audio),
        )

    resolved: dict[str, tuple[int, ResolvedAudioInput]] = {}

    def add_entry(priority: int, entry: ResolvedAudioInput) -> None:
        key = str(entry.audio_path)
        current = resolved.get(key)
        if current is None or priority >= current[0]:
            resolved[key] = (priority, entry)

    selectors = args.audios or []
    if not selectors:
        for language in DEFAULT_SAMPLE_LANGUAGES:
            add_entry(
                2,
                resolve_default_sample(
                    language=language,
                    source="default-language",
                    forced_language=language,
                ),
            )
        return [
            entry
            for _, entry in sorted(
                resolved.values(), key=lambda item: item[1].sample_label
            )
        ]

    for selector in selectors:
        language, audio_path, reference_path = parse_audio_spec(selector)
        validate_audio_selector_language(language)
        if language == "auto" and audio_path is None:
            for default_language in DEFAULT_SAMPLE_LANGUAGES:
                add_entry(
                    1,
                    resolve_default_sample(
                        language=default_language,
                        source="default-auto",
                        forced_language=None,
                    ),
                )
            continue
        if audio_path is None:
            if language not in DEFAULT_SAMPLES:
                raise ValueError(
                    f"No bundled sample is configured for language '{language}'."
                )
            add_entry(
                2,
                resolve_default_sample(
                    language=language,
                    source="default-language",
                    forced_language=language,
                ),
            )
            continue
        add_entry(3, resolve_custom_sample(language, audio_path, reference_path))

    return [
        entry
        for _, entry in sorted(resolved.values(), key=lambda item: item[1].sample_label)
    ]


def ensure_audio_file(audio_path: Path) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise ValueError(f"Audio path is not a file: {audio_path}")
    return audio_path.resolve()


def resolve_whisper_cpp_model_path(model_name: str) -> Path:
    """Resolve a whisper.cpp model file from the pinned local main snapshot."""
    model_filename = WHISPER_CPP_MODEL_FILES.get(model_name)
    if model_filename is None:
        raise ValueError(f"Unsupported whisper-cpp model: {model_name}")

    ref_path = WHISPER_CPP_MODEL_CACHE_ROOT / "refs" / "main"
    if not ref_path.is_file():
        raise FileNotFoundError(f"whisper.cpp model ref does not exist: {ref_path}")
    revision = ref_path.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError(
            "whisper.cpp model ref must contain a 40-character commit SHA: "
            f"{ref_path}"
        )

    snapshot_path = WHISPER_CPP_MODEL_CACHE_ROOT / "snapshots" / revision
    if not snapshot_path.is_dir():
        raise FileNotFoundError(
            f"whisper.cpp model snapshot does not exist: {snapshot_path}"
        )

    model_path = snapshot_path / model_filename
    if not model_path.is_file():
        raise FileNotFoundError(f"whisper.cpp model file does not exist: {model_path}")
    try:
        model_size = model_path.stat().st_size
    except OSError as exc:
        raise FileNotFoundError(
            f"whisper.cpp model file cannot be read: {model_path}"
        ) from exc
    if model_size <= 0:
        raise FileNotFoundError(
            f"whisper.cpp model file must be nonempty: {model_path}"
        )
    return model_path.resolve()


def resolve_gigaam_model_path(
    model_name: str | Path = GIGAAM_MODEL_VARIANT,
    model_path: Path | None = None,
) -> Path:
    """Resolve a selected GigaAM variant or an explicit local model directory."""
    if isinstance(model_name, Path):
        if model_path is not None:
            raise ValueError("GigaAM model path was provided more than once")
        model_path = model_name
        model_name = GIGAAM_MODEL_VARIANT
    if model_name not in GIGAAM_MODELS:
        raise ValueError(f"Unsupported GigaAM model: {model_name}")

    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.is_dir():
            raise ValueError(f"GigaAM model path is not a directory: {resolved}")
    else:
        ref_path = GIGAAM_MODEL_CACHE_ROOT / "refs" / model_name
        if not ref_path.is_file():
            raise FileNotFoundError(f"GigaAM model ref does not exist: {ref_path}")
        revision = ref_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                f"GigaAM model ref must contain a 40-character commit SHA: {ref_path}"
            )

        resolved = GIGAAM_MODEL_CACHE_ROOT / "snapshots" / revision
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"GigaAM model snapshot does not exist: {resolved}"
            )

    for relative_path in GIGAAM_REQUIRED_MODEL_FILES:
        required_path = resolved / relative_path
        if not required_path.is_file():
            raise FileNotFoundError(
                f"GigaAM model file does not exist: {required_path}"
            )
    return resolved.resolve()


def resolve_gigaam_multilingual_model_path(model_path: Path | None = None) -> Path:
    """Resolve a GigaAM Multilingual directory or the pinned local cache ref."""
    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.is_dir():
            raise ValueError(
                f"GigaAM Multilingual model path is not a directory: {resolved}"
            )
    else:
        ref_path = (
            GIGAAM_MULTILINGUAL_MODEL_CACHE_ROOT
            / "refs"
            / GIGAAM_MULTILINGUAL_MODEL_VARIANT
        )
        if not ref_path.is_file():
            raise FileNotFoundError(
                f"GigaAM Multilingual model ref does not exist: {ref_path}"
            )
        revision = ref_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                "GigaAM Multilingual model ref must contain a 40-character "
                f"commit SHA: {ref_path}"
            )

        resolved = (
            GIGAAM_MULTILINGUAL_MODEL_CACHE_ROOT / "snapshots" / revision
        )
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"GigaAM Multilingual model snapshot does not exist: {resolved}"
            )

    for relative_path in GIGAAM_MULTILINGUAL_REQUIRED_MODEL_FILES:
        required_path = resolved / relative_path
        if not required_path.is_file():
            raise FileNotFoundError(
                f"GigaAM Multilingual model file does not exist: {required_path}"
            )
    return resolved.resolve()


def resolve_gigaam_multilingual_mlx_model_path(model_path: Path | None = None) -> Path:
    """Resolve the GigaAM Multilingual MLX directory or pinned main snapshot."""
    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.is_dir():
            raise ValueError(
                f"GigaAM Multilingual MLX model path is not a directory: {resolved}"
            )
    else:
        ref_path = GIGAAM_MULTILINGUAL_MLX_MODEL_CACHE_ROOT / "refs" / "main"
        if not ref_path.is_file():
            raise FileNotFoundError(
                f"GigaAM Multilingual MLX model ref does not exist: {ref_path}"
            )
        revision = ref_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                "GigaAM Multilingual MLX model ref must contain a 40-character "
                f"commit SHA: {ref_path}"
            )
        resolved = GIGAAM_MULTILINGUAL_MLX_MODEL_CACHE_ROOT / "snapshots" / revision
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"GigaAM Multilingual MLX model snapshot does not exist: {resolved}"
            )

    for relative_path in GIGAAM_MULTILINGUAL_MLX_REQUIRED_MODEL_FILES:
        required_path = resolved / relative_path
        if not required_path.is_file():
            raise FileNotFoundError(
                f"GigaAM Multilingual MLX model file does not exist: {required_path}"
            )
    return resolved.resolve()


def resolve_vibevoice_model_path(model_path: Path | None = None) -> Path:
    """Resolve the VibeVoice directory or its pinned main snapshot."""
    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.is_dir():
            raise ValueError(f"VibeVoice model path is not a directory: {resolved}")
    else:
        ref_path = VIBEVOICE_MODEL_CACHE_ROOT / "refs" / "main"
        if not ref_path.is_file():
            raise FileNotFoundError(f"VibeVoice model ref does not exist: {ref_path}")
        revision = ref_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                "VibeVoice model ref must contain a 40-character commit SHA: "
                f"{ref_path}"
            )
        resolved = VIBEVOICE_MODEL_CACHE_ROOT / "snapshots" / revision
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"VibeVoice model snapshot does not exist: {resolved}"
            )

    for relative_path in VIBEVOICE_REQUIRED_MODEL_FILES:
        required_path = resolved / relative_path
        if not required_path.is_file():
            raise FileNotFoundError(
                f"VibeVoice model file does not exist: {required_path}"
            )
    return resolved.resolve()


def resolve_tone_model_path(model_path: Path | None = None) -> Path:
    """Resolve an explicit T-one directory or the pinned local cache ref."""
    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.is_dir():
            raise ValueError(f"T-one model path is not a directory: {resolved}")
    else:
        ref_path = TONE_MODEL_CACHE_ROOT / "refs" / "main"
        if not ref_path.is_file():
            raise FileNotFoundError(f"T-one model ref does not exist: {ref_path}")
        revision = ref_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                f"T-one model ref must contain a 40-character commit SHA: {ref_path}"
            )

        resolved = TONE_MODEL_CACHE_ROOT / "snapshots" / revision
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"T-one model snapshot does not exist: {resolved}"
            )

    model_file = resolved / "model.onnx"
    if not model_file.is_file():
        raise FileNotFoundError(f"T-one model file does not exist: {model_file}")
    return resolved.resolve()


def resolve_vosk_model_path(
    model_name: str | Path | None = None,
    model_path: Path | None = None,
) -> Path:
    """Resolve a selected Vosk layout or an explicit local model directory."""
    if isinstance(model_name, Path):
        if model_path is not None:
            raise ValueError("Vosk model path was provided more than once")
        model_path = model_name
        model_name = VOSK_MODEL_VARIANT
    selected_model_name = (
        VOSK_MODEL_VARIANT if model_name is None else model_name
    )
    if selected_model_name not in VOSK_MODELS:
        raise ValueError(f"Unsupported Vosk model: {selected_model_name}")

    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.is_dir():
            raise ValueError(f"Vosk model path is not a directory: {resolved}")
    else:
        configured_cache_root = VOSK_MODEL_CACHE_ROOTS[selected_model_name]
        if selected_model_name == VOSK_MODEL_VARIANT:
            cache_root = (
                VOSK_MODEL_CACHE_ROOT
                if VOSK_MODEL_CACHE_ROOT != _DEFAULT_VOSK_MODEL_CACHE_ROOT
                else configured_cache_root
            )
        else:
            cache_root = (
                VOSK_SMALL_MODEL_CACHE_ROOT
                if VOSK_SMALL_MODEL_CACHE_ROOT != _DEFAULT_VOSK_SMALL_MODEL_CACHE_ROOT
                else configured_cache_root
            )
        ref_path = cache_root / "refs" / "main"
        if not ref_path.is_file():
            raise FileNotFoundError(f"Vosk model ref does not exist: {ref_path}")
        revision = ref_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                f"Vosk model ref must contain a 40-character commit SHA: {ref_path}"
            )

        resolved = cache_root / "snapshots" / revision
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Vosk model snapshot does not exist: {resolved}"
            )

    for relative_path in VOSK_REQUIRED_MODEL_FILES_BY_VARIANT[selected_model_name]:
        required_path = resolved / relative_path
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Vosk model file does not exist: {required_path}"
            )
    return resolved.resolve()


def resolve_qwen3_asr_model_path(
    model_name: str | Path = QWEN3_ASR_MODEL_VARIANT,
    model_path: Path | None = None,
) -> Path:
    """Resolve a selected Qwen3-ASR variant or an explicit local model directory."""
    if isinstance(model_name, Path):
        if model_path is not None:
            raise ValueError("Qwen3-ASR model path was provided more than once")
        model_path = model_name
        model_name = QWEN3_ASR_MODEL_VARIANT
    if model_name not in QWEN3_ASR_MODELS:
        raise ValueError(f"Unsupported Qwen3-ASR model: {model_name}")

    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.is_dir():
            raise ValueError(f"Qwen3-ASR model path is not a directory: {resolved}")
    else:
        configured_cache_root = QWEN3_ASR_MODEL_CACHE_ROOTS[model_name]
        # Preserve the original 0.6B override point while allowing each
        # sibling variant to resolve from its own canonical repository root.
        cache_root = configured_cache_root
        if (
            model_name == QWEN3_ASR_MODEL_VARIANT
            and QWEN3_ASR_MODEL_CACHE_ROOT != _DEFAULT_QWEN3_ASR_MODEL_CACHE_ROOT
        ):
            cache_root = QWEN3_ASR_MODEL_CACHE_ROOT
        elif (
            model_name == QWEN3_ASR_1_7B_MODEL_VARIANT
            and QWEN3_ASR_1_7B_MODEL_CACHE_ROOT
            != _DEFAULT_QWEN3_ASR_1_7B_MODEL_CACHE_ROOT
        ):
            cache_root = QWEN3_ASR_1_7B_MODEL_CACHE_ROOT
        ref_path = cache_root / "refs" / "main"
        if not ref_path.is_file():
            raise FileNotFoundError(f"Qwen3-ASR model ref does not exist: {ref_path}")
        revision = ref_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                "Qwen3-ASR model ref must contain a 40-character commit SHA: "
                f"{ref_path}"
            )

        resolved = cache_root / "snapshots" / revision
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Qwen3-ASR model snapshot does not exist: {resolved}"
            )

    model_files = [
        path
        for path in resolved.rglob("*")
        if path.is_file()
        and path.suffix.lower()
        in {".safetensors", ".npz", ".bin", ".onnx", ".pt", ".pth", ".gguf"}
    ]
    if not model_files:
        raise FileNotFoundError(
            f"Qwen3-ASR model snapshot contains no model weight files: {resolved}"
        )
    empty_files = []
    for path in model_files:
        try:
            if path.stat().st_size <= 0:
                empty_files.append(path)
        except OSError as exc:
            raise FileNotFoundError(
                f"Qwen3-ASR model file cannot be read: {path}"
            ) from exc
    if empty_files:
        files = ", ".join(str(path) for path in empty_files)
        raise FileNotFoundError(f"Qwen3-ASR model files must be nonempty: {files}")

    return resolved.resolve()


def resolve_qwen3_asr_hf_model_path(model_path: Path | None = None) -> Path:
    """Resolve the official Qwen3-ASR HF directory or its pinned main snapshot."""
    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.is_dir():
            raise ValueError(
                f"Qwen3-ASR HF model path is not a directory: {resolved}"
            )
    else:
        ref_path = QWEN3_ASR_HF_MODEL_CACHE_ROOT / "refs" / "main"
        if not ref_path.is_file():
            raise FileNotFoundError(
                f"Qwen3-ASR HF model ref does not exist: {ref_path}"
            )
        revision = ref_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                "Qwen3-ASR HF model ref must contain a 40-character commit SHA: "
                f"{ref_path}"
            )
        resolved = QWEN3_ASR_HF_MODEL_CACHE_ROOT / "snapshots" / revision
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Qwen3-ASR HF model snapshot does not exist: {resolved}"
            )

    for relative_path in QWEN3_ASR_HF_REQUIRED_MODEL_FILES:
        required_path = resolved / relative_path
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Qwen3-ASR HF model file does not exist: {required_path}"
            )

    try:
        if (resolved / "model.safetensors").stat().st_size <= 0:
            raise FileNotFoundError(
                "Qwen3-ASR HF model file must be nonempty: "
                f"{resolved / 'model.safetensors'}"
            )
    except OSError as exc:
        raise FileNotFoundError(
            "Qwen3-ASR HF model file cannot be read: "
            f"{resolved / 'model.safetensors'}"
        ) from exc

    return resolved.resolve()


def resolve_podlodka_model_path(model_path: Path | None = None) -> Path:
    """Resolve a Podlodka directory or the pinned local cache ref."""
    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.is_dir():
            raise ValueError(f"Podlodka model path is not a directory: {resolved}")
    else:
        ref_path = PODLODKA_MODEL_CACHE_ROOT / "refs" / "main"
        if not ref_path.is_file():
            raise FileNotFoundError(f"Podlodka model ref does not exist: {ref_path}")
        revision = ref_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                "Podlodka model ref must contain a 40-character commit SHA: "
                f"{ref_path}"
            )

        resolved = PODLODKA_MODEL_CACHE_ROOT / "snapshots" / revision
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Podlodka model snapshot does not exist: {resolved}"
            )

    for relative_path in PODLODKA_REQUIRED_MODEL_FILES:
        required_path = resolved / relative_path
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Podlodka model file does not exist: {required_path}"
            )

    model_files = [
        resolved / name
        for name in PODLODKA_MODEL_WEIGHT_NAMES
        if (resolved / name).is_file()
    ]
    for pattern in PODLODKA_MODEL_WEIGHT_PATTERNS:
        model_files.extend(
            path for path in resolved.glob(pattern) if path.is_file()
        )
    if not model_files:
        raise FileNotFoundError(
            f"Podlodka model snapshot contains no model weight files: {resolved}"
        )
    empty_files = []
    for path in model_files:
        try:
            if path.stat().st_size <= 0:
                empty_files.append(path)
        except OSError as exc:
            raise FileNotFoundError(
                f"Podlodka model file cannot be read: {path}"
            ) from exc
    if empty_files:
        files = ", ".join(str(path) for path in empty_files)
        raise FileNotFoundError(f"Podlodka model files must be nonempty: {files}")

    return resolved.resolve()


def resolve_parakeet_hf_model_path(model_path: Path | None = None) -> Path:
    """Resolve the official Parakeet HF directory or its pinned main snapshot."""
    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.is_dir():
            raise ValueError(
                f"Parakeet HF model path is not a directory: {resolved}"
            )
    else:
        ref_path = PARAKEET_HF_MODEL_CACHE_ROOT / "refs" / "main"
        if not ref_path.is_file():
            raise FileNotFoundError(
                f"Parakeet HF model ref does not exist: {ref_path}"
            )
        revision = ref_path.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                "Parakeet HF model ref must contain a 40-character commit SHA: "
                f"{ref_path}"
            )

        resolved = PARAKEET_HF_MODEL_CACHE_ROOT / "snapshots" / revision
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Parakeet HF model snapshot does not exist: {resolved}"
            )

    for relative_path in PARAKEET_HF_REQUIRED_MODEL_FILES:
        required_path = resolved / relative_path
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Parakeet HF model file does not exist: {required_path}"
            )
    return resolved.resolve()


def parakeet_sherpa_quantization(model_name: str) -> str:
    try:
        return PARAKEET_SHERPA_MODEL_QUANTIZATION[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Parakeet Sherpa model: {model_name}") from exc


def resolve_parakeet_sherpa_model_path(
    model_name: str, model_path: Path | None = None
) -> Path:
    """Resolve Parakeet Sherpa artifacts and validate the selected variant layout."""
    quantization = parakeet_sherpa_quantization(model_name)
    if model_path is None:
        resolved = PARAKEET_SHERPA_DEFAULT_MODEL_PATH
    else:
        resolved = Path(model_path).expanduser()
    if not resolved.is_dir():
        raise ValueError(
            f"Parakeet Sherpa model path is not a directory: {resolved}"
        )

    for relative_path in PARAKEET_SHERPA_REQUIRED_MODEL_FILES[quantization]:
        required_path = resolved / relative_path
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Parakeet Sherpa {quantization} model file does not exist: "
                f"{required_path}"
            )
    return resolved.resolve()


def summarize_text(text: str) -> tuple[int, int]:
    return len(text), len(text.split())


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split()).strip()


def load_reference_transcript(reference_path: Path | None) -> str | None:
    if reference_path is None:
        return None
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference transcript does not exist: {reference_path}"
        )
    if not reference_path.is_file():
        raise ValueError(f"Reference transcript is not a file: {reference_path}")
    return normalize_transcript(reference_path.read_text(encoding="utf-8"))


def get_audio_duration_seconds(audio_path: Path) -> float:
    import wave

    if audio_path.suffix.lower() == ".wav":
        with wave.open(str(audio_path), "rb") as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
        if sample_rate <= 0:
            raise ValueError(f"Invalid WAV sample rate in {audio_path}")
        return frames / sample_rate

    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.samplerate <= 0:
            raise ValueError(f"Invalid sample rate in {audio_path}")
        return info.frames / info.samplerate
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is required to compute audio duration for non-WAV files"
        ) from exc


def resolve_insanely_fast_whisper_device(
    requested_device_id: str,
) -> tuple[str, bool, str]:
    if requested_device_id == "mps":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps", True, requested_device_id
        except ImportError:
            pass
        return "cpu", False, "cpu"

    if requested_device_id == "cpu":
        return "cpu", False, "cpu"

    return f"cuda:{requested_device_id}", False, requested_device_id


def compute_word_error_rate(reference: str, hypothesis: str) -> float:
    return jiwer.wer(reference, hypothesis)


def compute_character_error_rate(reference: str, hypothesis: str) -> float:
    return jiwer.cer(reference, hypothesis)


def score_transcript(
    transcript: str, reference_transcript: str | None
) -> tuple[float | None, float | None]:
    if reference_transcript is None:
        return None, None
    normalized_transcript = normalize_transcript(transcript)
    return (
        compute_word_error_rate(reference_transcript, normalized_transcript),
        compute_character_error_rate(reference_transcript, normalized_transcript),
    )


def build_run_result(
    *,
    backend: str,
    model_name: str,
    run_index: int,
    load_seconds: float | None,
    transcribe_seconds: float,
    transcript: str,
    detected_language: str | None,
    detected_language_probability: float | None,
    reference_transcript: str | None,
    audio_path: Path,
    sample_label: str,
    audio_duration_seconds: float,
    forced_language: str | None,
    total_seconds: float | None = None,
    peak_rss_mb: float | None = None,
) -> RunResult:
    chars, words = summarize_text(transcript)
    wer, cer = score_transcript(transcript, reference_transcript)
    return RunResult(
        audio=str(audio_path),
        sample_label=sample_label,
        audio_duration_seconds=audio_duration_seconds,
        forced_language=forced_language,
        backend=backend,
        model=model_name,
        backend_device=None,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        total_seconds=(
            total_seconds
            if total_seconds is not None
            else (load_seconds or 0.0) + transcribe_seconds
        ),
        transcript=transcript,
        transcript_chars=chars,
        transcript_words=words,
        wer=wer,
        cer=cer,
        detected_language=detected_language,
        detected_language_probability=detected_language_probability,
        status="ok",
        error=None,
        peak_rss_mb=peak_rss_mb,
    )


def build_error_result(
    *,
    backend: str,
    model_name: str,
    backend_device: str | None,
    run_index: int,
    error: str,
    audio_path: Path,
    sample_label: str,
    audio_duration_seconds: float,
    forced_language: str | None,
    peak_rss_mb: float | None = None,
) -> RunResult:
    return RunResult(
        audio=str(audio_path),
        sample_label=sample_label,
        audio_duration_seconds=audio_duration_seconds,
        forced_language=forced_language,
        backend=backend,
        model=model_name,
        backend_device=backend_device,
        run_index=run_index,
        load_seconds=None,
        transcribe_seconds=None,
        total_seconds=None,
        transcript=None,
        transcript_chars=None,
        transcript_words=None,
        wer=None,
        cer=None,
        detected_language=None,
        detected_language_probability=None,
        status="error",
        error=error,
        peak_rss_mb=peak_rss_mb,
    )


def args_for_audio_input(
    args: argparse.Namespace, audio_input: ResolvedAudioInput
) -> argparse.Namespace:
    per_audio_args = copy.copy(args)
    per_audio_args.language = audio_input.forced_language
    per_audio_args.reference_transcript = audio_input.reference_transcript_path
    per_audio_args.reference_transcript_text = audio_input.reference_transcript_text
    return per_audio_args


def hallucination_silence_threshold_for_backend(
    backend: str, args: argparse.Namespace
) -> float | None:
    capabilities = BACKEND_CAPABILITIES[backend]
    if not capabilities.supports_hallucination_silence_threshold:
        return None
    return args.hallucination_silence_threshold or None


def run_faster_whisper(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    transcribe_started = time.perf_counter()
    hal_threshold = hallucination_silence_threshold_for_backend("faster-whisper", args)
    segments, info = session.transcribe(
        str(audio_path),
        beam_size=args.beam_size,
        language=args.language,
        task=args.task,
        vad_filter=args.faster_whisper_vad_filter,
        condition_on_previous_text=args.condition_on_previous_text,
        word_timestamps=hal_threshold is not None,
        hallucination_silence_threshold=hal_threshold,
    )
    # faster-whisper yields segments lazily, so timing must include iteration.
    transcript = "".join(segment.text for segment in segments).strip()
    transcribe_seconds = time.perf_counter() - transcribe_started
    return build_run_result(
        backend="faster-whisper",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=getattr(info, "language", None),
        detected_language_probability=getattr(info, "language_probability", None),
        reference_transcript=args.reference_transcript_text,
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=args.language,
    )


def run_mlx_whisper(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    import mlx_whisper

    transcribe_started = time.perf_counter()
    hal_threshold = hallucination_silence_threshold_for_backend("mlx-whisper", args)
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=session["model_repo"],
        language=args.language,
        task=args.task,
        condition_on_previous_text=args.condition_on_previous_text,
        word_timestamps=hal_threshold is not None,
        hallucination_silence_threshold=hal_threshold,
        fp16=True,
        verbose=False,
    )
    transcribe_seconds = time.perf_counter() - transcribe_started
    transcript = (result.get("text") or "").strip()
    return build_run_result(
        backend="mlx-whisper",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=result.get("language"),
        detected_language_probability=result.get("language_probability"),
        reference_transcript=args.reference_transcript_text,
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=args.language,
    )


def format_gigaam_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def run_gigaam(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated GigaAM process and map its JSON contract to RunResult."""
    forced_language = getattr(args, "language", "ru")
    if model_name not in GIGAAM_MODELS:
        return build_error_result(
            backend="gigaam",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error=f"Unsupported GigaAM model: {model_name}",
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )
    try:
        execution = run_json_worker(
            getattr(args, "gigaam_python", DEFAULT_GIGAAM_PYTHON),
            GIGAAM_WORKER_MODULE,
            {
                "model_path": str(model_path),
                "audio_path": str(audio_path),
                "language": "ru",
                "variant": model_name,
            },
            getattr(args, "worker_timeout_seconds", 900.0),
            env=dict(GIGAAM_OFFLINE_ENV),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="gigaam",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload["transcript"]
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="gigaam",
                model_name=model_name,
                backend_device="subprocess",
                run_index=run_index,
                error=f"invalid GigaAM worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        result = build_run_result(
            backend="gigaam",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=float(execution.wall_seconds),
            transcript=transcript.strip(),
            detected_language=payload.get("language"),
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = "subprocess"
        return result

    return build_error_result(
        backend="gigaam",
        model_name=model_name,
        backend_device="subprocess",
        run_index=run_index,
        error=format_gigaam_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_gigaam_multilingual_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def gigaam_multilingual_language_hint(
    configured_override: str | None, resolved_language: str | None
) -> str | None:
    if configured_override is not None:
        return configured_override
    if resolved_language and resolved_language != "auto":
        return resolved_language
    return None


def run_gigaam_multilingual(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated GigaAM Multilingual process and map its JSON contract."""
    forced_language = getattr(args, "language", None)
    request: dict[str, Any] = {
        "model_path": str(model_path),
        "audio_path": str(audio_path),
        "variant": GIGAAM_MULTILINGUAL_MODEL_VARIANT,
    }
    language = gigaam_multilingual_language_hint(
        getattr(args, "gigaam_multilingual_language", None), forced_language
    )
    if language is not None:
        request["language"] = language

    try:
        execution = run_json_worker(
            getattr(
                args,
                "gigaam_multilingual_python",
                DEFAULT_GIGAAM_MULTILINGUAL_PYTHON,
            ),
            GIGAAM_MULTILINGUAL_WORKER_MODULE,
            request,
            getattr(args, "worker_timeout_seconds", 900.0),
            env=dict(GIGAAM_MULTILINGUAL_OFFLINE_ENV),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="gigaam-multilingual",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload.get("transcript", payload.get("text"))
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
            total_seconds = float(execution.wall_seconds)
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="gigaam-multilingual",
                model_name=model_name,
                backend_device="subprocess",
                run_index=run_index,
                error=f"invalid GigaAM Multilingual worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        detected_language = payload.get("language")
        if "language" not in payload and "detected_language" in payload:
            detected_language = payload["detected_language"]
        result = build_run_result(
            backend="gigaam-multilingual",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=total_seconds,
            transcript=transcript.strip(),
            detected_language=detected_language,
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = "subprocess"
        return result

    return build_error_result(
        backend="gigaam-multilingual",
        model_name=model_name,
        backend_device="subprocess",
        run_index=run_index,
        error=format_gigaam_multilingual_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_gigaam_multilingual_mlx_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def run_gigaam_multilingual_mlx(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated GigaAM Multilingual MLX process."""
    forced_language = getattr(args, "language", None)
    if model_name not in GIGAAM_MULTILINGUAL_MLX_MODELS:
        return build_error_result(
            backend="gigaam-multilingual-mlx",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error=f"Unsupported GigaAM Multilingual MLX model: {model_name}",
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )
    if getattr(args, "task", "transcribe") != "transcribe":
        return build_error_result(
            backend="gigaam-multilingual-mlx",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error="GigaAM Multilingual MLX supports only the transcribe task",
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    request: dict[str, Any] = {
        "model_path": str(model_path),
        "audio_path": str(audio_path),
        "variant": GIGAAM_MULTILINGUAL_MLX_MODEL_VARIANT,
        "chunk_seconds": getattr(
            args,
            "gigaam_multilingual_mlx_chunk_seconds",
            DEFAULT_GIGAAM_MULTILINGUAL_MLX_CHUNK_SECONDS,
        ),
        "overlap_seconds": getattr(
            args,
            "gigaam_multilingual_mlx_overlap_seconds",
            DEFAULT_GIGAAM_MULTILINGUAL_MLX_OVERLAP_SECONDS,
        ),
    }
    if forced_language and forced_language != "auto":
        request["language"] = forced_language

    try:
        execution = run_json_worker(
            getattr(
                args,
                "gigaam_multilingual_mlx_python",
                DEFAULT_GIGAAM_MULTILINGUAL_MLX_PYTHON,
            ),
            GIGAAM_MULTILINGUAL_MLX_WORKER_MODULE,
            request,
            getattr(args, "worker_timeout_seconds", 900.0),
            env=dict(GIGAAM_MULTILINGUAL_MLX_OFFLINE_ENV),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="gigaam-multilingual-mlx",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload.get("transcript", payload.get("text"))
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="gigaam-multilingual-mlx",
                model_name=model_name,
                backend_device="subprocess",
                run_index=run_index,
                error=f"invalid GigaAM Multilingual MLX worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        result = build_run_result(
            backend="gigaam-multilingual-mlx",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=float(execution.wall_seconds),
            transcript=transcript.strip(),
            detected_language=payload.get("language", payload.get("detected_language")),
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = "subprocess"
        return result

    return build_error_result(
        backend="gigaam-multilingual-mlx",
        model_name=model_name,
        backend_device="subprocess",
        run_index=run_index,
        error=format_gigaam_multilingual_mlx_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_vibevoice_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def run_vibevoice(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated VibeVoice ASR process."""
    forced_language = getattr(args, "language", None)
    if model_name not in VIBEVOICE_MODELS:
        return build_error_result(
            backend="vibevoice",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error=f"Unsupported VibeVoice model: {model_name}",
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )
    if getattr(args, "task", "transcribe") != "transcribe":
        return build_error_result(
            backend="vibevoice",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error="VibeVoice supports only the transcribe task",
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    request: dict[str, Any] = {
        "model_path": str(model_path),
        "audio_path": str(audio_path),
        "device": getattr(args, "vibevoice_device", DEFAULT_VIBEVOICE_DEVICE),
        "mode": getattr(args, "vibevoice_mode", DEFAULT_VIBEVOICE_MODE),
    }
    chunk_size = getattr(args, "vibevoice_acoustic_tokenizer_chunk_size", None)
    if chunk_size is not None:
        request["acoustic_tokenizer_chunk_size"] = chunk_size

    try:
        execution = run_json_worker(
            getattr(args, "vibevoice_python", DEFAULT_VIBEVOICE_PYTHON),
            VIBEVOICE_WORKER_MODULE,
            request,
            getattr(args, "worker_timeout_seconds", 900.0),
            env=dict(VIBEVOICE_OFFLINE_ENV),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="vibevoice",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload.get("transcript", payload.get("text"))
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="vibevoice",
                model_name=model_name,
                backend_device="subprocess",
                run_index=run_index,
                error=f"invalid VibeVoice worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        result = build_run_result(
            backend="vibevoice",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=float(execution.wall_seconds),
            transcript=transcript.strip(),
            detected_language=payload.get("language", payload.get("detected_language")),
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = "subprocess"
        return result

    return build_error_result(
        backend="vibevoice",
        model_name=model_name,
        backend_device="subprocess",
        run_index=run_index,
        error=format_vibevoice_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_tone_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def run_tone(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated T-one process and map its JSON contract to RunResult."""
    forced_language = getattr(args, "language", "ru")
    try:
        execution = run_json_worker(
            getattr(args, "tone_python", DEFAULT_TONE_PYTHON),
            TONE_WORKER_MODULE,
            {
                "model_path": str(model_path),
                "audio_path": str(audio_path),
                "decoder": getattr(args, "tone_decoder", "greedy"),
                "streaming": False,
            },
            getattr(args, "worker_timeout_seconds", 900.0),
            env=dict(TONE_OFFLINE_ENV),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="t-one",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload["transcript"]
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
            total_seconds = float(execution.wall_seconds)
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="t-one",
                model_name=model_name,
                backend_device="subprocess",
                run_index=run_index,
                error=f"invalid T-one worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        result = build_run_result(
            backend="t-one",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=total_seconds,
            transcript=transcript.strip(),
            detected_language=payload.get("language"),
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = "subprocess"
        return result

    return build_error_result(
        backend="t-one",
        model_name=model_name,
        backend_device="subprocess",
        run_index=run_index,
        error=format_tone_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_vosk_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def run_vosk(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated Sherpa-ONNX Vosk process and map its JSON contract."""
    forced_language = getattr(args, "language", "ru")
    try:
        execution = run_json_worker(
            getattr(args, "vosk_python", DEFAULT_VOSK_PYTHON),
            VOSK_WORKER_MODULE,
            {
                "model_path": str(model_path),
                "audio_path": str(audio_path),
                "decoding_method": getattr(
                    args, "vosk_decoding_method", VOSK_DEFAULT_DECODING_METHOD
                ),
                "quantization": "fp32",
                "streaming": False,
            },
            getattr(args, "worker_timeout_seconds", 900.0),
            env=dict(VOSK_OFFLINE_ENV),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="vosk",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload["transcript"]
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
            total_seconds = float(execution.wall_seconds)
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="vosk",
                model_name=model_name,
                backend_device="subprocess",
                run_index=run_index,
                error=f"invalid Vosk worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        result = build_run_result(
            backend="vosk",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=total_seconds,
            transcript=transcript.strip(),
            detected_language=payload.get("language"),
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = "subprocess"
        return result

    return build_error_result(
        backend="vosk",
        model_name=model_name,
        backend_device="subprocess",
        run_index=run_index,
        error=format_vosk_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_qwen3_asr_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def qwen3_asr_language_hint(
    configured_override: str | None, resolved_language: str | None
) -> str | None:
    if configured_override is not None:
        return configured_override
    if resolved_language and resolved_language != "auto":
        return resolved_language
    return None


def run_qwen3_asr(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated Qwen3-ASR process and map its JSON contract."""
    forced_language = getattr(args, "language", None)
    request: dict[str, Any] = {
        "model_path": str(model_path),
        "audio_path": str(audio_path),
        "max_tokens": getattr(args, "qwen3_asr_max_tokens", 8192),
        "temperature": getattr(args, "qwen3_asr_temperature", 0.0),
    }
    qwen_language = qwen3_asr_language_hint(
        getattr(args, "qwen3_asr_language", None), forced_language
    )
    if qwen_language is not None:
        request["language"] = qwen_language

    try:
        execution = run_json_worker(
            getattr(args, "qwen3_asr_python", DEFAULT_QWEN3_ASR_PYTHON),
            QWEN3_ASR_WORKER_MODULE,
            request,
            getattr(args, "worker_timeout_seconds", 900.0),
            env=dict(QWEN3_ASR_OFFLINE_ENV),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="qwen3-asr",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload.get("text", payload.get("transcript"))
            if not isinstance(transcript, str):
                raise TypeError("worker text must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
            total_seconds = float(execution.wall_seconds)
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="qwen3-asr",
                model_name=model_name,
                backend_device="subprocess",
                run_index=run_index,
                error=f"invalid Qwen3-ASR worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        detected_language = payload.get("language")
        if "language" not in payload and "detected_language" in payload:
            detected_language = payload["detected_language"]
        result = build_run_result(
            backend="qwen3-asr",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=total_seconds,
            transcript=transcript.strip(),
            detected_language=detected_language,
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = "subprocess"
        return result

    return build_error_result(
        backend="qwen3-asr",
        model_name=model_name,
        backend_device="subprocess",
        run_index=run_index,
        error=format_qwen3_asr_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_qwen3_asr_hf_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def qwen3_asr_hf_language_hint(resolved_language: str | None) -> str | None:
    if resolved_language and resolved_language != "auto":
        return resolved_language
    return None


def run_qwen3_asr_hf(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated official Qwen3-ASR Transformers process."""
    forced_language = getattr(args, "language", None)
    if model_name not in QWEN3_ASR_HF_MODELS:
        return build_error_result(
            backend="qwen3-asr-hf",
            model_name=model_name,
            backend_device=QWEN3_ASR_HF_BACKEND_DEVICE,
            run_index=run_index,
            error=f"Unsupported Qwen3-ASR HF model: {model_name}",
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )
    if getattr(args, "task", "transcribe") != "transcribe":
        return build_error_result(
            backend="qwen3-asr-hf",
            model_name=model_name,
            backend_device=QWEN3_ASR_HF_BACKEND_DEVICE,
            run_index=run_index,
            error="Qwen3-ASR HF supports only the transcribe task",
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    request: dict[str, Any] = {
        "model_path": str(model_path),
        "audio_path": str(audio_path),
        "device": getattr(
            args, "qwen3_asr_hf_device", DEFAULT_QWEN3_ASR_HF_DEVICE
        ),
        "max_new_tokens": getattr(
            args,
            "qwen3_asr_hf_max_tokens",
            getattr(
                args,
                "qwen3_asr_hf_max_new_tokens",
                QWEN3_ASR_HF_DEFAULT_MAX_TOKENS,
            ),
        ),
    }
    language = qwen3_asr_hf_language_hint(forced_language)
    if language is not None:
        request["language"] = language

    try:
        execution = run_json_worker(
            getattr(args, "qwen3_asr_hf_python", DEFAULT_QWEN3_ASR_HF_PYTHON),
            QWEN3_ASR_HF_WORKER_MODULE,
            request,
            getattr(args, "worker_timeout_seconds", 900.0),
            env=dict(QWEN3_ASR_HF_OFFLINE_ENV),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="qwen3-asr-hf",
            model_name=model_name,
            backend_device=QWEN3_ASR_HF_BACKEND_DEVICE,
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload.get("transcript", payload.get("text"))
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
            total_seconds = float(execution.wall_seconds)
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="qwen3-asr-hf",
                model_name=model_name,
                backend_device=QWEN3_ASR_HF_BACKEND_DEVICE,
                run_index=run_index,
                error=f"invalid Qwen3-ASR HF worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        result = build_run_result(
            backend="qwen3-asr-hf",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=total_seconds,
            transcript=transcript.strip(),
            detected_language=payload.get("language", payload.get("detected_language")),
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = QWEN3_ASR_HF_BACKEND_DEVICE
        return result

    return build_error_result(
        backend="qwen3-asr-hf",
        model_name=model_name,
        backend_device=QWEN3_ASR_HF_BACKEND_DEVICE,
        run_index=run_index,
        error=format_qwen3_asr_hf_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_whisper_cpp_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def whisper_cpp_language_hint(resolved_language: str | None) -> str | None:
    if resolved_language and resolved_language != "auto":
        return resolved_language
    return None


def run_whisper_cpp(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one official whisper.cpp CLI invocation through its JSON worker."""
    forced_language = getattr(args, "language", None)
    timeout_seconds = getattr(args, "worker_timeout_seconds", 900.0)
    request: dict[str, Any] = {
        "executable": str(
            getattr(args, "whisper_cpp_executable", DEFAULT_WHISPER_CPP_EXECUTABLE)
        ),
        "model_path": str(model_path),
        "audio_path": str(audio_path),
        "task": getattr(args, "task", "transcribe"),
        "beam_size": getattr(args, "beam_size", 5),
        "threads": getattr(args, "whisper_cpp_threads", WHISPER_CPP_DEFAULT_THREADS),
        "timeout_seconds": timeout_seconds,
    }
    language = whisper_cpp_language_hint(forced_language)
    if language is not None:
        request["language"] = language

    try:
        execution = run_json_worker(
            sys.executable,
            WHISPER_CPP_WORKER_MODULE,
            request,
            timeout_seconds,
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="whisper-cpp",
            model_name=model_name,
            backend_device=WHISPER_CPP_BACKEND_DEVICE,
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload["transcript"]
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            transcribe_seconds = float(payload["transcribe_seconds"])
            total_seconds = float(execution.wall_seconds)
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="whisper-cpp",
                model_name=model_name,
                backend_device=WHISPER_CPP_BACKEND_DEVICE,
                run_index=run_index,
                error=f"invalid whisper.cpp worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        result = build_run_result(
            backend="whisper-cpp",
            model_name=model_name,
            run_index=run_index,
            load_seconds=None,
            transcribe_seconds=transcribe_seconds,
            total_seconds=total_seconds,
            transcript=transcript.strip(),
            detected_language=payload.get("language"),
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = WHISPER_CPP_BACKEND_DEVICE
        return result

    return build_error_result(
        backend="whisper-cpp",
        model_name=model_name,
        backend_device=WHISPER_CPP_BACKEND_DEVICE,
        run_index=run_index,
        error=format_whisper_cpp_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_podlodka_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def podlodka_language_hint(
    configured_override: str | None, resolved_language: str | None
) -> str | None:
    if configured_override is not None:
        return None if configured_override == "auto" else configured_override
    if resolved_language and resolved_language != "auto":
        return resolved_language
    return None


def run_podlodka(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated local Podlodka process and map its JSON contract."""
    forced_language = getattr(args, "language", None)
    request: dict[str, Any] = {
        "model_path": str(model_path),
        "audio_path": str(audio_path),
        "max_new_tokens": getattr(
            args, "podlodka_max_new_tokens", PODLODKA_DEFAULT_MAX_NEW_TOKENS
        ),
    }
    language = podlodka_language_hint(
        getattr(args, "podlodka_language", None), forced_language
    )
    if language is not None:
        request["language"] = language

    try:
        execution = run_json_worker(
            getattr(args, "podlodka_python", DEFAULT_PODLODKA_PYTHON),
            PODLODKA_WORKER_MODULE,
            request,
            getattr(args, "worker_timeout_seconds", 900.0),
            env=dict(PODLODKA_OFFLINE_ENV),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="podlodka",
            model_name=model_name,
            backend_device="subprocess",
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload.get("transcript", payload.get("text"))
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
            total_seconds = float(execution.wall_seconds)
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="podlodka",
                model_name=model_name,
                backend_device="subprocess",
                run_index=run_index,
                error=f"invalid Podlodka worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        detected_language = payload.get("language")
        if "language" not in payload and "detected_language" in payload:
            detected_language = payload["detected_language"]
        result = build_run_result(
            backend="podlodka",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=total_seconds,
            transcript=transcript.strip(),
            detected_language=detected_language,
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = "subprocess"
        return result

    return build_error_result(
        backend="podlodka",
        model_name=model_name,
        backend_device="subprocess",
        run_index=run_index,
        error=format_podlodka_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_parakeet_hf_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def run_parakeet_hf(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated official Parakeet Transformers process."""
    forced_language = getattr(args, "language", None)
    request = {
        "model_path": str(model_path),
        "audio_path": str(audio_path),
        "device": getattr(args, "parakeet_hf_device", DEFAULT_PARAKEET_HF_DEVICE),
    }
    try:
        execution = run_json_worker(
            getattr(args, "parakeet_hf_python", DEFAULT_PARAKEET_HF_PYTHON),
            PARAKEET_HF_WORKER_MODULE,
            request,
            getattr(args, "worker_timeout_seconds", 900.0),
            env=dict(PARAKEET_HF_OFFLINE_ENV),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="parakeet-hf",
            model_name=model_name,
            backend_device=PARAKEET_BACKEND_DEVICE,
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload["transcript"]
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
            total_seconds = float(execution.wall_seconds)
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="parakeet-hf",
                model_name=model_name,
                backend_device=PARAKEET_BACKEND_DEVICE,
                run_index=run_index,
                error=f"invalid Parakeet HF worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        result = build_run_result(
            backend="parakeet-hf",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=total_seconds,
            transcript=transcript.strip(),
            detected_language=payload.get(
                "language", payload.get("detected_language")
            ),
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = PARAKEET_BACKEND_DEVICE
        return result

    return build_error_result(
        backend="parakeet-hf",
        model_name=model_name,
        backend_device=PARAKEET_BACKEND_DEVICE,
        run_index=run_index,
        error=format_parakeet_hf_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def format_parakeet_sherpa_worker_error(execution: Any) -> str:
    payload = getattr(execution, "payload", None)
    details: list[str] = []
    if isinstance(payload, dict) and payload.get("error"):
        details.append(str(payload["error"]))
    if getattr(execution, "error", None):
        details.append(str(execution.error))
    if not details:
        details.append(f"worker returned status {getattr(execution, 'status', 'unknown')}")
    return "; ".join(dict.fromkeys(details))


def run_parakeet_sherpa(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    model_path: Path,
) -> RunResult:
    """Run one isolated Parakeet Sherpa-ONNX process."""
    forced_language = getattr(args, "language", None)
    try:
        request = {
            "model_path": str(model_path),
            "audio_path": str(audio_path),
            "quantization": parakeet_sherpa_quantization(model_name),
            "threads": getattr(
                args, "parakeet_sherpa_threads", DEFAULT_PARAKEET_SHERPA_THREADS
            ),
        }
        execution = run_json_worker(
            getattr(args, "parakeet_sherpa_python", DEFAULT_PARAKEET_SHERPA_PYTHON),
            PARAKEET_SHERPA_WORKER_MODULE,
            request,
            getattr(args, "worker_timeout_seconds", 900.0),
        )
    except Exception as exc:  # pragma: no cover - defensive harness boundary.
        return build_error_result(
            backend="parakeet-sherpa",
            model_name=model_name,
            backend_device=PARAKEET_BACKEND_DEVICE,
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
        )

    payload = getattr(execution, "payload", None)
    if (
        getattr(execution, "status", None) == "ok"
        and isinstance(payload, dict)
        and payload.get("status", "ok") == "ok"
    ):
        try:
            transcript = payload["transcript"]
            if not isinstance(transcript, str):
                raise TypeError("worker transcript must be a string")
            load_value = payload.get("load_seconds")
            load_seconds = None if load_value is None else float(load_value)
            transcribe_seconds = float(payload["transcribe_seconds"])
            total_seconds = float(execution.wall_seconds)
        except (KeyError, TypeError, ValueError) as exc:
            return build_error_result(
                backend="parakeet-sherpa",
                model_name=model_name,
                backend_device=PARAKEET_BACKEND_DEVICE,
                run_index=run_index,
                error=f"invalid Parakeet Sherpa worker payload: {exc}",
                audio_path=audio_path,
                sample_label=args.sample_label,
                audio_duration_seconds=args.audio_duration_seconds,
                forced_language=forced_language,
                peak_rss_mb=getattr(execution, "peak_rss_mb", None),
            )

        result = build_run_result(
            backend="parakeet-sherpa",
            model_name=model_name,
            run_index=run_index,
            load_seconds=load_seconds,
            transcribe_seconds=transcribe_seconds,
            total_seconds=total_seconds,
            transcript=transcript.strip(),
            detected_language=payload.get(
                "language", payload.get("detected_language")
            ),
            detected_language_probability=payload.get("language_probability"),
            reference_transcript=getattr(args, "reference_transcript_text", None),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=forced_language,
            peak_rss_mb=getattr(execution, "peak_rss_mb", None),
        )
        result.backend_device = PARAKEET_BACKEND_DEVICE
        return result

    return build_error_result(
        backend="parakeet-sherpa",
        model_name=model_name,
        backend_device=PARAKEET_BACKEND_DEVICE,
        run_index=run_index,
        error=format_parakeet_sherpa_worker_error(execution),
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=forced_language,
        peak_rss_mb=getattr(execution, "peak_rss_mb", None),
    )


def run_insanely_fast_whisper(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    from insanely_fast_whisper.utils.result import build_result

    transcribe_started = time.perf_counter()
    outputs = session["pipe"](
        str(audio_path),
        chunk_length_s=30,
        ignore_warning=True,
        batch_size=args.insanely_fast_whisper_batch_size,
        generate_kwargs=session["generate_kwargs"],
        return_timestamps=True,
        return_language=True,
    )
    transcribe_seconds = time.perf_counter() - transcribe_started

    result = build_result([], outputs)
    transcript = (result.get("text") or "").strip()
    return build_run_result(
        backend="insanely-fast-whisper",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=outputs.get("language") or result.get("language"),
        detected_language_probability=None,
        reference_transcript=args.reference_transcript_text,
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=args.language,
    )


def run_mlx_audio(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    from mlx_audio.stt.generate import generate_transcription

    transcribe_started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_transcription(
            model=session,
            audio=str(audio_path),
            output_path=str(Path(tmpdir) / "transcript"),
            language=args.language,
            task=args.task,
            beam_size=args.beam_size,
            condition_on_previous_text=args.condition_on_previous_text,
        )
    transcribe_seconds = time.perf_counter() - transcribe_started
    if isinstance(result, dict):
        transcript = (result.get("text") or "").strip()
        detected_language = result.get("language")
    else:
        transcript = (getattr(result, "text", None) or "").strip()
        detected_language = getattr(result, "language", None)

    return build_run_result(
        backend="mlx-audio",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=detected_language,
        detected_language_probability=None,
        reference_transcript=args.reference_transcript_text,
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=args.language,
    )


def run_lightning_whisper_mlx(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    from lightning_whisper_mlx.transcribe import transcribe_audio

    hal_threshold = hallucination_silence_threshold_for_backend(
        "lightning-whisper-mlx", args
    )
    transcribe_started = time.perf_counter()
    result = transcribe_audio(
        str(audio_path),
        path_or_hf_repo=session["model_path"],
        language=args.language,
        task=args.task,
        condition_on_previous_text=args.condition_on_previous_text,
        batch_size=args.lightning_whisper_mlx_batch_size,
        word_timestamps=hal_threshold is not None,
        hallucination_silence_threshold=hal_threshold,
        fp16=True,
        verbose=False,
    )
    transcribe_seconds = time.perf_counter() - transcribe_started
    transcript = (result.get("text") or "").strip()
    return build_run_result(
        backend="lightning-whisper-mlx",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=result.get("language"),
        detected_language_probability=None,
        reference_transcript=args.reference_transcript_text,
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=args.language,
    )


def run_openai_whisper(
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    session: Any,
    load_seconds: float | None,
) -> RunResult:
    temperature = (
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        if args.openai_whisper_temperature_fallback
        else 0.0
    )
    hal_threshold = hallucination_silence_threshold_for_backend("openai-whisper", args)
    transcribe_started = time.perf_counter()
    result = session.transcribe(
        str(audio_path),
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        temperature=temperature,
        best_of=None,
        condition_on_previous_text=args.condition_on_previous_text,
        word_timestamps=hal_threshold is not None,
        hallucination_silence_threshold=hal_threshold,
        verbose=False,
    )
    transcribe_seconds = time.perf_counter() - transcribe_started
    transcript = (result.get("text") or "").strip()
    detected_language = result.get("language")
    detected_language_probability = None
    if args.language is None and detected_language is not None:
        try:
            from whisper.audio import log_mel_spectrogram, pad_or_trim

            mel_segment = pad_or_trim(
                log_mel_spectrogram(str(audio_path), session.dims.n_mels)
            ).to(session.device)
            _, probs = session.detect_language(mel_segment)
            detected_language_probability = probs.get(detected_language)
        except Exception:
            detected_language_probability = None
    return build_run_result(
        backend="openai-whisper",
        model_name=model_name,
        run_index=run_index,
        load_seconds=load_seconds,
        transcribe_seconds=transcribe_seconds,
        transcript=transcript,
        detected_language=detected_language,
        detected_language_probability=detected_language_probability,
        reference_transcript=args.reference_transcript_text,
        audio_path=audio_path,
        sample_label=args.sample_label,
        audio_duration_seconds=args.audio_duration_seconds,
        forced_language=args.language,
    )


def load_backend_session(
    backend: str, model_name: str, args: argparse.Namespace
) -> BackendSession:
    if backend == "whisper-cpp":
        model_path = resolve_whisper_cpp_model_path(model_name)
        return BackendSession(
            backend,
            model_name,
            WHISPER_CPP_BACKEND_DEVICE,
            {"model_path": model_path},
            None,
        )

    if backend == "gigaam":
        model_path = resolve_gigaam_model_path(
            model_name,
            getattr(args, "gigaam_model_path", None),
        )
        return BackendSession(
            backend,
            model_name,
            "subprocess",
            {"model_path": model_path},
            None,
        )

    if backend == "gigaam-multilingual":
        model_path = resolve_gigaam_multilingual_model_path(
            getattr(args, "gigaam_multilingual_model_path", None)
        )
        return BackendSession(
            backend,
            model_name,
            "subprocess",
            {"model_path": model_path},
            None,
        )

    if backend == "gigaam-multilingual-mlx":
        model_path = resolve_gigaam_multilingual_mlx_model_path(
            getattr(args, "gigaam_multilingual_mlx_model_path", None)
        )
        return BackendSession(
            backend,
            model_name,
            "subprocess",
            {"model_path": model_path},
            None,
        )

    if backend == "vibevoice":
        model_path = resolve_vibevoice_model_path(
            getattr(args, "vibevoice_model_path", None)
        )
        return BackendSession(
            backend,
            model_name,
            "subprocess",
            {"model_path": model_path},
            None,
        )

    if backend == "t-one":
        model_path = resolve_tone_model_path(
            getattr(args, "tone_model_path", None)
        )
        return BackendSession(
            backend,
            model_name,
            "subprocess",
            {"model_path": model_path},
            None,
        )

    if backend == "vosk":
        model_path = resolve_vosk_model_path(
            model_name,
            getattr(args, "vosk_model_path", None),
        )
        return BackendSession(
            backend,
            model_name,
            "subprocess",
            {"model_path": model_path},
            None,
        )

    if backend == "qwen3-asr-hf":
        model_path = resolve_qwen3_asr_hf_model_path(
            getattr(args, "qwen3_asr_hf_model_path", None)
        )
        return BackendSession(
            backend,
            model_name,
            QWEN3_ASR_HF_BACKEND_DEVICE,
            {"model_path": model_path},
            None,
        )

    if backend == "parakeet-hf":
        model_path = resolve_parakeet_hf_model_path(
            getattr(args, "parakeet_hf_model_path", None)
        )
        return BackendSession(
            backend,
            model_name,
            PARAKEET_BACKEND_DEVICE,
            {"model_path": model_path},
            None,
        )

    if backend == "parakeet-sherpa":
        model_path = resolve_parakeet_sherpa_model_path(
            model_name,
            getattr(args, "parakeet_sherpa_model_path", None),
        )
        return BackendSession(
            backend,
            model_name,
            PARAKEET_BACKEND_DEVICE,
            {"model_path": model_path},
            None,
        )

    if backend == "faster-whisper":
        from faster_whisper import WhisperModel

        load_started = time.perf_counter()
        session = WhisperModel(
            model_name,
            device=args.device,
            compute_type=args.compute_type,
        )
        resolved_device = str(
            getattr(getattr(session, "model", None), "device", args.device)
        )
        return BackendSession(
            backend,
            model_name,
            resolved_device,
            session,
            time.perf_counter() - load_started,
        )

    if backend == "mlx-whisper":
        import mlx.core as mx
        from mlx_whisper.load_models import load_model as mlx_whisper_load_model
        from mlx_whisper.transcribe import ModelHolder

        model_repo = MLX_WHISPER_REPOS.get(model_name)
        if model_repo is None:
            raise ValueError(f"Unsupported mlx-whisper model: {model_name}")
        load_started = time.perf_counter()
        ModelHolder.model = mlx_whisper_load_model(model_repo, dtype=mx.float16)
        ModelHolder.model_path = model_repo
        return BackendSession(
            backend,
            model_name,
            "mlx",
            {"model_repo": model_repo},
            time.perf_counter() - load_started,
        )

    if backend == "mlx-audio":
        from mlx_audio.stt.utils import load_model

        model_repo = MLX_AUDIO_WHISPER_REPOS.get(model_name)
        if model_repo is None:
            raise ValueError(f"Unsupported mlx-audio model: {model_name}")
        load_started = time.perf_counter()
        session = load_model(model_repo)
        return BackendSession(
            backend,
            model_name,
            "mlx",
            session,
            time.perf_counter() - load_started,
        )

    if backend == "lightning-whisper-mlx":
        import mlx.core as mx
        from lightning_whisper_mlx.transcribe import ModelHolder

        if model_name not in LIGHTNING_WHISPER_MLX_REPOS:
            raise ValueError(f"Unsupported lightning-whisper-mlx model: {model_name}")
        load_started = time.perf_counter()
        model_path = snapshot_download(
            repo_id=LIGHTNING_WHISPER_MLX_REPOS[model_name],
            allow_patterns=["config.json", "weights.npz"],
            local_files_only=True,
        )
        ModelHolder.get_model(model_path, dtype=mx.float16)
        return BackendSession(
            backend,
            model_name,
            "mlx",
            {"model_path": model_path},
            time.perf_counter() - load_started,
        )

    if backend == "insanely-fast-whisper":
        import torch
        from transformers import pipeline

        model_repo = INSANELY_FAST_WHISPER_REPOS.get(model_name)
        if model_repo is None:
            raise ValueError(f"Unsupported insanely-fast-whisper model: {model_name}")
        device, should_clear_mps_cache, resolved_device_id = (
            resolve_insanely_fast_whisper_device(args.insanely_fast_whisper_device_id)
        )
        attn = "flash_attention_2" if args.insanely_fast_whisper_flash else "sdpa"
        generate_kwargs = {
            "task": args.task,
            "language": args.language or None,
        }
        if model_repo.endswith(".en"):
            generate_kwargs.pop("task")

        load_started = time.perf_counter()
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_repo,
            dtype=torch.float16,
            device=device,
            model_kwargs={"attn_implementation": attn},
        )
        if should_clear_mps_cache:
            torch.mps.empty_cache()
        return BackendSession(
            backend,
            model_name,
            device,
            {
                "pipe": pipe,
                "generate_kwargs": generate_kwargs,
                "resolved_device_id": resolved_device_id,
            },
            time.perf_counter() - load_started,
        )

    if backend == "openai-whisper":
        import whisper

        whisper_model_name = OPENAI_WHISPER_REPOS.get(model_name)
        if whisper_model_name is None:
            raise ValueError(f"Unsupported openai-whisper model: {model_name}")
        device = args.device if args.device != "auto" else None
        load_started = time.perf_counter()
        session = whisper.load_model(whisper_model_name, device=device)
        return BackendSession(
            backend,
            model_name,
            str(getattr(session, "device", device or "cpu")),
            session,
            time.perf_counter() - load_started,
        )

    raise ValueError(f"Unsupported backend: {backend}")


def run_single_backend(
    backend: str,
    audio_path: Path,
    model_name: str,
    run_index: int,
    args: argparse.Namespace,
    backend_session: BackendSession,
    load_seconds: float | None,
) -> RunResult:
    try:
        session = backend_session.session
        if backend == "faster-whisper":
            result = run_faster_whisper(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        elif backend == "mlx-whisper":
            result = run_mlx_whisper(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        elif backend == "gigaam":
            result = run_gigaam(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "gigaam-multilingual":
            result = run_gigaam_multilingual(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "gigaam-multilingual-mlx":
            result = run_gigaam_multilingual_mlx(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "vibevoice":
            result = run_vibevoice(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "t-one":
            result = run_tone(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "vosk":
            result = run_vosk(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "parakeet-hf":
            result = run_parakeet_hf(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "parakeet-sherpa":
            result = run_parakeet_sherpa(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "qwen3-asr":
            result = run_qwen3_asr(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "qwen3-asr-hf":
            result = run_qwen3_asr_hf(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "whisper-cpp":
            result = run_whisper_cpp(
                audio_path,
                model_name,
                run_index,
                args,
                session["model_path"],
            )
        elif backend == "mlx-audio":
            result = run_mlx_audio(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        elif backend == "lightning-whisper-mlx":
            result = run_lightning_whisper_mlx(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        elif backend == "insanely-fast-whisper":
            result = run_insanely_fast_whisper(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        elif backend == "openai-whisper":
            result = run_openai_whisper(
                audio_path, model_name, run_index, args, session, load_seconds
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        result.backend_device = backend_session.device
        return result
    except (
        Exception
    ) as exc:  # pragma: no cover - benchmark scripts should continue after failures.
        return build_error_result(
            backend=backend,
            model_name=model_name,
            backend_device=backend_session.device,
            run_index=run_index,
            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            audio_path=audio_path,
            sample_label=args.sample_label,
            audio_duration_seconds=args.audio_duration_seconds,
            forced_language=args.language,
        )


def maybe_warmup(
    backend: str,
    audio_path: Path,
    model_name: str,
    args: argparse.Namespace,
    backend_session: BackendSession,
) -> None:
    if not args.warmup:
        return
    warmup_result = run_single_backend(
        backend, audio_path, model_name, 0, args, backend_session, None
    )
    if warmup_result.status != "ok":
        print(
            f"warmup failed for {backend} {model_name}: {warmup_result.error}",
            file=sys.stderr,
        )


def aggregate_results(results: list[RunResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[RunResult]] = {}
    for result in results:
        grouped.setdefault((result.audio, result.backend, result.model), []).append(
            result
        )

    aggregated: list[dict[str, Any]] = []
    for (audio, backend, model), group in sorted(grouped.items()):
        ok_runs = [
            item
            for item in group
            if item.status == "ok" and item.total_seconds is not None
        ]
        total_values = [
            item.total_seconds for item in ok_runs if item.total_seconds is not None
        ]
        load_values = [
            item.load_seconds for item in ok_runs if item.load_seconds is not None
        ]
        transcribe_values = [
            item.transcribe_seconds
            for item in ok_runs
            if item.transcribe_seconds is not None
        ]
        peak_rss_values = [
            item.peak_rss_mb for item in ok_runs if item.peak_rss_mb is not None
        ]
        wer_values = [item.wer for item in ok_runs if item.wer is not None]
        cer_values = [item.cer for item in ok_runs if item.cer is not None]

        aggregated.append(
            {
                "audio": audio,
                "sample_label": group[-1].sample_label,
                "forced_language": group[-1].forced_language,
                "audio_duration_seconds": group[-1].audio_duration_seconds,
                "backend": backend,
                "model": model,
                "backend_device": ok_runs[-1].backend_device if ok_runs else None,
                "runs": len(group),
                "successful_runs": len(ok_runs),
                "failed_runs": len(group) - len(ok_runs),
                "avg_total_seconds": mean_or_none(total_values),
                "median_total_seconds": median_or_none(total_values),
                "min_total_seconds": min(total_values) if total_values else None,
                "max_total_seconds": max(total_values) if total_values else None,
                "load_seconds": mean_or_none(load_values),
                "avg_transcribe_seconds": mean_or_none(transcribe_values),
                "stddev_transcribe_seconds": stdev_or_none(transcribe_values),
                "avg_peak_rss_mb": mean_or_none(peak_rss_values),
                "avg_rtf": (
                    mean_or_none(transcribe_values) / group[-1].audio_duration_seconds
                    if transcribe_values and group[-1].audio_duration_seconds > 0
                    else None
                ),
                "avg_wer": mean_or_none(wer_values),
                "avg_cer": mean_or_none(cer_values),
                "last_detected_language": ok_runs[-1].detected_language
                if ok_runs
                else None,
                "last_detected_language_probability": (
                    ok_runs[-1].detected_language_probability if ok_runs else None
                ),
                "last_transcript_chars": ok_runs[-1].transcript_chars
                if ok_runs
                else None,
                "last_transcript_words": ok_runs[-1].transcript_words
                if ok_runs
                else None,
                "last_peak_rss_mb": ok_runs[-1].peak_rss_mb if ok_runs else None,
                "last_wer": ok_runs[-1].wer if ok_runs else None,
                "last_cer": ok_runs[-1].cer if ok_runs else None,
                "errors": [item.error for item in group if item.error],
            }
        )
    return aggregated


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def stdev_or_none(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def print_summary(aggregated: list[dict[str, Any]]) -> None:
    headers = [
        "audio",
        "lang",
        "backend",
        "device",
        "model",
        "ok",
        "avg_total_s",
        "median_total_s",
        "load_s",
        "avg_transcribe_s",
        "stddev_transcribe_s",
        "avg_peak_rss_mb",
        "avg_rtf",
        "avg_wer",
        "avg_cer",
    ]
    rows = []
    for row in aggregated:
        rows.append(
            [
                row["sample_label"],
                row["forced_language"] or "auto",
                row["backend"],
                row.get("backend_device") or "-",
                row["model"],
                f"{row['successful_runs']}/{row['runs']}",
                format_float(row["avg_total_seconds"]),
                format_float(row["median_total_seconds"]),
                format_float(row["load_seconds"]),
                format_float(row["avg_transcribe_seconds"]),
                format_float(row["stddev_transcribe_seconds"]),
                format_float(row.get("avg_peak_rss_mb")),
                format_float(row["avg_rtf"]),
                format_float(row["avg_wer"]),
                format_float(row["avg_cer"]),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print("\nColumns:")
    print("audio: bundled sample label or custom audio stem")
    print("lang: forced language code, or auto when autodetection is used")
    print("backend: benchmarked engine")
    print("device: device or runtime label reported by the backend")
    print("model: normalized model name requested by the benchmark")
    print("ok: successful runs over total runs")
    print("avg_total_s: average end-to-end runtime in seconds")
    print("median_total_s: median end-to-end runtime in seconds")
    print("load_s: one-time model load time in seconds when measurable")
    print("avg_transcribe_s: average transcription time in seconds")
    print("stddev_transcribe_s: standard deviation of transcription time in seconds")
    print("avg_peak_rss_mb: average peak resident set size for successful runs")
    print("avg_rtf: average real-time factor (transcribe_s / audio_duration_s)")
    print("avg_wer: average word error rate against the reference transcript")
    print("avg_cer: average character error rate against the reference transcript")


def print_runs_table(results: list[RunResult]) -> None:
    headers = [
        "audio",
        "lang",
        "backend",
        "device",
        "model",
        "run",
        "status",
        "total_s",
        "load_s",
        "transcribe_s",
        "peak_rss_mb",
        "rtf",
        "wer",
        "cer",
        "error",
    ]
    rows = []
    for result in results:
        rtf = None
        if result.total_seconds is not None and result.transcribe_seconds is not None:
            rtf = (
                result.transcribe_seconds / result.audio_duration_seconds
                if result.audio_duration_seconds > 0
                else None
            )
        rows.append(
            [
                result.sample_label,
                result.forced_language or "auto",
                result.backend,
                result.backend_device or "-",
                result.model,
                str(result.run_index),
                result.status,
                format_float(result.total_seconds),
                format_float(result.load_seconds),
                format_float(result.transcribe_seconds),
                format_float(result.peak_rss_mb),
                format_float(rtf),
                format_float(result.wer),
                format_float(result.cer),
                result.error or "-",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    print("Runs:")
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print()


def print_skipped_summary(skipped: list[SkippedBenchmark]) -> None:
    if not skipped:
        return
    print("\nSkipped:")
    for item in skipped:
        print(f"{item.backend} {item.model}: {item.reason}")


def format_float(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_output_paths(output: Path | None) -> Path:
    if output is not None:
        return output

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("output") / f"benchmark_results_{timestamp}.json"


def build_metadata(
    args: argparse.Namespace, audio_inputs: list[ResolvedAudioInput]
) -> dict[str, Any]:
    gigaam_python = getattr(args, "gigaam_python", DEFAULT_GIGAAM_PYTHON)
    gigaam_model_path = getattr(args, "gigaam_model_path", None)
    gigaam_multilingual_python = getattr(
        args,
        "gigaam_multilingual_python",
        DEFAULT_GIGAAM_MULTILINGUAL_PYTHON,
    )
    gigaam_multilingual_model_path = getattr(
        args, "gigaam_multilingual_model_path", None
    )
    gigaam_multilingual_language = getattr(
        args, "gigaam_multilingual_language", None
    )
    gigaam_multilingual_mlx_python = getattr(
        args,
        "gigaam_multilingual_mlx_python",
        DEFAULT_GIGAAM_MULTILINGUAL_MLX_PYTHON,
    )
    gigaam_multilingual_mlx_model_path = getattr(
        args, "gigaam_multilingual_mlx_model_path", None
    )
    gigaam_multilingual_mlx_chunk_seconds = getattr(
        args,
        "gigaam_multilingual_mlx_chunk_seconds",
        DEFAULT_GIGAAM_MULTILINGUAL_MLX_CHUNK_SECONDS,
    )
    gigaam_multilingual_mlx_overlap_seconds = getattr(
        args,
        "gigaam_multilingual_mlx_overlap_seconds",
        DEFAULT_GIGAAM_MULTILINGUAL_MLX_OVERLAP_SECONDS,
    )
    effective_gigaam_multilingual_languages = [
        gigaam_multilingual_language_hint(
            gigaam_multilingual_language, item.forced_language
        )
        for item in audio_inputs
    ]
    effective_gigaam_multilingual_language_hint = (
        effective_gigaam_multilingual_languages[0]
        if effective_gigaam_multilingual_languages
        and all(
            language == effective_gigaam_multilingual_languages[0]
            for language in effective_gigaam_multilingual_languages
        )
        else None
    )
    tone_python = getattr(args, "tone_python", DEFAULT_TONE_PYTHON)
    tone_model_path = getattr(args, "tone_model_path", None)
    vosk_python = getattr(args, "vosk_python", DEFAULT_VOSK_PYTHON)
    vosk_model_path = getattr(args, "vosk_model_path", None)
    qwen3_asr_python = getattr(args, "qwen3_asr_python", DEFAULT_QWEN3_ASR_PYTHON)
    qwen3_asr_model_path = getattr(args, "qwen3_asr_model_path", None)
    qwen3_asr_language = getattr(args, "qwen3_asr_language", None)
    qwen3_asr_hf_python = getattr(
        args, "qwen3_asr_hf_python", DEFAULT_QWEN3_ASR_HF_PYTHON
    )
    qwen3_asr_hf_model_path = getattr(args, "qwen3_asr_hf_model_path", None)
    qwen3_asr_hf_device = getattr(
        args, "qwen3_asr_hf_device", DEFAULT_QWEN3_ASR_HF_DEVICE
    )
    qwen3_asr_hf_max_tokens = getattr(
        args,
        "qwen3_asr_hf_max_tokens",
        getattr(
            args,
            "qwen3_asr_hf_max_new_tokens",
            QWEN3_ASR_HF_DEFAULT_MAX_TOKENS,
        ),
    )
    effective_qwen3_asr_hf_languages = [
        qwen3_asr_hf_language_hint(item.forced_language) for item in audio_inputs
    ]
    effective_qwen3_asr_hf_language_hint = (
        effective_qwen3_asr_hf_languages[0]
        if effective_qwen3_asr_hf_languages
        and all(
            language == effective_qwen3_asr_hf_languages[0]
            for language in effective_qwen3_asr_hf_languages
        )
        else None
    )
    effective_qwen_languages = [
        qwen3_asr_language_hint(qwen3_asr_language, item.forced_language)
        for item in audio_inputs
    ]
    effective_qwen_language_hint = (
        effective_qwen_languages[0]
        if effective_qwen_languages
        and all(
            language == effective_qwen_languages[0]
            for language in effective_qwen_languages
        )
        else None
    )
    podlodka_python = getattr(args, "podlodka_python", DEFAULT_PODLODKA_PYTHON)
    podlodka_model_path = getattr(args, "podlodka_model_path", None)
    podlodka_language = getattr(args, "podlodka_language", None)
    effective_podlodka_languages = [
        podlodka_language_hint(podlodka_language, item.forced_language)
        for item in audio_inputs
    ]
    effective_podlodka_language_hint = (
        effective_podlodka_languages[0]
        if effective_podlodka_languages
        and all(
            language == effective_podlodka_languages[0]
            for language in effective_podlodka_languages
        )
        else None
    )
    parakeet_hf_python = getattr(
        args, "parakeet_hf_python", DEFAULT_PARAKEET_HF_PYTHON
    )
    parakeet_hf_model_path = getattr(args, "parakeet_hf_model_path", None)
    parakeet_hf_device = getattr(
        args, "parakeet_hf_device", DEFAULT_PARAKEET_HF_DEVICE
    )
    parakeet_sherpa_python = getattr(
        args, "parakeet_sherpa_python", DEFAULT_PARAKEET_SHERPA_PYTHON
    )
    parakeet_sherpa_model_path = getattr(args, "parakeet_sherpa_model_path", None)
    parakeet_sherpa_threads = getattr(
        args, "parakeet_sherpa_threads", DEFAULT_PARAKEET_SHERPA_THREADS
    )
    vibevoice_python = getattr(args, "vibevoice_python", DEFAULT_VIBEVOICE_PYTHON)
    vibevoice_model_path = getattr(args, "vibevoice_model_path", None)
    vibevoice_device = getattr(args, "vibevoice_device", DEFAULT_VIBEVOICE_DEVICE)
    vibevoice_mode = getattr(args, "vibevoice_mode", DEFAULT_VIBEVOICE_MODE)
    vibevoice_acoustic_tokenizer_chunk_size = getattr(
        args, "vibevoice_acoustic_tokenizer_chunk_size", None
    )
    qwen3_asr_hf_capabilities = BACKEND_CAPABILITIES.get(
        "qwen3-asr-hf",
        BackendCapabilities(
            supported_models=QWEN3_ASR_HF_MODELS,
            supports_hallucination_silence_threshold=False,
            multilingual=True,
            supported_tasks=("transcribe",),
            supports_segment_timestamps=False,
            supports_condition_on_previous_text=False,
            device=QWEN3_ASR_HF_BACKEND_DEVICE,
        ),
    )
    parakeet_hf_capabilities = BACKEND_CAPABILITIES.get(
        "parakeet-hf",
        BackendCapabilities(
            supported_models=PARAKEET_HF_MODELS,
            supports_hallucination_silence_threshold=False,
            multilingual=True,
            supported_tasks=("transcribe",),
            supports_segment_timestamps=True,
            supports_condition_on_previous_text=False,
            device=PARAKEET_BACKEND_DEVICE,
        ),
    )
    parakeet_sherpa_capabilities = BACKEND_CAPABILITIES.get(
        "parakeet-sherpa",
        BackendCapabilities(
            supported_models=PARAKEET_SHERPA_MODELS,
            supports_hallucination_silence_threshold=False,
            multilingual=True,
            supported_tasks=("transcribe",),
            supports_segment_timestamps=True,
            supports_condition_on_previous_text=False,
            device=PARAKEET_BACKEND_DEVICE,
        ),
    )
    whisper_cpp_executable = getattr(
        args, "whisper_cpp_executable", DEFAULT_WHISPER_CPP_EXECUTABLE
    )
    whisper_cpp_threads = getattr(
        args, "whisper_cpp_threads", WHISPER_CPP_DEFAULT_THREADS
    )
    whisper_cpp_capabilities = BACKEND_CAPABILITIES.get(
        "whisper-cpp",
        BackendCapabilities(
            supported_models=WHISPER_CPP_MODELS,
            supports_hallucination_silence_threshold=False,
            multilingual=True,
            supported_tasks=("transcribe", "translate"),
            supports_segment_timestamps=True,
            supports_condition_on_previous_text=False,
            device=WHISPER_CPP_BACKEND_DEVICE,
        ),
    )
    gigaam_multilingual_mlx_capabilities = BACKEND_CAPABILITIES.get(
        "gigaam-multilingual-mlx",
        BackendCapabilities(
            supported_models=GIGAAM_MULTILINGUAL_MLX_MODELS,
            supports_hallucination_silence_threshold=False,
            multilingual=True,
            supported_tasks=("transcribe",),
            supports_segment_timestamps=True,
            supports_condition_on_previous_text=False,
            device="subprocess",
        ),
    )
    vibevoice_capabilities = BACKEND_CAPABILITIES.get(
        "vibevoice",
        BackendCapabilities(
            supported_models=VIBEVOICE_MODELS,
            supports_hallucination_silence_threshold=False,
            multilingual=True,
            supported_tasks=("transcribe",),
            supports_segment_timestamps=True,
            supports_condition_on_previous_text=False,
            device="subprocess",
        ),
    )
    return {
        "profile": getattr(args, "profile", None),
        "audios": [
            {
                "audio": str(item.audio_path),
                "sample_label": item.sample_label,
                "reference_transcript": str(item.reference_transcript_path)
                if item.reference_transcript_path is not None
                else None,
                "forced_language": item.forced_language,
                "audio_duration_seconds": item.audio_duration_seconds,
                "source": item.source,
            }
            for item in audio_inputs
        ],
        "models": args.models,
        "backends": args.backends,
        "benchmark_pairs": (
            [list(pair) for pair in iter_benchmark_pairs(args)]
            if uses_exact_profile_pairs(args)
            else None
        ),
        "runs": args.runs,
        "task": args.task,
        "audio_selectors": args.audios,
        "beam_size": args.beam_size,
        "compute_type": args.compute_type,
        "device": args.device,
        "faster_whisper_vad_filter": args.faster_whisper_vad_filter,
        "condition_on_previous_text": args.condition_on_previous_text,
        "hallucination_silence_threshold": args.hallucination_silence_threshold,
        "openai_whisper_temperature_fallback": args.openai_whisper_temperature_fallback,
        "lightning_whisper_mlx_batch_size": args.lightning_whisper_mlx_batch_size,
        "insanely_fast_whisper_device_id": args.insanely_fast_whisper_device_id,
        "insanely_fast_whisper_batch_size": args.insanely_fast_whisper_batch_size,
        "insanely_fast_whisper_flash": args.insanely_fast_whisper_flash,
        "gigaam_python": str(gigaam_python),
        "gigaam_model_path": (
            str(gigaam_model_path) if gigaam_model_path is not None else None
        ),
        "gigaam_model_variant": GIGAAM_MODEL_VARIANT,
        "gigaam_offline": True,
        "gigaam_multilingual_python": str(gigaam_multilingual_python),
        "gigaam_multilingual_model_path": (
            str(gigaam_multilingual_model_path)
            if gigaam_multilingual_model_path is not None
            else None
        ),
        "gigaam_multilingual_model_variant": GIGAAM_MULTILINGUAL_MODEL_VARIANT,
        "gigaam_multilingual_language": gigaam_multilingual_language,
        "gigaam_multilingual_effective_language_hint": (
            effective_gigaam_multilingual_language_hint
        ),
        "gigaam_multilingual_supported_languages": list(
            GIGAAM_MULTILINGUAL_SUPPORTED_LANGUAGES
        ),
        "gigaam_multilingual_offline": True,
        "gigaam_multilingual_mlx_python": str(gigaam_multilingual_mlx_python),
        "gigaam_multilingual_mlx_model_path": (
            str(gigaam_multilingual_mlx_model_path)
            if gigaam_multilingual_mlx_model_path is not None
            else None
        ),
        "gigaam_multilingual_mlx_model_variant": (
            GIGAAM_MULTILINGUAL_MLX_MODEL_VARIANT
        ),
        "gigaam_multilingual_mlx_model_cache_root": str(
            GIGAAM_MULTILINGUAL_MLX_MODEL_CACHE_ROOT
        ),
        "gigaam_multilingual_mlx_required_model_files": [
            str(path) for path in GIGAAM_MULTILINGUAL_MLX_REQUIRED_MODEL_FILES
        ],
        "gigaam_multilingual_mlx_chunk_seconds": gigaam_multilingual_mlx_chunk_seconds,
        "gigaam_multilingual_mlx_overlap_seconds": gigaam_multilingual_mlx_overlap_seconds,
        "gigaam_multilingual_mlx_multilingual": (
            gigaam_multilingual_mlx_capabilities.multilingual
        ),
        "gigaam_multilingual_mlx_supported_languages": list(
            GIGAAM_MULTILINGUAL_MLX_SUPPORTED_LANGUAGES
        ),
        "gigaam_multilingual_mlx_supported_tasks": list(
            gigaam_multilingual_mlx_capabilities.supported_tasks
        ),
        "gigaam_multilingual_mlx_word_timestamps": True,
        "gigaam_multilingual_mlx_segment_timestamps": (
            gigaam_multilingual_mlx_capabilities.supports_segment_timestamps
        ),
        "gigaam_multilingual_mlx_supports_condition_on_previous_text": (
            gigaam_multilingual_mlx_capabilities.supports_condition_on_previous_text
        ),
        "gigaam_multilingual_mlx_supports_hallucination_silence_threshold": (
            gigaam_multilingual_mlx_capabilities.supports_hallucination_silence_threshold
        ),
        "gigaam_multilingual_mlx_timestamp_semantics": (
            GIGAAM_MULTILINGUAL_MLX_TIMESTAMP_SEMANTICS
        ),
        "gigaam_multilingual_mlx_device": "subprocess",
        "gigaam_multilingual_mlx_offline_env": dict(
            GIGAAM_MULTILINGUAL_MLX_OFFLINE_ENV
        ),
        "gigaam_multilingual_mlx_offline": True,
        "tone_python": str(tone_python),
        "tone_model_path": (
            str(tone_model_path) if tone_model_path is not None else None
        ),
        "tone_model_variant": TONE_MODEL_VARIANT,
        "tone_decoder": getattr(args, "tone_decoder", "greedy"),
        "tone_streaming": False,
        "tone_offline": True,
        "vosk_python": str(vosk_python),
        "vosk_model_path": (
            str(vosk_model_path) if vosk_model_path is not None else None
        ),
        "vosk_model_variant": VOSK_MODEL_VARIANT,
        "vosk_decoding_method": getattr(
            args, "vosk_decoding_method", VOSK_DEFAULT_DECODING_METHOD
        ),
        "vosk_quantization": "fp32",
        "vosk_streaming": False,
        "vosk_offline": True,
        "qwen3_asr_python": str(qwen3_asr_python),
        "qwen3_asr_model_path": (
            str(qwen3_asr_model_path) if qwen3_asr_model_path is not None else None
        ),
        "qwen3_asr_model_variant": QWEN3_ASR_MODEL_VARIANT,
        "qwen3_asr_language": qwen3_asr_language,
        "qwen3_asr_effective_language_hint": effective_qwen_language_hint,
        "qwen3_asr_max_tokens": getattr(args, "qwen3_asr_max_tokens", 8192),
        "qwen3_asr_temperature": getattr(args, "qwen3_asr_temperature", 0.0),
        "qwen3_asr_offline": True,
        "qwen3_asr_hf_python": str(qwen3_asr_hf_python),
        "qwen3_asr_hf_model_path": (
            str(qwen3_asr_hf_model_path)
            if qwen3_asr_hf_model_path is not None
            else None
        ),
        "qwen3_asr_hf_model_variant": QWEN3_ASR_HF_MODEL_VARIANT,
        "qwen3_asr_hf_model_cache_root": str(QWEN3_ASR_HF_MODEL_CACHE_ROOT),
        "qwen3_asr_hf_required_model_files": [
            str(path) for path in QWEN3_ASR_HF_REQUIRED_MODEL_FILES
        ],
        "qwen3_asr_hf_device": qwen3_asr_hf_device,
        "qwen3_asr_hf_max_tokens": qwen3_asr_hf_max_tokens,
        "qwen3_asr_hf_max_new_tokens": qwen3_asr_hf_max_tokens,
        "qwen3_asr_hf_effective_language_hint": (
            effective_qwen3_asr_hf_language_hint
        ),
        "qwen3_asr_hf_multilingual": qwen3_asr_hf_capabilities.multilingual,
        "qwen3_asr_hf_supported_tasks": list(
            qwen3_asr_hf_capabilities.supported_tasks
        ),
        "qwen3_asr_hf_segment_timestamps": (
            qwen3_asr_hf_capabilities.supports_segment_timestamps
        ),
        "qwen3_asr_hf_supports_condition_on_previous_text": (
            qwen3_asr_hf_capabilities.supports_condition_on_previous_text
        ),
        "qwen3_asr_hf_supports_hallucination_silence_threshold": (
            qwen3_asr_hf_capabilities.supports_hallucination_silence_threshold
        ),
        "qwen3_asr_hf_timestamp_semantics": QWEN3_ASR_HF_TIMESTAMP_SEMANTICS,
        "qwen3_asr_hf_timestamp_reason": QWEN3_ASR_HF_TIMESTAMP_REASON,
        "qwen3_asr_hf_forced_aligner": False,
        "qwen3_asr_hf_device_label": QWEN3_ASR_HF_BACKEND_DEVICE,
        "qwen3_asr_hf_offline_env": dict(QWEN3_ASR_HF_OFFLINE_ENV),
        "qwen3_asr_hf_offline": True,
        "whisper_cpp_python": sys.executable,
        "whisper_cpp_executable": str(whisper_cpp_executable),
        "whisper_cpp_threads": whisper_cpp_threads,
        "whisper_cpp_model_cache_root": str(WHISPER_CPP_MODEL_CACHE_ROOT),
        "whisper_cpp_model_files": dict(WHISPER_CPP_MODEL_FILES),
        "whisper_cpp_model_variants": list(WHISPER_CPP_MODEL_VARIANTS),
        "whisper_cpp_multilingual": whisper_cpp_capabilities.multilingual,
        "whisper_cpp_supported_tasks": list(
            whisper_cpp_capabilities.supported_tasks
        ),
        "whisper_cpp_segment_timestamps": (
            whisper_cpp_capabilities.supports_segment_timestamps
        ),
        "whisper_cpp_supports_condition_on_previous_text": (
            whisper_cpp_capabilities.supports_condition_on_previous_text
        ),
        "whisper_cpp_supports_hallucination_silence_threshold": (
            whisper_cpp_capabilities.supports_hallucination_silence_threshold
        ),
        "whisper_cpp_device": WHISPER_CPP_BACKEND_DEVICE,
        "whisper_cpp_capabilities": {
            "multilingual": whisper_cpp_capabilities.multilingual,
            "tasks": list(whisper_cpp_capabilities.supported_tasks),
            "segment_timestamps": whisper_cpp_capabilities.supports_segment_timestamps,
            "condition_on_previous_text": (
                whisper_cpp_capabilities.supports_condition_on_previous_text
            ),
            "hallucination_silence_threshold": (
                whisper_cpp_capabilities.supports_hallucination_silence_threshold
            ),
            "device": WHISPER_CPP_BACKEND_DEVICE,
        },
        "whisper_cpp_offline": True,
        "podlodka_python": str(podlodka_python),
        "podlodka_model_path": (
            str(podlodka_model_path) if podlodka_model_path is not None else None
        ),
        "podlodka_model_variant": PODLODKA_MODEL_VARIANT,
        "podlodka_language": podlodka_language,
        "podlodka_effective_language_hint": effective_podlodka_language_hint,
        "podlodka_max_new_tokens": getattr(
            args, "podlodka_max_new_tokens", PODLODKA_DEFAULT_MAX_NEW_TOKENS
        ),
        "podlodka_offline": True,
        "parakeet_hf_python": str(parakeet_hf_python),
        "parakeet_hf_model_path": (
            str(parakeet_hf_model_path)
            if parakeet_hf_model_path is not None
            else None
        ),
        "parakeet_hf_model_variant": PARAKEET_HF_MODEL_VARIANT,
        "parakeet_hf_model_cache_root": str(PARAKEET_HF_MODEL_CACHE_ROOT),
        "parakeet_hf_required_model_files": [
            str(path) for path in PARAKEET_HF_REQUIRED_MODEL_FILES
        ],
        "parakeet_hf_device": parakeet_hf_device,
        "parakeet_hf_multilingual": parakeet_hf_capabilities.multilingual,
        "parakeet_hf_supported_tasks": list(
            parakeet_hf_capabilities.supported_tasks
        ),
        "parakeet_hf_segment_timestamps": (
            parakeet_hf_capabilities.supports_segment_timestamps
        ),
        "parakeet_hf_supports_condition_on_previous_text": (
            parakeet_hf_capabilities.supports_condition_on_previous_text
        ),
        "parakeet_hf_supports_hallucination_silence_threshold": (
            parakeet_hf_capabilities.supports_hallucination_silence_threshold
        ),
        "parakeet_hf_timestamp_semantics": PARAKEET_HF_TIMESTAMP_SEMANTICS,
        "parakeet_hf_offline_env": dict(PARAKEET_HF_OFFLINE_ENV),
        "parakeet_hf_offline": True,
        "parakeet_sherpa_python": str(parakeet_sherpa_python),
        "parakeet_sherpa_model_path": (
            str(parakeet_sherpa_model_path)
            if parakeet_sherpa_model_path is not None
            else None
        ),
        "parakeet_sherpa_default_model_path": str(
            PARAKEET_SHERPA_DEFAULT_MODEL_PATH
        ),
        "parakeet_sherpa_model_variants": list(PARAKEET_SHERPA_MODEL_VARIANTS),
        "parakeet_sherpa_model_quantization": dict(
            PARAKEET_SHERPA_MODEL_QUANTIZATION
        ),
        "parakeet_sherpa_required_model_files": {
            quantization: [str(path) for path in paths]
            for quantization, paths in PARAKEET_SHERPA_REQUIRED_MODEL_FILES.items()
        },
        "parakeet_sherpa_threads": parakeet_sherpa_threads,
        "parakeet_sherpa_multilingual": parakeet_sherpa_capabilities.multilingual,
        "parakeet_sherpa_supported_tasks": list(
            parakeet_sherpa_capabilities.supported_tasks
        ),
        "parakeet_sherpa_segment_timestamps": (
            parakeet_sherpa_capabilities.supports_segment_timestamps
        ),
        "parakeet_sherpa_supports_condition_on_previous_text": (
            parakeet_sherpa_capabilities.supports_condition_on_previous_text
        ),
        "parakeet_sherpa_supports_hallucination_silence_threshold": (
            parakeet_sherpa_capabilities.supports_hallucination_silence_threshold
        ),
        "parakeet_sherpa_timestamp_semantics": PARAKEET_SHERPA_TIMESTAMP_SEMANTICS,
        "parakeet_sherpa_device": PARAKEET_BACKEND_DEVICE,
        "parakeet_sherpa_offline": True,
        "vibevoice_python": str(vibevoice_python),
        "vibevoice_model_path": (
            str(vibevoice_model_path) if vibevoice_model_path is not None else None
        ),
        "vibevoice_model_variant": VIBEVOICE_MODEL_VARIANT,
        "vibevoice_model_cache_root": str(VIBEVOICE_MODEL_CACHE_ROOT),
        "vibevoice_required_model_files": [
            str(path) for path in VIBEVOICE_REQUIRED_MODEL_FILES
        ],
        "vibevoice_device": vibevoice_device,
        "vibevoice_mode": vibevoice_mode,
        "vibevoice_acoustic_tokenizer_chunk_size": (
            vibevoice_acoustic_tokenizer_chunk_size
        ),
        "vibevoice_multilingual": vibevoice_capabilities.multilingual,
        "vibevoice_supported_tasks": list(vibevoice_capabilities.supported_tasks),
        "vibevoice_segment_timestamps": (
            vibevoice_capabilities.supports_segment_timestamps
        ),
        "vibevoice_supports_condition_on_previous_text": (
            vibevoice_capabilities.supports_condition_on_previous_text
        ),
        "vibevoice_timestamp_semantics": VIBEVOICE_TIMESTAMP_SEMANTICS,
        "vibevoice_device_label": "subprocess",
        "vibevoice_offline_env": dict(VIBEVOICE_OFFLINE_ENV),
        "vibevoice_offline": True,
        "worker_timeout_seconds": getattr(args, "worker_timeout_seconds", 900.0),
        "warmup": args.warmup,
        "show_full_table": args.show_full_table,
        "platform": platform.platform(),
        "python_version": sys.version,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = resolve_output_paths(args.output)
    audio_inputs = resolve_audio_inputs(args)
    configured_gigaam_model_path = getattr(args, "gigaam_model_path", None)
    configured_vosk_model_path = getattr(args, "vosk_model_path", None)
    configured_qwen_model_path = getattr(args, "qwen3_asr_model_path", None)
    configured_qwen_hf_model_path = getattr(args, "qwen3_asr_hf_model_path", None)

    results: list[RunResult] = []
    skipped: list[SkippedBenchmark] = []
    for audio_input in audio_inputs:
        per_audio_args = args_for_audio_input(args, audio_input)
        per_audio_args.sample_label = audio_input.sample_label
        per_audio_args.audio_duration_seconds = audio_input.audio_duration_seconds
        # Keep default cache resolution variant-specific even after a previous
        # pair stores its resolved path on the base args for metadata.
        per_audio_args.gigaam_model_path = configured_gigaam_model_path
        per_audio_args.vosk_model_path = configured_vosk_model_path
        per_audio_args.qwen3_asr_model_path = configured_qwen_model_path
        per_audio_args.qwen3_asr_hf_model_path = configured_qwen_hf_model_path
        for backend, model_name in iter_benchmark_pairs(args):
            if backend == "gigaam" and audio_input.forced_language != "ru":
                skipped.append(
                    SkippedBenchmark(
                        audio=str(audio_input.audio_path),
                        sample_label=audio_input.sample_label,
                        forced_language=audio_input.forced_language,
                        backend=backend,
                        model=model_name,
                        reason=GIGAAM_RU_ONLY_REASON,
                    )
                )
                print(
                    f"Skipping {backend} on sample {audio_input.sample_label} model "
                    f"{model_name} ({GIGAAM_RU_ONLY_REASON}).",
                    file=sys.stderr,
                )
                continue
            if backend == "t-one" and audio_input.forced_language != "ru":
                skipped.append(
                    SkippedBenchmark(
                        audio=str(audio_input.audio_path),
                        sample_label=audio_input.sample_label,
                        forced_language=audio_input.forced_language,
                        backend=backend,
                        model=model_name,
                        reason=TONE_RU_ONLY_REASON,
                    )
                )
                print(
                    f"Skipping {backend} on sample {audio_input.sample_label} model "
                    f"{model_name} ({TONE_RU_ONLY_REASON}).",
                    file=sys.stderr,
                )
                continue
            if backend == "vosk" and audio_input.forced_language != "ru":
                skipped.append(
                    SkippedBenchmark(
                        audio=str(audio_input.audio_path),
                        sample_label=audio_input.sample_label,
                        forced_language=audio_input.forced_language,
                        backend=backend,
                        model=model_name,
                        reason=VOSK_RU_ONLY_REASON,
                    )
                )
                print(
                    f"Skipping {backend} on sample {audio_input.sample_label} model "
                    f"{model_name} ({VOSK_RU_ONLY_REASON}).",
                    file=sys.stderr,
                )
                continue
            supported = BACKEND_CAPABILITIES[backend].supported_models
            if supported is not None and model_name not in supported:
                skipped.append(
                    SkippedBenchmark(
                        audio=str(audio_input.audio_path),
                        sample_label=audio_input.sample_label,
                        forced_language=audio_input.forced_language,
                        backend=backend,
                        model=model_name,
                        reason="not supported",
                    )
                )
                print(
                    f"Skipping {backend} on sample {audio_input.sample_label} model {model_name} (not supported).",
                    file=sys.stderr,
                )
                continue
            if backend == "gigaam-multilingual":
                print(
                    f"Benchmarking {backend} on sample {audio_input.sample_label} model {model_name}...",
                    file=sys.stderr,
                )
                try:
                    gigaam_multilingual_model_path = (
                        resolve_gigaam_multilingual_model_path(
                            getattr(args, "gigaam_multilingual_model_path", None)
                        )
                    )
                except Exception as exc:  # pragma: no cover - benchmark scripts should continue after failures.
                    error = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()
                    print(f"  load error: {error}", file=sys.stderr)
                    for run_index in range(1, args.runs + 1):
                        results.append(
                            build_error_result(
                                backend=backend,
                                model_name=model_name,
                                backend_device="subprocess",
                                run_index=run_index,
                                error=error,
                                audio_path=audio_input.audio_path,
                                sample_label=audio_input.sample_label,
                                audio_duration_seconds=audio_input.audio_duration_seconds,
                                forced_language=audio_input.forced_language,
                            )
                        )
                    continue

                args.gigaam_multilingual_model_path = gigaam_multilingual_model_path
                if args.warmup:
                    warmup_result = run_gigaam_multilingual(
                        audio_input.audio_path,
                        model_name,
                        0,
                        per_audio_args,
                        gigaam_multilingual_model_path,
                    )
                    if warmup_result.status != "ok":
                        print(
                            f"warmup failed for {backend} {model_name}: {warmup_result.error}",
                            file=sys.stderr,
                        )
                for run_index in range(1, args.runs + 1):
                    result = run_gigaam_multilingual(
                        audio_input.audio_path,
                        model_name,
                        run_index,
                        per_audio_args,
                        gigaam_multilingual_model_path,
                    )
                    results.append(result)
                    if result.status == "ok":
                        print(
                            f"  run {run_index}: total={format_float(result.total_seconds)}s",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  run {run_index}: error={result.error}",
                            file=sys.stderr,
                        )
                continue
            if backend == "podlodka":
                print(
                    f"Benchmarking {backend} on sample {audio_input.sample_label} model {model_name}...",
                    file=sys.stderr,
                )
                try:
                    podlodka_model_path = resolve_podlodka_model_path(
                        getattr(args, "podlodka_model_path", None)
                    )
                except Exception as exc:  # pragma: no cover - benchmark scripts should continue after failures.
                    error = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()
                    print(f"  load error: {error}", file=sys.stderr)
                    for run_index in range(1, args.runs + 1):
                        results.append(
                            build_error_result(
                                backend=backend,
                                model_name=model_name,
                                backend_device="subprocess",
                                run_index=run_index,
                                error=error,
                                audio_path=audio_input.audio_path,
                                sample_label=audio_input.sample_label,
                                audio_duration_seconds=audio_input.audio_duration_seconds,
                                forced_language=audio_input.forced_language,
                            )
                        )
                    continue

                args.podlodka_model_path = podlodka_model_path
                if args.warmup:
                    warmup_result = run_podlodka(
                        audio_input.audio_path,
                        model_name,
                        0,
                        per_audio_args,
                        podlodka_model_path,
                    )
                    if warmup_result.status != "ok":
                        print(
                            f"warmup failed for {backend} {model_name}: {warmup_result.error}",
                            file=sys.stderr,
                        )
                for run_index in range(1, args.runs + 1):
                    result = run_podlodka(
                        audio_input.audio_path,
                        model_name,
                        run_index,
                        per_audio_args,
                        podlodka_model_path,
                    )
                    results.append(result)
                    if result.status == "ok":
                        print(
                            f"  run {run_index}: total={format_float(result.total_seconds)}s",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  run {run_index}: error={result.error}",
                            file=sys.stderr,
                        )
                continue
            if backend == "qwen3-asr":
                print(
                    f"Benchmarking {backend} on sample {audio_input.sample_label} model {model_name}...",
                    file=sys.stderr,
                )
                try:
                    qwen_model_path = resolve_qwen3_asr_model_path(
                        model_name,
                        configured_qwen_model_path,
                    )
                except Exception as exc:  # pragma: no cover - benchmark scripts should continue after failures.
                    error = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()
                    print(f"  load error: {error}", file=sys.stderr)
                    for run_index in range(1, args.runs + 1):
                        results.append(
                            build_error_result(
                                backend=backend,
                                model_name=model_name,
                                backend_device="subprocess",
                                run_index=run_index,
                                error=error,
                                audio_path=audio_input.audio_path,
                                sample_label=audio_input.sample_label,
                                audio_duration_seconds=audio_input.audio_duration_seconds,
                                forced_language=audio_input.forced_language,
                            )
                        )
                    continue

                args.qwen3_asr_model_path = qwen_model_path
                if args.warmup:
                    warmup_result = run_qwen3_asr(
                        audio_input.audio_path,
                        model_name,
                        0,
                        per_audio_args,
                        qwen_model_path,
                    )
                    if warmup_result.status != "ok":
                        print(
                            f"warmup failed for {backend} {model_name}: {warmup_result.error}",
                            file=sys.stderr,
                        )
                for run_index in range(1, args.runs + 1):
                    result = run_qwen3_asr(
                        audio_input.audio_path,
                        model_name,
                        run_index,
                        per_audio_args,
                        qwen_model_path,
                    )
                    results.append(result)
                    if result.status == "ok":
                        print(
                            f"  run {run_index}: total={format_float(result.total_seconds)}s",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  run {run_index}: error={result.error}",
                            file=sys.stderr,
                        )
                continue
            if backend == "qwen3-asr-hf":
                print(
                    f"Benchmarking {backend} on sample {audio_input.sample_label} model {model_name}...",
                    file=sys.stderr,
                )
                try:
                    qwen_hf_model_path = resolve_qwen3_asr_hf_model_path(
                        configured_qwen_hf_model_path
                    )
                except Exception as exc:  # pragma: no cover - benchmark scripts should continue after failures.
                    error = "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip()
                    print(f"  load error: {error}", file=sys.stderr)
                    for run_index in range(1, args.runs + 1):
                        results.append(
                            build_error_result(
                                backend=backend,
                                model_name=model_name,
                                backend_device=QWEN3_ASR_HF_BACKEND_DEVICE,
                                run_index=run_index,
                                error=error,
                                audio_path=audio_input.audio_path,
                                sample_label=audio_input.sample_label,
                                audio_duration_seconds=audio_input.audio_duration_seconds,
                                forced_language=audio_input.forced_language,
                            )
                        )
                    continue

                args.qwen3_asr_hf_model_path = qwen_hf_model_path
                if args.warmup:
                    warmup_result = run_qwen3_asr_hf(
                        audio_input.audio_path,
                        model_name,
                        0,
                        per_audio_args,
                        qwen_hf_model_path,
                    )
                    if warmup_result.status != "ok":
                        print(
                            f"warmup failed for {backend} {model_name}: {warmup_result.error}",
                            file=sys.stderr,
                        )
                for run_index in range(1, args.runs + 1):
                    result = run_qwen3_asr_hf(
                        audio_input.audio_path,
                        model_name,
                        run_index,
                        per_audio_args,
                        qwen_hf_model_path,
                    )
                    results.append(result)
                    if result.status == "ok":
                        print(
                            f"  run {run_index}: total={format_float(result.total_seconds)}s",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"  run {run_index}: error={result.error}",
                            file=sys.stderr,
                        )
                continue
            print(
                f"Benchmarking {backend} on sample {audio_input.sample_label} model {model_name}...",
                file=sys.stderr,
            )
            try:
                backend_session = load_backend_session(
                    backend, model_name, per_audio_args
                )
            except Exception as exc:  # pragma: no cover - benchmark scripts should continue after failures.
                error = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                print(f"  load error: {error}", file=sys.stderr)
                for run_index in range(1, args.runs + 1):
                    results.append(
                        build_error_result(
                            backend=backend,
                            model_name=model_name,
                            backend_device=None,
                            run_index=run_index,
                            error=error,
                            audio_path=audio_input.audio_path,
                            sample_label=audio_input.sample_label,
                            audio_duration_seconds=audio_input.audio_duration_seconds,
                            forced_language=audio_input.forced_language,
                        )
                    )
                continue

            if backend == "gigaam":
                args.gigaam_model_path = backend_session.session["model_path"]
            if backend == "gigaam-multilingual-mlx":
                if isinstance(backend_session.session, dict):
                    args.gigaam_multilingual_mlx_model_path = backend_session.session[
                        "model_path"
                    ]
            if backend == "t-one":
                if isinstance(backend_session.session, dict):
                    args.tone_model_path = backend_session.session["model_path"]
            if backend == "vosk":
                if isinstance(backend_session.session, dict):
                    args.vosk_model_path = backend_session.session["model_path"]
            if backend == "parakeet-hf":
                if isinstance(backend_session.session, dict):
                    args.parakeet_hf_model_path = backend_session.session["model_path"]
            if backend == "parakeet-sherpa":
                if isinstance(backend_session.session, dict):
                    args.parakeet_sherpa_model_path = backend_session.session[
                        "model_path"
                    ]
            if backend == "vibevoice":
                if isinstance(backend_session.session, dict):
                    args.vibevoice_model_path = backend_session.session["model_path"]
            maybe_warmup(
                backend,
                audio_input.audio_path,
                model_name,
                per_audio_args,
                backend_session,
            )
            for run_index in range(1, args.runs + 1):
                result = run_single_backend(
                    backend,
                    audio_input.audio_path,
                    model_name,
                    run_index,
                    per_audio_args,
                    backend_session,
                    backend_session.load_seconds if run_index == 1 else None,
                )
                results.append(result)
                if result.status == "ok":
                    print(
                        f"  run {run_index}: total={format_float(result.total_seconds)}s",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  run {run_index}: error={result.error}",
                        file=sys.stderr,
                    )

    aggregated = aggregate_results(results)
    payload = {
        "metadata": build_metadata(args, audio_inputs),
        "skipped": [asdict(item) for item in skipped],
        "summary": aggregated,
        "runs": [asdict(result) for result in results],
    }
    write_json(args.output, payload)
    if args.show_full_table:
        print_runs_table(results)
    print_summary(aggregated)
    print_skipped_summary(skipped)
    print(f"\nWrote JSON results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
