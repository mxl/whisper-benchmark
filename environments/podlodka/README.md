# Whisper-Podlodka Transformers environment

This file documents the T0.10 integration for the cached Hugging Face-native
`bond005/whisper-podlodka-turbo` model. The model card declares the Apache-2.0
license and RU+EN support. The benchmark scope is direct local inference with
Transformers `AutoProcessor` and `WhisperForConditionalGeneration`; it does
not add an MLX conversion or integrate `whisper-large-v3-ru-podlodka`.

## Process and environment

The worker is launched as an isolated subprocess with the repository's
`.venv/bin/python` interpreter. This is a process-isolation boundary, not a
new Podlodka model format or an MLX environment. The worker receives a local
model snapshot path and local audio, and runs with the Hugging Face offline
flags:

```bash
export HF_HOME=/Volumes/512GB/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The repository's normal environment setup is the prerequisite:

```bash
uv sync
test -x .venv/bin/python
```

No Podlodka-specific requirements file is declared here. The integration uses
the existing Transformers dependency without introducing a second runtime or
downloading model files during a benchmark.

## Exact local snapshot

The model must already be present in the canonical cache. Resolve the cached
`refs/main` ref and pass the resulting snapshot directory to the worker:

```bash
MODEL_CACHE=/Volumes/512GB/hf/hub/models--bond005--whisper-podlodka-turbo
MODEL_SHA=$(tr -d '\n' < "$MODEL_CACHE/refs/main")
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_SHA"

test -d "$MODEL_PATH"
test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/model.safetensors"
printf '%s\n' "$MODEL_PATH"
```

The current local cache resolves `refs/main` to snapshot
`da87efd100d2111281b1672ad6bd386722b32251`. The integration must resolve the
ref at runtime and must not replace the local path with a Hub ID or URL.

## Benchmark contract

The separate profile is intended to run with:

```bash
uv run stt-benchmark benchmark --profile podlodka
```

It is not part of the default `main` profile. The worker reads audio with
`soundfile`, resamples it to 16 kHz when needed, and runs deterministic fixed
chunks (30 seconds by default) through the local processor and model. Its
default `max_new_tokens=444` leaves room for Podlodka's four-token Whisper
decoder prompt within the model's 448 target positions. It returns
chunk-boundary segment timestamps; word-level alignment is outside
this scope. No pipeline, torchcodec, or FFmpeg audio path is used.

`--podlodka-language` takes precedence over any language inferred from the
audio selector. Without the override, a forced `ru` or `en` selector supplies a
language hint. `--podlodka-language auto` and an `auto`/unforced input omit the
language hint.

The integration overrides are:

- `--podlodka-python` or `PODLODKA_PYTHON` — worker Python executable;
- `--podlodka-model-path` or `PODLODKA_MODEL_PATH` — exact local snapshot
  directory.

T0.10 is complete. The offline three-run profile passed on both bundled
samples; the recorded metrics are in `evidence/2026-09-01/podlodka.json`,
`RESULTS.md`, and the task log.
