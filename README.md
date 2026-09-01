# STT Benchmark

Current fixed-corpus results, quantization comparisons and extended-model
blockers are recorded in [`RESULTS.md`](RESULTS.md).

This repo benchmarks speech-to-text engines on the same audio files. The
current implementation includes six Whisper-family runtimes, GigaAM-v3 and
GigaAM Multilingual, T-one, Zipformer2 ONNX, Qwen3-ASR MLX and official HF,
Podlodka, official Whisper.cpp, Parakeet HF/Sherpa, GigaAM Multilingual MLX and
VibeVoice-ASR.

The repo ships bundled English and Russian benchmark samples. The `.mp3` sample assets are stored with Git LFS; transcripts and attribution files are regular Git files.

## What it measures

- `faster-whisper`: model load time and transcription time separately
- `mlx-whisper`: model load time and transcription time separately
- `mlx-audio`: model load time and transcription time separately
- `lightning-whisper-mlx`: model load time and transcription time via the Apple Silicon MLX implementation
- `insanely-fast-whisper`: model load time and transcription time separately using Hugging Face Whisper checkpoints
- `openai-whisper`: model load time and transcription time via the original Python package
- `gigaam`: official RU-only GigaAM-v3 `e2e_rnnt` model through an isolated worker subprocess
- `gigaam-multilingual`: official `ai-sage/GigaAM-Multilingual` `large_ctc` model through the shared GigaAM environment and PyTorch path
- `t-one`: official RU-only T-one model through an isolated worker subprocess
- `vosk`: cached RU Zipformer2 ONNX model through an isolated `sherpa-onnx` worker subprocess
- `qwen3-asr`: cached multilingual RU+EN 0.6B 8-bit MLX model through an isolated worker subprocess; segment-level timestamps only
- `qwen3-asr-hf`: official 1.7B Transformers model with deterministic fixed chunks; ForcedAligner is separate
- `whisper-cpp`: official `whisper-cli` with FP16/Q5/Q8 and Metal
- `parakeet-hf` / `parakeet-sherpa`: official Transformers and derived FP32/INT8 Sherpa-ONNX runtimes
- `gigaam-multilingual-mlx`: Apple Silicon FP16 CTC runtime
- `vibevoice`: heavyweight long-form Transformers runtime; rich parsed output is separate from plain transcription
- optional WER/CER scoring against a reference transcript
- repeated runs per backend/model pair
- JSON output for analysis

## Setup

Create a Python virtual environment and install dependencies with `uv`:

```bash
/opt/homebrew/bin/uv sync
```

The official GigaAM-v3 and GigaAM Multilingual workers share a separate
Python 3.13.12 environment. The multilingual path reuses `.venvs/gigaam`; it
does not create a second GigaAM virtualenv:

```bash
uv venv --python 3.13.12 .venvs/gigaam
source .venvs/gigaam/bin/activate
uv pip install -r environments/gigaam/requirements.txt
deactivate
```

The official T-one backend uses a separate Python 3.12 environment:

```bash
uv venv --python 3.12 .venvs/t-one
uv pip install --python .venvs/t-one/bin/python -r environments/t-one/requirements.txt
```

The Vosk-named Zipformer2 ONNX backend uses a separate Python 3.13.12 environment:

```bash
uv venv --python 3.13.12 .venvs/vosk
uv pip install --python .venvs/vosk/bin/python -r environments/vosk/requirements.txt
```

Qwen3-ASR uses the existing main `.venv`, which contains `mlx-audio 0.4.2`
and MLX for the 0.6B/1.7B MLX workers. The official 1.7B Transformers worker
uses `.venvs/qwen3-asr-hf`. Additional focused environments are created from
their checked-in requirement files:

```bash
for env in parakeet-hf parakeet-sherpa qwen3-asr-hf \
  gigaam-multilingual-mlx vibevoice
do
  uv venv --python 3.13 ".venvs/$env"
  uv pip install --python ".venvs/$env/bin/python" \
    -r "environments/$env/requirements.txt"
done
```

Whisper.cpp uses the official Homebrew `/opt/homebrew/bin/whisper-cli` binary.

Primary CLI:

```bash
uv run stt-benchmark --help
```

Available commands:

```bash
uv run stt-benchmark benchmark --help
uv run stt-benchmark download-models --help
uv run stt-benchmark prepare-samples --help
uv run stt-benchmark smoke-test --help
```

The existing script entry points still work and are kept for compatibility.

`mlx-whisper` requires Apple Silicon and MLX. On macOS you will also usually want `ffmpeg` installed:

```bash
brew install ffmpeg
```

`mlx-audio` currently requires Python `3.10+`, so it may need a newer virtualenv than the rest of this repo if your local environment is older.

`insanely-fast-whisper` is benchmarked in-process through `transformers.pipeline(...)`. On macOS it will usually run with `mps`; if MPS is unavailable, the benchmark falls back to CPU when `--insanely-fast-whisper-device-id mps` is used.

By default, MLX model repos are resolved as `mlx-community/whisper-<model>-mlx`, except `large-v3-turbo`, which resolves to `mlx-community/whisper-large-v3-turbo`.

`lightning-whisper-mlx` uses direct model-to-repo mappings in this benchmark, including `large-v3-turbo`, which resolves to `mlx-community/whisper-turbo`.

## Run the benchmark

The default `main` profile runs exactly six pairs on the bundled Russian sample:

- `mlx-whisper` / `large-v3-turbo`
- `gigaam` / `e2e_rnnt`
- `gigaam-multilingual` / `gigaam-multilingual-large-ctc`
- `t-one` / `t-one-greedy`
- `vosk` / `vosk-ru`
- `qwen3-asr` / `qwen3-asr-0.6b-8bit`

```bash
uv run stt-benchmark benchmark                 # default: main
uv run stt-benchmark benchmark --profile main
```

The separate `whisper` profile preserves all six current Whisper runtimes and all six current model sizes (`tiny`, `base`, `small`, `medium`, `large-v3`, and `large-v3-turbo`) across the bundled English and Russian samples:

```bash
uv run stt-benchmark benchmark --profile whisper
```

Focused profiles:

```bash
uv run stt-benchmark benchmark --profile podlodka
uv run stt-benchmark benchmark --profile whisper-cpp
uv run stt-benchmark benchmark --profile parakeet
uv run stt-benchmark benchmark --profile ru-variants
uv run stt-benchmark benchmark --profile qwen --worker-timeout-seconds 1800
uv run stt-benchmark benchmark --profile gigaam-multilingual
uv run stt-benchmark benchmark --profile vibevoice --worker-timeout-seconds 1800
```

Explicit `--models` or `--backends` options switch pair generation to the Cartesian product of the selected models and backends. `--audio` selects samples but does not change the exact `main` pairs.

### Podlodka integration

The `podlodka` profile is separate from `main` and does not change the
default six-pair matrix. It covers the cached Hugging Face-native
`bond005/whisper-podlodka-turbo` model, one `podlodka` backend/model pair, and
the bundled RU and EN samples. The model is licensed Apache-2.0 and its
benchmark scope is RU+EN ASR.

Run it with:

```bash
uv run stt-benchmark benchmark --profile podlodka
```

The profile completed three offline cold runs on both bundled samples:

| Language | Duration | Median total | Avg transcribe | RTF | WER | CER | Avg peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN | 179.810 s | 34.251 s | 12.093 s | 0.0673 | 0.0628 | 0.0577 | 592.745 MiB |
| RU | 298.120 s | 45.987 s | 32.561 s | 0.1092 | 0.0650 | 0.0426 | 640.901 MiB |

Evidence is in `evidence/2026-09-01/podlodka.json`. These are fixed-corpus
observations, not universal performance claims.

The worker runs direct local Transformers inference in an isolated subprocess
using the repository's `.venv/bin/python`: `AutoProcessor` plus
`WhisperForConditionalGeneration`. The parent process resolves and passes an
exact local snapshot directory; the worker never receives a Hub repo ID or URL
and runs with:

```bash
export HF_HOME=/Volumes/512GB/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The current local cache observation is:

```text
/Volumes/512GB/hf/hub/models--bond005--whisper-podlodka-turbo/snapshots/da87efd100d2111281b1672ad6bd386722b32251
```

The resolver should read `refs/main` at runtime rather than hard-code that
observation:

```bash
MODEL_CACHE=/Volumes/512GB/hf/hub/models--bond005--whisper-podlodka-turbo
MODEL_SHA=$(tr -d '\n' < "$MODEL_CACHE/refs/main")
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_SHA"
test -d "$MODEL_PATH"
```

The integration-level overrides are `--podlodka-python` / `PODLODKA_PYTHON`
for the worker interpreter and `--podlodka-model-path` /
`PODLODKA_MODEL_PATH` for an already-resolved local snapshot. The default
interpreter is `.venv/bin/python`; the default model path is the cached
`refs/main` snapshot.

Language precedence is explicit:

1. `--podlodka-language` wins over the audio selector.
2. If it is omitted and the input has a forced concrete language (`ru` or
   `en`), that language is passed to `generate`.
3. `--podlodka-language auto`, or an `auto`/unforced input without an explicit
   override, omits the forced language hint and lets the model detect it.

The output contract is deterministic chunk-boundary offsets from 30-second
fixed chunks; this scope does not claim word-level timestamps or a separate
alignment pass. Podlodka's four-token Whisper prompt leaves 444 of its 448
decoder target positions for generated tokens. No MLX conversion is planned,
and
`whisper-large-v3-ru-podlodka` is not integrated.

The standard Transformers ASR pipeline is deliberately not used: the installed
Transformers 5.5.3 path imports TorchCodec even for raw audio, while the local
Homebrew FFmpeg 9 libraries are outside TorchCodec's supported range.

### GigaAM-v3 integration

GigaAM-v3 runs by default in `.venvs/gigaam` as an isolated subprocess. The runner resolves the local `e2e_rnnt` cache ref to an exact snapshot under `/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-v3/snapshots/`, passes `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, enforces `--worker-timeout-seconds` (900 seconds by default), and exchanges one JSON request/response over stdin/stdout. On macOS, `/usr/bin/time -l` supplies peak RSS.

The official GigaAM-v3 `transcribe_longform` path is intentionally not used because it requires gated pyannote/HF_TOKEN access. When short-form transcription returns the exact too-long signal, the worker reads the local audio and uses deterministic 25-second WAV chunks with zero overlap through the short-form API.

### GigaAM Multilingual

The separate `gigaam-multilingual` integration uses the official
`ai-sage/GigaAM-Multilingual` `large_ctc` revision. It is a 600M character-wise
CTC model with the published language set RU, EN, KK, KY, and UZ. It uses the
official PyTorch/Transformers path, with CPU or MPS on Apple Silicon; it is not
an MLX port.

The runner resolves the local `refs/large_ctc` ref in the canonical cache at
runtime, validates its snapshot, and passes that local directory to the worker.
The current offline cache resolves it to:

```text
/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-Multilingual/snapshots/3905cd51c3ed4e88c8edf33f3302969ba480a327
```

The worker sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. It uses
short-form transcription and, after the exact too-long signal, deterministic
25-second WAV chunks with zero overlap instead of gated `transcribe_longform`.
Unlike the RU-only GigaAM-v3 path, EN and `auto` inputs are not skipped as
`ru-only model`. The separate `gigaam-multilingual` profile compares this
official path with the cached `ai-babai/gigaam-multilingual-mlx` FP16 worker on
both bundled samples.

### T-one integration

T-one runs in the isolated `.venvs/t-one` subprocess. The official GitHub
source is pinned to commit
`3c5b6c015038173840e62cea99e10cdb1c759116`. The worker resolves the exact
local Hugging Face snapshot from `t-tech/T-one`; the current cached `main` ref
is snapshot `106f3b0b32a9e107eb613312e4ebc61ff3d53926` under
`/Volumes/512GB/hf/hub/models--t-tech--T-one/snapshots/`. It runs CPU-only at
8 kHz, loads locally with `from_local`, and uses greedy decoding by default.
Inference uses `model.onnx`; `kenlm.bin` is optional for greedy decoding and
required for beam decoding. The worker returns phrase timestamps. Its official
streaming/long-audio path is available with the worker's `streaming` option,
but the `main` profile uses offline mode.
T-one is RU-only; EN and `auto` inputs are skipped with reason `ru-only model`.

### Vosk-named Zipformer2 ONNX integration

The cached `alphacep/vosk-model-ru` repository is a Zipformer2 ONNX
transducer model, not a classic Vosk model package. The isolated `.venvs/vosk`
environment uses `sherpa-onnx==1.13.6`; the classic Vosk Python API is
intentionally excluded. The runner resolves the exact local `refs/main`
snapshot `df6a54a4d8e5d43e82675e4f5dba2d507731a0d1` under
`/Volumes/512GB/hf/hub/models--alphacep--vosk-model-ru/snapshots/` and does
not download at runtime.

The main `vosk` / `vosk-ru` pair runs CPU-only with 16 kHz float32 audio,
`modified_beam_search`, and the FP32 model files. The worker detects both the
full `am-onnx` layout and the cached small-model `am` layout. For compatibility
with both layouts it decodes deterministic 20-second chunks with zero overlap;
returned token/frame start timestamps are converted to absolute audio times.
The backend is RU-only, so EN and `auto` inputs are skipped with reason
`ru-only model`. `alphacep/vosk-model-small-ru` is cached and supported by the
worker's layout detection, but is not part of `main` yet.

The separate `ru-variants` profile compares both full and small Zipformer2
models alongside GigaAM-v3 RNNT/CTC. Neither Vosk variant is the classic Vosk
Python API.

### Qwen3-ASR integration

The `qwen3-asr` / `qwen3-asr-0.6b-8bit` pair uses the cached
`mlx-community/Qwen3-ASR-0.6B-8bit` snapshot
`89e96d92ba34aca20b3e29fb10cc284097d1219f` under
`/Volumes/512GB/hf/hub/models--mlx-community--Qwen3-ASR-0.6B-8bit/snapshots/`.
The runner resolves and passes that local snapshot path, sets
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, and does not download at
runtime. It uses the existing main `.venv` (`mlx-audio 0.4.2` and MLX), an
isolated subprocess, and Apple Silicon MLX. The model is multilingual for RU
and EN. It returns segment-level timestamps, not word-level timestamps, and
does not run alignment.

Language precedence is explicit: `--qwen3-asr-language` wins. If it is absent,
the concrete language of a forced bundled sample is passed to Qwen3-ASR (the
default `main` sample therefore uses `ru`); for `auto`/`None`, the language hint
is omitted so the model can detect it.

The separate `qwen` profile adds MLX 1.7B 8-bit and the official Transformers
1.7B model. The official worker runs from `.venvs/qwen3-asr-hf`, uses
deterministic 30-second chunks for long audio, and does not run ForcedAligner.
Alignment remains a separate optional profile.

### Current results

Repeated baseline, Whisper.cpp quantization, Parakeet, RU variants and extended
model observations are maintained in [`RESULTS.md`](RESULTS.md). That report
also records exact evidence files, heavyweight runtimes and blocked variants.

Run two repetitions and add warmup:

```bash
uv run stt-benchmark benchmark \
  --runs 2 \
  --warmup \
  --output results.json
```

Print the full per-run table before the summary table:

```bash
uv run stt-benchmark benchmark --show-full-table
```

Run only the bundled English sample with forced English:

```bash
uv run stt-benchmark benchmark --audio en
```

In the default `main` profile this keeps the exact six-pair matrix: MLX Whisper,
GigaAM Multilingual, and Qwen3-ASR run, while GigaAM-v3, T-one, and Vosk are
recorded in `skipped` as `ru-only model`. Use
`--backends mlx-whisper gigaam-multilingual qwen3-asr` when you want to run the
English sample without the RU-only skips.

Run all bundled samples with language autodetection instead of forcing a language:

```bash
uv run stt-benchmark benchmark --audio auto
```

The multilingual GigaAM pair also runs for `auto`; only RU-only GigaAM-v3,
T-one, and Vosk are skipped for that selector.

Mix selectors so that specific sample language settings override `auto`:

```bash
uv run stt-benchmark benchmark --audio auto --audio ru
```

Benchmark a custom audio file with a reference transcript:

```bash
uv run stt-benchmark benchmark \
  --audio en:/path/to/audio.mp3:/path/to/reference.txt
```

Only benchmark selected models:

```bash
uv run stt-benchmark benchmark --audio en --models tiny base small
```

Only benchmark one backend:

```bash
uv run stt-benchmark benchmark --audio en --backends faster-whisper
```

Benchmark only insanely-fast-whisper:

```bash
uv run stt-benchmark benchmark --audio en --backends insanely-fast-whisper
```

Benchmark only openai-whisper:

```bash
uv run stt-benchmark benchmark --audio en --backends openai-whisper
```

Benchmark only mlx-audio:

```bash
uv run stt-benchmark benchmark --audio en --backends mlx-audio
```

Benchmark only lightning-whisper-mlx:

```bash
uv run stt-benchmark benchmark --audio en --backends lightning-whisper-mlx
```

Reduce end-of-audio hallucination loops on supported backends:

```bash
uv run stt-benchmark benchmark --audio en \
  --hallucination-silence-threshold 2.0
```

Set `--hallucination-silence-threshold 0` to disable it.

Use a different `faster-whisper` compute type:

```bash
uv run stt-benchmark benchmark --audio en --compute-type int8 --device cpu
```

Download only selected model weights:

```bash
uv run stt-benchmark download-models --mlx-whisper --models tiny large-v3
```

## Smoke Test

Run a quick end-to-end sanity check against the bundled English sample:

```bash
uv run stt-benchmark smoke-test
```

This defaults to `mlx-whisper` with the `tiny` model on the bundled English sample and writes JSON output to `output/smoke_test_results.json`.

## Run Tests

Run the unit test suite with:

```bash
.venv/bin/python3 -m unittest test_benchmark.py
```

## Output

The script prints a summary table like:

```text
audio                  lang  backend         device  model  ok   avg_total_s  median_total_s  load_s  avg_transcribe_s
librispeech_1089_134686  en    faster-whisper  cpu     tiny   3/3  1.234        1.210           0.456   0.778
ruls_sample_8169_13240   ru    mlx-whisper     mlx     tiny   3/3  0.987        0.981           0.123   0.864
```

When a sample has a reference transcript, the summary also includes `avg_wer` and `avg_cer`.
WER and CER are computed with `jiwer` after a fixed multilingual-safe normalization step: Unicode NFKC normalization, lowercasing, punctuation removal, and whitespace collapsing.

When `--show-full-table` is enabled, the script first prints one row per timed run and then prints the aggregated summary table.

It also writes:

- `summary`: aggregated stats per audio/backend/model combination
- `runs`: one row per timed run
- `metadata`: benchmark configuration and machine details
- each run's `peak_rss_mb` when available, plus aggregated peak-RSS fields
- `metadata.benchmark_pairs` for the exact `main` profile pairs
- full transcript text and optional per-run `wer` / `cer`

Unsupported backend/model combinations are skipped and do not appear in the summary.

## Notes on fairness

- Keep the same audio file, language setting, and task across compared backends.
- Run benchmarks sequentially, not in parallel.
- Use one backend at a time, one model at a time, and one language file at a time.
- The benchmark passes the overlapping knobs that are available across backends, including `language`, `task`, and `condition_on_previous_text`, but exact decoding parity is still not possible across all implementations.
- `--beam-size` is currently passed to `faster-whisper`, `mlx-audio`, and `openai-whisper`. It is intentionally not passed to `mlx-whisper` or `lightning-whisper-mlx` because both backends expose the option but still raise `NotImplementedError` for beam search at runtime.
- `--hallucination-silence-threshold` is supported by `faster-whisper`, `mlx-whisper`, `lightning-whisper-mlx`, and `openai-whisper`. It is not supported by `mlx-audio` or `insanely-fast-whisper`.
- `mlx-audio` only supports the overlapping Whisper-style model repos configured in this benchmark: `tiny`, `base`, `small`, `medium`, `large-v3`, and `large-v3-turbo`.
- `lightning-whisper-mlx` supports `tiny`, `base`, `small`, `medium`, `large-v3`, and `large-v3-turbo`. In this benchmark `large-v3-turbo` is loaded from `mlx-community/whisper-turbo` because the standard MLX turbo repo format used by other backends is not directly compatible with `lightning-whisper-mlx`.
- `insanely-fast-whisper` uses Hugging Face Whisper checkpoints like `openai/whisper-medium` and `openai/whisper-large-v3-turbo` rather than the CTranslate2 or MLX model formats used by the other backends.
- `openai-whisper` uses Whisper model names like `tiny`, `base`, `small`, `medium`, and `turbo`; in this benchmark `large-v3-turbo` is mapped to `turbo`.
- The default `main` profile intentionally contains one MLX Whisper baseline, one official RU-only GigaAM-v3 RNNT baseline, one official multilingual GigaAM `large_ctc` baseline, one official T-one baseline, one Vosk-named Zipformer2 ONNX baseline, and one experimental Qwen3-ASR MLX baseline. Use `--profile whisper` for runtime-specific Whisper comparisons.
- The first run may include model downloads and backend-specific compile or JIT overhead. For cleaner comparisons, run once to populate caches, then run the benchmark again.
- Large models can take substantial disk and memory.
