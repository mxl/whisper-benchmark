# Whisper Backend Benchmark

This repo benchmarks `faster-whisper`, `mlx-whisper`, `mlx-audio`, `lightning-whisper-mlx`, `insanely-fast-whisper`, and `openai-whisper` on the same audio file across multiple Whisper model sizes.

## What it measures

- `faster-whisper`: model load time and transcription time separately
- `mlx-whisper`: model load time and transcription time separately
- `mlx-audio`: model load time and transcription time separately
- `lightning-whisper-mlx`: model load time and transcription time via the Apple Silicon MLX implementation
- `insanely-fast-whisper`: model load time and transcription time separately using Hugging Face Whisper checkpoints
- `openai-whisper`: model load time and transcription time via the original Python package
- optional WER/CER scoring against a reference transcript
- repeated runs per backend/model pair
- JSON output for analysis and optional CSV output for spreadsheets

## Setup

Create a Python virtual environment and install dependencies with `uv`:

```bash
/opt/homebrew/bin/uv sync
```

`mlx-whisper` requires Apple Silicon and MLX. On macOS you will also usually want `ffmpeg` installed:

```bash
brew install ffmpeg
```

`mlx-audio` currently requires Python `3.10+`, so it may need a newer virtualenv than the rest of this repo if your local environment is older.

`insanely-fast-whisper` is benchmarked in-process through `transformers.pipeline(...)`. On macOS it will usually run with `mps`; if MPS is unavailable, the benchmark falls back to CPU when `--insanely-fast-whisper-device-id mps` is used.

By default, MLX model repos are resolved as `mlx-community/whisper-<model>-mlx`, except `large-v3-turbo`, which resolves to `mlx-community/whisper-large-v3-turbo`.

`lightning-whisper-mlx` uses direct model-to-repo mappings in this benchmark, including `large-v3-turbo`, which resolves to `mlx-community/whisper-turbo`.

## Run the benchmark

Basic run across `tiny`, `base`, `small`, `medium`, `large-v3`, and `large-v3-turbo`:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3
```

Run two repetitions, add warmup, and write CSV too:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 \
  --runs 2 \
  --warmup \
  --output results.json \
  --csv-output results.csv
```

Only benchmark selected models:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 --models tiny base small
```

Only benchmark one backend:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 --backends faster-whisper
```

Benchmark only insanely-fast-whisper:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 --backends insanely-fast-whisper
```

Benchmark only openai-whisper:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 --backends openai-whisper
```

Benchmark only mlx-audio:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 --backends mlx-audio
```

Benchmark only lightning-whisper-mlx:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 --backends lightning-whisper-mlx
```

Force English transcription:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 --language en
```

Score against a ground-truth transcript:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 \
  --reference-transcript /path/to/reference.txt
```

Reduce end-of-audio hallucination loops on supported backends:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 \
  --hallucination-silence-threshold 2.0
```

Set `--hallucination-silence-threshold 0` to disable it.

Use a different `faster-whisper` compute type:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 --compute-type int8 --device cpu
```

Override the MLX repo naming pattern if needed:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 \
  --mlx-prefix mlx-community/whisper- \
  --mlx-suffix -mlx
```

Download only selected model weights:

```bash
.venv/bin/python download_models.py --mlx-whisper --models tiny large-v3
```

## Output

The script prints a summary table like:

```text
backend         model      ok    avg_total_s   median_total_s   avg_load_s   avg_transcribe_s
faster-whisper  tiny       3/3   1.234         1.210            0.456        0.778
mlx-whisper     tiny       3/3   0.987         0.981            0.123        0.864
```

When `--reference-transcript` is provided, the summary also includes `avg_wer` and `avg_cer`.
WER and CER are computed with `jiwer` after a fixed multilingual-safe normalization step: Unicode NFKC normalization, lowercasing, punctuation removal, and whitespace collapsing.

It also writes:

- `summary`: aggregated stats per backend/model pair
- `runs`: one row per timed run
- `metadata`: benchmark configuration and machine details
- full transcript text and optional per-run `wer` / `cer`

Unsupported backend/model combinations are skipped and do not appear in the summary.

## Notes on fairness

- Keep the same audio file, language setting, and task across both backends.
- Run benchmarks sequentially, not in parallel.
- Use one backend at a time, one model at a time, and one language file at a time.
- The benchmark passes the overlapping knobs that are available across backends, including `language`, `task`, and `condition_on_previous_text`, but exact decoding parity is still not possible across all implementations.
- `--hallucination-silence-threshold` is supported by `faster-whisper`, `mlx-whisper`, `lightning-whisper-mlx`, and `openai-whisper`. It is not supported by `mlx-audio` or `insanely-fast-whisper`.
- `mlx-audio` only supports the overlapping Whisper-style model repos configured in this benchmark: `tiny`, `base`, `small`, `medium`, `large-v3`, and `large-v3-turbo`.
- `lightning-whisper-mlx` supports `tiny`, `base`, `small`, `medium`, `large-v3`, and `large-v3-turbo`. In this benchmark `large-v3-turbo` is loaded from `mlx-community/whisper-turbo` because the standard MLX turbo repo format used by other backends is not directly compatible with `lightning-whisper-mlx`.
- `insanely-fast-whisper` uses Hugging Face Whisper checkpoints like `openai/whisper-medium` and `openai/whisper-large-v3-turbo` rather than the CTranslate2 or MLX model formats used by the other backends.
- `openai-whisper` uses Whisper model names like `tiny`, `base`, `small`, `medium`, and `turbo`; in this benchmark `large-v3-turbo` is mapped to `turbo`.
- The first run may include model downloads and backend-specific compile or JIT overhead. For cleaner comparisons, run once to populate caches, then run the benchmark again.
- Large models can take substantial disk and memory.
