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
python3 -m venv .venv
.venv/bin/python -m pip install uv
/opt/homebrew/bin/uv pip install -r requirements.txt
```

`mlx-whisper` requires Apple Silicon and MLX. On macOS you will also usually want `ffmpeg` installed:

```bash
brew install ffmpeg
```

`mlx-audio` currently requires Python `3.10+`, so it may need a newer virtualenv than the rest of this repo if your local environment is older.

`insanely-fast-whisper` is benchmarked in-process through `transformers.pipeline(...)`. On macOS it will usually be run with `mps`, and the benchmark currently defaults to `--insanely-fast-whisper-device-id mps`.

By default, MLX model repos are resolved as `mlx-community/whisper-<model>-mlx`, except `large-v3-turbo`, which resolves to `mlx-community/whisper-large-v3-turbo`.

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

Ignore case and punctuation differences during WER/CER scoring:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 \
  --reference-transcript /path/to/reference.txt \
  --score-normalization basic
```

Use Whisper's English normalization for WER/CER reporting that matches common Whisper evaluations:

```bash
.venv/bin/python benchmark_whisper.py /path/to/audio.mp3 \
  --reference-transcript /path/to/reference.txt \
  --score-normalization whisper-english
```

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

## Output

The script prints a summary table like:

```text
backend         model      ok    avg_total_s   median_total_s   avg_load_s   avg_transcribe_s
faster-whisper  tiny       3/3   1.234         1.210            0.456        0.778
mlx-whisper     tiny       3/3   0.987         0.981            0.123        0.864
```

When `--reference-transcript` is provided, the summary also includes `avg_wer` and `avg_cer`.
`--score-normalization raw` keeps literal scoring. `--score-normalization basic` lowercases text and strips punctuation before WER/CER scoring. `--score-normalization whisper-english` uses `openai-whisper`'s `EnglishTextNormalizer`, which is the standard normalization used in Whisper English WER reporting.

It also writes:

- `summary`: aggregated stats per backend/model pair
- `runs`: one row per timed run
- `metadata`: benchmark configuration and machine details
- full transcript text and optional per-run `wer` / `cer`

## Notes on fairness

- Keep the same audio file, language setting, and task across both backends.
- Run benchmarks sequentially, not in parallel.
- Use one backend at a time, one model at a time, and one language file at a time.
- The benchmark passes the overlapping knobs that are available across backends, including `language`, `task`, and `condition_on_previous_text`, but exact decoding parity is still not possible across all implementations.
- `mlx-audio` only supports the overlapping Whisper-style model repos configured in this benchmark: `tiny`, `base`, `small`, `medium`, `large-v3`, and `large-v3-turbo`.
- `lightning-whisper-mlx` only supports the overlapping model names configured in this benchmark: `tiny`, `base`, `small`, `medium`, and `large-v3`. It does not support `large-v3-turbo`.
- `insanely-fast-whisper` uses Hugging Face Whisper checkpoints like `openai/whisper-medium` and `openai/whisper-large-v3-turbo` rather than the CTranslate2 or MLX model formats used by the other backends.
- `openai-whisper` uses Whisper model names like `tiny`, `base`, `small`, `medium`, and `turbo`; in this benchmark `large-v3-turbo` is mapped to `turbo`.
- The first run may include model downloads and backend-specific compile or JIT overhead. For cleaner comparisons, run once to populate caches, then run the benchmark again.
- Large models can take substantial disk and memory.
