# Qwen3-ASR MLX environment

Qwen3-ASR runs in the repository's existing main environment, which already
provides `mlx-audio` and `mlx`. Do not create a Qwen-specific virtualenv and do
not install anything from this directory. The dependency declaration remains in
the repository [`pyproject.toml`](../../pyproject.toml).

## Verified local model

The verified model is:

```text
mlx-community/Qwen3-ASR-0.6B-8bit
```

The worker must receive the model's local snapshot directory, not a Hub repo ID
or URL. Resolve the exact snapshot through the cached `main` ref:

```bash
MODEL_CACHE=/Volumes/512GB/hf/hub/models--mlx-community--Qwen3-ASR-0.6B-8bit
MODEL_SHA=$(tr -d '\n' < "$MODEL_CACHE/refs/main")
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_SHA"

test -d "$MODEL_PATH"
printf '%s\n' "$MODEL_PATH"
```

`refs/main` contains the snapshot SHA. Runtime model loading is local-only; no
Hub resolution, download, or network fallback is allowed.

## Runtime contract

- Run on Apple Silicon with the MLX backend (`mlx-audio`/`mlx`); this is not a
  CUDA environment.
- Set `HF_HOME=/Volumes/512GB/hf`, `HF_HUB_OFFLINE=1`, and
  `TRANSFORMERS_OFFLINE=1` for worker runs.
- Pass the local audio path to `mlx-audio`; its audio handling supplies the
  model's 16 kHz input. No separate audio-resampling package is added here.
- Omit `language` or set it to `auto` for auto-detection; the worker normalizes
  `"auto"` to `None` before calling the MLX API. A forced language code such as
  `en` or `ru` is passed through unchanged.
- Use `max_tokens=8192`.
- Results expose segment-level timestamps only. They are not word-level
  timestamps, and this worker does not run alignment.

The Transformers-native Qwen3-ASR integration is deferred. That path requires
Transformers `>=5.13`, but Transformers is not added or pinned for this MLX
worker. `ForcedAligner` is a separate component and is not part of this worker.

## Direct worker smoke

From the repository root, resolve `MODEL_PATH` as above and invoke the worker
directly in the main environment:

```bash
MODEL_CACHE=/Volumes/512GB/hf/hub/models--mlx-community--Qwen3-ASR-0.6B-8bit
MODEL_SHA=$(tr -d '\n' < "$MODEL_CACHE/refs/main")
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_SHA"
test -d "$MODEL_PATH"

printf '%s\n' \
  '{"model_path":"'"$MODEL_PATH"'","audio_path":"samples/librispeech_1089_134686.mp3","max_tokens":8192}' \
  | HF_HOME=/Volumes/512GB/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    uv run python -m stt_benchmark.workers.qwen3_asr
```

This command uses only the resolved local snapshot and the bundled local audio;
it does not download or install packages or model files.
