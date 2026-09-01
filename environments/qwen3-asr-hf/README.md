# Official Qwen3-ASR 1.7B HF environment

This environment runs `Qwen/Qwen3-ASR-1.7B-hf` with the native Transformers
5.13 API. It is isolated from the repository's MLX Qwen worker and does not use
the `qwen-asr` toolkit, a Transformers pipeline, TorchCodec, or the ForcedAligner.

## Create the environment

Run from the repository root:

```bash
uv venv --python 3.12 .venvs/qwen3-asr-hf
uv pip install --python .venvs/qwen3-asr-hf/bin/python \
  -r environments/qwen3-asr-hf/requirements.txt
```

## Exact local model snapshot

The cached model card is `Qwen/Qwen3-ASR-1.7B-hf`. Resolve its exact `main`
snapshot and pass that snapshot directory to the worker; never pass a Hub ID or
URL:

```bash
MODEL_CACHE=/Volumes/512GB/hf/hub/models--Qwen--Qwen3-ASR-1.7B-hf
MODEL_SHA=$(tr -d '\n' < "$MODEL_CACHE/refs/main")
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_SHA"

test -d "$MODEL_PATH"
printf '%s\n' "$MODEL_PATH"
```

The worker validates the local config, processor config, tokenizer content, and
model weights before importing Transformers. Both `AutoProcessor` and
`AutoModelForMultimodalLM` receive the exact path with `local_files_only=True`.
Offline environment flags prevent Hub resolution or network fallback.

## Runtime contract

- `device` is `auto`, `mps`, or `cpu`; `auto` selects MPS when available and
  otherwise CPU. The model is moved directly with `model.to(device).eval()`.
- Audio is read by `soundfile` as float32, downmixed to mono, and resampled with
  scipy to the processor's expected rate (16 kHz for the cached card).
- Inference uses the official `processor.apply_transcription_request(audio=...)`
  API with the raw NumPy waveform. Language is omitted for auto-detection and is
  passed as a code/name when explicitly forced.
- Generation is deterministic with `do_sample=False` and defaults to
  `max_new_tokens=4096`, which leaves room for the bundled long samples.
- The worker slices generated IDs after the prompt, decodes parsed language and
  transcription, and falls back to `transcription_only` when parsed decoding is
  unavailable.
- `segments` is always empty. Timestamps are explicitly unsupported because this
  worker does not run `Qwen3-ForcedAligner`.
- Every invocation emits one JSON object on stdout with status, timings,
  effective configuration, and structured validation/load/transcription errors.

## Direct worker smoke

This is a real local-only smoke invocation; unit tests inject fake processor,
model, and audio objects and do not load or download the checkpoint.

```bash
MODEL_CACHE=/Volumes/512GB/hf/hub/models--Qwen--Qwen3-ASR-1.7B-hf
MODEL_SHA=$(tr -d '\n' < "$MODEL_CACHE/refs/main")
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_SHA"
test -d "$MODEL_PATH"

printf '%s\n' \
  '{"model_path":"'"$MODEL_PATH"'","audio_path":"samples/librispeech_1089_134686.mp3","language":"auto","device":"auto","max_new_tokens":4096}' \
  | HF_HOME=/Volumes/512GB/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=. \
    .venvs/qwen3-asr-hf/bin/python -m stt_benchmark.workers.qwen3_asr_hf
```
