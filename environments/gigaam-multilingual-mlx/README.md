# GigaAM Multilingual MLX environment

This environment is for the standalone `gigaam_multilingual_mlx` worker. It is
separate from the official PyTorch GigaAM environment and requires Python 3.12
or 3.13 on Apple Silicon/macOS 14+.

The runtime is pinned to the documented `gigaam-multilingual-mlx==0.2.0` API and
the compatible MLX `0.32.x` range. The package's official audio loader uses
`ffmpeg` to decode local media, downmix it to mono, resample it to 16 kHz, and
return float32 samples. Install the executable separately:

```bash
brew install ffmpeg
uv venv --python 3.13.12 .venvs/gigaam-multilingual-mlx
uv pip install --python .venvs/gigaam-multilingual-mlx/bin/python \
  -r environments/gigaam-multilingual-mlx/requirements.txt
```

## Local model

Use only the exact local FP16 artifact snapshot already present in the canonical
cache:

```text
/Volumes/512GB/hf/hub/models--ai-babai--gigaam-multilingual-mlx/snapshots/2532f20238d7de763dfa45b1baaaf1d50a1726f9
```

The snapshot contains `config.json`, `manifest.json`, and
`model.safetensors`. The worker receives that directory directly and never
receives a Hub repository ID, revision, or URL. The package loader is called
with `local_files_only=True` and verifies the artifact manifest and hashes.

## Worker contract

The worker exchanges one JSON object on stdin and one JSON object on stdout.
Diagnostics from MLX or the model package are redirected to stderr. A successful
request looks like:

```json
{
  "model_path": "/Volumes/512GB/hf/hub/models--ai-babai--gigaam-multilingual-mlx/snapshots/2532f20238d7de763dfa45b1baaaf1d50a1726f9",
  "audio_path": "samples/librispeech_1089_134686.mp3",
  "language": "en"
}
```

Run it from the repository root:

```bash
printf '%s\n' '{"model_path":"/Volumes/512GB/hf/hub/models--ai-babai--gigaam-multilingual-mlx/snapshots/2532f20238d7de763dfa45b1baaaf1d50a1726f9","audio_path":"samples/librispeech_1089_134686.mp3","language":"en"}' |
  HF_HOME=/Volumes/512GB/hf \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  .venvs/gigaam-multilingual-mlx/bin/python -m stt_benchmark.workers.gigaam_multilingual_mlx
```

Supported language metadata values are `ru`, `en`, `kk`, `ky`, `uz`, and
`auto`. The MLX CTC API has no language-conditioning or language-detection
output, so this value is recorded as request metadata rather than passed to the
model or claimed as detected language.

The worker uses the documented `load_model(local_path)` API, the model's
float32 input waveform, and deterministic greedy CTC decoding. It processes
long audio in deterministic 20-second chunks with 2 seconds of overlap by
default; `chunk_seconds` and `overlap_seconds` can be supplied in the request.
The official model API returns token frames. The worker exposes the resulting
approximate word emission timestamps in `timestamps` and includes chunk details
when token frames are available. These are not forced-alignment timestamps.

Every success includes `load_seconds`, `transcribe_seconds`, `duration_seconds`,
`timestamps`, language metadata, and `effective_config`. Errors are returned as
`{"status":"error","error_type":...,"error":...}` and produce a non-zero
worker exit code.
