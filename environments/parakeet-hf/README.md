# Official Parakeet HF environment

This environment runs the official `nvidia/parakeet-tdt-0.6b-v3` model through
Transformers 5.9.0's `AutoProcessor` and `AutoModelForTDT`. It is separate from
the repository's main environment because older Transformers releases do not
recognize `parakeet_tdt`.

## Create the environment

Run from the repository root:

```bash
uv venv --python 3.12 .venvs/parakeet-hf
uv pip install --python .venvs/parakeet-hf/bin/python \
  -r environments/parakeet-hf/requirements.txt
```

## Local-only model

Resolve a local Hugging Face snapshot and pass the snapshot directory to the
worker. Do not pass a Hub ID or URL:

```bash
MODEL_CACHE=/Volumes/512GB/hf/hub/models--nvidia--parakeet-tdt-0.6b-v3
MODEL_SHA=$(tr -d '\n' < "$MODEL_CACHE/refs/main")
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_SHA"
test -d "$MODEL_PATH"
```

The snapshot must contain `config.json`, model weights (a single
`model.safetensors` or a safetensors index), `processor_config.json`, and
tokenizer content (`tokenizer.json` in the official snapshot). The worker uses
`local_files_only=True` for both Transformers loaders and never downloads
files.

## Runtime contract

- Use `device` `auto`, `mps`, or `cpu`; `auto` selects MPS when available and
  otherwise CPU.
- Audio is read with soundfile as float32, downmixed to mono, and resampled
  with scipy to the processor's sampling rate.
- Generation requests `return_dict_in_generate=True`; token timestamps are
  decoded from the official `durations` output when available.
- TorchCodec, `pipeline`, and `datasets` are not used.
- Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for direct runs.

## Direct worker smoke

```bash
printf '%s\n' \
  '{"model_path":"'"$MODEL_PATH"'","audio_path":"samples/librispeech_1089_134686.mp3","device":"auto"}' \
  | HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
    .venvs/parakeet-hf/bin/python -m stt_benchmark.workers.parakeet_hf
```

The command requires the environment to be installed and uses only the
resolved local model snapshot and local audio.
