# Official VibeVoice-ASR HF environment

This environment runs `microsoft/VibeVoice-ASR-HF` with the native
Transformers 5.9.0 API. It is isolated because the repository's main
environment does not provide the required VibeVoice model class.

## Create the environment

Run from the repository root:

```bash
uv venv --python 3.12 .venvs/vibevoice
uv pip install --python .venvs/vibevoice/bin/python \
  -r environments/vibevoice/requirements.txt
```

## Exact local model snapshot

The inspected official snapshot is:

```text
/Volumes/512GB/hf/hub/models--microsoft--VibeVoice-ASR-HF/snapshots/f22241c2062b3b25272bf117397e03d73381037a
```

Pass that snapshot directory to the worker, never the Hub ID or a URL. The
worker validates `config.json`, the safetensors index, all eight
`model-00001-of-00008.safetensors` through `model-00008-of-00008.safetensors`
shards, tokenizer content, and the processor config before loading. The
official snapshot also includes its tokenizer config and chat template. Both
Transformers loaders use `local_files_only=True`.

## Runtime contract

- `device` is `auto`, `mps`, or `cpu`; `auto` selects MPS when available and
  otherwise CPU.
- Audio is read as float32 with `soundfile`, downmixed to mono, and resampled
  with scipy to the processor's official 24 kHz rate.
- Inference uses `AutoProcessor.apply_transcription_request(audio=waveform)`
  followed by direct `VibeVoiceAsrForConditionalGeneration.generate`.
- The worker does not use `pipeline`, TorchCodec, URL audio loading, or network
  fallback.
- Generation is deterministic (`do_sample=False`).
- `mode` defaults to `transcription_only`. `mode: "parsed"` additionally
  returns the official Who/When/What dictionaries with their `Start`, `End`,
  `Speaker`, and `Content` fields preserved.
- `acoustic_tokenizer_chunk_size` is optional. When supplied, it is passed to
  the official `generate` API and must be a multiple of 3200, the model's
  acoustic-tokenizer hop length.
- Runs set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

## Direct worker smoke

The model is an 8.3B-parameter checkpoint and can require substantial memory.
The command below is intentionally a real smoke invocation; unit tests inject
fake processor/model/audio objects and do not load the checkpoint.

```bash
MODEL_PATH=/Volumes/512GB/hf/hub/models--microsoft--VibeVoice-ASR-HF/snapshots/f22241c2062b3b25272bf117397e03d73381037a
test -d "$MODEL_PATH"

printf '%s\n' \
  '{"model_path":"'"$MODEL_PATH"'","audio_path":"samples/librispeech_1089_134686.mp3","mode":"transcription_only","device":"auto"}' \
  | HF_HOME=/Volumes/512GB/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=. \
    .venvs/vibevoice/bin/python -m stt_benchmark.workers.vibevoice
```
