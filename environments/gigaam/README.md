# Official GigaAM environment

This is the shared isolated Python environment for the official
`ai-sage/GigaAM-v3` `e2e_rnnt` worker and the official
`ai-sage/GigaAM-Multilingual` `large_ctc` worker. Both reuse Python
**3.13.12**, `.venvs/gigaam`, and the exact pins in
[`requirements.txt`](requirements.txt). Both paths use the official
PyTorch/Transformers implementation, with CPU or MPS where available, and
offline local snapshots. Do not create a separate GigaAM Multilingual
virtualenv.

The cached GigaAM-v3 model README contains only MIT license metadata, so it does
not provide dependency instructions. The Multilingual model README documents
the PyTorch/Transformers path. No standalone `gigaam` package is pinned: the
official snapshots supply `modeling_gigaam.py`, loaded through Transformers'
trusted local remote code. Unrecorded transitive versions are left to `uv`; no
versions are guessed here.

`pyannote-audio` is required when the current GigaAM-v3 modeling file is
imported. The worker does not call the official VAD-based
`transcribe_longform`, so it does not download a pyannote model and does not
require `HF_TOKEN`; the same no-longform policy applies to Multilingual.

## Create the environment

Run from the repository root:

```bash
uv venv --python 3.13.12 .venvs/gigaam
uv pip install --python .venvs/gigaam/bin/python \
  -r environments/gigaam/requirements.txt
```

The model must already be present in the canonical offline HF cache. The
proven GigaAM-v3 snapshots and the currently cached Multilingual snapshot are:

```text
/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-v3/snapshots/cec030b4c4f35d928e4a9044a3bdb29ebd499fac  # e2e_ctc
/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-v3/snapshots/7655ad717f8122257385bb4b2f373db3697e8680  # e2e_rnnt
/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-Multilingual/snapshots/3905cd51c3ed4e88c8edf33f3302969ba480a327  # large_ctc, current local ref
```

The Multilingual runner must read `refs/large_ctc` and resolve its revision to
the corresponding snapshot at runtime. The SHA above is the current local
offline cache observation, not a value to hard-code in the integration.

## Long audio and offline policy

The official remote-code `transcribe_longform` implementation is intentionally
not used by either worker. It invokes a Hugging Face-hosted pyannote
segmentation model and therefore needs remote authentication and model access,
which violates the benchmark's completely offline policy. Neither worker needs
`HF_TOKEN` at runtime.

When short-form `transcribe` returns the exact too-long error, the worker reads
the audio locally with `soundfile==0.14.0`, downmixes it to mono, and writes
deterministic 16 kHz WAV chunks. It transcribes each chunk with the same
official short-form `model.transcribe` method, using 25 seconds per chunk by
default and no overlap. Both workers use this fixed-chunk offline behavior.
The bundled MP3 corpus was verified to read as 16 kHz in the isolated
environment; other sample rates are rejected explicitly for now.

## Offline worker smoke

This invokes the worker directly with the proven RNNT snapshot and bundled RU
sample; it must not resolve a model or audio over the network:

```bash
printf '%s\n' '{"model_path":"/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-v3/snapshots/7655ad717f8122257385bb4b2f373db3697e8680","audio_path":"samples/ruls_sample_8169_13240.mp3","language":"ru","variant":"e2e_rnnt"}' |
  HF_HOME=/Volumes/512GB/hf \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  .venvs/gigaam/bin/python -m stt_benchmark.workers.gigaam
```

The pinned `soundfile` dependency reads the bundled MP3 input in this isolated
environment; no network access or remote long-form model is used.

That command is the GigaAM-v3 smoke. The separate Multilingual `large_ctc`
path has completed offline RU/EN smoke and benchmark checks; see
[`TASKS.md`](../../TASKS.md) for the recorded benchmark observations. It
supports RU, EN, KK, KY, and UZ and does not apply the v3 `ru-only model` skip.

## Multilingual offline worker smoke

The integration resolves the Multilingual model through the local
`refs/large_ctc` file and passes the resulting snapshot directory to the
worker. From the repository root, this runs the worker directly for the bundled
EN sample without network access:

```bash
cache=/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-Multilingual
revision="$(tr -d '\n' < "$cache/refs/large_ctc")"
printf '%s\n' "{\"model_path\":\"$cache/snapshots/$revision\",\"audio_path\":\"samples/librispeech_1089_134686.mp3\",\"language\":\"en\",\"variant\":\"large_ctc\"}" |
  HF_HOME=/Volumes/512GB/hf \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  .venvs/gigaam/bin/python -m stt_benchmark.workers.gigaam_multilingual
```

## Integration overrides

The integration honors these optional environment variables:

- `GIGAAM_PYTHON`: worker interpreter override; otherwise use
  `.venvs/gigaam/bin/python`.
- `GIGAAM_MODEL_PATH`: already-resolved local snapshot directory override;
  otherwise select the configured official snapshot above.

These are integration-level overrides. The worker itself receives the
resolved `model_path` in its JSON request and always loads it with
`local_files_only=True`.

GigaAM-v3 remains a separate RU-only integration. GigaAM Multilingual is a
separate `large_ctc` model path and supports RU, EN, KK, KY, and UZ; EN and
`auto` must not be skipped for it.
