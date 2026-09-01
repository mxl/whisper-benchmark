# Official T-one environment

This is the isolated environment for the official
[voicekit-team/T-one](https://github.com/voicekit-team/T-one) streaming CTC
pipeline. The source is installed from GitHub at commit
`3c5b6c015038173840e62cea99e10cdb1c759116`.

There is no official PyPI `tone` package: the PyPI name is unrelated. Do not
replace the GitHub requirement with `tone==...`. The dependency ranges in
[`requirements.txt`](requirements.txt) come from the official `pyproject.toml`
at the pinned commit. `miniaudio` is included because the benchmark uses
`read_audio`; it is supplied by T-one's official `demo` extra.

## Create the environment

Run from the repository root. Python 3.12 is within T-one's officially
supported Python 3.9–3.12 range:

```bash
uv venv --python 3.12 .venvs/t-one
uv pip install --python .venvs/t-one/bin/python \
  -r environments/t-one/requirements.txt
```

## Required local model snapshot

The benchmark never downloads a model. Resolve the exact Hugging Face cache
snapshot from the cached `main` ref:

```bash
MODEL_CACHE=/Volumes/512GB/hf/hub/models--t-tech--T-one
MODEL_SHA=$(tr -d '\n' < "$MODEL_CACHE/refs/main")
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_SHA"

test -f "$MODEL_PATH/model.onnx"
printf '%s\n' "$MODEL_PATH"
```

In other words, `refs/main` contains `<sha>`, and the model directory is
`/Volumes/512GB/hf/hub/models--t-tech--T-one/snapshots/<sha>`. The snapshot
must contain `model.onnx`. `kenlm.bin` is optional for the benchmark's default
greedy decoder, but required when beam decoding is selected.

## Offline, CPU-only benchmark policy

T-one is used with 8 kHz audio on CPU. Keep Hugging Face access offline and
pass the resolved local snapshot to the worker:

```bash
export HF_HOME=/Volumes/512GB/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The benchmark worker uses T-one's local `from_local` path and greedy decoding
by default. It must not use the online `from_hugging_face` path. `kenlm` stays
installed because it is an official runtime dependency and is needed for
beam-decoding runs when the local snapshot provides `kenlm.bin`.
