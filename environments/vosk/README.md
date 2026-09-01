# Vosk-named Russian Zipformer2 environment

## Runtime choice

The cached repositories
[`alphacep/vosk-model-ru`](https://huggingface.co/alphacep/vosk-model-ru)
and
[`alphacep/vosk-model-small-ru`](https://huggingface.co/alphacep/vosk-model-small-ru)
contain Zipformer2 ONNX transducer models, version 0.54. They are not classic
Kaldi Vosk model packages. The isolated environment therefore uses the PyPI
package `sherpa-onnx`, not `vosk`.

The classic `vosk` API is intentionally not used. `vosk` is also intentionally
absent from [`requirements.txt`](requirements.txt); the worker loads the local
ONNX encoder/decoder/joiner triplet through sherpa-onnx instead.

The NumPy range matches the existing isolated T-one ONNX environment in this
repository. `soundfile==0.14.0` matches the current isolated audio-runtime pin
used by the GigaAM worker and provides the worker's local float32 audio input.

## Create the isolated environment

Run from the repository root. This task does not execute package or model
downloads:

```bash
uv venv --python 3.13.12 .venvs/vosk
uv pip install --python .venvs/vosk/bin/python \
  -r environments/vosk/requirements.txt
```

## Resolve the cached big model

Do not hard-code a snapshot revision. Resolve the local snapshot through the
cached `refs/main` file:

```bash
MODEL_CACHE=/Volumes/512GB/hf/hub/models--alphacep--vosk-model-ru
MODEL_SHA=$(tr -d '\n' < "$MODEL_CACHE/refs/main")
MODEL_PATH="$MODEL_CACHE/snapshots/$MODEL_SHA"

test -d "$MODEL_PATH"
test -f "$MODEL_PATH/am-onnx/encoder.onnx"
test -f "$MODEL_PATH/am-onnx/decoder.onnx"
test -f "$MODEL_PATH/am-onnx/joiner.onnx"
test -f "$MODEL_PATH/lang/tokens.txt"
printf '%s\n' "$MODEL_PATH"
```

The required big-model layout is `am-onnx/{encoder,decoder,joiner}.onnx`
plus `lang/tokens.txt`:

```text
$MODEL_PATH/
├── am-onnx/
│   ├── encoder.onnx
│   ├── decoder.onnx
│   └── joiner.onnx
└── lang/
    └── tokens.txt
```

The same `am-onnx/` directory also contains the optional
`encoder.int8.onnx`, `decoder.int8.onnx`, and `joiner.int8.onnx` artifacts.
Select the int8 triplet only for an int8 run; otherwise use the FP32 triplet
above.

The small snapshot is resolved in the same way from
`/Volumes/512GB/hf/hub/models--alphacep--vosk-model-small-ru/refs/main`, but
its model files are under `am/{encoder,decoder,joiner}.onnx` rather than
`am-onnx/`:

```text
$SMALL_MODEL_PATH/
├── am/
│   ├── encoder.onnx
│   ├── decoder.onnx
│   └── joiner.onnx
└── lang/
    └── tokens.txt
```

The small snapshot also has the corresponding `encoder.int8.onnx`,
`decoder.int8.onnx`, and `joiner.int8.onnx` files in `am/`.

## CPU and audio contract

The worker is CPU-only. Configure the sherpa-onnx model with provider `cpu`
and an explicit CPU thread count; the smoke command below uses
`num_threads=4`, matching the repository's existing sherpa-onnx CPU smoke
setting. Do not use a CUDA, MPS, or other accelerator provider for this
environment.

The worker input contract is 16 kHz audio represented as `float32` samples.
Prepare local audio at 16,000 Hz before invoking the worker; no model or audio
resource is resolved from the network.

## Offline worker smoke

After resolving `MODEL_PATH` in the same shell, run the big-model worker
directly:

```bash
printf '%s\n' \
  '{"model_path":"'"$MODEL_PATH"'","audio_path":"samples/ruls_sample_8169_13240.mp3","language":"ru","variant":"big","provider":"cpu","num_threads":4}' \
  | HF_HOME=/Volumes/512GB/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    .venvs/vosk/bin/python -m stt_benchmark.workers.vosk
```

The request passes an already-resolved local model directory to
`stt_benchmark.workers.vosk`. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` keep the worker invocation aligned with the
repository's offline subprocess policy; no Hugging Face snapshot resolution,
model download, or classic Vosk API call is expected at runtime.

For the small model, resolve `SMALL_MODEL_PATH` from its own `refs/main` file,
use that value as `model_path`, and change `variant` to `small`. The optional
int8 filenames are selected by the worker's int8 variant/configuration rather
than by the classic `vosk.Model` interface.
