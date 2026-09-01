# Parakeet Sherpa-ONNX environment

This environment runs the local NeMo Parakeet TDT Sherpa-ONNX export on the
CPU. It does not resolve model names, access Hugging Face, or download audio or
model files at runtime.

## Create the isolated environment

Run from the repository root:

```bash
uv venv --python 3.13.12 .venvs/parakeet-sherpa
uv pip install --python .venvs/parakeet-sherpa/bin/python \
  -r environments/parakeet-sherpa/requirements.txt
```

## Required local artifacts

Pass the exact directory containing one of these complete layouts as
`model_path`:

```text
model/
├── encoder.onnx
├── encoder.weights
├── decoder.onnx
├── joiner.onnx
├── tokens.txt
└── bpe.vocab
```

The INT8 layout uses the exact quantized ONNX names instead:

```text
model/
├── encoder.int8.onnx
├── decoder.int8.onnx
├── joiner.int8.onnx
├── tokens.txt
└── bpe.vocab
```

The worker never falls back between FP32 and INT8 artifacts. FP16 is not
supported: the Parakeet Sherpa FP16 export is blocked by ONNX Runtime internal
`Cast` type conflicts.

## Offline worker invocation

The request is one JSON object. `quantization` defaults to `fp32` and `threads`
defaults to `4`; `threads` must be positive.

```bash
printf '%s\n' \
  '{"model_path":"/path/to/model","audio_path":"/path/to/audio.wav","quantization":"int8","threads":4}' \
  | .venvs/parakeet-sherpa/bin/python -m stt_benchmark.workers.parakeet_sherpa
```

Audio is read as float32, downmixed to mono, and resampled with SciPy to the
model's 16 kHz sample rate when necessary. Sherpa-ONNX receives one complete
offline stream using CPU provider, greedy search, and `max_active_paths=4`.

Successful responses include load/transcription timings, the selected files,
quantization, CPU configuration, and an `effective_config` object. When the
Sherpa result exposes both `tokens` and `timestamps`, `segments` contains token
start records with `token/frame starts` semantics. Otherwise `segments` and
`timestamps` are empty and timestamp semantics are reported as `unsupported`.
