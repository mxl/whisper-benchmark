FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir --timeout 300 --retries 5 \
    "nemo_toolkit[asr]" \
    "numpy<2" \
    ipython \
    kaldi-native-fbank \
    librosa \
    "onnx==1.17.0" \
    "onnxruntime==1.17.1" \
    soundfile

# Pin sherpa-onnx export script at a known commit
ARG SHERPA_ONNX_REF=master
RUN git clone https://github.com/k2-fsa/sherpa-onnx.git /tmp/sherpa-onnx \
    && cd /tmp/sherpa-onnx \
    && git checkout ${SHERPA_ONNX_REF} \
    && cp -r scripts/nemo/parakeet-tdt-0.6b-v3 /workspace/export-scripts \
    && cp scripts/nemo/generate_bpe_vocab.py /workspace/export-scripts/ \
    && rm -rf /tmp/sherpa-onnx

WORKDIR /workspace/export

CMD ["bash"]
