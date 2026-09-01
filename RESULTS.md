# Fixed-Corpus Benchmark Results

Date: 2026-09-01. Hardware: Apple M1 Max, macOS. All inference used exact
local artifacts under `/Volumes/512GB/hf` with model downloads disabled. These
are observations on the two bundled samples, not universal rankings.

## Corpus

| Sample | Language | Duration |
|---|---|---:|
| `librispeech_1089_134686.mp3` | EN | 179.810 s |
| `ruls_sample_8169_13240.mp3` | RU | 298.120 s |

## Baseline

Three cold runs per row. `RTF` is transcription time divided by audio duration.

| Lang | Backend/model | Median total | RTF | WER | CER | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| RU | MLX Whisper large-v3-turbo | 27.083 s | 0.090 | 0.043 | 0.021 | n/a |
| RU | GigaAM-v3 RNNT | 59.249 s | 0.167 | 0.052 | 0.013 | 1828 MiB |
| RU | GigaAM Multilingual large CTC | 114.323 s | 0.322 | 0.025 | 0.005 | 2775 MiB |
| RU | T-one greedy | 66.925 s | 0.230 | 0.027 | 0.004 | 540 MiB |
| RU | Zipformer2 full | 14.042 s | 0.042 | 0.037 | 0.009 | 897 MiB |
| RU | Qwen3-ASR 0.6B MLX 8-bit | 33.296 s | 0.068 | 0.112 | 0.032 | 1802 MiB |
| EN | MLX Whisper large-v3-turbo | 15.408 s | 0.090 | 0.010 | 0.004 | n/a |
| EN | GigaAM Multilingual large CTC | 88.201 s | 0.377 | 0.061 | 0.021 | 2766 MiB |
| EN | Qwen3-ASR 0.6B MLX 8-bit | 23.932 s | 0.048 | 0.006 | 0.002 | 1669 MiB |

Evidence: `evidence/2026-09-01/main-ru.json`,
`evidence/2026-09-01/main-en.json` and
`evidence/2026-09-01/podlodka.json`.

### Podlodka

| Lang | Median total | RTF | WER | CER | Peak RSS |
|---|---:|---:|---:|---:|---:|
| EN | 34.251 s | 0.067 | 0.063 | 0.058 | 593 MiB |
| RU | 45.987 s | 0.109 | 0.065 | 0.043 | 641 MiB |

## Whisper.cpp Quantization

Three cold runs with official `whisper-cli 1.9.2` and Metal.

| Lang | Model | Quant | Median total | RTF | WER | CER | RSS |
|---|---|---|---:|---:|---:|---:|---:|
| EN | tiny | FP16 | 6.973 s | 0.036 | 0.056 | 0.020 | 313 MiB |
| EN | tiny | Q5_1 | 6.760 s | 0.037 | 0.061 | 0.017 | 235 MiB |
| EN | tiny | Q8_0 | 6.790 s | 0.037 | 0.065 | 0.019 | 253 MiB |
| EN | turbo | FP16 | 15.099 s | 0.107 | 0.006 | 0.002 | 1918 MiB |
| EN | turbo | Q5_0 | 19.078 s | 0.105 | 0.004 | 0.002 | 886 MiB |
| EN | turbo | Q8_0 | 17.085 s | 0.095 | 0.008 | 0.003 | 1190 MiB |
| RU | tiny | FP16 | 18.256 s | 0.064 | 0.255 | 0.062 | 330 MiB |
| RU | tiny | Q5_1 | 20.557 s | 0.069 | 0.238 | 0.056 | 254 MiB |
| RU | tiny | Q8_0 | 15.686 s | 0.051 | 0.237 | 0.054 | 263 MiB |
| RU | turbo | FP16 | 29.676 s | 0.098 | 0.027 | 0.011 | 1949 MiB |
| RU | turbo | Q5_0 | 35.465 s | 0.114 | 0.057 | 0.037 | 907 MiB |
| RU | turbo | Q8_0 | 29.859 s | 0.103 | 0.027 | 0.011 | 1220 MiB |

Q8 retained FP16 RU accuracy at lower RSS. Q5 reduced memory further but
degraded RU quality on this sample. Evidence:
`evidence/2026-09-01/whisper-cpp.json`.

## Parakeet

Three cold runs per row.

| Lang | Runtime | Median total | RTF | WER | CER | RSS |
|---|---|---:|---:|---:|---:|---:|
| EN | official HF | 45.104 s | 0.114 | 0.004 | 0.001 | 758 MiB |
| EN | Sherpa FP32 | 66.002 s | 0.304 | 0.004 | 0.001 | 5120 MiB |
| EN | Sherpa INT8 | 36.795 s | 0.202 | 0.008 | 0.002 | 3866 MiB |
| RU | official HF | 67.927 s | 0.096 | 0.048 | 0.011 | 889 MiB |
| RU | Sherpa FP32 | 119.584 s | 0.364 | 0.042 | 0.010 | 5463 MiB |
| RU | Sherpa INT8 | 87.101 s | 0.270 | 0.072 | 0.021 | 4224 MiB |

Sherpa FP16 remains blocked by internal ONNX Cast/type conflicts. Evidence:
`evidence/2026-09-01/parakeet.json`.

## RU Variants

Three cold runs per row.

| Backend/model | Median total | RTF | WER | CER | RSS |
|---|---:|---:|---:|---:|---:|
| GigaAM-v3 RNNT | 119.488 s | 0.300 | 0.052 | 0.013 | 1821 MiB |
| GigaAM-v3 CTC | 54.950 s | 0.146 | 0.062 | 0.017 | 1819 MiB |
| Zipformer2 full | 29.346 s | 0.098 | 0.037 | 0.009 | 858 MiB |
| Zipformer2 small | 41.170 s | 0.128 | 0.058 | 0.013 | 524 MiB |

Evidence: `evidence/2026-09-01/ru-variants.json`.

## Extended Models

One cold run per row because of their higher cost. Peak RSS is the subprocess
resident-set measurement from `/usr/bin/time -l`; for MPS/Metal runtimes it is
not complete unified-memory or GPU-allocation accounting.

| Lang | Runtime | Total | RTF | WER | CER | RSS |
|---|---|---:|---:|---:|---:|---:|
| EN | Qwen 0.6B MLX 8-bit | 38.974 s | 0.078 | 0.006 | 0.002 | 1678 MiB |
| EN | Qwen 1.7B MLX 8-bit | 59.504 s | 0.122 | 0.004 | 0.001 | 3073 MiB |
| EN | Qwen 1.7B official HF | 229.193 s | 0.917 | 0.008 | 0.002 | 596 MiB |
| RU | Qwen 0.6B MLX 8-bit | 51.471 s | 0.101 | 0.112 | 0.032 | 1820 MiB |
| RU | Qwen 1.7B MLX 8-bit | 83.028 s | 0.152 | 0.052 | 0.011 | 3194 MiB |
| RU | Qwen 1.7B official HF | 800.014 s | 2.437 | 0.068 | 0.018 | 598 MiB |
| EN | GigaAM Multilingual official | 114.286 s | 0.446 | 0.061 | 0.021 | 2770 MiB |
| EN | GigaAM Multilingual MLX FP16 | 19.223 s | 0.017 | 0.059 | 0.030 | 1238 MiB |
| RU | GigaAM Multilingual official | 227.836 s | 0.637 | 0.025 | 0.005 | 2773 MiB |
| RU | GigaAM Multilingual MLX FP16 | 24.041 s | 0.026 | 0.033 | 0.016 | 1273 MiB |
| EN | VibeVoice-ASR HF | 545.228 s | 1.721 | 0.006 | 0.002 | 762 MiB |
| RU | VibeVoice-ASR HF | 671.448 s | 1.412 | 0.068 | 0.014 | 792 MiB |

The official Qwen worker uses deterministic 30-second chunks because a single
full EN request returned only `language!`. ForcedAligner is excluded from these
timings. VibeVoice is accurate on EN but slower than real time. Evidence:
`evidence/2026-09-01/qwen.json`,
`evidence/2026-09-01/gigaam-multilingual.json`,
`evidence/2026-09-01/vibevoice-en-profile.json` and
`evidence/2026-09-01/vibevoice-ru.json`.

## Blockers

- Borealis was not executed. Reviewed local remote code unconditionally loads
  assets from `openai/whisper-large-v3` and `Qwen/Qwen3-4B`; neither dependency
  is in the canonical cache. Network access would violate the benchmark.
- Parakeet Sherpa FP16 is blocked by invalid mixed-type ONNX graphs.
- ForcedAligner remains a separate second-stage alignment benchmark.
- Podlodka MLX and classic Vosk are optional parity tasks and do not block this
  report.

## Commands

```bash
uv run stt-benchmark benchmark --profile main --audio ru --runs 3
uv run stt-benchmark benchmark --profile main --audio en --runs 3
uv run stt-benchmark benchmark --profile podlodka --runs 3
uv run stt-benchmark benchmark --profile whisper-cpp --runs 3
uv run stt-benchmark benchmark --profile parakeet --runs 3
uv run stt-benchmark benchmark --profile ru-variants --runs 3
uv run stt-benchmark benchmark --profile qwen --runs 1 --worker-timeout-seconds 1800
uv run stt-benchmark benchmark --profile gigaam-multilingual --runs 1
uv run stt-benchmark benchmark --profile vibevoice --runs 1 --worker-timeout-seconds 1800
```
