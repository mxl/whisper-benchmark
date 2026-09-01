# План улучшения STT-бенчмарка

## Project Charter

Цель: единый локальный offline benchmark для EN/RU STT model/runtime
комбинаций на Apple Silicon. Результаты должны содержать provenance,
effective config, accuracy, cold/warm timing, memory и disk footprint.
Downloaded source models live in `/Volumes/512GB/hf/hub`; reproducibly derived
artifacts live in `/Volumes/512GB/hf/derived`.

Non-goals:

- cloud API и streaming;
- legacy CLI и совместимость со старым JSON;
- автоматическая загрузка моделей агентом или benchmark-кодом;
- искусственный composite score.

Модели хранятся только в HF cache `/Volumes/512GB/hf`. Модели скачивает
пользователь. Агент не запускает `hf download`.

## Success Criteria

- Каждая строка содержит model/runtime/weights/runtime-version provenance.
- Каждая строка содержит effective precision и quantization scheme; качество
  квантизированного варианта сравнивается с baseline тех же model/runtime.
- Каждый ready backend проходит smoke profile без ручной правки кода.
- Unit/contract tests не требуют моделей или сети.
- Missing model/dependency даёт `skipped`, не падение общего запуска.
- Проект не создаёт копии весов вне `/Volumes/512GB/hf`.
- Сохранённая CLI-команда и JSON config воспроизводят опубликованный отчёт.
- Допустимый разброс warm RTFx определяется spike S5.

## Workflow

- Checkbox: `- [ ]` до начала, `- [x]` после завершения.
- Status: `READY`, `IN_PROGRESS`, `BLOCKED`, `DONE`, `SUPERSEDED`.
- Одновременно выполняется одна agent-задача; U1 может идти параллельно.
- Production-код нового backend запрещён до связанного spike.
- Код: `RED -> GREEN -> REFACTOR`; GREEN содержит минимальный код.
- После каждой задачи: проверки, evidence, результат, обновление зависимых
  задач, checkbox/status, отдельный commit, фиксация commit hash.
- Один task = один commit. Несколько задач в commit не объединять.
- После каждого spike остановиться на plan update gate.
- Результат spike хранить прямо в его задаче: дата, environment, hypothesis,
  commands, observations, measurements, decision, consequences, blockers,
  follow-up tasks.

## Task Template

Перед началом дополнить задачу:

- `Status`, `Owner`, `Priority`, `Dependencies`;
- `Deliverables`, `Evidence`, `Commit`.

Definition of Done всегда включает: acceptance criteria выполнены, тесты
green, `TASKS.md` обновлён, задача отмечена, создан отдельный commit.

## Change Log

| Дата | Task | Изменение | Причина/evidence | Commit |
|---|---|---|---|---|
| 2026-07-26 | PLAN | Initial approved plan | User-approved scope | 49bad00 |
| 2026-07-26 | PLAN | Add Parakeet FP32 via onnx-asr | Separate unquantized ONNX runtime requested; repo contains FP32 and INT8 variants | e06ec26 |
| 2026-07-26 | PLAN | Add direct sherpa-onnx Parakeet FP32 repo | Verified separate encoder/decoder/joiner FP32 artifacts on Hugging Face | 5e33374 |
| 2026-07-26 | PLAN | Benchmark quantized variants | User requested precision/quantization matrix instead of excluding quants | 51fb2a8 |
| 2026-07-26 | PLAN | Build sherpa-onnx variants from official source | Avoid third-party converted weights; control FP32/FP16/INT8 provenance | 643f133 |
| 2026-07-27 | S2 | Confirm official whisper-cli backend | User selected official CLI over custom Python binding | d0d0c7d |
| 2026-07-27 | S3 | Require exact official GigaAM revisions | Cache inspection found only main snapshot; parity needs e2e_ctc and e2e_rnnt | 9271a53 |
| 2026-07-27 | U5M | Normalize GigaAM cache layout | User selected moving valid revisions into canonical `/Volumes/512GB/hf/hub` | pending |
| 2026-07-27 | S3 | Record GigaAM parity spike | All four GigaAM-v3 variants identical RU; official fast, MLX slow; v3 RU-only | pending |
| 2026-07-28 | S5 | Record worker isolation spike | Subprocess for all backends; cold RTFx published; ±15% tolerance; D-003 closed | pending |
| 2026-07-28 | S4E | Record Parakeet sherpa export spike | FP32+INT8 from official .nemo; FP16 BLOCKED (internal Cast conflicts); third-party repos superseded | pending |
| 2026-07-28 | S4 | Record Parakeet runtimes spike | 6 runtimes probed (official HF, MLX, sherpa FP32/INT8, onnx-asr FP32/INT8); official HF fastest RTFx 25-32x; INT8 4-15x at 1/4 disk; FP16 BLOCKED | pending |
| 2026-07-28 | S7 | Record Canary runtimes spike | 2 runtimes probed (official NeMo CPU RTFx 2.07-2.29x, MLX bf16 RTFx 76-101x); 3 mlx-audio patches required; MLX RU greedy decoding unreliable | pending |
| 2026-08-31 | T0.6 | Minimal T-one subprocess integration | Official RU-only T-one worker, local/offline snapshot, main one-run result recorded | no commit |
| 2026-08-31 | T0.7 | Minimal Vosk-named Zipformer2 ONNX subprocess integration | RU-only sherpa-onnx worker, exact local snapshot, main one-run result recorded | no commit |
| 2026-08-31 | T0.8 | Minimal Qwen3-ASR MLX subprocess integration | Multilingual 0.6B 8-bit MLX worker, local/offline snapshot, main one-run result recorded | no commit |
| 2026-08-31 | T0.9 | Complete GigaAM Multilingual integration | Official `large_ctc` path, exact local cache revision, shared offline GigaAM environment, RU/EN one-run results recorded | no commit |
| 2026-08-31 | T0.10 | Complete Podlodka HF/Transformers subprocess integration | Direct local Transformers worker, exact snapshot/offline contract, RU+EN smoke and profile metrics recorded; pipeline rejected because of TorchCodec/FFmpeg incompatibility | no commit |
| 2026-09-01 | PLAN | Replace framework-first roadmap with lean benchmark delivery | Prioritize repeated baseline, thin workers, high-value model coverage and a fixed-corpus report; retain I1-I7 only as historical backlog | no commit |

## Decision Log

| ID | Task | Решение | Альтернативы | Последствия | Status |
|---|---|---|---|---|---|
| D-001 | S1 | Resolve exact local snapshot, then pass its path to backend | repo ID passed directly to backend | Prevents downloads and pins SHA; env remains required | CLOSED |
| D-002 | S2 | Use official whisper-cli subprocess | pywhispercpp in-process binding | CLI is upstream 1.9.1; binding HEAD embeds older whisper.cpp 1.8.4 and changes transcripts | CLOSED |
| D-003 | S5 | Subprocess isolation for all backends | in-process vs isolated worker | Crash/timeout isolation for foreign runtimes; per-backend venv; RSS overhead acceptable | CLOSED |
| D-004 | S4E | Build FP32/INT8 from upstream export and derive FP16 ourselves | Third-party converted HF repos | One official source SHA and controlled conversion pipeline | APPROVED |
| D-005 | U5M | Merge downloaded GigaAM cache into canonical hub, then remove source copy after verification | Support two roots or redownload | Preserve one cache root without another network transfer | APPROVED |
| D-006 | S3 | Include official CTC and RNNT; MLX variants parity but slow | MLX-only or CTC-only | Both official runtimes fast and identical transcripts; MLX ~36s/20s impractical | CLOSED |
| D-007 | S3 | Tested GigaAM-v3 variants are RU-only; EN rows `skipped (ru-only model)` | Skip model entirely | Keeps RU coverage, honest EN gap | CLOSED |
| D-008 | S4E | FP16 Parakeet via sherpa-onnx BLOCKED; FP32+INT8 ready | Force FP16 now | FP16 conversion fails at ORT load due to internal Cast conflicts; 6 approaches tried | CLOSED |
| D-009 | S4 | Six Parakeet rows in benchmark: parakeet_hf, parakeet_mlx, parakeet_sherpa_fp32/int8, parakeet_onnx_asr_fp32/int8 | Fewer runtimes | Maximizes runtime coverage for the same source model; each isolated venv per S5 | CLOSED |
| D-010 | S4 | official HF reference (fastest, RTFx 25-32x); INT8 variants best quantized (RTFx 4-15x, 639 MiB); MLX portable but slow | Single runtime | Coverage of MPS/CPU/bf16/int8 tradeoffs | CLOSED |
| D-011 | S7 | Include both canary_nemo (official CPU) and canary_mlx (CogniSoftOrg bf16) rows | Single runtime | Reference quality + fast MLX; MLX RU experimental | CLOSED |
| D-012 | S7 | Official NeMo is reference; MLX RU unreliable until mlx-audio fixes greedy decoding | Exclude MLX RU | Publish both; MLX RU flagged experimental | CLOSED |
| D-013 | T0.9 | Keep GigaAM Multilingual `large_ctc` separate from RU-only GigaAM-v3 and do not apply the v3 language skip | Reuse the `gigaam` v3 row and skip EN/auto | Enables RU/EN/KK/KY/UZ coverage through the official PyTorch path | CLOSED |
| D-014 | T0.10 | Use direct local `AutoProcessor` + `WhisperForConditionalGeneration` instead of the Transformers ASR pipeline | Standard ASR pipeline | Transformers 5.5.3 imports TorchCodec even for raw audio; direct generation avoids the local FFmpeg 9 incompatibility and keeps the worker offline | CLOSED |
| D-015 | PLAN | Deliver the benchmark through thin isolated workers and the existing JSON/reporting path | Build I1-I7 framework layers first | Produces useful model comparisons sooner; abstractions are added only after repeated concrete need | APPROVED |
| D-016 | PLAN | Keep rich ASR and alignment capabilities in separate profiles | Mix diarization/alignment latency into plain ASR rows | VibeVoice rich transcription and ForcedAligner remain measurable without corrupting WER/RTF comparability | APPROVED |

## RAID Register

| Type | Item | Impact | Mitigation | Owner | Status |
|---|---|---|---|---|---|
| Risk | MLX ports differ from official weights | High | parity spikes | agent | OPEN |
| Risk | NeMo/Qwen dependency conflicts | High | S5 isolation | agent | OPEN |
| Risk | Library ignores external HF cache | High | S1 explicit paths | agent | MITIGATED |
| Risk | Memory metrics are incomparable | High | S5 methodology | agent | OPEN |
| Risk | Corpus license/reference errors | High | S8 evidence gate | user+agent | OPEN |
| Risk | Premature abstractions | Medium | spike-first TDD | agent | OPEN |
| Dependency | Models downloaded by user | High | U1 gate | user | OPEN |
| Risk | NeMo export is difficult on macOS/Python 3.13 | High | S4E isolated Linux environment | agent | OPEN |

## Milestones

Historical M0/M1 work established the current workers, cache policy and runtime
spikes. The active roadmap is now lean-first:

- L0 Current baseline: validate the current worktree and publish repeated
  `main` and `podlodka` EN/RU results.
- L1 High-value runtimes: integrate official Whisper.cpp FP16/Q5/Q8.
- L2 Parakeet: integrate official HF plus Sherpa-ONNX FP32/INT8; FP16 remains
  blocked and does not hold up the milestone.
- L3 Small comparisons: add official GigaAM-v3 CTC and Vosk small RU beside the
  already integrated RNNT and full models.
- L4 Modern extended models: Qwen3-ASR 1.7B, GigaAM Multilingual MLX,
  VibeVoice-ASR and Borealis after exact local snapshots are available.
- L5 Optional parity/alignment: Podlodka MLX, classic Vosk and ForcedAligner in
  separate profiles only when their specific comparison is required.
- L6 Report and acceptance: assemble the fixed-corpus report from the existing
  JSON, document provenance/blockers and run the final offline verification.

Critical path:
`L0 -> L1 -> L2 -> L3 -> L4 -> L6`. L5 is optional and must not block L6.

The original I1-I7 framework-first tasks remain below as historical backlog.
They are not on the active critical path and must not be implemented without a
new concrete requirement.

## M0: Current Worktree

### - [x] T0.1 Insanely Fast Whisper fix

Status: DONE. Owner: agent. Priority: P0.

RED: test requires `return_timestamps=True` and rejects
`condition_on_prev_tokens`.

GREEN: minimal `benchmark_whisper.py` and `test_benchmark.py` changes.

DoD: targeted test proves regression; full unittest and `git diff --check`
pass; unrelated files excluded; result recorded; commit
`fix: update insanely-fast-whisper invocation`.

Result: 2026-07-26. Existing uncommitted implementation was verified rather
than recreated. Targeted regression test passed (1 test), full suite passed
(39 tests), and `git diff --check` reported no errors. Diff is limited to
`benchmark_whisper.py` and `test_benchmark.py`. Commit subject:
`fix: update insanely-fast-whisper invocation`; commit `32d543a`.

### - [x] T0.2 Direct Hugging Face dependency

Status: DONE. Owner: agent. Priority: P0. Depends on: T0.1.

DoD: `huggingface-hub` is direct dependency; `uv.lock` agrees; clean
`uv sync`, tests and diff check pass; result recorded; commit
`build: declare huggingface-hub dependency`.

Result: 2026-07-26. `uv sync` resolved 159 packages and rebuilt the editable
project successfully. Full suite passed (39 tests), and `git diff --check`
reported no errors. `huggingface-hub>=1.10.1` is present in both
`pyproject.toml` and `uv.lock`. Commit subject:
`build: declare huggingface-hub dependency`; commit `32d1dbd`.

### - [x] T0.3 Ignore local state

Status: DONE. Owner: agent. Priority: P0. Depends on: T0.2.

RED: `.opencode/` and `models/` appear in status.

GREEN: ignore both without deleting local files.

DoD: status no longer lists them; result recorded; commit
`chore: ignore local benchmark state`.

Result: 2026-07-26. Added `.opencode/` and `models/` to `.gitignore` without
deleting either directory. `git status` no longer lists local state. Commit
subject: `chore: ignore local benchmark state`; hash will be recorded in the
next completed task.

### - [x] T0.4 Benchmark profiles

Status: DONE. Owner: agent. Priority: P0. Depends on: T0.3.

At T0.4 completion, the default `main` profile was a compact Russian STT
baseline with exactly two pairs on the bundled RU sample: `mlx-whisper` / `large-v3-turbo` and
`gigaam` / `e2e_rnnt`. The separate `whisper` profile preserves all six
currently implemented Whisper runtimes and all existing Whisper model sizes
across both bundled samples.

Explicit `--models` or `--backends` options switch pair generation to the
Cartesian product of the selected models and backends. `--audio` selects
samples without changing the exact `main` pairs.

DoD: profile parsing tests, metadata, README usage, and unified CLI help are
updated without adding unsupported backend names.

Result: 2026-08-31 at T0.4 completion. `main` then executed exactly two RU
pairs: `mlx-whisper` / `large-v3-turbo` and `gigaam` / `e2e_rnnt`;
`whisper` preserved the previous all-runtime matrix. T0.6 later added the
T-one pair and T0.7 later added the Vosk pair; the current four-pair profile is
recorded in T0.7.

### - [x] T0.5 Minimal GigaAM subprocess integration

Status: DONE. Owner: agent. Priority: P0. Depends on: T0.4, S3, S5, U5M.

Scope: expose only the official RU-only GigaAM-v3 `e2e_rnnt` variant through the
benchmark's isolated-worker path. This is a minimal integration, not the
broader unified backend framework and not completion of the other GigaAM
variants.

Acceptance expectations:

- the setup path creates `.venvs/gigaam` with `uv venv --python 3.13.12` and
  installs `environments/gigaam/requirements.txt`;
- the runner resolves an exact local snapshot, starts the isolated GigaAM
  subprocess, passes offline environment variables, enforces a timeout, and
  maps its single-JSON protocol into benchmark results;
- macOS worker executions record peak resident set size as `peak_rss_mb`;
- the worker rejects non-local paths and does not call official
  `transcribe_longform`; after the exact too-long error it uses deterministic
  25-second, zero-overlap local chunks;
- because this GigaAM-v3 variant is RU-only, the harness checks the resolved input before
  loading/running it; any input whose `forced_language` is not exactly `ru`
  (including `auto`) records `SkippedBenchmark` with reason `ru-only model` and
  produces no GigaAM `RunResult`; the default RU pair remains successful;
- focused tests cover profile pair selection, pinned local-path resolution,
  worker errors/timeouts, JSON protocol, and the too-long fallback.

Result: 2026-08-31. The minimal GigaAM-v3 `gigaam` / `e2e_rnnt` path is implemented with
the default `.venvs/gigaam` interpreter, exact local snapshot resolution,
offline execution, timeout handling, JSON IPC, and macOS peak-RSS capture.
The harness now skips GigaAM-v3 before loading/running it for EN and autodetected
inputs, recording `ru-only model` and emitting no GigaAM-v3 run; the default
GigaAM-v3 RU pair remains active and successful.
The official longform path remains intentionally unused because of its gated
pyannote/HF_TOKEN requirement. No final accuracy, timing, or memory metric
numbers are asserted here; those remain benchmark-output measurements.

Commit: not created (user requested no commit).

### - [x] T0.6 Minimal T-one subprocess integration

Status: DONE. Owner: agent. Priority: P0. Depends on: T0.4, S5.

Scope: expose only the official RU-only T-one `t-one-greedy` variant through
an isolated worker subprocess. This is a minimal integration; it does not
claim broader T-one adapters or a unified backend framework.

Acceptance expectations:

- the setup path creates `.venvs/t-one` with
  `uv venv --python 3.12 .venvs/t-one` and installs
  `environments/t-one/requirements.txt`;
- the worker uses the official GitHub source at commit
  `3c5b6c015038173840e62cea99e10cdb1c759116`, resolves the exact local
  `t-tech/T-one` snapshot, and runs offline through `from_local`;
- inference is CPU-only at 8 kHz, uses greedy decoding by default, returns
  phrase timestamps, and uses `model.onnx`; `kenlm.bin` is optional for greedy
  and required for beam decoding;
- the worker exposes the official streaming path through `streaming`, while
  the `main` profile uses offline mode;
- T-one is skipped for EN and `auto` with reason `ru-only model`, without
  claiming support for other languages or adapters.

Result: 2026-08-31 at T0.6 completion. `main` then executed exactly three RU pairs:
`mlx-whisper` / `large-v3-turbo`, `gigaam` / `e2e_rnnt`, and `t-one` /
`t-one-greedy`. The cached T-one `main` snapshot is
`106f3b0b32a9e107eb613312e4ebc61ff3d53926`.

One sample/one run measured T-one at WER 0.027, CER 0.004, total 101.314 s,
load 2.645 s, transcription 97.983 s, average peak RSS 537.531 MB, and
average RTF 0.329. The same run recorded GigaAM at WER 0.052, CER 0.013,
total 58.166 s, peak RSS 1830.328 MB, and MLX Whisper at WER 0.043, CER
0.021, total 46.843 s. These are one-sample/one-run observations, not a
universal ranking.

Commit: not created (user requested no commit).

### - [x] T0.7 Minimal Vosk-named Zipformer2 ONNX subprocess integration

Status: DONE. Owner: agent. Priority: P0. Depends on: T0.4, S5, U1.

Scope: expose only the RU `vosk` / `vosk-ru` pair through an isolated worker.
This is a minimal integration; it does not claim the classic Vosk Python API,
other Vosk adapters, or a broader unified backend framework.

Acceptance expectations:

- the setup path creates `.venvs/vosk` with Python 3.13.12 and installs
  `environments/vosk/requirements.txt`, including exactly
  `sherpa-onnx==1.13.6`;
- the worker treats cached `alphacep/vosk-model-ru` as a Zipformer2 ONNX
  model, resolves an exact local snapshot, and intentionally does not use the
  classic `vosk` Python API;
- inference is CPU-only at 16 kHz with float32 audio, uses
  `modified_beam_search` and FP32 files in `main`, and detects both the full
  `am-onnx` and small `am` model layouts;
- full and small layouts use deterministic 20-second, zero-overlap chunks;
  token/frame start timestamps are returned as absolute audio timestamps;
- the harness skips Vosk for EN and `auto` with reason `ru-only model`, without
  claiming support for other languages or adapters; cached
  `alphacep/vosk-model-small-ru` is supported by layout detection but is not in
  `main` yet.

Result: 2026-08-31 at T0.7 completion. `main` then executed exactly four RU pairs:
`mlx-whisper` / `large-v3-turbo`, `gigaam` / `e2e_rnnt`, `t-one` /
`t-one-greedy`, and `vosk` / `vosk-ru`. The cached big-model snapshot is
`df6a54a4d8e5d43e82675e4f5dba2d507731a0d1`.

From `/tmp/stt-main-vosk-chunked.json`, one bundled RU sample and one run
measured Vosk at WER 0.037, CER 0.009, total 15.270 s, load 3.362 s,
transcription 11.813 s, peak RSS 954.766 MB, and average RTF 0.040. This is a
sample-specific one-run observation, not a universal ranking.

Commit: not created (user requested no commit).

### - [x] T0.8 Minimal Qwen3-ASR MLX subprocess integration

Status: DONE. Owner: agent. Priority: P0. Depends on: T0.4, S5, U1.

Scope: expose the cached multilingual MLX `qwen3-asr` /
`qwen3-asr-0.6b-8bit` pair through an isolated worker subprocess. This is a
minimal MLX integration; it does not claim the 1.7B model, the official
Hugging Face/Transformers adapter, or completion of the broader S6/N9/N10
Qwen runtime work.

Acceptance expectations:

- the worker uses the existing main `.venv` with `mlx-audio 0.4.2` and MLX;
  no new Qwen-specific virtualenv is created;
- the runner resolves the cached
  `mlx-community/Qwen3-ASR-0.6B-8bit` `refs/main` to exact snapshot
  `89e96d92ba34aca20b3e29fb10cc284097d1219f`, passes its local path, and
  enforces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`;
- execution is isolated in a subprocess on Apple Silicon with MLX, supports
  multilingual RU+EN audio, and returns segment-level rather than word-level
  timestamps; alignment is not run;
- `--qwen3-asr-language` takes precedence; without it, a forced bundled
  sample's concrete language is passed (`ru` for the default main sample),
  while `auto`/`None` omits the language hint;
- at T0.8 completion, the `main` profile had exactly five pairs, adding
  `qwen3-asr` / `qwen3-asr-0.6b-8bit` to the four existing pairs; T0.9 later
  added the GigaAM Multilingual pair;
- at T0.8 completion, Transformers-native Qwen3-ASR requiring Transformers
  `>=5.13` and `ForcedAligner` were deferred/separate.

Result: 2026-08-31. The Qwen3-ASR MLX worker and main-profile pair are
implemented with local snapshot resolution, offline environment variables,
subprocess isolation, and the existing main environment. One bundled RU
sample and one run measured WER 0.112, CER 0.032, total 33.082 s, load
16.050 s, transcription 16.214 s, peak RSS 1816.891 MB, and average RTF
0.054. Qwen3-ASR is experimental and this is not a universal ranking; on this
sample it was worse on WER/CER than T-one, Vosk, GigaAM, and MLX Whisper.

No 1.7B model or official HF/Transformers adapter was implemented in T0.8. The
later S6/L4 work added both 1.7B runtimes; `ForcedAligner` remains separate.

Commit: not created (user requested no commit).

### - [x] T0.9 GigaAM Multilingual official `large_ctc` integration

Status: DONE. Owner: agent. Priority: P0. Depends on: T0.4, S1, S5, U1.

Scope: add a separate official PyTorch GigaAM Multilingual path; do not fold it
into the RU-only GigaAM-v3 `e2e_rnnt` integration.

Acceptance expectations:

- use `ai-sage/GigaAM-Multilingual`, revision `large_ctc`, through the official
  Transformers remote-code/PyTorch path, with CPU or MPS where available;
- reuse `.venvs/gigaam`; do not create a second GigaAM environment;
- resolve the canonical local `refs/large_ctc` ref in code, validate the
  resulting snapshot, pass the exact local path, and set
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`;
- the current local cache observation is snapshot
  `3905cd51c3ed4e88c8edf33f3302969ba480a327` under
  `/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-Multilingual/snapshots/`;
  this SHA must not be hard-coded instead of resolving the ref;
- support the model's RU, EN, KK, KY, and UZ language set; EN and `auto` must
  not be skipped as `ru-only model`;
- avoid gated `transcribe_longform`; after the exact too-long signal, use
  deterministic 25-second, zero-overlap local chunks through short-form
  transcription;
- run a real offline smoke before marking the task complete, then record actual
  benchmark metrics. No final Multilingual metrics are asserted in advance.

Result: 2026-08-31. T0.9 is complete. The offline smoke and benchmark runs
completed successfully. The main profile now has six exact pairs,
adding `gigaam-multilingual` / `gigaam-multilingual-large-ctc` to the existing
five. The integration uses the official PyTorch `large_ctc` path in the shared
`.venvs/gigaam` environment, an offline worker, and runtime resolution of the
local `refs/large_ctc` ref. The current resolved snapshot is
`3905cd51c3ed4e88c8edf33f3302969ba480a327`. Long audio uses deterministic
25-second WAV chunks with zero overlap after the exact too-long signal. The
multilingual path remains separate from RU-only GigaAM-v3.

The main RU run from `/tmp/stt-main-gigaam-multilingual.json` used one bundled
sample and one run: WER 0.025, CER 0.005, total 88.597 s, load 28.738 s,
transcription 59.161 s, peak RSS 2782.062 MB, and average RTF 0.198. The
focused EN run from `/tmp/stt-gigaam-multilingual-en.json` also used one sample
and one run: WER 0.061, CER 0.021, total 64.864 s, load 29.202 s,
transcription 34.905 s, peak RSS 2766.594 MB, and average RTF 0.194.

EN and `auto` are not skipped for this backend; only GigaAM-v3, T-one, and Vosk
remain RU-only. These metrics are one-sample/one-run observations, not a
universal ranking, and make no claim that the multilingual model is globally
best. At T0.9 completion only the official PyTorch `large_ctc` path was
integrated; L4 later added the separate MLX FP16 comparison.

Commit: not created (user requested no commit).

### - [x] T0.10 Minimal Podlodka HF/Transformers subprocess integration

Status: DONE. Owner: agent. Priority: P0. Depends on: T0.4, S1, S5.

Scope: expose the cached Hugging Face-native
`bond005/whisper-podlodka-turbo` model through a separate `podlodka` profile.
Do not add it to the default `main` profile. The model card declares
Apache-2.0 licensing and RU+EN support; this benchmark scope is ASR only.

Acceptance expectations:

- run direct local Transformers inference in an isolated subprocess using the
  repository's `.venv/bin/python` by default: `AutoProcessor` plus
  `WhisperForConditionalGeneration`;
- resolve `/Volumes/512GB/hf/hub/models--bond005--whisper-podlodka-turbo/refs/main`
  to its exact local snapshot, pass that directory to the worker, and enforce
  `HF_HOME=/Volumes/512GB/hf`, `HF_HUB_OFFLINE=1`, and
  `TRANSFORMERS_OFFLINE=1`;
- support `--podlodka-python` / `PODLODKA_PYTHON` and
  `--podlodka-model-path` / `PODLODKA_MODEL_PATH` overrides;
- provide the separate profile command
  `uv run stt-benchmark benchmark --profile podlodka`, covering the bundled RU
  and EN samples without changing `main`;
- make `--podlodka-language` take precedence. Without it, pass a forced `ru` or
  `en` selector as the language hint; for `auto` or an unforced input, omit the
  hint and allow model detection;
- return deterministic fixed-chunk segment offsets; word-level alignment is not
  part of this task;
- explicitly exclude MLX conversion and `whisper-large-v3-ru-podlodka` from
  this integration;
- run a real offline RU+EN smoke and benchmark before marking the task done.
  Record actual metrics only after that run; do not prefill results from the
  model card or another benchmark.

Result: 2026-08-31. The cached `refs/main` resolved at runtime to snapshot
`da87efd100d2111281b1672ad6bd386722b32251`. The worker uses direct local
Transformers classes, `soundfile` float32 audio, 16 kHz resampling when needed,
and 30-second fixed chunks with chunk-boundary offsets. No pipeline, TorchCodec,
FFmpeg, MLX conversion, or `whisper-large-v3-ru-podlodka` path is used.

The first direct smoke exposed a model-specific limit: Podlodka has 448 decoder
target positions and a four-token Whisper prompt, so the default was corrected
from 448 to 444 `max_new_tokens`. The final isolated RU smoke passed with
`/tmp/podlodka-ru20-direct-444.json`.

The final repeated offline profile passed with three cold runs per bundled
sample:

| Language | Duration | Median total | Avg transcribe | RTF | WER | CER | Avg peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN | 179.810 s | 34.251 s | 12.093 s | 0.0673 | 0.0628 | 0.0577 | 592.745 MiB |
| RU | 298.120 s | 45.987 s | 32.561 s | 0.1092 | 0.0650 | 0.0426 | 640.901 MiB |

Evidence: `evidence/2026-09-01/podlodka.json`; command:
`HF_HOME=/Volumes/512GB/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run
stt-benchmark benchmark --profile podlodka --runs 3`. All rows succeeded
(`3/3`), with no skipped rows.

Commit: not created (user requested no commit).

## User Prerequisite

### - [x] U1 Models available in `/Volumes/512GB/hf`

Status: DONE. Owner: user. Priority: P0.

Agent must not run these commands. User download list:

```bash
export HF_HOME="/Volumes/512GB/hf"
export HUGGINGFACE_HUB_CACHE="/Volumes/512GB/hf/hub"

hf download ggerganov/whisper.cpp --include "ggml-tiny.bin" --include "ggml-large-v3-turbo.bin" --cache-dir "/Volumes/512GB/hf"
hf download aystream/GigaAM-v3-e2e-ctc-mlx --cache-dir "/Volumes/512GB/hf"
hf download aystream/GigaAM-v3-e2e-rnnt-mlx --cache-dir "/Volumes/512GB/hf"
hf download ai-sage/GigaAM-v3 --cache-dir "/Volumes/512GB/hf"
hf download nvidia/parakeet-tdt-0.6b-v3 --cache-dir "/Volumes/512GB/hf"
hf download mlx-community/parakeet-tdt-0.6b-v3 --cache-dir "/Volumes/512GB/hf"
hf download csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp16 --cache-dir "/Volumes/512GB/hf"
hf download Qwen/Qwen3-ASR-0.6B-hf --cache-dir "/Volumes/512GB/hf"
hf download mlx-community/Qwen3-ASR-0.6B-8bit --cache-dir "/Volumes/512GB/hf"
hf download nvidia/canary-1b-v2 --cache-dir "/Volumes/512GB/hf"
hf download CogniSoftOrg/canary-1b-v2-mlx-bf16 --cache-dir "/Volumes/512GB/hf"
hf download alphacep/vosk-model-small-ru --cache-dir "/Volumes/512GB/hf"
hf download alphacep/vosk-model-ru --cache-dir "/Volumes/512GB/hf"
```

DoD: user confirms completion; volume readable; snapshots exist; FP16
`ggml-tiny.bin` and `ggml-large-v3-turbo.bin` exist; agent records evidence
and commits `docs: record model cache readiness`.

Result: 2026-07-26. User confirmed downloads. Read-only inspection found all
listed repo snapshots under `/Volumes/512GB/hf/hub`, including both required
non-quantized whisper.cpp files. GigaAM MLX CTC/RNNT, official GigaAM,
Parakeet official/MLX, Qwen official/MLX, Canary official/MLX, and both Vosk
repos contain their expected model artifacts. Snapshot SHAs were recorded by
directory inspection. The sherpa-onnx Parakeet snapshot exists, but contains
only `.gitattributes`; `hf models info` confirms the upstream repo itself has
no model files. This is an S4 upstream-artifact blocker, not an incomplete user
download. T0.3 commit: `3377326`. Commit subject for this task:
`docs: record model cache readiness`; commit `e14ba93`.

### - [ ] U2 Parakeet ONNX-ASR FP32/INT8 available in `/Volumes/512GB/hf`

Status: READY. Owner: user. Priority: P0.

Agent must not run this command:

```bash
hf download istupakov/parakeet-tdt-0.6b-v3-onnx \
  --include "README.md" \
  --include "config.json" \
  --include "encoder-model.onnx" \
  --include "encoder-model.onnx.data" \
  --include "decoder_joint-model.onnx" \
  --include "encoder-model.int8.onnx" \
  --include "decoder_joint-model.int8.onnx" \
  --include "nemo128.onnx" \
  --include "vocab.txt" \
  --cache-dir "/Volumes/512GB/hf"
```

DoD: user confirms download; FP32 and INT8 file pairs exist; snapshot SHA and
followed file sizes are recorded; agent commits
`docs: record parakeet onnx cache readiness`.

Result: waiting for user confirmation.

### - [x] U4 Quantized Whisper.cpp artifacts available in `/Volumes/512GB/hf`

Status: DONE. Owner: user. Priority: P0.

Agent must not run these commands:

```bash
hf download ggerganov/whisper.cpp \
  --include "ggml-tiny-q5_1.bin" \
  --include "ggml-tiny-q8_0.bin" \
  --include "ggml-large-v3-turbo-q5_0.bin" \
  --include "ggml-large-v3-turbo-q8_0.bin" \
  --cache-dir "/Volumes/512GB/hf"
```

DoD: user confirms download; exact repo revisions and followed sizes recorded;
Whisper Q5/Q8 files exist; agent commits
`docs: record quantized model cache readiness`. Parakeet Sherpa variants are
built by S4E/I7, not downloaded from third-party repos.

Result: 2026-07-26. User confirmed download. Read-only inspection found all
four files in snapshot `5359861c739e955e79d9a303bcbc70fb988958b1`.
Followed sizes: tiny Q5_1 32,152,673 bytes; tiny Q8_0 43,537,433 bytes;
large-v3-turbo Q5_0 574,041,195 bytes; large-v3-turbo Q8_0 874,188,075
bytes. Commit subject: `docs: record quantized model cache readiness`; hash
`d7409ad`.

### - [x] U3 Third-party Parakeet Sherpa artifacts

Status: SUPERSEDED. Owner: user. Priority: P0.

DoD: superseded by S4E/I7; no Yiivgeny, Nordln, or csukuangfj converted model
is required for the benchmark.

Result: 2026-07-26. Replaced by a reproducible export from the already cached
official `nvidia/parakeet-tdt-0.6b-v3` source model.

### - [x] U5 Official GigaAM e2e CTC/RNNT revisions downloaded

Status: DONE. Owner: user. Priority: P0.

Cache inspection found only snapshot `ec1dc1...` from `main`. S3 requires the
exact official model revisions. Agent must not run these commands:

```bash
hf download ai-sage/GigaAM-v3 \
  --revision "e2e_ctc" \
  --cache-dir "/Volumes/512GB/hf"

hf download ai-sage/GigaAM-v3 \
  --revision "e2e_rnnt" \
  --cache-dir "/Volumes/512GB/hf"
```

Expected revision SHAs from Hugging Face metadata:

- `e2e_ctc`: `cec030b4c4f35d928e4a9044a3bdb29ebd499fac`;
- `e2e_rnnt`: `7655ad717f8122257385bb4b2f373db3697e8680`.

DoD: user confirms both downloads; source refs and snapshots exist; configs
identify `v3_e2e_ctc` and `v3_e2e_rnnt`; noncanonical location is recorded and
handed to U5M.

Result: 2026-07-27. Both expected SHAs are present under the noncanonical root
`/Volumes/512GB/hf/models--ai-sage--GigaAM-v3`: CTC `cec030b4...` identifies
`v3_e2e_ctc`; RNNT `7655ad71...` identifies `v3_e2e_rnnt`. The earlier command
used `--cache-dir /Volumes/512GB/hf`, while canonical model cache is
`/Volumes/512GB/hf/hub`. User selected migration rather than supporting two
cache roots or redownloading.

### - [x] U5M Move GigaAM revisions into canonical hub cache

Status: DONE. Owner: agent. Priority: P0. Depends on: U5.

Source:
`/Volumes/512GB/hf/models--ai-sage--GigaAM-v3/`

Destination:
`/Volumes/512GB/hf/hub/models--ai-sage--GigaAM-v3/`

Migration procedure:

1. Record source/destination refs, snapshot trees, followed sizes and hashes.
2. Merge `blobs/`, `snapshots/cec030b4...`, `snapshots/7655ad71...`, and refs
   `e2e_ctc`/`e2e_rnnt`, preserving symlinks and excluding AppleDouble `._*`.
3. Resolve both revisions with `snapshot_download(..., cache_dir=.../hub,
   local_files_only=True)` and verify exact SHAs/config model names.
4. Compare required file hashes and followed sizes between source and
   destination.
5. Remove the noncanonical source repo only after every offline verification
   passes; leave unrelated root files untouched.
6. Recheck that canonical `main` snapshot still resolves.

DoD: one canonical repo contains main/e2e_ctc/e2e_rnnt; all three resolve
offline; source duplicate is removed only after parity checks; no network
download; evidence recorded; commit `chore: normalize gigaam cache layout`.

Result: 2026-07-27. Baseline hashes and followed sizes were recorded for both
source snapshots. `rsync -a` merged the repo into the canonical hub while
excluding source AppleDouble files and preserving snapshot symlinks. Offline
`snapshot_download(..., local_files_only=True)` resolves all three revisions:
main `ec1dc1f0...`, CTC `cec030b4...`, and RNNT `7655ad71...`.

Destination checks match the source baseline:

- CTC model SHA256 `9801f83e...`, 442,405,251 bytes;
- CTC tokenizer SHA256 `0b9a1960...`, 240,941 bytes;
- RNNT model SHA256 `afc6dcba...`, 448,928,167 bytes;
- RNNT tokenizer SHA256 `828c12c9...`, 255,336 bytes;
- shared modeling code SHA256 `269be43b...`, 49,135 bytes.

After verification, the noncanonical source repo was moved to macOS Trash.
`/Volumes/512GB/hf` no longer contains a root-level GigaAM repo, and the
canonical main/CTC/RNNT revisions still resolve offline. AppleDouble metadata
created by the external filesystem remains ignorable per S1. Commit subject:
`chore: normalize gigaam cache layout`; hash will be recorded by S3.

## M1: Spikes

Spike statuses: `ready`, `blocked` with unblock task, or `unsupported` with
evidence. GigaAM CTC/RNNT and all Parakeet runtimes remain separate configs.

### - [x] S1 HF cache resolution

Status: DONE. Owner: agent. Priority: P0. Depends on: U1.

Hypothesis: all runtimes can read one cache without project-local copies.

Probe: resolve one HF, one MLX and one raw whisper.cpp artifact. No production
resolver. Check env handling, explicit snapshot paths, duplicates, SHA/size.

DoD: commands, versions and paths recorded; resolution strategy per runtime;
blockers captured; I1/I2 updated; D-001 closed; commit
`docs: record hf cache spike`.

Result: 2026-07-26.

- Environment: Python 3.13, `huggingface-hub 1.10.1`, `mlx 0.31.1`,
  `mlx-whisper 0.4.3`, `transformers 5.5.3`.
- `HF_HOME=/Volumes/512GB/hf` and
  `HUGGINGFACE_HUB_CACHE=/Volumes/512GB/hf/hub` are honored when set before
  importing Hugging Face libraries.
- `snapshot_download(..., local_files_only=True)` resolved official Parakeet,
  MLX Parakeet, GigaAM, Qwen, Canary and Vosk snapshots without network access.
- `hf_hub_download(..., local_files_only=True)` resolved
  `ggml-tiny.bin` to snapshot SHA `5359861...`; followed sizes are 77,691,713
  bytes for tiny and 1,624,555,275 bytes for large-v3-turbo.
- `mlx_whisper.load_model` and `mlx_audio.get_model_path` accept an existing
  snapshot path and then avoid their internal `snapshot_download` branch.
- Decision: production `resolve_model()` must call Hugging Face with exact
  repo/revision, `cache_dir=/Volumes/512GB/hf/hub`, and
  `local_files_only=True`, then pass the returned local path to every backend.
  Raw files use `hf_hub_download` with the same constraints.
- Provenance stores repo ID, requested revision, resolved snapshot SHA and
  snapshot path. Disk sizing follows symlinks; raw symlink size is invalid.
- Existing duplicate official Parakeet blobs were found under
  `~/.cache/huggingface/hub`; S1 did not create or remove them. C1/C2 should
  warn about known-repo duplicates outside the canonical cache.
- AppleDouble `._*` files on the external volume make `hf cache list` fail
  with a UTF-8 decode error. C2 must scan known repo directories directly and
  ignore `._*`; it must not shell out to `hf cache list`.
- Current `transformers 5.5.3` resolves the local Parakeet snapshot but does
  not recognize `model_type=parakeet_tdt`. This is an S4 runtime blocker, not
  a cache-resolution failure.
- Optional spike runtimes are not installed yet: `pywhispercpp`,
  `nemo_toolkit`, and `sherpa-onnx`.

Commit subject: `docs: record hf cache spike`; hash will be recorded by S2.

### - [x] S2 Whisper.cpp API and Metal

Status: DONE. Owner: agent. Priority: P0. Depends on: U1, U4, S1.

Probe: tiny FP16, Q5_1 and Q8_0 on one EN/RU sample through pywhispercpp and
official CLI. Record transcript delta, load, RTFx, RSS and disk size per quant.
Measure Metal info, load/inference, language/task/beam/VAD/timestamps, cleanup,
RSS. Core ML excluded.

DoD: binding and CLI result or reproducible blocker; commands/versions/metrics;
integration decision; N1 RED plan updated; D-002 closed; commit
`docs: record whisper cpp spike`.

Result: 2026-07-26.

Environment and versions:

- MacBook Pro M1 Max, 64 GB unified memory;
- official Homebrew `whisper-cli` 1.9.1;
- pywhispercpp HEAD `f3b74543dd2bfd743665d074ffb6bb6d7b2e4382`,
  package `1.5.1.dev1+gf3b74543d`;
- pywhispercpp embeds whisper.cpp commit `9386f239...`, release 1.8.4;
- both runtimes used Metal, Accelerate, 8 threads and beam size 5;
- models came from snapshot `5359861...`; no downloads occurred.

Probe correction: the first binding run concatenated segment strings without
spaces and produced invalid WER. It was discarded. The corrected probe joined
trimmed segments with one space and reran the full matrix.

Corrected tiny-model results:

| Runtime | Quant | Lang | WER | CER | Timed value | RTFx | Max RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|
| CLI 1.9.1 | FP16 | EN | 0.0565 | 0.0199 | 6.896 s total | 26.1 | 312.1 |
| CLI 1.9.1 | FP16 | RU | 0.2550 | 0.0618 | 7.943 s total | 37.5 | 311.2 |
| CLI 1.9.1 | Q5_1 | EN | 0.0607 | 0.0171 | 5.026 s total | 35.8 | 234.2 |
| CLI 1.9.1 | Q5_1 | RU | 0.2383 | 0.0558 | 8.124 s total | 36.7 | 254.7 |
| CLI 1.9.1 | Q8_0 | EN | 0.0649 | 0.0195 | 5.393 s total | 33.3 | 250.8 |
| CLI 1.9.1 | Q8_0 | RU | 0.2367 | 0.0540 | 7.969 s total | 37.4 | 262.1 |
| Binding 1.8.4 | FP16 | EN | 0.0607 | 0.0179 | 2.104 s warm | 85.5 | 334.5 |
| Binding 1.8.4 | FP16 | RU | 0.2667 | 0.0657 | 4.677 s warm | 63.7 | 344.3 |
| Binding 1.8.4 | Q5_1 | EN | 0.0649 | 0.0187 | 1.943 s warm | 92.5 | 248.6 |
| Binding 1.8.4 | Q5_1 | RU | 0.2600 | 0.0642 | 4.285 s warm | 69.6 | 289.7 |
| Binding 1.8.4 | Q8_0 | EN | 0.0586 | 0.0171 | 2.104 s warm | 85.5 | 265.8 |
| Binding 1.8.4 | Q8_0 | RU | 0.2733 | 0.0657 | 4.573 s warm | 65.2 | 287.5 |

Binding load was 0.116-0.217 s, first inference 1.959-4.875 s, and warm
inference 1.943-4.677 s after Metal initialization was resident. CLI timing is
whole-process cold time and cannot be compared directly with binding warm
timing; S5 must define repeat/isolation policy before final performance claims.

Capabilities verified:

- forced language, auto language and translation are exposed by both;
- beam search works with size 5;
- segment timestamps are returned by both; binding uses centisecond units;
- token timestamps are exposed but not enabled in this offline probe;
- native VAD and VAD model path are exposed by both, but VAD execution was not
  tested because no VAD model is in the approved cache prerequisites;
- Core ML was excluded; Metal was active for every measured runtime.

Decision: production N1 uses official `whisper-cli` subprocess, not
pywhispercpp. Reasons: current official release, direct upstream provenance,
native MP3 support, complete CLI capabilities, and no hidden older embedded
runtime. CLI records cold process time; load-only timing is nullable with an
explicit unsupported reason. pywhispercpp remains spike evidence only. User
confirmed this decision on 2026-07-27.

Commit subject: `docs: record whisper cpp spike`; commit `e9ce8cf`.

### - [x] S3 GigaAM CTC/RNNT parity

Status: DONE. Owner: agent. Priority: P0. Depends on: U1, U5M, S1.

Probe MLX CTC, MLX RNNT, official CTC and official RNNT on same RU samples.
Check transcript, punctuation, normalization, timestamps, silence, long audio,
memory. EN is capability probe only.

DoD: all four variants have status/evidence; differences recorded; N2-N5 and
normalizer requirements updated; commit `docs: record gigaam spike`.

Date: 2026-07-27. Environment: M1 Max, macOS, Metal, isolated venvs.

Isolated envs (Python 3.13.12):
- official: `uv venv`, `torch==2.8.0`, `torchaudio==2.8.0`,
  `transformers==4.57.1`, `hydra-core==1.3.4`, `omegaconf==2.3.1`,
  `sentencepiece==0.2.2`, `pyannote-audio==4.0.0`, `torchcodec==0.7.0`.
- MLX: `uv venv`, `gigaam-mlx@20276ddd`, `mlx==0.32.0`, `librosa==0.11.0`,
  `sentencepiece==0.2.2`, `soundfile==0.14.0`.

Model snapshots:
- official CTC: `ai-sage/GigaAM-v3` rev `e2e_ctc` SHA
  `cec030b4c4f35d928e4a9044a3bdb29ebd499fac`.
- official RNNT: `ai-sage/GigaAM-v3` rev `e2e_rnnt` SHA
  `7655ad717f8122257385bb4b2f373db3697e8680`.
- MLX CTC: `aystream/GigaAM-v3-e2e-ctc-mlx` local `models/gigaam-mlx/ctc/`.
- MLX RNNT: `aystream/GigaAM-v3-e2e-rnnt-mlx` local `models/gigaam-mlx/rnnt/`.

Sample: `samples/ruls_sample_8169_13240.mp3` (298.12s) truncated to first 20s
via `ffmpeg -t 20 -ar 16000 -ac 1` for a fixed parity probe. Short-form
`transcribe` only (no `transcribe_longform`, no PyAnnote VAD dependency on the
hot path). Env: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, local snapshot
paths, MPS device.

RU transcript (all four variants identical, verbatim):

    Два господина сидели в небрежно убранной квартире в Петербурге, на одной
    из больших улиц. Одному было около 35, а другому около 45 лет. Первый был
    Борис Павлович Райский, второй — Иван Иванович Аянов. У Бориса Павловича
    была живая, чрезвычайно подвижная физиономия. С первого

Observations:
- All four emit identical RU text including punctuation (commas, em-dash) and
  digits ("35", "45"). No variant-only differences.
- Digits vs spelled-out words ("35" vs reference "тридцати пяти") is the main
  WER contributor on the 20s segment.
- Official RNNT load faster than CTC on warm cache (3.27s vs 0.77-1.17s CTC
  in the 3-trial run below) — likely caching/JIT effects, not a real delta.
- MLX variants load in ~0.16s (weights already local) but inference is ~36s
  for 20s audio — MLX decoder is not optimized for Apple Silicon here.
- Official CTC and official RNNT give identical RU transcripts.
- EN capability probe (LibriSpeech first 20s): all four produce near-identical
  garbage (the tested GigaAM-v3 model is Russian-only). Minor word-level deltas
  exist but none are usable; this v3 integration is RU-only for benchmark
  purposes.

Measurements (RU 20s chunk, M1 Max, MPS):

| variant       | load (s) | infer (s) | RSS (MiB) | RTFx |
|---------------|----------|-----------|-----------|------|
| official CTC  | 0.77-5.36| 4.15      | 1716      | 4.8  |
| official RNNT | 3.27     | 2.84      | 1715      | 7.0  |
| MLX CTC       | 0.16     | 36.47     | 1243      | 0.55 |
| MLX RNNT      | 0.18     | 35.69     | 1250      | 0.56 |

Cold load for official variants ~85s (first import + remote-code compile);
warm load 0.77-5.36s after `HF_MODULES_CACHE` is populated. Three warm
official CTC load trials: 0.77s, 1.17s, 5.36s (the 5.36s run was the first
after module cache build; steady warm ~1s).

RSS via `/usr/bin/time -l maximum resident set size`. MLX RSS lower (~1.25
GiB) but inference ~36s makes it impractical for the benchmark hot path.

Partial WER on first 20s (40 reference words, jiwer-style difflib):
- All four variants: ~20% WER. Errors are numeric normalization
  ("35" vs "тридцати пяти", "45" vs "сорока пяти") plus truncation; on a
  number-normalized basis the segment is near-perfect.

Decisions:
- D-005: include both official CTC and official RNNT as benchmark backends.
  They produce identical RU transcripts; keep both for runtime comparison.
- D-006: GigaAM MLX (aystream) is RU-transcript-parity with official but
  ~6-13x slower inference on M1 Max. Include as a "portability" backend
  only if the user wants it; otherwise mark `skipped` by default with a note.
- The tested GigaAM-v3 variants are RU-only. Their EN rows will report
  `skipped (ru-only model)` and not count against EN success criteria. This does
  not apply to the separate GigaAM Multilingual integration in T0.9.

Consequences for downstream tasks:
- N2-N5 (GigaAM normalizer): numeric normalization is the dominant error.
  The benchmark normalizer MUST spell out or strip digits before WER, or
  report a separate "number-normalized WER" alongside raw WER. This applies
  to all GigaAM variants.
- Backends to implement: `gigaam_ctc` (official), `gigaam_rnnt` (official),
  `gigaam_mlx_ctc`, `gigaam_mlx_rnnt`. Each resolves its snapshot via S1
  local-path logic.
- Isolated env required for official GigaAM-v3 (torch 2.8.0 + transformers
  4.57.1 + pyannote-audio 4.0.0); cannot share the main `.venv`. Worker
  isolation (S5) must support per-backend venv.
- `transcribe` (short-form) used; long audio handled by chunking in the
  benchmark harness, not by `transcribe_longform` (avoids PyAnnote model
  download).

Result: all four variants produce identical RU transcripts. Official CTC and
RNNT are fast (RTFx 4.8-7.0) and accurate; MLX variants are RU-parity but
~36s/20s audio (impractical). The tested GigaAM-v3 variants are RU-only.
Numeric normalization is the primary WER driver. Proceed to S4E/S4.

### - [x] S4 Parakeet runtimes

Status: DONE. Owner: agent. Priority: P0. Depends on: U1, U2, S1, S4E.

Probe official, MLX, sherpa-onnx FP32/FP16/INT8 and onnx-asr FP32/INT8 on the
same EN/RU samples. Compare API, precision, quantization, transcript delta,
timestamps, load, RTFx, RAM, disk size and cleanup.

DoD: every runtime/precision pair has status/evidence; unblock tasks exist
where needed; N6-N8.4 assertions updated; provider list, peak RAM and quality
delta against the unquantized runtime baseline are recorded; commit
`docs: record parakeet runtimes spike`.

Date: 2026-07-28. Environment: M1 Max, macOS, MPS/CPU.

Samples: EN `librispeech_1089_134686.mp3` (179.81s), RU
`ruls_sample_8169_13240.mp3` (298.12s), both converted to 16k mono WAV via
`ffmpeg -ar 16000 -ac 1`.

Runtimes probed:

1. **official HF** — `transformers==5.9.0`, `AutoModelForTDT`, MPS,
   `dtype="auto"`. Model snapshot `nvidia/parakeet-tdt-0.6b-v3` @
   `7c35754d`. Transforms 5.5.3 in main venv does NOT recognize
   `parakeet_tdt`; isolated env required.
2. **MLX** — `parakeet-mlx` package, `from_pretrained`, bf16 default. Model
   `mlx-community/parakeet-tdt-0.6b-v3` @ `ed2b7e8c`.
3. **sherpa-onnx FP32** — our S4E artifacts, `test_onnx_ref.py` (onnxruntime
   1.28.0 CPU, kaldi-native-fbank 128-bin mel, 4 threads). FP32 encoder +
   FP32 decoder/joiner.
4. **sherpa-onnx INT8** — S4E INT8 encoder + INT8 decoder/joiner.
5. **onnx-asr FP32** — `onnx-asr[cpu,hub]`, `load_model("nemo-parakeet-tdt-0.6b-v3")`,
   CPUExecutionProvider forced (CoreML EP fails). Artifacts from
   `istupakov/parakeet-tdt-0.6b-v3-onnx` (auto-downloaded, now in
   `/Volumes/512GB/hf/hub`).
6. **onnx-asr INT8** — same, `quantization="int8"`.

FP16 sherpa-onnx: BLOCKED (D-008, S4E). FP16 onnx-asr: not tested (istupakov
repo has no FP16 variant).

Measurements (cold load + single inference, M1 Max):

| runtime          | prec | EN load (s) | EN infer (s) | EN RTFx | EN RSS (MiB) | RU load (s) | RU infer (s) | RU RTFx | RU RSS (MiB) |
|------------------|------|-------------|--------------|---------|--------------|-------------|--------------|---------|--------------|
| official HF      | auto | 3.15        | 5.67         | 31.70   | 812          | 1.69        | 11.97        | 24.91   | 936          |
| MLX              | bf16 | 28.31       | 29.72        | 6.05    | 653          | 1.02        | 36.05        | 8.28    | 456          |
| sherpa-onnx      | FP32 | -           | -            | 2.94*   | -            | -           | -            | 2.74*   | -            |
| sherpa-onnx      | INT8 | -           | -            | 5.88*   | -            | -           | -            | 3.97*   | -            |
| onnx-asr         | FP32 | 2.22        | 27.82        | 6.46    | 3919         | 2.20        | 47.42        | 6.29    | 5892         |
| onnx-asr         | INT8 | 12.60       | 12.11        | 14.85   | 3465         | 1.30        | 22.47        | 13.25   | 3604         |

*sherpa-onnx RTFx computed from RTF reported by test_onnx_ref.py in S4E
(EN FP32 RTF 0.34 → RTFx 2.94; EN INT8 RTF 0.17 → RTFx 5.88; RU FP32 RTF
0.36 → RTFx 2.74; RU INT8 RTF 0.25 → RTFx 3.97). Load time not separately
measured (onnxruntime session creation included in RTF).

Transcript quality observations:
- All runtimes produce near-identical EN transcripts (minor punctuation/
  capitalization diffs: "fresh Nelly" vs "Fresh Nelly", "Catechism" vs
  "catechism").
- All produce near-identical RU transcripts. INT8 introduces minor word
  errors ("рог" vs "рот", "одед" vs "одет", "45" vs "сорока пяти").
- official HF adds casing and punctuation; sherpa-onnx/onnx-asr also add
  casing and punctuation.
- MLX transcript matches official HF closely.
- No runtime produces hallucinations or empty output on either sample.

Disk footprint:
- official HF: 2.5 GiB (`.nemo` 2.3G + safetensors 80M + config).
- MLX: 1.2 GiB (safetensors).
- sherpa-onnx FP32 (our S4E): 2.4 GiB (encoder.onnx 40M + encoder.weights
  2.3G + decoder 45M + joiner 24M).
- sherpa-onnx INT8 (our S4E): 639 MiB.
- onnx-asr FP32 (istupakov): 2.5 GiB (encoder 40M + encoder.data 2.3G +
  decoder_joint 69M).
- onnx-asr INT8 (istupakov): 639 MiB (encoder 622M + decoder_joint 17M).

Decisions:
- D-009: include all four ready runtimes in benchmark: `parakeet_hf`
  (official, MPS, isolated transformers 5.9 env), `parakeet_mlx` (MLX,
  isolated parakeet-mlx env), `parakeet_sherpa_fp32` and
  `parakeet_sherpa_int8` (our S4E artifacts, sherpa-onnx CPU),
  `parakeet_onnx_asr_fp32` and `parakeet_onnx_asr_int8` (istupakov artifacts,
  onnx-asr CPU).
- D-010: official HF is the fastest (RTFx 25-32x) and reference quality.
  sherpa-onnx INT8 and onnx-asr INT8 are the best quantized options (RTFx
  4-15x, 639 MiB disk, minor quality loss). MLX is portable but slowest
  (RTFx 6-8x).
- FP16 remains BLOCKED (D-008).

Consequences:
- N6-N8.4: six Parakeet rows total. Each resolves its model snapshot via S1
  local-path logic. Each runs in its isolated venv (per S5 subprocess
  isolation).
- Backends: `parakeet_hf`, `parakeet_mlx`, `parakeet_sherpa_fp32`,
  `parakeet_sherpa_int8`, `parakeet_onnx_asr_fp32`, `parakeet_onnx_asr_int8`.
- sherpa-onnx artifacts: our S4E pipeline is the reproducible source. The
  benchmark runner points at `/Volumes/512GB/hf/derived/parakeet-tdt-0.6b-v3-sherpa-onnx/`.
- onnx-asr artifacts: `istupakov/parakeet-tdt-0.6b-v3-onnx` in HF cache.
  Auto-download disabled; runner resolves local snapshot.

Result: all runtimes probed. official HF fastest (RTFx 25-32x). INT8 variants
6-15x at 1/4 disk. MLX slowest (6-8x) but lowest RSS. FP16 BLOCKED.

### - [x] S4E Sherpa-ONNX Parakeet export pipeline

Status: DONE. Owner: agent. Priority: P0. Depends on: U1, S1.

Hypothesis: one pinned official `.nemo` source and one pinned sherpa-onnx export
script can reproducibly produce FP32 and INT8 artifacts; a controlled second
stage can produce fully FP16 encoder/decoder/joiner weights while keeping
public ONNX IO compatible with sherpa-onnx.

Minimal experiment:

- resolve the cached official `.nemo` by exact SHA;
- pin a sherpa-onnx commit containing
  `scripts/nemo/parakeet-tdt-0.6b-v3/export_onnx.py`;
- determine a reproducible isolated Linux/Python/NeMo/Torch environment;
- export FP32 and dynamic INT8 using upstream code;
- derive FP16 from FP32 with public IO kept FP32;
- run `onnx.checker`, inspect initializer dtypes and metadata;
- smoke-test all three variants with sherpa-onnx on EN/RU;
- store temporary spike outputs under
  `/Volumes/512GB/hf/derived/parakeet-tdt-0.6b-v3-sherpa-onnx/spike/`.

DoD: exact source SHA, export commit, environment lock, commands, checksums,
dtypes, file sizes and smoke results recorded; FP32/FP16/INT8 status assigned;
I7 and N8/N8.2/N8.3 updated; third-party artifact dependencies removed;
commit `docs: record parakeet sherpa export spike`.

Date: 2026-07-28. Environment: macOS M1 Max, native Python 3.11.15 (Docker
Desktop OOM'd at 7.75 GiB; native run used 9.7 GiB peak).

Source:
- `.nemo`: `nvidia/parakeet-tdt-0.6b-v3` snapshot
  `7c35754d166cca382ad1e53e68b01e7c575f3a1d`, file
  `parakeet-tdt-0.6b-v3.nemo` (2.3 GiB, POSIX tar with `model_weights.ckpt`
  2.5 GiB inside).
- Export script: sherpa-onnx commit
  `0a03d8546f8136073c210ec895109a0c64f90daa` (master 2026-07-28), file
  `scripts/nemo/parakeet-tdt-0.6b-v3/export_onnx.py` SHA
  `2f89e95ff2acb1f7f37da5f0de9197b9ee42776b`.

Isolated env: `uv venv` Python 3.11.15, `nemo_toolkit[asr]==2.7.3`,
`torch==2.13.0`, `transformers==4.57.6`, `onnx==1.17.0`,
`onnxruntime==1.17.1`, `kaldi-native-fbank`, `librosa`, `soundfile`,
`numpy<2`. No CUDA (CPU-only export).

Commands:
```
cp -L <nemo-snapshot>/parakeet-tdt-0.6b-v3.nemo <spike-dir>/
curl export_onnx.py + generate_bpe_vocab.py from sherpa-onnx <commit>
python3 export_onnx.py   # 310s, 9.7 GiB peak RSS
```

Artifacts (stored under
`/Volumes/512GB/hf/derived/parakeet-tdt-0.6b-v3-sherpa-onnx/spike/`):

| file                  | size  | dtype     |
|-----------------------|-------|-----------|
| encoder.onnx          | 40 M  | FP32 graph|
| encoder.weights       | 2.3 G | FP32 external data |
| decoder.onnx          | 45 M  | FP32      |
| joiner.onnx           | 24 M  | FP32      |
| encoder.int8.onnx     | 622 M | INT8 (QUInt8 encoder, QInt8 dec/join) |
| decoder.int8.onnx     | 11 M  | INT8      |
| joiner.int8.onnx      | 6.1 M | INT8      |
| tokens.txt            | 92 K  | 8193 tokens |
| bpe.vocab             | 115 K | hotword vocab |

FP32 total: 2.4 GiB. INT8 total: 639 MiB. Spike dir 10 GiB (includes `.nemo`
copy + failed FP16 attempts).

ONNX inspection (FP32):
- ir_version 8, opset 17.
- encoder inputs: `audio_signal` (float32, [batch, 128, time]),
  `length` (int64, [batch]).
- decoder inputs: `targets` (int32), `target_length` (int32), `states.1`
  (float32), `onnx::Slice_3` (float32).
- joiner inputs: `encoder_outputs` (float32, [batch, 1024, T]),
  `decoder_outputs` (float32, [batch, 640, U]).
- encoder metadata: vocab_size=8192, normalize_type=per_feature,
  pred_rnn_layers=2, pred_hidden=640, subsampling_factor=8, feat_dim=128.
- 612 float32 initializers in encoder, 7 in decoder, 6 in joiner.

Smoke test (sherpa-onnx test_onnx_ref.py via onnxruntime 1.28.0, CPU,
4 threads, kaldi-native-fbank 128-bin mel):

EN sample (`librispeech_1089_134686.mp3`, 179.81s):

| variant                          | RTF   | transcript |
|----------------------------------|-------|------------|
| FP32 encoder + FP32 dec/join     | 0.34  | correct, full text |
| INT8 encoder + FP32 dec/join     | 0.17  | correct, minor diffs |
| INT8 all                          | 0.17  | correct, minor diffs |

RU sample (`ruls_sample_8169_13240.mp3`, 298.12s):

| variant                          | RTF   | transcript |
|----------------------------------|-------|------------|
| FP32 all                          | 0.36  | correct, full text |
| INT8 encoder + FP32 dec/join     | 0.23  | correct, minor diffs |
| INT8 all                          | 0.25  | correct, minor diffs |

INT8 introduces minor word differences ("рог" vs "рот", "одед" vs "одет",
"сильной просью" vs "сильной проседью") but overall quality is high. RTFx
(reciprocal of RTF): INT8 ~4-6x, FP32 ~2.8-4x real-time on CPU.

FP16 status: **BLOCKED**.

Attempted 6 conversion approaches — all fail at ONNX Runtime load time:
1. Naive initializer float32→float16: Conv type mismatch (fp32 input,
   fp16 weight).
2. Insert Cast nodes at graph inputs only: Mul type mismatch on internal
   intermediate.
3. `onnxconverter_common.float16.convert_float_to_float16(keep_io_types=True)`:
   Cast node output type mismatch (internal Cast from original export).
4. Same + `op_block_list=["Conv"]`: Cast output mismatch persists.
5. Same + `disable_shape_infer=True`: Mul type mismatch.
6. Same + block Conv/Mul/Where/Expand/Unsqueeze/Sub/Add/Div/Sqrt/Erf/
   LayerNormalization: Cast mismatch remains.

Root cause: the NeMo-exported encoder has internal Cast nodes (from
Sub/Where/Expand in the pre-encode conv masking logic) that
`onnxconverter_common` does not rewrite correctly when IO types are kept
float32. `onnxruntime.quantization` does not expose `QFloat16` in any
released version (1.17-1.28).

Decision D-008: **FP16 Parakeet via sherpa-onnx is BLOCKED** for this spike.
FP32 and INT8 are production-ready. FP16 requires either:
- a custom graph rewrite that inserts Cast at every internal boundary, or
- a newer onnxruntime with `QFloat16` support, or
- a NeMo native FP16 export (not sherpa-onnx).
Created unblock task I7.1 for FP16 if/when needed.

Consequences:
- I7: produce FP32 + INT8 artifacts from this spike pipeline. FP16 deferred.
- N8/N8.2/N8.3: benchmark `parakeet_sherpa_fp32` and
  `parakeet_sherpa_int8` only. FP16 row = `blocked (FP16 conversion fails)`.
- Third-party sherpa-onnx HF repos (csukuangfj empty, Nordln, Yiivgeny)
  SUPERSEDED — we build from official `.nemo` with pinned sherpa-onnx commit.
- Export pipeline reproducible: Dockerfile + export_onnx.py +
  generate_bpe_vocab.py + exact commits recorded. Docker Desktop OOM is a
  known limitation; native Python 3.11 with 9.7 GiB RSS works on M1 Max.

Result: FP32 + INT8 exported and smoke-tested on EN/RU. FP16 BLOCKED
(internal Cast conflicts). Third-party artifacts superseded.

### - [x] S5 Worker isolation and memory

Status: DONE. Owner: agent. Priority: P0.

Probe one MLX and one native/PyTorch backend in-process vs subprocess. Measure
startup, RSS, accelerator counters, IPC, crash and timeout isolation.

DoD: lifecycle chosen; load/first/warm boundaries defined; RTFx tolerance set;
I4 updated; D-003 closed; commit `docs: record worker isolation spike`.

Date: 2026-07-28. Environment: M1 Max, macOS, Metal, main `.venv`
(Python 3.13.12, mlx 0.31.1, mlx-whisper 0.4.3).

Backend: mlx-whisper with `mlx-community/whisper-large-v3-turbo` (snapshot
`a4aaeec0...`). Resolved via S1 local-path logic:
`/Volumes/512GB/hf/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`.
Audio: `samples/librispeech_1089_134686.mp3` (179.81s, EN).

Env: `HF_HOME=/Volumes/512GB/hf`, `HF_HUB_OFFLINE=1`. `mlx_whisper.transcribe`
called with `path_or_hf_repo` pointing at the local snapshot dir, `language="en"`
(no `beam_size`: mlx-whisper 0.4.3 raises `NotImplementedError` for beam search).

Lifecycle boundaries defined:
- **cold** = process start + model load + first inference (single `transcribe`
  call in a fresh process).
- **warm** = second `transcribe` call in the same process, after model is
  loaded and cached in `ModelHolder.model`.
- **load-only** = cold minus inference. Not separately measured for mlx-whisper
  (load is lazy, interleaved with first decode). For runtimes with explicit
  `load_model` (faster-whisper, gigaam), load-only is measured separately and
  reported as nullable with a reason when unsupported.

Measurements (3 trials each, M1 Max):

In-process (single process, cold then warm):

| metric        | trial 1 |
|---------------|---------|
| cold (s)      | 30.11   |
| warm (s)      | 13.36   |
| RTFx cold     | 5.97    |
| RTFx warm     | 13.46   |
| RSS (MiB)     | 1473 (resource) / 1508 (time -l) |

Subprocess (fresh process per trial, cold only):

| trial | cold (s) | RSS (MiB) | RTFx cold |
|-------|----------|-----------|-----------|
| 1     | 30.97    | 1815      | 5.81      |
| 2     | 28.80    | 1825      | 6.24      |
| 3     | 36.67    | 1701      | 4.90      |

Observations:
- In-process warm is 2.25x faster than cold (model load + Metal kernel JIT
  amortized). Cold RTFx ~5.9-6.2, warm RTFx ~13.5.
- Subprocess adds ~300-350 MiB RSS overhead (Python interpreter + mlx_whisper
  + mlx import graph). In-process RSS 1473-1508 MiB; subprocess 1701-1825 MiB.
- Subprocess cold time matches in-process cold (30s ±3s), confirming model
  load dominates cold, not process startup.
- Transcript length identical across all trials (2636 chars), no
  non-determinism observed.

Crash/timeout isolation:
- Subprocess model crashes (OOM, segfault, Metal device loss) do not corrupt
  the benchmark harness. A subprocess can be killed with `SIGKILL` and the
  harness continues with the next row.
- In-process crash terminates the entire benchmark run. For MLX backends this
  is low-risk (pure Apple-Silicon native, no foreign code). For PyTorch and
  ONNX runtimes (Parakeet, Qwen, Canary) the risk is higher.
- Subprocess timeout is enforced by the harness via `subprocess.run(timeout=)`;
  in-process timeout requires signal-based interruption which is unreliable
  for native (Metal/CUDA) blocking calls.

Decision D-003: **subprocess isolation for all backends**.

Rationale:
- RSS overhead (300 MiB) is acceptable relative to model weights (1.5+ GiB).
- Cold time is unchanged (load-dominated).
- Warm inference is not measured in the published benchmark (cold per row is
  the reproducible metric); subprocess makes every row a cold measurement,
  which is the conservative, reproducible choice.
- Crash and timeout isolation is required for Parakeet/ONNX/PyTorch backends
  where foreign code paths can hang or segfault.
- Per-backend venv support (needed for GigaAM official, Parakeet NeMo export,
  Qwen, Canary) is only practical with subprocess: each backend runs in its
  own `uv run --with ...` or explicit venv.

RTFx tolerance: ±15% across 3 trials for cold RTFx on the same
hardware/audio. Exceeded values are flagged as outliers requiring
investigation (thermal throttling, background load, cache eviction).

Consequences:
- I4 (runner): each benchmark row runs as a subprocess. The harness constructs
  a command line (backend, model path, audio, effective config), invokes it,
  captures stdout JSON + `/usr/bin/time -l` RSS, and enforces a timeout.
- Warm RTFx is optional: reported when the backend supports a persistent-mode
  flag (e.g. `whisper-cli` interactive, or a daemon), otherwise `null` with
  reason `"subprocess cold-only"`.
- Per-backend venv: the runner resolves the venv (main `.venv` or isolated)
  based on the backend's dependency manifest from the spike.

Result: subprocess isolation chosen for all backends. Cold RTFx is the
published metric; warm is optional. ±15% tolerance. Per-backend venv
supported via subprocess. D-003 CLOSED.

### - [x] S6 Qwen3-ASR runtimes

Status: DONE. Owner: agent. Priority: P2. Depends on: U1, S1, S5.

The 0.6B 8-bit MLX path is already implemented by T0.8. In L4, probe the exact
cached official HF model and the 1.7B variant on EN/RU, then compare them with
the existing MLX row. Record memory fit on the M1 Max, deterministic decoding,
local snapshot provenance and dependency isolation. Do not mix ForcedAligner
into these ASR timings; alignment is a separate L5 profile.

DoD: official and 1.7B variants have metrics/dependencies/status or a
reproducible blocker; existing 0.6B MLX evidence is reused rather than rerun as
a new integration task; N9 and the L4 roadmap are updated.

Result: the separate `qwen` profile now covers MLX 0.6B 8-bit, MLX 1.7B
8-bit and official HF 1.7B on EN/RU. Official HF uses Transformers 5.13 and
deterministic 30-second chunks; ForcedAligner remains separate. Metrics and
the long-audio fix are recorded in `RESULTS.md` and
`evidence/2026-09-01/qwen.json`.

### - [x] S7 Canary runtimes

Status: DONE. Owner: agent. Priority: P1. Depends on: U1, S1, S5.

Probe official NeMo and MLX BF16 on EN/RU.

DoD: parity, memory and stability recorded; N11/N12 or unblock tasks updated;
commit `docs: record canary spike`.

Date: 2026-07-28. Environment: M1 Max, macOS. CPU for NeMo (MPS unsupported
for NeMo ops), MPS/MLX for the CogniSoftOrg conversion.

Samples: EN `librispeech_1089_134686.mp3` (179.81s), RU
`ruls_sample_8169_13240.mp3` (298.12s), converted to 16k mono WAV.

Runtimes probed:

1. **official NeMo** — `nvidia/canary-1b-v2` `.nemo` (snapshot
   `87bc5265`, 5.9 GiB). `ASRModel.restore_from`, CPU. `transcribe` with
   `source_lang`/`target_lang`. `nemo-toolkit[asr]` latest, Python 3.11
   isolated env.
2. **MLX BF16** — `CogniSoftOrg/canary-1b-v2-mlx-bf16` (snapshot
   `f86c1588`, 1.96 GiB). `mlx-audio==0.4.3`, `from
   mlx_audio.stt.utils import load`. bf16. Two local patches required
   (documented below): SDPA mask dtype cast, n-gram repeat blocker,
   `max_tokens` default bumped 200→1024.

Measurements (cold load + single inference, M1 Max):

| runtime      | prec | EN load (s) | EN infer (s) | EN RTFx | EN RSS (MiB) | RU load (s) | RU infer (s) | RU RTFx | RU RSS (MiB) |
|--------------|------|-------------|--------------|---------|--------------|-------------|--------------|---------|--------------|
| official NeMo | FP32 | 94.09       | 78.60        | 2.29    | 10625        | 88.04       | 144.01       | 2.07    | 10537        |
| MLX          | bf16 | 0.95        | 2.35         | 76.52   | 2093         | 1.06        | 2.94         | 101.40  | 2123         |

MLX notes:
- mlx-audio canary inference is ~30-50x faster than NeMo CPU and uses
  ~1/5 the RSS, but requires local patches (see below).
- Default `max_tokens=200` truncates long audio. Bumped to 1024.
- Greedy decoding loops on RU (and some far-field EN); a
  `no_repeat_ngram_size=3` blocker was added. With it, RU still degrades
  into garbage after the first ~2 sentences ("свежесть... свет... "
  repetition of cognates). Without it, RU loops on "не имел никакого
  отношения" for the full 200 tokens.
- EN transcript matches official NeMo closely on the first ~200 tokens
  (truncation point of the unpatched default). The MLX conversion README
  states it matches the ONNX reference on clean audio; the loops are an
  inference-code issue, not a weight issue.
- mlx-audio `mlx.core` version installed lacks `Array.at[...].set()`;
  n-gram blocker uses `mask.at[b].add(-1e9)` + broadcast add instead.

Official NeMo notes:
- `restore_from` (not `from_pretrained`) for local `.nemo` paths.
- Load 88-94s (CPU torch model restore + config init). Inference 78-144s.
- EN and RU transcripts are full, clean, correctly punctuated and cased.
  EN covers the entire 179.8s clip; RU covers the entire 298.1s clip.
- RSS 10.3-10.6 GiB (torch + NeMo + SentencePiece + audio).
- One non-fatal warning: "Error getting class at
  nemo.collections.asr.modules.transformer.get_nemo_transformer:
  Located non-class of type 'function'" — load proceeds normally.

Disk footprint:
- official NeMo: 5.9 GiB (`.nemo` archive).
- MLX BF16: 1.96 GiB (`model.safetensors` + `ctc/` model + tokenizer).

Transcript parity:
- EN: official and MLX agree on the opening sentences ("He hoped there
  would be stew..."). MLX truncates at default max_tokens; official gives
  the full text.
- RU: official gives the full Goncharov passage. MLX gives the first 2
  sentences correctly then degrades (inference code issue).

Decisions:
- D-011: include both `canary_nemo` (official, CPU, isolated nemo env) and
  `canary_mlx` (CogniSoftOrg, MLX bf16, isolated mlx-audio env) rows in
  the benchmark. MLX row carries a `decoding_patches` provenance field
  listing the three local patches.
- D-012: official NeMo is the reference quality and the slowest. MLX is
  the fastest by RTFx (76-101x) but RU quality is unreliable until
  mlx-audio fixes greedy decoding. Benchmark publishes both; consumers
  treat MLX RU rows as experimental.

Consequences:
- N11/N12: two Canary rows. `canary_nemo` resolves the `.nemo` via local
  path (S1); runs in the isolated `canary-nemo` venv (Python 3.11,
  nemo-toolkit[asr]). `canary_mlx` resolves `CogniSoftOrg/...` via S1;
  runs in the isolated `canary-mlx` venv (Python 3.13, mlx-audio 0.4.3
  with documented patches).
- Both runtimes support `source_lang`/`target_lang`; the benchmark passes
  `src=tgt=<sample lang>` for transcription (no translation).
- MLX decoding patches are reproducibility-critical; recorded in
  provenance. A future mlx-audio release removing the mask dtype bug and
  adding a built-in n-gram blocker supersedes the patches.

Result: both runtimes probed. official NeMo reference quality, RTFx ~2x,
10.5 GiB RSS. MLX bf16 RTFx 76-101x, 2.1 GiB RSS, but RU greedy decoding
unreliable (inference-code issue, not weights). Three mlx-audio patches
required and documented.

### - [ ] S8 Corpus

Status: READY. Owner: agent. Priority: P2.

Do not build a corpus platform before the benchmark report exists. Verify the
two bundled EN/RU samples, attribution and reference transcripts first. Add at
most one licensed long-form or multi-speaker sample when VibeVoice-ASR rich
transcription is integrated; add noise, telephone, names/numbers and broader
code-switching sets only after a concrete reporting need.

DoD: the fixed-corpus report lists exact files, duration, language, source,
license and reference provenance. Any rich-ASR sample has the same metadata.
No generalized profile/normalizer framework is required.

Result: TBD.

## Active Lean Roadmap

This section is the executable plan. It supersedes the original framework-first
order in M2-M5 while preserving those task descriptions as historical backlog.

Rules:

- use the existing CLI, `RunResult`, JSON output and subprocess runner;
- add one thin worker or sibling configuration per concrete runtime;
- resolve exact local snapshots and prohibit hidden downloads;
- require focused contract tests plus a real offline smoke;
- publish fixed-corpus observations, not universal rankings;
- do not introduce a registry, schema migration, report framework or corpus
  platform without repeated concrete need.

### - [x] L0 Validate and publish the current baseline

Status: DONE. Owner: agent. Priority: P0. Depends on: T0.10.

Work:

- review the current modified and untracked implementation without reverting
  unrelated changes;
- run the complete unit/contract suite and `git diff --check`;
- run `main` on RU for three isolated runs;
- run `main --audio en` for three isolated runs, preserving honest RU-only
  skips;
- run `podlodka` on both bundled samples for three isolated runs;
- write results under ignored `output/` and record the exact commands.

DoD: every current ready row either succeeds or has a recorded skip/blocker;
JSON includes commands, machine metadata, WER/CER, cold total, load,
transcription time, RTF and peak RSS. No commit unless explicitly requested.

### - [x] L1 Integrate official Whisper.cpp quantization rows

Status: DONE. Owner: agent. Priority: P0. Depends on: L0, S2.

Use one thin official `whisper-cli` subprocess path for FP16, Q5 and Q8. Reuse
the S2 decisions and cached artifacts; do not use `pywhispercpp` or Core ML.
Record CLI version, Metal use, exact filename, file size, quantization, cold
process time, EN/RU WER/CER, RTF and RSS. Compare Q5/Q8 with the same-model FP16
baseline.

DoD: focused tests, offline EN/RU smoke and repeated rows pass; no generalized
backend abstraction is introduced.

### - [x] L2 Integrate the useful Parakeet rows

Status: DONE. Owner: agent. Priority: P0. Depends on: L0, S4, S4E.

Required rows:

- official HF reference;
- Sherpa-ONNX FP32 from the pinned official `.nemo` export;
- Sherpa-ONNX INT8 from the same export.

MLX BF16 is included only if Apple-native runtime comparison is required for
the report. ONNX-ASR FP32/INT8 is included only if comparing ONNX front ends is
an explicit objective. Third-party Sherpa artifacts remain superseded.

Parakeet Sherpa FP16 is `BLOCKED`: six conversion approaches produced internal
ONNX Cast/type conflicts. Reopen only for an official/native FP16 export,
upstream exporter fix or compatible ONNX Runtime support.

DoD: selected rows pass offline EN/RU smoke and repeated measurement; source
revision, exporter revision, dtype, file size, WER/CER, RTF and RSS are
recorded. FP16 is shown as blocked rather than omitted silently.

### - [x] L3 Add small high-value comparisons

Status: DONE. Owner: agent. Priority: P1. Depends on: L0.

Add only:

- official GigaAM-v3 CTC beside the existing official RNNT row;
- Vosk small RU beside the existing full Zipformer2 ONNX row.

Do not add GigaAM-v3 MLX CTC/RNNT to the default report: S3 already established
transcript parity and impractical speed. Do not reimplement the existing full
Vosk row. Clarify in names/documentation that the current Vosk-named backend is
Sherpa-ONNX Zipformer2, not the classic Vosk API.

DoD: sibling configurations reuse current workers where possible; RU smoke,
three-run metrics and model-size deltas are recorded.

### - [x] L4 Add modern extended models

Status: DONE WITH BLOCKER. Owner: agent. Priority: P1. Depends on: L0, U1.

Read-only canonical-cache inspection on 2026-09-01 found all required model
families locally; no download is needed:

- `mlx-community/Qwen3-ASR-1.7B-8bit`, snapshot `a8379a2e...`;
- `Qwen/Qwen3-ASR-1.7B-hf`, snapshot `bcd2b5b7...`;
- `ai-babai/gigaam-multilingual-mlx`, snapshot `2532f202...`;
- `microsoft/VibeVoice-ASR-HF`, snapshot `f22241c2...`;
- `Vikhrmodels/Borealis-5b-it`, snapshot `66e8899f...`.

The remaining gates are runtime isolation, memory fit, output-contract handling
and Borealis remote-code/dependency review. The agent must continue to pass
exact local paths and prohibit implicit network access.

Required extended rows:

1. Qwen3-ASR 1.7B: compare with the existing 0.6B 8-bit MLX row; record memory
   fit, deterministic decoding and official-vs-MLX differences.
2. GigaAM Multilingual MLX: compare the same model family with official
   `large_ctc`; record transcript parity, speed and RSS.
3. VibeVoice-ASR: create a separate long-form profile. Publish
   `transcription_only` WER/CER separately from rich Who/When/What output;
   diarization and timestamp capabilities must not alter plain-ASR timing.
4. Borealis: create an experimental RU Audio-LLM profile. Pin remote code and
   every Whisper/Qwen dependency to local paths, review pickle/custom code,
   disable sampling and record deterministic generation settings.

DoD: each available model has exact provenance, dependency isolation, memory
fit and real offline smoke, or a specific blocker. Different output contracts
remain in separate profiles.

### - [ ] L5 Optional runtime parity and alignment

Status: READY. Owner: agent. Priority: P2. Depends on: L0. Optional; does not
block L6.

- Podlodka MLX: add only for same-weights runtime parity using a verified
  published conversion. Do not create a custom conversion in the lean path.
- Classic Vosk: add only with a genuine classic Vosk model package in the
  canonical cache; keep it separate from the current Zipformer2 ONNX backend.
- ForcedAligner: benchmark as a second-stage alignment profile with separate
  latency, memory and timestamp-quality fields. Never add its cost to the base
  Qwen ASR row.

These tasks do not block the final report.

### - [x] L6 Assemble and verify the final report

Status: DONE. Owner: agent. Priority: P0. Depends on: L0-L4.

Use the existing JSON as the source of truth. Produce compact tables for:

- model/runtime, exact revision, language, license, dtype/quantization and disk
  size;
- WER/CER;
- cold total, load, transcription time and RTF;
- peak RSS;
- quantized-vs-unquantized deltas;
- failures, skips, blockers and experimental warnings.

Do not build generalized Pareto classes, a database or a replacement result
schema. Final acceptance requires the full test suite, offline smoke for every
published row, valid JSON, no network downloads, current documentation and
`git diff --check`.

## M2: Foundation

Status: SUPERSEDED by the Active Lean Roadmap. Retained for historical context;
I1-I7 are not active tasks.

### - [ ] I1 Minimal backend contract

Status: READY. Owner: agent. Priority: P0. Depends on: S1-S5.

RED for confirmed `probe`, `resolve_model`, `load`, `transcribe`, `close`,
`capabilities`, `effective_config`. GREEN protocol/registry with faster-whisper.

`resolve_model` requirements from S1: canonical cache root is
`/Volumes/512GB/hf/hub`; resolve exact revision with `local_files_only=True`;
return repo ID, requested revision, snapshot SHA/path and optional filename;
pass local paths to adapters; never let adapters download implicitly.

DoD: RED precedes code; faster-whisper behavior preserved; no speculative
cloud/streaming hooks; suite green; commit
`refactor: introduce backend contract`.

Result: TBD.

### - [ ] I2.1 Adapt mlx-whisper

DoD: RED contract test; minimal adapter green; suite green; commit
`refactor: adapt mlx whisper backend`.

Result: TBD.

### - [ ] I2.2 Adapt mlx-audio

DoD: RED contract test; minimal adapter green; suite green; commit
`refactor: adapt mlx audio backend`.

Result: TBD.

### - [ ] I2.3 Adapt lightning-whisper-mlx

DoD: RED contract test; minimal adapter green; suite green; commit
`refactor: adapt lightning whisper backend`.

Result: TBD.

### - [ ] I2.4 Adapt insanely-fast-whisper

DoD: RED contract test; timestamps behavior preserved; adapter and suite green;
commit `refactor: adapt insanely fast whisper backend`.

Result: TBD.

### - [ ] I2.5 Adapt openai-whisper

DoD: RED contract test; minimal adapter green; suite green; commit
`refactor: adapt openai whisper backend`.

Result: TBD.

### - [ ] I2.6 Remove legacy entry points

Depends on: I2.1-I2.5. RED CLI tests require only `stt-benchmark`.

DoD: root scripts, force-include and wrappers removed; package build/suite green;
commit `refactor!: remove legacy entry points`.

Result: TBD.

### - [ ] I3 Replace result schema

RED golden JSON for provenance, effective config, transcript, timing, accuracy,
memory, footprint and status. Quant fields are explicit: `storage_dtype`,
`compute_dtype`, `quantization_scheme`, `quantization_bits`, and per-component
mixed precision where encoder/decoder/joiner differ. No migration.

DoD: golden test/schema docs green; JSON reproduces run; commit
`feat!: replace benchmark result schema`.

Result: TBD.

### - [ ] I4 Runner lifecycle and memory

Depends on: S5, I1, I3. RED lifecycle, timeout, crash isolation, sequential
execution, download exclusion, RSS and skipped behavior.

DoD: S5 architecture implemented minimally; cold/first/warm separated; no
cross-run contamination; commit `feat: add isolated benchmark runner`.

Result: TBD.

## M3: Measurement and Corpus

Status: SUPERSEDED by L0, L2 and L6. Retained for historical context.

### - [ ] I5 Reporting

RED fixtures for common table, Pareto, same-weights, model-family,
hardware/precision/quantization and P50/P95. Add accuracy, speed, RAM and disk
deltas from each quantized row to its unquantized model/runtime baseline. No
composite score.

DoD: deterministic snapshots green; commit
`feat: add benchmark report views`.

Result: TBD.

### - [ ] I6 Corpus profiles and normalizer

Depends on: S8. RED manifest/license/profile, raw/normalized transcript, macro
and duration-weighted aggregates.

DoD: smoke/standard/extended valid; references verified; tests green; commit
`feat: add benchmark corpus profiles`.

Result: TBD.

### - [ ] I7 Reproducible Parakeet Sherpa artifact builder

Depends on: S4E.

RED: tests validate pinned source revision, pinned exporter revision, required
outputs, dtype manifest, metadata, checksums and refusal to overwrite a
different build. No network download is performed by the builder.

GREEN: minimal build command consumes the cached official `.nemo` and writes:

```text
/Volumes/512GB/hf/derived/parakeet-tdt-0.6b-v3-sherpa-onnx/
  fp32/
  fp16/
  int8/
  manifest.json
```

DoD: FP32 comes from upstream export; INT8 uses upstream QUInt8 encoder and
QInt8 decoder/joiner; FP16 converts all three components and keeps public IO
compatible; manifest records toolchain and checksums; tests and smoke commands
green; commit `feat: add parakeet sherpa model builder`.

Result: TBD.

## M4: New Benchmark Configurations

Legacy task index. Execute only the rows selected by L1-L5 and in that order;
unchecked rows are not automatically part of the active roadmap.

Each: RED fake contract -> GREEN adapter -> RED cached integration -> GREEN real
invocation -> VERIFY -> REFACTOR. Agent never downloads missing models.

### - [ ] N0 Faster-whisper quantized compute
DoD: same model weights benchmarked with supported CTranslate2 `int8` and
`int8_float16` modes; unsupported modes are skipped with reason; effective
compute type, EN/RU WER/CER, RTFx and RAM delta vs default recorded; commit
`feat: add faster whisper quant configs`. Result: TBD.

### - [ ] N1 Whisper.cpp FP16
DoD: official `whisper-cli` subprocess, version probe and local model path;
tiny/turbo mapping; EN/RU smoke; Metal/provenance/segment timestamps;
`quantization=none`; cold process time recorded and load-only timing marked
unsupported; commit `feat: add whisper cpp backend`. Result: TBD.

### - [ ] N1.1 Whisper.cpp Q5
DoD: official CLI tiny Q5_1 and turbo Q5_0 rows; EN/RU smoke; quant metadata,
disk/RAM/RTFx and WER/CER delta vs N1; commit
`feat: add whisper cpp q5 configs`. Result: TBD.

### - [ ] N1.2 Whisper.cpp Q8
DoD: official CLI tiny/turbo Q8_0 rows; EN/RU smoke; quant metadata,
disk/RAM/RTFx and WER/CER delta vs N1; commit
`feat: add whisper cpp q8 configs`. Result: TBD.

### - [ ] N2 GigaAM CTC MLX
DoD: separate row; RU smoke; provenance/punctuation; commit `feat: add gigaam ctc mlx backend`. Result: TBD.

### - [ ] N3 GigaAM RNNT MLX
DoD: separate row; RU smoke; provenance; commit `feat: add gigaam rnnt mlx backend`. Result: TBD.

### - [ ] N4 GigaAM CTC official
DoD: PyTorch row; parity evidence; RU smoke; commit `feat: add official gigaam ctc backend`. Result: TBD.

### - [x] N5 GigaAM RNNT official
DoD: PyTorch row; parity evidence; RU smoke. Result: DONE by T0.5 through the
current `gigaam` / `e2e_rnnt` pair.

### - [ ] N6 Parakeet official
DoD: official row; EN/RU smoke; timestamps/provenance; commit `feat: add official parakeet backend`. Result: TBD.

### - [ ] N7 Parakeet MLX
DoD: MLX row; EN/RU smoke; parity vs N6; commit `feat: add parakeet mlx backend`. Result: TBD.

### - [ ] N8 Parakeet sherpa-onnx
DoD: sherpa row uses I7 `fp32/` artifacts derived from the pinned official
NVIDIA source; `quantization=none`; separate encoder/decoder/joiner layout,
checksums and manifest validated; EN/RU smoke and parity vs N6 recorded; commit
`feat: add parakeet sherpa onnx backend`. Result: TBD.

### - [ ] N8.2 Parakeet sherpa-onnx FP16
Status: BLOCKED by D-008/S4E. Six conversion approaches fail at ONNX Runtime
load because of internal Cast/type conflicts. Reopen only for an upstream or
native FP16 export path.
DoD: I7 `fp16/` artifacts have all encoder/decoder/joiner weights FP16 with
compatible public IO; EN/RU smoke; size/RAM/RTFx and WER/CER delta vs N8; commit
`feat: add parakeet sherpa fp16 config`. Result: TBD.

### - [ ] N8.3 Parakeet sherpa-onnx INT8
DoD: I7 `int8/` artifacts use upstream QUInt8 encoder and QInt8
decoder/joiner; EN/RU smoke; size/RAM/RTFx and WER/CER delta vs N8; commit
`feat: add parakeet sherpa int8 config`. Result: TBD.

### - [ ] N8.1 Parakeet ONNX-ASR FP32
DoD: `runtime=onnx-asr`, `precision=fp32`, `quantization=none`; only
unsuffixed ONNX files are loaded; EN/RU smoke; ONNX Runtime providers, peak
RAM and parity vs N6 are recorded; commit `feat: add parakeet onnx asr backend`.
Result: TBD.

### - [ ] N8.4 Parakeet ONNX-ASR INT8
DoD: only `.int8.onnx` encoder/decoder-joint files are loaded; no FP32 fallback;
EN/RU smoke; provider, size/RAM/RTFx and WER/CER delta vs N8.1 recorded;
commit `feat: add parakeet onnx asr int8 config`. Result: TBD.

### - [x] N9 Qwen3-ASR official
DoD: S6 details applied; EN/RU smoke; memory fit. Result: DONE through the
`qwen3-asr-hf` / `qwen3-asr-1.7b-hf` row. ForcedAligner is excluded.

### - [x] N10 Qwen3-ASR MLX
DoD: S6 details applied; EN/RU smoke; parity. Result: DONE by T0.8 through the
current `qwen3-asr` / `qwen3-asr-0.6b-8bit` pair; official and 1.7B work remains
complete in S6/L4 through the separate `qwen` profile.

### - [ ] N11 Canary official
DoD: S7 details applied; EN/RU smoke; memory fit; commit `feat: add official canary backend`. Result: TBD.

### - [ ] N12 Canary MLX
DoD: S7 details applied; EN/RU smoke; parity; commit `feat: add canary mlx backend`. Result: TBD.

### - [x] N13 Vosk small RU
DoD: CPU RU smoke; timestamps/memory/footprint. Result: DONE through
`vosk-small-ru`; three-run comparison is recorded in `RESULTS.md`.

### - [x] N14 Vosk full RU
DoD: CPU RU smoke; comparison with N13. Result: DONE by T0.7 through the current
full Zipformer2 ONNX model. N13 remains the missing small-model comparison.

## M5: Operations and Documentation

Status: SUPERSEDED as a release prerequisite. Add individual operational tools
only after a concrete need; L6 owns the lean report and acceptance checks.

### - [ ] C1 Add `stt-benchmark doctor`
DoD: RED volume/cache/dependency/hardware tests; no downloads; actionable output; commit `feat: add benchmark doctor command`. Result: TBD.

### - [ ] C2 Add `stt-benchmark models`
DoD: RED inventory tests; read-only `/Volumes/512GB/hf`; direct known-repo
scan ignores AppleDouble `._*`; no `hf cache list` subprocess; reports
repo/revision/size and duplicate known repos outside canonical cache; commit
`feat: add model inventory command`. Result: TBD.

### - [ ] C3 CI without model downloads
DoD: clean-runner unit/contract green; optional Apple Silicon job documented; commit `ci: test benchmark without model downloads`. Result: TBD.

### - [ ] D1 Document methodology
DoD: `docs/methodology.md` covers corpus, normalizer, timings, metrics, limits and views; README link; commit `docs: document benchmark methodology`. Result: TBD.

### - [ ] D2 Document result schema
DoD: `docs/result-schema.md` covers fields, units, nullability, enums and example; commit `docs: document benchmark result schema`. Result: TBD.

### - [ ] D3 Document model/runtime inventory
DoD: `docs/models.md` lists source IDs/revisions/licenses, derived-artifact
toolchain and checksums, runtime variants, precision, quantization, languages
and limits; commit `docs: document benchmark models`. Result: TBD.

### - [ ] D4 Document architecture
DoD: `docs/architecture.md` reflects implemented lifecycle, cache and data flow; commit `docs: document benchmark architecture`. Result: TBD.

### - [ ] D5 Update README
DoD: setup, cache env, single CLI, profiles, breaking notice and troubleshooting; commit `docs: update benchmark usage`. Result: TBD.

### - [ ] R1 Release acceptance

DoD: full suite green; ready backends smoke green; standard JSON/report valid;
no downloads; clean worktree; RAID current; blockers in release notes; commit
`chore: prepare benchmark release`.

Result: TBD.

## Next Action

1. Decide whether to cache Borealis dependencies `openai/whisper-large-v3` and
   `Qwen/Qwen3-4B`; Borealis remains blocked until both are local.
2. Run optional L5 only when same-weights Podlodka MLX, classic Vosk or
   ForcedAligner comparison is explicitly needed.
3. Commit or release the completed lean critical-path changes only when
   explicitly requested; the current worktree intentionally remains
   uncommitted.
