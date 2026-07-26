# План улучшения STT-бенчмарка

## Project Charter

Цель: единый локальный offline benchmark для EN/RU STT model/runtime
комбинаций на Apple Silicon. Результаты должны содержать provenance,
effective config, accuracy, cold/warm timing, memory и disk footprint.

Non-goals:

- cloud API и streaming;
- квантизированные Whisper/GGML веса;
- legacy CLI и совместимость со старым JSON;
- автоматическая загрузка моделей агентом или benchmark-кодом;
- искусственный composite score.

Модели хранятся только в HF cache `/Volumes/512GB/hf`. Модели скачивает
пользователь. Агент не запускает `hf download`.

## Success Criteria

- Каждая строка содержит model/runtime/weights/runtime-version provenance.
- Каждый ready backend проходит smoke profile без ручной правки кода.
- Unit/contract tests не требуют моделей или сети.
- Missing model/dependency даёт `skipped`, не падение общего запуска.
- Проект не создаёт копии весов вне `/Volumes/512GB/hf`.
- Сохранённая CLI-команда и JSON config воспроизводят опубликованный отчёт.
- Допустимый разброс warm RTFx определяется spike S5.

## Workflow

- Checkbox: `- [ ]` до начала, `- [x]` после завершения.
- Status: `READY`, `IN_PROGRESS`, `BLOCKED`, `DONE`.
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
| 2026-07-26 | PLAN | Add Parakeet FP32 via onnx-asr | Separate unquantized ONNX runtime requested; repo contains FP32 and INT8 variants | pending |
| 2026-07-26 | PLAN | Add direct sherpa-onnx Parakeet FP32 repo | Verified separate encoder/decoder/joiner FP32 artifacts on Hugging Face | pending |

## Decision Log

| ID | Task | Решение | Альтернативы | Последствия | Status |
|---|---|---|---|---|---|
| D-001 | S1 | Resolve exact local snapshot, then pass its path to backend | repo ID passed directly to backend | Prevents downloads and pins SHA; env remains required | CLOSED |
| D-002 | S2 | TBD | pywhispercpp vs whisper-cli | TBD | OPEN |
| D-003 | S5 | TBD | in-process vs isolated worker | TBD | OPEN |

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

## Milestones

- M0 Baseline: PLAN, T0.1-T0.3. Exit: clean worktree, tests green.
- M1 Research: U1, S1-S8. Exit: evidence/status for every runtime.
- M2 Foundation: I1-I4. Exit: existing backends use one schema/runner.
- M3 Measurement: I5-I6. Exit: reports and corpus profiles reproducible.
- M4 Rollout: N1-N14 plus N8.1. Exit: every ready variant has benchmark row.
- M5 Readiness: C1-C3, D1-D5, R1. Exit: docs, CI, release gate complete.

Critical path:
`PLAN -> T0 -> U1/S1 -> S2-S5 -> I1 -> I2 -> I3 -> I4 -> I5/I6 -> N* -> C*/D* -> R1`.

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

### - [ ] U2 Parakeet ONNX-ASR FP32 available in `/Volumes/512GB/hf`

Status: READY. Owner: user. Priority: P0.

Agent must not run this command:

```bash
hf download istupakov/parakeet-tdt-0.6b-v3-onnx \
  --include "README.md" \
  --include "config.json" \
  --include "encoder-model.onnx" \
  --include "encoder-model.onnx.data" \
  --include "decoder_joint-model.onnx" \
  --include "nemo128.onnx" \
  --include "vocab.txt" \
  --cache-dir "/Volumes/512GB/hf"
```

DoD: user confirms download; only unsuffixed FP32 ONNX artifacts are required;
`.int8.onnx` files are absent or ignored; snapshot SHA and followed file sizes
are recorded; agent commits `docs: record parakeet onnx cache readiness`.

Result: waiting for user confirmation.

### - [ ] U3 Parakeet Sherpa-ONNX FP32 available in `/Volumes/512GB/hf`

Status: READY. Owner: user. Priority: P0.

Verified repo: `Yiivgeny/parakeet-tdt-0.6b-v3-sherpa-onnx-fp32`, revision
`2ed28fb1c13002afd5d0756be4d3cbe9be5170e5`. Agent must not run this
command. Download direct model files, not the duplicate compressed archive:

```bash
hf download Yiivgeny/parakeet-tdt-0.6b-v3-sherpa-onnx-fp32 \
  --revision "2ed28fb1c13002afd5d0756be4d3cbe9be5170e5" \
  --include "README.md" \
  --include "SHA256SUMS" \
  --include "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/bpe.vocab" \
  --include "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/decoder.onnx" \
  --include "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/encoder.onnx" \
  --include "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/encoder.weights" \
  --include "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/joiner.onnx" \
  --include "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp32/tokens.txt" \
  --cache-dir "/Volumes/512GB/hf"
```

Expected followed sizes: encoder weights 2,435,420,160 bytes, encoder graph
41,838,346 bytes, decoder 47,233,744 bytes, joiner 25,286,331 bytes. The
2,421,191,352-byte `.tar.bz2` is intentionally excluded to avoid storing the
same model twice.

DoD: user confirms download; exact revision and SHA256 files are present;
model directory contains separate encoder/decoder/joiner files; agent records
evidence and commits `docs: record parakeet sherpa fp32 cache readiness`.

Result: waiting for user confirmation.

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

### - [ ] S2 Whisper.cpp API and Metal

Status: READY. Owner: agent. Priority: P0. Depends on: U1, S1.

Probe: tiny FP16 on one EN/RU sample through pywhispercpp and official CLI.
Measure Metal info, load/inference, language/task/beam/VAD/timestamps, cleanup,
RSS. Core ML excluded.

DoD: binding and CLI result or reproducible blocker; commands/versions/metrics;
integration decision; N1 RED plan updated; D-002 closed; commit
`docs: record whisper cpp spike`.

Result: TBD.

### - [ ] S3 GigaAM CTC/RNNT parity

Status: READY. Owner: agent. Priority: P0. Depends on: U1, S1.

Probe MLX CTC, MLX RNNT, official CTC and official RNNT on same RU samples.
Check transcript, punctuation, normalization, timestamps, silence, long audio,
memory. EN is capability probe only.

DoD: all four variants have status/evidence; differences recorded; N2-N5 and
normalizer requirements updated; commit `docs: record gigaam spike`.

Result: TBD.

### - [ ] S4 Parakeet runtimes

Status: READY. Owner: agent. Priority: P0. Depends on: U1, U2, U3, S1.

Probe official, MLX, sherpa-onnx and onnx-asr FP32 on same EN/RU samples.
Compare API, precision, transcripts, timestamps, load, RTFx, RAM and cleanup.

DoD: all four have status/evidence; unblock tasks exist where needed; N6-N8.1
assertions updated; onnx-asr provider list and peak RAM are recorded; commit
`docs: record parakeet runtimes spike`.

Result: TBD.

### - [ ] S5 Worker isolation and memory

Status: READY. Owner: agent. Priority: P0.

Probe one MLX and one native/PyTorch backend in-process vs subprocess. Measure
startup, RSS, accelerator counters, IPC, crash and timeout isolation.

DoD: lifecycle chosen; load/first/warm boundaries defined; RTFx tolerance set;
I4 updated; D-003 closed; commit `docs: record worker isolation spike`.

Result: TBD.

### - [ ] S6 Qwen3-ASR runtimes

Status: READY. Owner: agent. Priority: P1. Depends on: U1, S1, S5.

Probe official HF and MLX 8-bit on EN/RU. Forced aligner excluded.

DoD: both have metrics/dependencies/status; N9/N10 updated; commit
`docs: record qwen asr spike`.

Result: TBD.

### - [ ] S7 Canary runtimes

Status: READY. Owner: agent. Priority: P1. Depends on: U1, S1, S5.

Probe official NeMo and MLX BF16 on EN/RU.

DoD: parity, memory and stability recorded; N11/N12 or unblock tasks updated;
commit `docs: record canary spike`.

Result: TBD.

### - [ ] S8 Corpus

Status: READY. Owner: agent. Priority: P1.

Research clean, long, noise, telephone, multi-speaker, names/numbers and
code-switching EN/RU sources, licenses and reference quality.

DoD: selected sources/licenses and exact manifest/profile sizes recorded; I6
updated; commit `docs: record benchmark corpus spike`.

Result: TBD.

## M2: Foundation

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
memory, footprint and status. No migration.

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

### - [ ] I5 Reporting

RED fixtures for common table, Pareto, same-weights, model-family,
hardware/precision and P50/P95. No composite score.

DoD: deterministic snapshots green; commit
`feat: add benchmark report views`.

Result: TBD.

### - [ ] I6 Corpus profiles and normalizer

Depends on: S8. RED manifest/license/profile, raw/normalized transcript, macro
and duration-weighted aggregates.

DoD: smoke/standard/extended valid; references verified; tests green; commit
`feat: add benchmark corpus profiles`.

Result: TBD.

## M4: New Benchmark Configurations

Each: RED fake contract -> GREEN adapter -> RED cached integration -> GREEN real
invocation -> VERIFY -> REFACTOR. Agent never downloads missing models.

### - [ ] N1 Whisper.cpp FP16
DoD: tiny/turbo mapping; EN/RU smoke; Metal/provenance/timestamps; commit `feat: add whisper cpp backend`. Result: TBD.

### - [ ] N2 GigaAM CTC MLX
DoD: separate row; RU smoke; provenance/punctuation; commit `feat: add gigaam ctc mlx backend`. Result: TBD.

### - [ ] N3 GigaAM RNNT MLX
DoD: separate row; RU smoke; provenance; commit `feat: add gigaam rnnt mlx backend`. Result: TBD.

### - [ ] N4 GigaAM CTC official
DoD: PyTorch row; parity evidence; RU smoke; commit `feat: add official gigaam ctc backend`. Result: TBD.

### - [ ] N5 GigaAM RNNT official
DoD: PyTorch row; parity evidence; RU smoke; commit `feat: add official gigaam rnnt backend`. Result: TBD.

### - [ ] N6 Parakeet official
DoD: official row; EN/RU smoke; timestamps/provenance; commit `feat: add official parakeet backend`. Result: TBD.

### - [ ] N7 Parakeet MLX
DoD: MLX row; EN/RU smoke; parity vs N6; commit `feat: add parakeet mlx backend`. Result: TBD.

### - [ ] N8 Parakeet sherpa-onnx
DoD: sherpa row uses
`Yiivgeny/parakeet-tdt-0.6b-v3-sherpa-onnx-fp32` revision `2ed28fb...`;
precision is FP32; separate encoder/decoder/joiner layout validated; EN/RU
smoke and parity vs N6 recorded; commit
`feat: add parakeet sherpa onnx backend`. Result: TBD.

### - [ ] N8.1 Parakeet ONNX-ASR FP32
DoD: `runtime=onnx-asr`, `precision=fp32`; only unsuffixed ONNX files are
loaded; EN/RU smoke; ONNX Runtime providers, peak RAM and parity vs N6 are
recorded; no INT8 fallback; commit `feat: add parakeet onnx asr backend`.
Result: TBD.

### - [ ] N9 Qwen3-ASR official
DoD: S6 details applied; EN/RU smoke; memory fit; commit `feat: add official qwen3 asr backend`. Result: TBD.

### - [ ] N10 Qwen3-ASR MLX
DoD: S6 details applied; EN/RU smoke; parity; commit `feat: add qwen3 asr mlx backend`. Result: TBD.

### - [ ] N11 Canary official
DoD: S7 details applied; EN/RU smoke; memory fit; commit `feat: add official canary backend`. Result: TBD.

### - [ ] N12 Canary MLX
DoD: S7 details applied; EN/RU smoke; parity; commit `feat: add canary mlx backend`. Result: TBD.

### - [ ] N13 Vosk small RU
DoD: CPU RU smoke; timestamps/memory/footprint; commit `feat: add vosk small ru backend`. Result: TBD.

### - [ ] N14 Vosk full RU
DoD: CPU RU smoke; comparison with N13; commit `feat: add vosk full ru backend`. Result: TBD.

## M5: Operations and Documentation

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
DoD: `docs/models.md` lists IDs, revisions, licenses, variants, precision, languages and limits; commit `docs: document benchmark models`. Result: TBD.

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

1. Commit this file as `docs: add benchmark improvement tasks`.
2. Record commit hash in Change Log on T0.1 commit.
3. Execute T0.1, then T0.2, then T0.3.
4. Wait for user confirmation of U1.
5. Execute S1 and stop at plan update gate.
