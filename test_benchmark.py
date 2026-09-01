import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import benchmark_whisper
import download_models
import prepare_samples
import smoke_test
from stt_benchmark import cli as stt_cli
from stt_benchmark.subprocess_runner import WorkerExecution


class NormalizeTranscriptTests(unittest.TestCase):
    def test_normalize_transcript_lowercases_and_removes_punctuation(self) -> None:
        text = "  Hello,   WORLD!  Shelley's flour-fattened sauce.  "
        self.assertEqual(
            benchmark_whisper.normalize_transcript(text),
            "hello world shelley s flour fattened sauce",
        )

    def test_load_reference_transcript_uses_same_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reference.txt"
            path.write_text("HELLO,   WORLD!\n", encoding="utf-8")
            self.assertEqual(
                benchmark_whisper.load_reference_transcript(path),
                "hello world",
            )


class ErrorResultTests(unittest.TestCase):
    def test_build_error_result_sets_error_fields(self) -> None:
        result = benchmark_whisper.build_error_result(
            backend="mlx-whisper",
            model_name="tiny",
            backend_device="mlx",
            run_index=2,
            error="boom",
            audio_path=Path("audio.mp3"),
            sample_label="audio",
            audio_duration_seconds=10.0,
            forced_language="en",
        )
        self.assertEqual(result.audio, "audio.mp3")
        self.assertEqual(result.sample_label, "audio")
        self.assertEqual(result.forced_language, "en")
        self.assertEqual(result.backend, "mlx-whisper")
        self.assertEqual(result.model, "tiny")
        self.assertEqual(result.backend_device, "mlx")
        self.assertEqual(result.run_index, 2)
        self.assertEqual(result.status, "error")
        self.assertEqual(result.error, "boom")
        self.assertIsNone(result.transcript)
        self.assertIsNone(result.total_seconds)


class BackendCapabilitiesTests(unittest.TestCase):
    def test_hallucination_threshold_helper_respects_backend_capabilities(self) -> None:
        args = argparse.Namespace(hallucination_silence_threshold=2.0)
        self.assertEqual(
            benchmark_whisper.hallucination_silence_threshold_for_backend(
                "mlx-whisper", args
            ),
            2.0,
        )
        self.assertIsNone(
            benchmark_whisper.hallucination_silence_threshold_for_backend(
                "mlx-audio", args
            )
        )


class OutputHelperTests(unittest.TestCase):
    def test_resolve_output_paths_uses_explicit_path_when_provided(self) -> None:
        json_path = Path("custom/results.json")
        self.assertEqual(
            benchmark_whisper.resolve_output_paths(json_path),
            json_path,
        )

    def test_resolve_output_paths_generates_timestamped_default_path(self) -> None:
        fake_now = mock.Mock()
        fake_now.strftime.return_value = "20260413_120000"
        with mock.patch.object(benchmark_whisper, "datetime") as mock_datetime:
            mock_datetime.now.return_value = fake_now
            json_path = benchmark_whisper.resolve_output_paths(None)
        self.assertEqual(
            json_path,
            Path("output") / "benchmark_results_20260413_120000.json",
        )

    def test_build_metadata_includes_current_benchmark_options(self) -> None:
        args = argparse.Namespace(
            profile="main",
            audios=["en", "ru"],
            models=["tiny", "large-v3"],
            backends=["mlx-whisper", "openai-whisper"],
            models_explicit=True,
            backends_explicit=True,
            runs=2,
            task="transcribe",
            beam_size=5,
            compute_type="default",
            device="auto",
            faster_whisper_vad_filter=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            openai_whisper_temperature_fallback=True,
            lightning_whisper_mlx_batch_size=12,
            insanely_fast_whisper_device_id="mps",
            insanely_fast_whisper_batch_size=1,
            insanely_fast_whisper_flash=False,
            gigaam_python=Path(".venvs/gigaam/bin/python"),
            gigaam_model_path=Path("/tmp/gigaam-snapshot"),
            gigaam_multilingual_python=Path(".venvs/gigaam/bin/python"),
            gigaam_multilingual_model_path=Path("/tmp/gigaam-multilingual-snapshot"),
            gigaam_multilingual_language=None,
            tone_python=Path(".venvs/t-one/bin/python"),
            tone_model_path=Path("/tmp/t-one-snapshot"),
            tone_decoder="beam",
            vosk_python=Path(".venvs/vosk/bin/python"),
            vosk_model_path=Path("/tmp/vosk-snapshot"),
            vosk_decoding_method="greedy_search",
            qwen3_asr_python=Path(".venv/bin/python"),
            qwen3_asr_model_path=Path("/tmp/qwen3-asr-snapshot"),
            qwen3_asr_language=None,
            qwen3_asr_max_tokens=8192,
            qwen3_asr_temperature=0.0,
            worker_timeout_seconds=900.0,
            warmup=True,
            show_full_table=False,
        )
        audio_inputs = [
            benchmark_whisper.ResolvedAudioInput(
                audio_path=Path("samples/en.mp3"),
                reference_transcript_path=Path("samples/en.txt"),
                reference_transcript_text="hello world",
                forced_language="en",
                selector_language="en",
                sample_label="en_sample",
                source="default-language",
                audio_duration_seconds=12.5,
            ),
            benchmark_whisper.ResolvedAudioInput(
                audio_path=Path("samples/ru.mp3"),
                reference_transcript_path=Path("samples/ru.txt"),
                reference_transcript_text="privet mir",
                forced_language="ru",
                selector_language="ru",
                sample_label="ru_sample",
                source="default-language",
                audio_duration_seconds=13.5,
            ),
        ]

        metadata = benchmark_whisper.build_metadata(
            args=args, audio_inputs=audio_inputs
        )

        self.assertEqual(metadata["audio_selectors"], ["en", "ru"])
        self.assertEqual(metadata["profile"], "main")
        self.assertEqual(len(metadata["audios"]), 2)
        self.assertEqual(metadata["audios"][0]["audio"], "samples/en.mp3")
        self.assertEqual(metadata["audios"][0]["forced_language"], "en")
        self.assertEqual(metadata["models"], ["tiny", "large-v3"])
        self.assertEqual(metadata["backends"], ["mlx-whisper", "openai-whisper"])
        self.assertIsNone(metadata["benchmark_pairs"])
        self.assertEqual(metadata["runs"], 2)
        self.assertEqual(metadata["task"], "transcribe")
        self.assertEqual(metadata["condition_on_previous_text"], False)
        self.assertEqual(metadata["hallucination_silence_threshold"], 2.0)
        self.assertEqual(metadata["warmup"], True)
        self.assertEqual(metadata["show_full_table"], False)
        self.assertEqual(metadata["gigaam_python"], ".venvs/gigaam/bin/python")
        self.assertEqual(metadata["gigaam_model_path"], "/tmp/gigaam-snapshot")
        self.assertEqual(
            metadata["gigaam_multilingual_python"], ".venvs/gigaam/bin/python"
        )
        self.assertEqual(
            metadata["gigaam_multilingual_model_path"],
            "/tmp/gigaam-multilingual-snapshot",
        )
        self.assertEqual(
            metadata["gigaam_multilingual_model_variant"], "large_ctc"
        )
        self.assertIsNone(metadata["gigaam_multilingual_language"])
        self.assertIsNone(metadata["gigaam_multilingual_effective_language_hint"])
        self.assertTrue(metadata["gigaam_multilingual_offline"])
        self.assertEqual(metadata["tone_python"], ".venvs/t-one/bin/python")
        self.assertEqual(metadata["tone_model_path"], "/tmp/t-one-snapshot")
        self.assertEqual(metadata["tone_model_variant"], "t-one-greedy")
        self.assertEqual(metadata["tone_decoder"], "beam")
        self.assertFalse(metadata["tone_streaming"])
        self.assertTrue(metadata["tone_offline"])
        self.assertEqual(metadata["vosk_python"], ".venvs/vosk/bin/python")
        self.assertEqual(metadata["vosk_model_path"], "/tmp/vosk-snapshot")
        self.assertEqual(metadata["vosk_model_variant"], "vosk-ru")
        self.assertEqual(metadata["vosk_decoding_method"], "greedy_search")
        self.assertEqual(metadata["vosk_quantization"], "fp32")
        self.assertFalse(metadata["vosk_streaming"])
        self.assertTrue(metadata["vosk_offline"])
        self.assertEqual(metadata["qwen3_asr_python"], ".venv/bin/python")
        self.assertEqual(
            metadata["qwen3_asr_model_path"], "/tmp/qwen3-asr-snapshot"
        )
        self.assertEqual(
            metadata["qwen3_asr_model_variant"], "qwen3-asr-0.6b-8bit"
        )
        self.assertIsNone(metadata["qwen3_asr_language"])
        self.assertIsNone(metadata["qwen3_asr_effective_language_hint"])
        self.assertEqual(metadata["qwen3_asr_max_tokens"], 8192)
        self.assertEqual(metadata["qwen3_asr_temperature"], 0.0)
        self.assertTrue(metadata["qwen3_asr_offline"])
        self.assertEqual(metadata["worker_timeout_seconds"], 900.0)
        self.assertNotIn("mlx_prefix", metadata)
        self.assertNotIn("mlx_suffix", metadata)
        self.assertIn("platform", metadata)
        self.assertIn("python_version", metadata)


class DeviceResolutionTests(unittest.TestCase):
    def test_resolve_insanely_fast_whisper_device_cpu(self) -> None:
        device, should_clear_mps_cache, resolved_device_id = (
            benchmark_whisper.resolve_insanely_fast_whisper_device("cpu")
        )

        self.assertEqual(device, "cpu")
        self.assertFalse(should_clear_mps_cache)
        self.assertEqual(resolved_device_id, "cpu")

    def test_aggregate_results_uses_load_seconds_key(self) -> None:
        result = benchmark_whisper.RunResult(
            audio="samples/en.mp3",
            sample_label="en_sample",
            audio_duration_seconds=10.0,
            forced_language="en",
            backend="mlx-whisper",
            model="tiny",
            backend_device="mlx",
            run_index=1,
            load_seconds=1.25,
            transcribe_seconds=2.5,
            total_seconds=3.75,
            transcript="hello world",
            transcript_chars=11,
            transcript_words=2,
            wer=None,
            cer=None,
            detected_language="en",
            detected_language_probability=None,
            status="ok",
            error=None,
            peak_rss_mb=None,
        )

        aggregated = benchmark_whisper.aggregate_results([result])

        self.assertEqual(aggregated[0]["load_seconds"], 1.25)
        self.assertNotIn("avg_load_seconds", aggregated[0])
        self.assertEqual(aggregated[0]["audio"], "samples/en.mp3")
        self.assertEqual(aggregated[0]["forced_language"], "en")

    def test_print_summary_includes_backend_device_column(self) -> None:
        aggregated = [
            {
                "audio": "samples/en.mp3",
                "sample_label": "en_sample",
                "forced_language": "en",
                "backend": "mlx-whisper",
                "backend_device": "mlx",
                "model": "tiny",
                "runs": 1,
                "successful_runs": 1,
                "avg_total_seconds": 3.75,
                "median_total_seconds": 3.75,
                "load_seconds": 1.25,
                "avg_transcribe_seconds": 2.5,
                "stddev_transcribe_seconds": None,
                "avg_rtf": 0.25,
                "avg_wer": None,
                "avg_cer": None,
            }
        ]

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            benchmark_whisper.print_summary(aggregated)

        output = stdout.getvalue()
        self.assertIn("device", output)
        self.assertIn("mlx", output)
        self.assertIn("en_sample", output)

    def test_print_runs_table_includes_per_run_details(self) -> None:
        results = [
            benchmark_whisper.RunResult(
                audio="samples/en.mp3",
                sample_label="en_sample",
                audio_duration_seconds=10.0,
                forced_language="en",
                backend="mlx-whisper",
                model="tiny",
                backend_device="mlx",
                run_index=1,
                load_seconds=1.25,
                transcribe_seconds=2.5,
                total_seconds=3.75,
                transcript="hello world",
                transcript_chars=11,
                transcript_words=2,
                wer=None,
                cer=None,
                detected_language="en",
                detected_language_probability=None,
                status="ok",
                error=None,
                peak_rss_mb=None,
            )
        ]

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            benchmark_whisper.print_runs_table(results)

        output = stdout.getvalue()
        self.assertIn("Runs:", output)
        self.assertIn("en_sample", output)
        self.assertIn("mlx-whisper", output)
        self.assertIn("mlx", output)
        self.assertIn("ok", output)

    def test_write_json_writes_pretty_json_with_trailing_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.json"
            payload = {"hello": "world", "count": 2}

            benchmark_whisper.write_json(output_path, payload)

            content = output_path.read_text(encoding="utf-8")
            self.assertTrue(content.endswith("\n"))
            self.assertEqual(json.loads(content), payload)
            self.assertIn('\n  "hello": "world",\n', content)


class BenchmarkCliTests(unittest.TestCase):
    def test_main_profile_defaults_to_six_russian_baselines(self) -> None:
        args = benchmark_whisper.parse_args([])

        self.assertEqual(args.profile, "main")
        self.assertEqual(args.audios, ["ru"])
        self.assertEqual(
            args.models,
            [
                "large-v3-turbo",
                "e2e_rnnt",
                "gigaam-multilingual-large-ctc",
                "t-one-greedy",
                "vosk-ru",
                "qwen3-asr-0.6b-8bit",
            ],
        )
        self.assertEqual(
            args.backends,
            [
                "mlx-whisper",
                "gigaam",
                "gigaam-multilingual",
                "t-one",
                "vosk",
                "qwen3-asr",
            ],
        )
        self.assertFalse(args.models_explicit)
        self.assertFalse(args.backends_explicit)
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            [
                ("mlx-whisper", "large-v3-turbo"),
                ("gigaam", "e2e_rnnt"),
                ("gigaam-multilingual", "gigaam-multilingual-large-ctc"),
                ("t-one", "t-one-greedy"),
                ("vosk", "vosk-ru"),
                ("qwen3-asr", "qwen3-asr-0.6b-8bit"),
            ],
        )

    def test_main_audio_override_keeps_exact_profile_pairs(self) -> None:
        args = benchmark_whisper.parse_args(["--audio", "en"])

        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            [
                ("mlx-whisper", "large-v3-turbo"),
                ("gigaam", "e2e_rnnt"),
                ("gigaam-multilingual", "gigaam-multilingual-large-ctc"),
                ("t-one", "t-one-greedy"),
                ("vosk", "vosk-ru"),
                ("qwen3-asr", "qwen3-asr-0.6b-8bit"),
            ],
        )

    def test_qwen_profile_uses_bundled_en_ru_and_exact_mlx_hf_pairs(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "qwen"])

        self.assertEqual(args.audios, [])
        self.assertEqual(
            args.models,
            [
                *benchmark_whisper.QWEN3_ASR_MODEL_VARIANTS,
                benchmark_whisper.QWEN3_ASR_HF_MODEL_VARIANT,
            ],
        )
        self.assertEqual(args.backends, ["qwen3-asr", "qwen3-asr-hf"])
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            benchmark_whisper.QWEN_BENCHMARK_PAIRS,
        )

        with (
            mock.patch.object(
                benchmark_whisper,
                "ensure_audio_file",
                side_effect=lambda path: path.resolve(),
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_reference_transcript",
                side_effect=lambda path: f"normalized:{path.name}",
            ),
            mock.patch.object(
                benchmark_whisper,
                "get_audio_duration_seconds",
                return_value=1.0,
            ),
        ):
            audio_inputs = benchmark_whisper.resolve_audio_inputs(args)

        self.assertEqual([item.selector_language for item in audio_inputs], ["en", "ru"])
        self.assertEqual([item.forced_language for item in audio_inputs], ["en", "ru"])

        main_args = benchmark_whisper.parse_args([])
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(main_args),
            benchmark_whisper.MAIN_BENCHMARK_PAIRS,
        )
        self.assertNotIn("qwen3-asr-hf", main_args.backends)

    def test_qwen_profile_does_not_change_main_defaults(self) -> None:
        args = benchmark_whisper.parse_args([])

        self.assertEqual(args.models[-1], "qwen3-asr-0.6b-8bit")
        self.assertNotIn("qwen3-asr-1.7b-8bit", args.models)
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            benchmark_whisper.MAIN_BENCHMARK_PAIRS,
        )

    def test_ru_variants_profile_uses_only_bundled_ru_and_exact_pairs(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "ru-variants"])

        self.assertEqual(args.audios, ["ru"])
        self.assertEqual(args.models, ["e2e_rnnt", "e2e_ctc", "vosk-ru", "vosk-small-ru"])
        self.assertEqual(args.backends, ["gigaam", "vosk"])
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            benchmark_whisper.RU_VARIANTS_BENCHMARK_PAIRS,
        )

        with (
            mock.patch.object(
                benchmark_whisper,
                "ensure_audio_file",
                side_effect=lambda path: path.resolve(),
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_reference_transcript",
                side_effect=lambda path: f"normalized:{path.name}",
            ),
            mock.patch.object(
                benchmark_whisper,
                "get_audio_duration_seconds",
                return_value=1.0,
            ),
        ):
            audio_inputs = benchmark_whisper.resolve_audio_inputs(args)

        self.assertEqual([item.selector_language for item in audio_inputs], ["ru"])
        self.assertEqual([item.forced_language for item in audio_inputs], ["ru"])

    def test_ru_variants_profile_does_not_change_main_pairs(self) -> None:
        main_args = benchmark_whisper.parse_args([])
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(main_args),
            benchmark_whisper.MAIN_BENCHMARK_PAIRS,
        )
        self.assertNotIn("e2e_ctc", main_args.models)
        self.assertNotIn("vosk-small-ru", main_args.models)

    def test_whisper_profile_preserves_all_current_runtimes_and_models(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "whisper"])

        self.assertEqual(args.audios, [])
        self.assertEqual(args.models, benchmark_whisper.DEFAULT_MODELS)
        self.assertEqual(
            args.backends,
            benchmark_whisper.CURRENT_WHISPER_BACKENDS,
        )
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            [
                (backend, model)
                for model in benchmark_whisper.DEFAULT_MODELS
                for backend in benchmark_whisper.CURRENT_WHISPER_BACKENDS
            ],
        )

    def test_podlodka_profile_uses_one_pair_on_all_bundled_samples(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "podlodka"])

        self.assertEqual(args.audios, [])
        self.assertEqual(args.models, ["whisper-podlodka-turbo"])
        self.assertEqual(args.backends, ["podlodka"])
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            [("podlodka", "whisper-podlodka-turbo")],
        )
        with (
            mock.patch.object(
                benchmark_whisper,
                "ensure_audio_file",
                side_effect=lambda path: path.resolve(),
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_reference_transcript",
                side_effect=lambda path: f"normalized:{path.name}",
            ),
            mock.patch.object(
                benchmark_whisper,
                "get_audio_duration_seconds",
                side_effect=lambda _path: 10.0,
            ),
        ):
            audio_inputs = benchmark_whisper.resolve_audio_inputs(args)

        self.assertEqual([item.selector_language for item in audio_inputs], ["en", "ru"])
        self.assertEqual([item.forced_language for item in audio_inputs], ["en", "ru"])

    def test_podlodka_profile_runs_without_skips_on_both_samples(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "podlodka"])
        args.output = Path("/tmp/test-output.json")
        args.runs = 1
        audio_inputs = [
            benchmark_whisper.ResolvedAudioInput(
                audio_path=Path("en.mp3"),
                reference_transcript_path=None,
                reference_transcript_text=None,
                forced_language="en",
                selector_language="en",
                sample_label="en",
                source="default-language",
                audio_duration_seconds=1.0,
            ),
            benchmark_whisper.ResolvedAudioInput(
                audio_path=Path("ru.mp3"),
                reference_transcript_path=None,
                reference_transcript_text=None,
                forced_language="ru",
                selector_language="ru",
                sample_label="ru",
                source="default-language",
                audio_duration_seconds=1.0,
            ),
        ]
        run_calls: list[tuple[str, str, str]] = []

        def fake_run(
            audio_path: Path,
            model_name: str,
            run_index: int,
            run_args: argparse.Namespace,
            _model_path: Path,
        ) -> benchmark_whisper.RunResult:
            run_calls.append((str(audio_path), model_name, run_args.language))
            return benchmark_whisper.build_run_result(
                backend="podlodka",
                model_name=model_name,
                run_index=run_index,
                load_seconds=0.1,
                transcribe_seconds=0.2,
                transcript="ok",
                detected_language=run_args.language,
                detected_language_probability=None,
                reference_transcript=None,
                audio_path=audio_path,
                sample_label=run_args.sample_label,
                audio_duration_seconds=run_args.audio_duration_seconds,
                forced_language=run_args.language,
            )

        written_payload: dict[str, object] = {}
        with (
            mock.patch.object(benchmark_whisper, "parse_args", return_value=args),
            mock.patch.object(
                benchmark_whisper, "resolve_output_paths", return_value=args.output
            ),
            mock.patch.object(
                benchmark_whisper, "resolve_audio_inputs", return_value=audio_inputs
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_podlodka_model_path",
                return_value=Path("/models/podlodka"),
            ),
            mock.patch.object(
                benchmark_whisper, "run_podlodka", side_effect=fake_run
            ),
            mock.patch.object(
                benchmark_whisper,
                "write_json",
                side_effect=lambda _path, payload: written_payload.update(payload),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            exit_code = benchmark_whisper.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run_calls,
            [
                ("en.mp3", "whisper-podlodka-turbo", "en"),
                ("ru.mp3", "whisper-podlodka-turbo", "ru"),
            ],
        )
        self.assertEqual(written_payload["skipped"], [])

    def test_podlodka_options_use_cli_values_over_environment(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "PODLODKA_PYTHON": "/env/bin/python",
                "PODLODKA_MODEL_PATH": "/env/model",
            },
            clear=True,
        ):
            args = benchmark_whisper.parse_args(
                [
                    "--profile",
                    "podlodka",
                    "--podlodka-python",
                    "/cli/bin/python",
                    "--podlodka-model-path",
                    "/cli/model",
                    "--podlodka-language",
                    "English",
                    "--podlodka-max-new-tokens",
                    "123",
                ]
            )

        self.assertEqual(args.podlodka_python, Path("/cli/bin/python"))
        self.assertEqual(args.podlodka_model_path, Path("/cli/model"))
        self.assertEqual(args.podlodka_language, "English")
        self.assertEqual(args.podlodka_max_new_tokens, 123)

    def test_podlodka_options_use_environment_and_builtin_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "PODLODKA_PYTHON": "/env/bin/python",
                "PODLODKA_MODEL_PATH": "/env/model",
            },
            clear=True,
        ):
            args = benchmark_whisper.parse_args([])

        self.assertEqual(args.podlodka_python, Path("/env/bin/python"))
        self.assertEqual(args.podlodka_model_path, Path("/env/model"))
        self.assertIsNone(args.podlodka_language)
        self.assertEqual(args.podlodka_max_new_tokens, 444)

        with mock.patch.dict("os.environ", {}, clear=True):
            fallback_args = benchmark_whisper.parse_args([])

        self.assertEqual(
            fallback_args.podlodka_python, benchmark_whisper.DEFAULT_PODLODKA_PYTHON
        )
        self.assertIsNone(fallback_args.podlodka_model_path)
        self.assertIsNone(fallback_args.podlodka_language)
        self.assertEqual(fallback_args.podlodka_max_new_tokens, 444)

    def test_resolve_podlodka_model_path_uses_pinned_snapshot_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--bond005--whisper-podlodka-turbo"
            revision = "f" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (cache_root / "refs" / "main").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")

            with mock.patch.object(
                benchmark_whisper, "PODLODKA_MODEL_CACHE_ROOT", cache_root
            ):
                resolved = benchmark_whisper.resolve_podlodka_model_path()

        self.assertEqual(resolved, snapshot.resolve())

    def test_resolve_podlodka_model_path_requires_config_and_nonempty_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            (model_path / "model.safetensors").write_bytes(b"weights")
            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_podlodka_model_path(model_path)

            (model_path / "config.json").write_text("{}", encoding="utf-8")
            (model_path / "model.safetensors").write_bytes(b"")
            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_podlodka_model_path(model_path)

    def test_run_podlodka_maps_success_timing_scores_and_rss(self) -> None:
        args = argparse.Namespace(
            language="ru",
            podlodka_language=None,
            podlodka_max_new_tokens=444,
            reference_transcript_text="привет мир",
            sample_label="audio",
            audio_duration_seconds=3.0,
            podlodka_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "Привет, мир!",
                "timestamps": [{"text": "Привет", "start": 0.0, "end": 0.5}],
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
                "language": "ru",
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=512.5,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_podlodka(
                Path("audio.mp3"),
                "whisper-podlodka-turbo",
                2,
                args,
                Path("/models/podlodka"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "podlodka")
        self.assertEqual(result.model, "whisper-podlodka-turbo")
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.transcript, "Привет, мир!")
        self.assertEqual(result.load_seconds, 1.25)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 512.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            "stt_benchmark.workers.podlodka",
            {
                "model_path": "/models/podlodka",
                "audio_path": "audio.mp3",
                "max_new_tokens": 444,
                "language": "ru",
            },
            12.5,
            env=benchmark_whisper.PODLODKA_OFFLINE_ENV,
        )

    def test_run_podlodka_language_precedence_and_auto(self) -> None:
        args = argparse.Namespace(
            language="ru",
            podlodka_language="English",
            podlodka_max_new_tokens=123,
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
            podlodka_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={"status": "ok", "text": "hello", "transcribe_seconds": 0.5},
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=0.75,
            peak_rss_mb=None,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            benchmark_whisper.run_podlodka(
                Path("audio.mp3"), "whisper-podlodka-turbo", 1, args, Path("/model")
            )
            args.podlodka_language = None
            args.language = "en"
            benchmark_whisper.run_podlodka(
                Path("audio.mp3"), "whisper-podlodka-turbo", 2, args, Path("/model")
            )
            args.language = "auto"
            benchmark_whisper.run_podlodka(
                Path("audio.mp3"), "whisper-podlodka-turbo", 3, args, Path("/model")
            )

        requests = [call.args[2] for call in run_worker.call_args_list]
        self.assertEqual(requests[0]["language"], "English")
        self.assertEqual(requests[1]["language"], "en")
        self.assertNotIn("language", requests[2])
        self.assertEqual(requests[0]["max_new_tokens"], 123)

    def test_run_podlodka_maps_worker_errors_and_rss(self) -> None:
        args = argparse.Namespace(
            language="ru",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
        )
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=64.0,
                error="worker timed out after 12.5 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "load failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", side_effect=executions
        ):
            timeout_result = benchmark_whisper.run_podlodka(
                Path("audio.mp3"),
                "whisper-podlodka-turbo",
                1,
                args,
                Path("/models/podlodka"),
            )
            worker_error_result = benchmark_whisper.run_podlodka(
                Path("audio.mp3"),
                "whisper-podlodka-turbo",
                2,
                args,
                Path("/models/podlodka"),
            )

        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(timeout_result.peak_rss_mb, 64.0)
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("load failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

    def test_explicit_options_override_profile_defaults(self) -> None:
        args = benchmark_whisper.parse_args(
            [
                "--profile",
                "whisper",
                "--audio",
                "ru",
                "--models",
                "large-v3-turbo",
                "--backends",
                "mlx-whisper",
            ]
        )

        self.assertEqual(args.audios, ["ru"])
        self.assertEqual(args.models, ["large-v3-turbo"])
        self.assertEqual(args.backends, ["mlx-whisper"])

    def test_explicit_main_model_or_backend_override_uses_cartesian_pairs(self) -> None:
        model_override = benchmark_whisper.parse_args(
            ["--models", "large-v3-turbo"]
        )
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(model_override),
            [
                ("mlx-whisper", "large-v3-turbo"),
                ("gigaam", "large-v3-turbo"),
                ("gigaam-multilingual", "large-v3-turbo"),
                ("t-one", "large-v3-turbo"),
                ("vosk", "large-v3-turbo"),
                ("qwen3-asr", "large-v3-turbo"),
            ],
        )

        backend_override = benchmark_whisper.parse_args(
            ["--backends", "mlx-whisper"]
        )
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(backend_override),
            [
                ("mlx-whisper", "large-v3-turbo"),
                ("mlx-whisper", "e2e_rnnt"),
                ("mlx-whisper", "gigaam-multilingual-large-ctc"),
                ("mlx-whisper", "t-one-greedy"),
                ("mlx-whisper", "vosk-ru"),
                ("mlx-whisper", "qwen3-asr-0.6b-8bit"),
            ],
        )

    def test_gigaam_options_use_cli_values_over_environment(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "GIGAAM_PYTHON": "/env/bin/python",
                "GIGAAM_MODEL_PATH": "/env/model",
            },
            clear=False,
        ):
            args = benchmark_whisper.parse_args(
                [
                    "--gigaam-python",
                    "/cli/bin/python",
                    "--gigaam-model-path",
                    "/cli/model",
                    "--worker-timeout-seconds",
                    "12.5",
                ]
            )

        self.assertEqual(args.gigaam_python, Path("/cli/bin/python"))
        self.assertEqual(args.gigaam_model_path, Path("/cli/model"))
        self.assertEqual(args.worker_timeout_seconds, 12.5)

    def test_gigaam_options_use_environment_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "GIGAAM_PYTHON": "/env/bin/python",
                "GIGAAM_MODEL_PATH": "/env/model",
            },
            clear=False,
        ):
            args = benchmark_whisper.parse_args([])

        self.assertEqual(args.gigaam_python, Path("/env/bin/python"))
        self.assertEqual(args.gigaam_model_path, Path("/env/model"))
        self.assertEqual(args.worker_timeout_seconds, 900.0)

    def test_gigaam_multilingual_options_use_cli_values_over_environment(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "GIGAAM_MULTILINGUAL_PYTHON": "/env/bin/python",
                "GIGAAM_MULTILINGUAL_MODEL_PATH": "/env/model",
            },
            clear=True,
        ):
            args = benchmark_whisper.parse_args(
                [
                    "--gigaam-multilingual-python",
                    "/cli/bin/python",
                    "--gigaam-multilingual-model-path",
                    "/cli/model",
                    "--gigaam-multilingual-language",
                    "kk",
                    "--worker-timeout-seconds",
                    "12.5",
                ]
            )

        self.assertEqual(args.gigaam_multilingual_python, Path("/cli/bin/python"))
        self.assertEqual(args.gigaam_multilingual_model_path, Path("/cli/model"))
        self.assertEqual(args.gigaam_multilingual_language, "kk")
        self.assertEqual(args.worker_timeout_seconds, 12.5)

    def test_gigaam_multilingual_options_use_environment_and_builtin_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "GIGAAM_MULTILINGUAL_PYTHON": "/env/bin/python",
                "GIGAAM_MULTILINGUAL_MODEL_PATH": "/env/model",
            },
            clear=True,
        ):
            args = benchmark_whisper.parse_args([])

        self.assertEqual(args.gigaam_multilingual_python, Path("/env/bin/python"))
        self.assertEqual(args.gigaam_multilingual_model_path, Path("/env/model"))
        self.assertIsNone(args.gigaam_multilingual_language)

        with mock.patch.dict("os.environ", {}, clear=True):
            fallback_args = benchmark_whisper.parse_args([])

        self.assertEqual(
            fallback_args.gigaam_multilingual_python,
            benchmark_whisper.DEFAULT_GIGAAM_MULTILINGUAL_PYTHON,
        )
        self.assertIsNone(fallback_args.gigaam_multilingual_model_path)
        self.assertIsNone(fallback_args.gigaam_multilingual_language)

    def test_resolve_gigaam_multilingual_model_path_uses_pinned_snapshot_and_required_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--ai-sage--GigaAM-Multilingual"
            revision = "e" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (cache_root / "refs" / "large_ctc").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            for relative_path in benchmark_whisper.GIGAAM_MULTILINGUAL_REQUIRED_MODEL_FILES:
                (snapshot / relative_path).write_bytes(b"model")

            with mock.patch.object(
                benchmark_whisper,
                "GIGAAM_MULTILINGUAL_MODEL_CACHE_ROOT",
                cache_root,
            ):
                resolved = benchmark_whisper.resolve_gigaam_multilingual_model_path()

        self.assertEqual(resolved, snapshot.resolve())

    def test_resolve_gigaam_multilingual_model_path_requires_all_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            for relative_path in benchmark_whisper.GIGAAM_MULTILINGUAL_REQUIRED_MODEL_FILES[:-1]:
                (model_path / relative_path).write_bytes(b"model")

            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_gigaam_multilingual_model_path(model_path)

    def test_qwen3_asr_options_use_cli_values_over_environment(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "QWEN3_ASR_PYTHON": "/env/bin/python",
                "QWEN3_ASR_MODEL_PATH": "/env/model",
            },
            clear=False,
        ):
            args = benchmark_whisper.parse_args(
                [
                    "--qwen3-asr-python",
                    "/cli/bin/python",
                    "--qwen3-asr-model-path",
                    "/cli/model",
                    "--qwen3-asr-language",
                    "English",
                    "--qwen3-asr-max-tokens",
                    "123",
                    "--qwen3-asr-temperature",
                    "0.25",
                    "--worker-timeout-seconds",
                    "12.5",
                ]
            )

        self.assertEqual(args.qwen3_asr_python, Path("/cli/bin/python"))
        self.assertEqual(args.qwen3_asr_model_path, Path("/cli/model"))
        self.assertEqual(args.qwen3_asr_language, "English")
        self.assertEqual(args.qwen3_asr_max_tokens, 123)
        self.assertEqual(args.qwen3_asr_temperature, 0.25)
        self.assertEqual(args.worker_timeout_seconds, 12.5)

    def test_qwen3_asr_options_use_environment_and_builtin_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "QWEN3_ASR_PYTHON": "/env/bin/python",
                "QWEN3_ASR_MODEL_PATH": "/env/model",
            },
            clear=False,
        ):
            args = benchmark_whisper.parse_args([])

        self.assertEqual(args.qwen3_asr_python, Path("/env/bin/python"))
        self.assertEqual(args.qwen3_asr_model_path, Path("/env/model"))
        self.assertIsNone(args.qwen3_asr_language)
        self.assertEqual(args.qwen3_asr_max_tokens, 8192)
        self.assertEqual(args.qwen3_asr_temperature, 0.0)

        with mock.patch.dict("os.environ", {}, clear=True):
            fallback_args = benchmark_whisper.parse_args([])

        self.assertEqual(
            fallback_args.qwen3_asr_python,
            benchmark_whisper.DEFAULT_QWEN3_ASR_PYTHON,
        )
        self.assertIsNone(fallback_args.qwen3_asr_model_path)
        self.assertIsNone(fallback_args.qwen3_asr_language)
        self.assertEqual(fallback_args.qwen3_asr_max_tokens, 8192)
        self.assertEqual(fallback_args.qwen3_asr_temperature, 0.0)

    def test_resolve_qwen3_asr_model_path_uses_pinned_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--mlx-community--Qwen3-ASR-0.6B-8bit"
            revision = "d" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (cache_root / "refs" / "main").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")

            with mock.patch.object(
                benchmark_whisper, "QWEN3_ASR_MODEL_CACHE_ROOT", cache_root
            ):
                resolved = benchmark_whisper.resolve_qwen3_asr_model_path()

        self.assertEqual(resolved, snapshot.resolve())

    def test_resolve_qwen3_asr_model_path_selects_each_variant_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_roots = {}
            snapshots = {}
            for index, variant in enumerate(benchmark_whisper.QWEN3_ASR_MODEL_VARIANTS):
                cache_root = Path(tmpdir) / f"models-{index}"
                revision = f"{index + 1:x}" * 40
                snapshot = cache_root / "snapshots" / revision
                (cache_root / "refs").mkdir(parents=True)
                snapshot.mkdir(parents=True)
                (cache_root / "refs" / "main").write_text(
                    f"{revision}\n", encoding="utf-8"
                )
                (snapshot / "model.safetensors").write_bytes(b"weights")
                cache_roots[variant] = cache_root
                snapshots[variant] = snapshot

            with mock.patch.object(
                benchmark_whisper, "QWEN3_ASR_MODEL_CACHE_ROOTS", cache_roots
            ):
                resolved = {
                    variant: benchmark_whisper.resolve_qwen3_asr_model_path(variant)
                    for variant in benchmark_whisper.QWEN3_ASR_MODEL_VARIANTS
                }

        self.assertEqual(
            resolved,
            {variant: snapshot.resolve() for variant, snapshot in snapshots.items()},
        )

    def test_resolve_qwen3_asr_model_path_requires_nonempty_weight_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            (model_path / "model.safetensors").touch()

            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_qwen3_asr_model_path(model_path)

    def test_run_qwen3_asr_maps_success_and_uses_resolved_audio_language_by_default(
        self,
    ) -> None:
        args = argparse.Namespace(
            language="ru",
            qwen3_asr_language=None,
            qwen3_asr_max_tokens=8192,
            qwen3_asr_temperature=0.0,
            reference_transcript_text="привет мир",
            sample_label="audio",
            audio_duration_seconds=3.0,
            qwen3_asr_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "Привет, мир!",
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
                "language": "Russian",
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=768.5,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_qwen3_asr(
                Path("audio.mp3"),
                "qwen3-asr-0.6b-8bit",
                2,
                args,
                Path("/models/qwen3-asr"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "qwen3-asr")
        self.assertEqual(result.model, "qwen3-asr-0.6b-8bit")
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.run_index, 2)
        self.assertEqual(result.transcript, "Привет, мир!")
        self.assertEqual(result.detected_language, "Russian")
        self.assertEqual(result.load_seconds, 1.25)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 768.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            "stt_benchmark.workers.qwen3_asr",
            {
                "model_path": "/models/qwen3-asr",
                "audio_path": "audio.mp3",
                "language": "ru",
                "max_tokens": 8192,
                "temperature": 0.0,
            },
            12.5,
            env=benchmark_whisper.QWEN3_ASR_OFFLINE_ENV,
        )

    def test_run_qwen3_asr_passes_explicit_language(self) -> None:
        args = argparse.Namespace(
            language="ru",
            qwen3_asr_language="English",
            qwen3_asr_max_tokens=123,
            qwen3_asr_temperature=0.25,
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
            qwen3_asr_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "text": "hello",
                "transcribe_seconds": 0.5,
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=0.75,
            peak_rss_mb=None,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_qwen3_asr(
                Path("audio.mp3"),
                "qwen3-asr-0.6b-8bit",
                1,
                args,
                Path("/models/qwen3-asr"),
            )

        self.assertEqual(result.transcript, "hello")
        request = run_worker.call_args.args[2]
        self.assertEqual(request["language"], "English")
        self.assertEqual(request["max_tokens"], 123)
        self.assertEqual(request["temperature"], 0.25)

    def test_run_qwen3_asr_passes_selected_sibling_model_path(self) -> None:
        args = argparse.Namespace(
            language="en",
            qwen3_asr_language=None,
            qwen3_asr_max_tokens=8192,
            qwen3_asr_temperature=0.0,
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
            qwen3_asr_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "text": "hello",
                "transcribe_seconds": 0.5,
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=0.75,
            peak_rss_mb=None,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_qwen3_asr(
                Path("audio.mp3"),
                benchmark_whisper.QWEN3_ASR_1_7B_MODEL_VARIANT,
                1,
                args,
                Path("/models/qwen3-asr-1.7b"),
            )

        self.assertEqual(result.model, benchmark_whisper.QWEN3_ASR_1_7B_MODEL_VARIANT)
        self.assertEqual(
            run_worker.call_args.args[2]["model_path"],
            "/models/qwen3-asr-1.7b",
        )

    def test_run_qwen3_asr_omits_language_for_auto(self) -> None:
        args = argparse.Namespace(
            language="auto",
            qwen3_asr_language=None,
            qwen3_asr_max_tokens=8192,
            qwen3_asr_temperature=0.0,
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
            qwen3_asr_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "text": "hello",
                "transcribe_seconds": 0.5,
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=0.75,
            peak_rss_mb=None,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            benchmark_whisper.run_qwen3_asr(
                Path("audio.mp3"),
                "qwen3-asr-0.6b-8bit",
                1,
                args,
                Path("/models/qwen3-asr"),
            )

        request = run_worker.call_args.args[2]
        self.assertNotIn("language", request)

    def test_run_qwen3_asr_maps_timeout_and_worker_errors_with_rss(self) -> None:
        args = argparse.Namespace(
            language=None,
            qwen3_asr_language=None,
            qwen3_asr_max_tokens=8192,
            qwen3_asr_temperature=0.0,
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
            qwen3_asr_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=64.0,
                error="worker timed out after 12.5 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "load failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", side_effect=executions
        ):
            timeout_result = benchmark_whisper.run_qwen3_asr(
                Path("audio.mp3"),
                "qwen3-asr-0.6b-8bit",
                1,
                args,
                Path("/models/qwen3-asr"),
            )
            worker_error_result = benchmark_whisper.run_qwen3_asr(
                Path("audio.mp3"),
                "qwen3-asr-0.6b-8bit",
                2,
                args,
                Path("/models/qwen3-asr"),
            )

        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(timeout_result.peak_rss_mb, 64.0)
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("load failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

    def test_main_default_matrix_runs_only_supported_pairs(self) -> None:
        args = benchmark_whisper.parse_args([])
        args.output = Path("/tmp/test-output.json")
        args.runs = 1
        audio_input = benchmark_whisper.ResolvedAudioInput(
            audio_path=Path("audio.mp3"),
            reference_transcript_path=None,
            reference_transcript_text=None,
            forced_language="ru",
            selector_language="ru",
            sample_label="audio",
            source="default-language",
            audio_duration_seconds=1.0,
        )
        load_calls: list[tuple[str, str]] = []
        run_calls: list[tuple[str, str]] = []

        def fake_load(backend: str, model: str, _args: argparse.Namespace):
            load_calls.append((backend, model))
            session: object = object()
            if backend in {"gigaam", "t-one", "vosk"}:
                session = {"model_path": Path(f"/tmp/{backend}-model")}
            return benchmark_whisper.BackendSession(
                backend=backend,
                model=model,
                device="subprocess"
                if backend in {"gigaam", "t-one", "vosk"}
                else "mlx",
                session=session,
                load_seconds=None,
            )

        def fake_run(
            backend: str,
            _audio_path: Path,
            model: str,
            _run_index: int,
            _args: argparse.Namespace,
            _session: benchmark_whisper.BackendSession,
            _load_seconds: float | None,
        ) -> benchmark_whisper.RunResult:
            run_calls.append((backend, model))
            return benchmark_whisper.build_run_result(
                backend=backend,
                model_name=model,
                run_index=1,
                load_seconds=None,
                transcribe_seconds=0.1,
                transcript="ok",
                detected_language="ru",
                detected_language_probability=None,
                reference_transcript=None,
                audio_path=Path("audio.mp3"),
                sample_label="audio",
                audio_duration_seconds=1.0,
                forced_language="ru",
            )

        def fake_qwen_run(
            _audio_path: Path,
            model: str,
            _run_index: int,
            _args: argparse.Namespace,
            _model_path: Path,
        ) -> benchmark_whisper.RunResult:
            run_calls.append(("qwen3-asr", model))
            return benchmark_whisper.build_run_result(
                backend="qwen3-asr",
                model_name=model,
                run_index=1,
                load_seconds=0.2,
                transcribe_seconds=0.1,
                transcript="ok",
                detected_language="Russian",
                detected_language_probability=None,
                reference_transcript=None,
                audio_path=Path("audio.mp3"),
                sample_label="audio",
                audio_duration_seconds=1.0,
                forced_language="ru",
            )

        def fake_gigaam_multilingual_run(
            audio_path: Path,
            model: str,
            run_index: int,
            _args: argparse.Namespace,
            _model_path: Path,
        ) -> benchmark_whisper.RunResult:
            run_calls.append(("gigaam-multilingual", model))
            return benchmark_whisper.build_run_result(
                backend="gigaam-multilingual",
                model_name=model,
                run_index=run_index,
                load_seconds=0.2,
                transcribe_seconds=0.1,
                transcript="ok",
                detected_language="ru",
                detected_language_probability=None,
                reference_transcript=None,
                audio_path=audio_path,
                sample_label=_args.sample_label,
                audio_duration_seconds=_args.audio_duration_seconds,
                forced_language=_args.language,
            )

        written_payload: dict[str, object] = {}
        with (
            mock.patch.object(benchmark_whisper, "parse_args", return_value=args),
            mock.patch.object(
                benchmark_whisper,
                "resolve_output_paths",
                return_value=args.output,
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_audio_inputs",
                return_value=[audio_input],
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_backend_session",
                side_effect=fake_load,
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_qwen3_asr_model_path",
                return_value=Path("/tmp/qwen3-asr-model"),
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_gigaam_multilingual_model_path",
                return_value=Path("/tmp/gigaam-multilingual-model"),
            ),
            mock.patch.object(benchmark_whisper, "maybe_warmup"),
            mock.patch.object(
                benchmark_whisper,
                "run_single_backend",
                side_effect=fake_run,
            ),
            mock.patch.object(
                benchmark_whisper,
                "run_qwen3_asr",
                side_effect=fake_qwen_run,
            ),
            mock.patch.object(
                benchmark_whisper,
                "run_gigaam_multilingual",
                side_effect=fake_gigaam_multilingual_run,
            ),
            mock.patch.object(
                benchmark_whisper,
                "write_json",
                side_effect=lambda _path, payload: written_payload.update(payload),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            exit_code = benchmark_whisper.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            load_calls,
            [
                ("mlx-whisper", "large-v3-turbo"),
                ("gigaam", "e2e_rnnt"),
                ("t-one", "t-one-greedy"),
                ("vosk", "vosk-ru"),
            ],
        )
        self.assertEqual(
            run_calls,
            [
                ("mlx-whisper", "large-v3-turbo"),
                ("gigaam", "e2e_rnnt"),
                ("gigaam-multilingual", "gigaam-multilingual-large-ctc"),
                ("t-one", "t-one-greedy"),
                ("vosk", "vosk-ru"),
                ("qwen3-asr", "qwen3-asr-0.6b-8bit"),
            ],
        )
        self.assertEqual(written_payload["skipped"], [])
        self.assertIsNone(written_payload["metadata"]["qwen3_asr_language"])
        self.assertEqual(
            written_payload["metadata"]["qwen3_asr_effective_language_hint"],
            "ru",
        )
        self.assertEqual(
            written_payload["metadata"]["benchmark_pairs"],
            [
                ["mlx-whisper", "large-v3-turbo"],
                ["gigaam", "e2e_rnnt"],
                ["gigaam-multilingual", "gigaam-multilingual-large-ctc"],
                ["t-one", "t-one-greedy"],
                ["vosk", "vosk-ru"],
                ["qwen3-asr", "qwen3-asr-0.6b-8bit"],
            ],
        )

    def _run_main_with_stubbed_backends(
        self,
        args: argparse.Namespace,
        audio_inputs: list[benchmark_whisper.ResolvedAudioInput],
    ) -> tuple[
        dict[str, object],
        list[tuple[str, str]],
        list[tuple[str, str]],
        mock.Mock,
    ]:
        args.output = Path("/tmp/test-output.json")
        args.runs = 1
        load_calls: list[tuple[str, str]] = []
        run_calls: list[tuple[str, str]] = []

        def fake_load(backend: str, model: str, _args: argparse.Namespace):
            load_calls.append((backend, model))
            session: object = object()
            if backend in {"gigaam", "t-one", "vosk"}:
                session = {"model_path": Path(f"/tmp/{backend}-model")}
            return benchmark_whisper.BackendSession(
                backend=backend,
                model=model,
                device="subprocess"
                if backend in {"gigaam", "t-one", "vosk"}
                else "mlx",
                session=session,
                load_seconds=None,
            )

        def fake_run(
            backend: str,
            audio_path: Path,
            model: str,
            run_index: int,
            _args: argparse.Namespace,
            _session: benchmark_whisper.BackendSession,
            _load_seconds: float | None,
        ) -> benchmark_whisper.RunResult:
            run_calls.append((backend, model))
            return benchmark_whisper.build_run_result(
                backend=backend,
                model_name=model,
                run_index=run_index,
                load_seconds=None,
                transcribe_seconds=0.1,
                transcript="ok",
                detected_language=_args.language,
                detected_language_probability=None,
                reference_transcript=_args.reference_transcript_text,
                audio_path=audio_path,
                sample_label=_args.sample_label,
                audio_duration_seconds=_args.audio_duration_seconds,
                forced_language=_args.language,
            )

        def fake_qwen_run(
            audio_path: Path,
            model: str,
            run_index: int,
            _args: argparse.Namespace,
            _model_path: Path,
        ) -> benchmark_whisper.RunResult:
            run_calls.append(("qwen3-asr", model))
            return benchmark_whisper.build_run_result(
                backend="qwen3-asr",
                model_name=model,
                run_index=run_index,
                load_seconds=0.2,
                transcribe_seconds=0.1,
                transcript="ok",
                detected_language=None,
                detected_language_probability=None,
                reference_transcript=getattr(_args, "reference_transcript_text", None),
                audio_path=audio_path,
                sample_label=_args.sample_label,
                audio_duration_seconds=_args.audio_duration_seconds,
                forced_language=_args.language,
            )

        def fake_gigaam_multilingual_run(
            audio_path: Path,
            model: str,
            run_index: int,
            _args: argparse.Namespace,
            _model_path: Path,
        ) -> benchmark_whisper.RunResult:
            run_calls.append(("gigaam-multilingual", model))
            return benchmark_whisper.build_run_result(
                backend="gigaam-multilingual",
                model_name=model,
                run_index=run_index,
                load_seconds=0.2,
                transcribe_seconds=0.1,
                transcript="ok",
                detected_language=_args.language,
                detected_language_probability=None,
                reference_transcript=getattr(_args, "reference_transcript_text", None),
                audio_path=audio_path,
                sample_label=_args.sample_label,
                audio_duration_seconds=_args.audio_duration_seconds,
                forced_language=_args.language,
            )

        written_payload: dict[str, object] = {}
        with (
            mock.patch.object(benchmark_whisper, "parse_args", return_value=args),
            mock.patch.object(
                benchmark_whisper,
                "resolve_output_paths",
                return_value=args.output,
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_audio_inputs",
                return_value=audio_inputs,
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_backend_session",
                side_effect=fake_load,
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_qwen3_asr_model_path",
                return_value=Path("/tmp/qwen3-asr-model"),
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_gigaam_multilingual_model_path",
                return_value=Path("/tmp/gigaam-multilingual-model"),
            ),
            mock.patch.object(benchmark_whisper, "maybe_warmup"),
            mock.patch.object(
                benchmark_whisper,
                "run_single_backend",
                side_effect=fake_run,
            ),
            mock.patch.object(
                benchmark_whisper,
                "run_qwen3_asr",
                side_effect=fake_qwen_run,
            ),
            mock.patch.object(
                benchmark_whisper,
                "run_gigaam_multilingual",
                side_effect=fake_gigaam_multilingual_run,
            ),
            mock.patch.object(
                benchmark_whisper,
                "run_json_worker",
            ) as run_worker,
            mock.patch.object(
                benchmark_whisper,
                "write_json",
                side_effect=lambda _path, payload: written_payload.update(payload),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            exit_code = benchmark_whisper.main()

        self.assertEqual(exit_code, 0)
        return written_payload, load_calls, run_calls, run_worker

    def test_main_english_override_runs_multilingual_and_skips_ru_only_backends(
        self,
    ) -> None:
        args = benchmark_whisper.parse_args(["--audio", "en"])
        audio_input = benchmark_whisper.ResolvedAudioInput(
            audio_path=Path("audio.mp3"),
            reference_transcript_path=None,
            reference_transcript_text=None,
            forced_language="en",
            selector_language="en",
            sample_label="audio",
            source="default-language",
            audio_duration_seconds=1.0,
        )

        payload, load_calls, run_calls, run_worker = (
            self._run_main_with_stubbed_backends(args, [audio_input])
        )

        self.assertEqual(load_calls, [("mlx-whisper", "large-v3-turbo")])
        self.assertEqual(
            run_calls,
            [
                ("mlx-whisper", "large-v3-turbo"),
                ("gigaam-multilingual", "gigaam-multilingual-large-ctc"),
                ("qwen3-asr", "qwen3-asr-0.6b-8bit"),
            ],
        )
        self.assertEqual(
            payload["skipped"],
            [
                {
                    "audio": "audio.mp3",
                    "sample_label": "audio",
                    "forced_language": "en",
                    "backend": "gigaam",
                    "model": "e2e_rnnt",
                    "reason": "ru-only model",
                },
                {
                    "audio": "audio.mp3",
                    "sample_label": "audio",
                    "forced_language": "en",
                    "backend": "t-one",
                    "model": "t-one-greedy",
                    "reason": "ru-only model",
                },
                {
                    "audio": "audio.mp3",
                    "sample_label": "audio",
                    "forced_language": "en",
                    "backend": "vosk",
                    "model": "vosk-ru",
                    "reason": "ru-only model",
                },
            ],
        )
        self.assertEqual(
            [run["backend"] for run in payload["runs"]],
            ["mlx-whisper", "gigaam-multilingual", "qwen3-asr"],
        )
        run_worker.assert_not_called()

    def test_main_auto_skips_ru_only_backends_for_all_resolved_inputs(self) -> None:
        args = benchmark_whisper.parse_args(["--audio", "auto"])
        audio_inputs = [
            benchmark_whisper.ResolvedAudioInput(
                audio_path=Path("en.mp3"),
                reference_transcript_path=None,
                reference_transcript_text=None,
                forced_language=None,
                selector_language="en",
                sample_label="en",
                source="default-auto",
                audio_duration_seconds=1.0,
            ),
            benchmark_whisper.ResolvedAudioInput(
                audio_path=Path("ru.mp3"),
                reference_transcript_path=None,
                reference_transcript_text=None,
                forced_language=None,
                selector_language="ru",
                sample_label="ru",
                source="default-auto",
                audio_duration_seconds=1.0,
            ),
        ]

        payload, load_calls, run_calls, run_worker = (
            self._run_main_with_stubbed_backends(args, audio_inputs)
        )

        self.assertEqual(
            load_calls,
            [
                ("mlx-whisper", "large-v3-turbo"),
                ("mlx-whisper", "large-v3-turbo"),
            ],
        )
        self.assertEqual(
            run_calls,
            [
                ("mlx-whisper", "large-v3-turbo"),
                ("gigaam-multilingual", "gigaam-multilingual-large-ctc"),
                ("qwen3-asr", "qwen3-asr-0.6b-8bit"),
                ("mlx-whisper", "large-v3-turbo"),
                ("gigaam-multilingual", "gigaam-multilingual-large-ctc"),
                ("qwen3-asr", "qwen3-asr-0.6b-8bit"),
            ],
        )
        self.assertEqual(len(payload["skipped"]), 6)
        self.assertTrue(
            all(
                item["backend"] in {"gigaam", "t-one", "vosk"}
                and item["reason"] == "ru-only model"
                and item["forced_language"] is None
                for item in payload["skipped"]
            )
        )
        self.assertNotIn("gigaam", [run["backend"] for run in payload["runs"]])
        self.assertNotIn("t-one", [run["backend"] for run in payload["runs"]])
        self.assertNotIn("vosk", [run["backend"] for run in payload["runs"]])
        self.assertEqual(
            [run["backend"] for run in payload["runs"]],
            [
                "mlx-whisper",
                "gigaam-multilingual",
                "qwen3-asr",
                "mlx-whisper",
                "gigaam-multilingual",
                "qwen3-asr",
            ],
        )
        run_worker.assert_not_called()

    def test_resolve_gigaam_model_path_uses_pinned_local_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--ai-sage--GigaAM-v3"
            revision = "a" * 40
            (cache_root / "refs").mkdir(parents=True)
            (cache_root / "snapshots" / revision).mkdir(parents=True)
            (cache_root / "refs" / "e2e_rnnt").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            for relative_path in benchmark_whisper.GIGAAM_REQUIRED_MODEL_FILES:
                (cache_root / "snapshots" / revision / relative_path).write_bytes(
                    b"model"
                )

            with mock.patch.object(
                benchmark_whisper, "GIGAAM_MODEL_CACHE_ROOT", cache_root
            ):
                resolved = benchmark_whisper.resolve_gigaam_model_path()

        self.assertEqual(resolved, (cache_root / "snapshots" / revision).resolve())

    def test_resolve_gigaam_model_path_selects_each_variant_ref_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--ai-sage--GigaAM-v3"
            revisions = {"e2e_rnnt": "a" * 40, "e2e_ctc": "b" * 40}
            (cache_root / "refs").mkdir(parents=True)
            for variant, revision in revisions.items():
                snapshot = cache_root / "snapshots" / revision
                snapshot.mkdir(parents=True)
                (cache_root / "refs" / variant).write_text(
                    f"{revision}\n", encoding="utf-8"
                )
                for relative_path in benchmark_whisper.GIGAAM_REQUIRED_MODEL_FILES:
                    (snapshot / relative_path).write_bytes(b"model")

            with mock.patch.object(
                benchmark_whisper, "GIGAAM_MODEL_CACHE_ROOT", cache_root
            ):
                rnnt = benchmark_whisper.resolve_gigaam_model_path("e2e_rnnt")
                ctc = benchmark_whisper.resolve_gigaam_model_path("e2e_ctc")

        self.assertEqual(rnnt, (cache_root / "snapshots" / revisions["e2e_rnnt"]).resolve())
        self.assertEqual(ctc, (cache_root / "snapshots" / revisions["e2e_ctc"]).resolve())

    def test_resolve_gigaam_model_path_requires_selected_variant_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            for relative_path in benchmark_whisper.GIGAAM_REQUIRED_MODEL_FILES[:-1]:
                (model_path / relative_path).write_bytes(b"model")

            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_gigaam_model_path("e2e_ctc", model_path)

    def test_run_gigaam_maps_success_and_scores_transcript(self) -> None:
        args = argparse.Namespace(
            language="ru",
            reference_transcript_text="привет мир",
            sample_label="audio",
            audio_duration_seconds=3.0,
            gigaam_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "Привет, мир!",
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
                "language": "ru",
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=512.5,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_gigaam(
                audio_path=Path("audio.mp3"),
                model_name="e2e_rnnt",
                run_index=2,
                args=args,
                model_path=Path("/models/gigaam"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "gigaam")
        self.assertEqual(result.model, "e2e_rnnt")
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.run_index, 2)
        self.assertEqual(result.transcript, "Привет, мир!")
        self.assertEqual(result.load_seconds, 1.25)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 512.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            "stt_benchmark.workers.gigaam",
            {
                "model_path": "/models/gigaam",
                "audio_path": "audio.mp3",
                "language": "ru",
                "variant": "e2e_rnnt",
            },
            12.5,
            env=benchmark_whisper.GIGAAM_OFFLINE_ENV,
        )

    def test_run_gigaam_requests_selected_ctc_variant(self) -> None:
        args = argparse.Namespace(
            language="ru",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
            gigaam_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "text",
                "transcribe_seconds": 0.5,
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=0.75,
            peak_rss_mb=None,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_gigaam(
                Path("audio.mp3"),
                "e2e_ctc",
                1,
                args,
                Path("/models/gigaam-ctc"),
            )

        self.assertEqual(result.status, "ok")
        request = run_worker.call_args.args[2]
        self.assertEqual(request["variant"], "e2e_ctc")
        self.assertEqual(request["model_path"], "/models/gigaam-ctc")

    def test_run_gigaam_maps_timeout_and_worker_errors(self) -> None:
        args = argparse.Namespace(
            language="ru",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
        )
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=None,
                error="worker timed out after 1.0 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "load failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper,
            "run_json_worker",
            side_effect=executions,
        ):
            timeout_result = benchmark_whisper.run_gigaam(
                Path("audio.mp3"),
                "e2e_rnnt",
                1,
                args,
                Path("/models/gigaam"),
            )
            worker_error_result = benchmark_whisper.run_gigaam(
                Path("audio.mp3"),
                "e2e_rnnt",
                2,
                args,
                Path("/models/gigaam"),
            )

        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("load failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

    def test_run_gigaam_multilingual_uses_ru_en_and_omits_auto_language(self) -> None:
        args = argparse.Namespace(
            language="ru",
            gigaam_multilingual_language=None,
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
            gigaam_multilingual_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "text",
                "transcribe_seconds": 0.5,
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=0.75,
            peak_rss_mb=None,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            benchmark_whisper.run_gigaam_multilingual(
                Path("audio.mp3"),
                "gigaam-multilingual-large-ctc",
                1,
                args,
                Path("/models/gigaam-multilingual"),
            )
            args.language = "en"
            benchmark_whisper.run_gigaam_multilingual(
                Path("audio.mp3"),
                "gigaam-multilingual-large-ctc",
                2,
                args,
                Path("/models/gigaam-multilingual"),
            )
            args.language = "auto"
            benchmark_whisper.run_gigaam_multilingual(
                Path("audio.mp3"),
                "gigaam-multilingual-large-ctc",
                3,
                args,
                Path("/models/gigaam-multilingual"),
            )

        requests = [call.args[2] for call in run_worker.call_args_list]
        self.assertEqual(requests[0]["language"], "ru")
        self.assertEqual(requests[1]["language"], "en")
        self.assertNotIn("language", requests[2])

    def test_run_gigaam_multilingual_maps_success_timing_scores_and_rss(self) -> None:
        args = argparse.Namespace(
            language="ru",
            gigaam_multilingual_language=None,
            reference_transcript_text="привет мир",
            sample_label="audio",
            audio_duration_seconds=3.0,
            gigaam_multilingual_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "Привет, мир!",
                "timestamps": [{"text": "Привет", "start": 0.0, "end": 0.5}],
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
                "language": "ru",
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=512.5,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_gigaam_multilingual(
                Path("audio.mp3"),
                "gigaam-multilingual-large-ctc",
                2,
                args,
                Path("/models/gigaam-multilingual"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "gigaam-multilingual")
        self.assertEqual(result.model, "gigaam-multilingual-large-ctc")
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.transcript, "Привет, мир!")
        self.assertEqual(result.load_seconds, 1.25)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 512.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            "stt_benchmark.workers.gigaam_multilingual",
            {
                "model_path": "/models/gigaam-multilingual",
                "audio_path": "audio.mp3",
                "variant": "large_ctc",
                "language": "ru",
            },
            12.5,
            env=benchmark_whisper.GIGAAM_MULTILINGUAL_OFFLINE_ENV,
        )

    def test_run_gigaam_multilingual_maps_timeout_and_worker_errors_with_rss(
        self,
    ) -> None:
        args = argparse.Namespace(
            language="en",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
        )
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=64.0,
                error="worker timed out after 12.5 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "load failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", side_effect=executions
        ):
            timeout_result = benchmark_whisper.run_gigaam_multilingual(
                Path("audio.mp3"),
                "gigaam-multilingual-large-ctc",
                1,
                args,
                Path("/models/gigaam-multilingual"),
            )
            worker_error_result = benchmark_whisper.run_gigaam_multilingual(
                Path("audio.mp3"),
                "gigaam-multilingual-large-ctc",
                2,
                args,
                Path("/models/gigaam-multilingual"),
            )

        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(timeout_result.peak_rss_mb, 64.0)
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("load failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

    def test_tone_options_use_cli_values_over_environment(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "TONE_PYTHON": "/env/bin/python",
                "TONE_MODEL_PATH": "/env/model",
            },
            clear=False,
        ):
            args = benchmark_whisper.parse_args(
                [
                    "--tone-python",
                    "/cli/bin/python",
                    "--tone-model-path",
                    "/cli/model",
                    "--tone-decoder",
                    "beam",
                ]
            )

        self.assertEqual(args.tone_python, Path("/cli/bin/python"))
        self.assertEqual(args.tone_model_path, Path("/cli/model"))
        self.assertEqual(args.tone_decoder, "beam")

    def test_tone_options_use_environment_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "TONE_PYTHON": "/env/bin/python",
                "TONE_MODEL_PATH": "/env/model",
            },
            clear=False,
        ):
            args = benchmark_whisper.parse_args([])

        self.assertEqual(args.tone_python, Path("/env/bin/python"))
        self.assertEqual(args.tone_model_path, Path("/env/model"))
        self.assertEqual(args.tone_decoder, "greedy")

    def test_resolve_tone_model_path_uses_pinned_snapshot_and_model_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--t-tech--T-one"
            revision = "b" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (cache_root / "refs" / "main").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            (snapshot / "model.onnx").write_bytes(b"model")

            with mock.patch.object(
                benchmark_whisper, "TONE_MODEL_CACHE_ROOT", cache_root
            ):
                resolved = benchmark_whisper.resolve_tone_model_path()

        self.assertEqual(resolved, snapshot.resolve())

    def test_resolve_tone_model_path_requires_model_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()

            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_tone_model_path(model_path)

    def test_load_tone_session_resolves_and_stores_local_model_path(self) -> None:
        model_path = Path("/models/t-one")
        args = argparse.Namespace(tone_model_path=None)
        with mock.patch.object(
            benchmark_whisper, "resolve_tone_model_path", return_value=model_path
        ) as resolve:
            session = benchmark_whisper.load_backend_session(
                "t-one", "t-one-greedy", args
            )

        resolve.assert_called_once_with(None)
        self.assertEqual(session.backend, "t-one")
        self.assertEqual(session.model, "t-one-greedy")
        self.assertEqual(session.device, "subprocess")
        self.assertEqual(session.session, {"model_path": model_path})
        self.assertIsNone(session.load_seconds)

    def test_run_tone_maps_success_timestamps_and_peak_rss(self) -> None:
        args = argparse.Namespace(
            language="ru",
            reference_transcript_text="привет мир",
            sample_label="audio",
            audio_duration_seconds=3.0,
            tone_python=Path("/venv/bin/python"),
            tone_decoder="greedy",
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "Привет, мир!",
                "timestamps": [
                    {"text": "Привет", "start_time": 0.0, "end_time": 0.5},
                ],
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=256.5,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_tone(
                Path("audio.mp3"),
                "t-one-greedy",
                2,
                args,
                Path("/models/t-one"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "t-one")
        self.assertEqual(result.model, "t-one-greedy")
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.transcript, "Привет, мир!")
        self.assertEqual(result.load_seconds, 1.25)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 256.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            "stt_benchmark.workers.tone",
            {
                "model_path": "/models/t-one",
                "audio_path": "audio.mp3",
                "decoder": "greedy",
                "streaming": False,
            },
            12.5,
            env=benchmark_whisper.TONE_OFFLINE_ENV,
        )

    def test_run_tone_maps_timeout_and_worker_errors(self) -> None:
        args = argparse.Namespace(
            language="ru",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
        )
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=None,
                error="worker timed out after 1.0 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "load failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", side_effect=executions
        ):
            timeout_result = benchmark_whisper.run_tone(
                Path("audio.mp3"),
                "t-one-greedy",
                1,
                args,
                Path("/models/t-one"),
            )
            worker_error_result = benchmark_whisper.run_tone(
                Path("audio.mp3"),
                "t-one-greedy",
                2,
                args,
                Path("/models/t-one"),
            )

        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("load failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

    def test_vosk_options_use_cli_values_over_environment(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "VOSK_PYTHON": "/env/bin/python",
                "VOSK_MODEL_PATH": "/env/model",
            },
            clear=False,
        ):
            args = benchmark_whisper.parse_args(
                [
                    "--vosk-python",
                    "/cli/bin/python",
                    "--vosk-model-path",
                    "/cli/model",
                    "--vosk-decoding-method",
                    "greedy_search",
                    "--worker-timeout-seconds",
                    "12.5",
                ]
            )

        self.assertEqual(args.vosk_python, Path("/cli/bin/python"))
        self.assertEqual(args.vosk_model_path, Path("/cli/model"))
        self.assertEqual(args.vosk_decoding_method, "greedy_search")
        self.assertEqual(args.worker_timeout_seconds, 12.5)

    def test_vosk_options_use_environment_and_builtin_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "VOSK_PYTHON": "/env/bin/python",
                "VOSK_MODEL_PATH": "/env/model",
            },
            clear=False,
        ):
            args = benchmark_whisper.parse_args([])

        self.assertEqual(args.vosk_python, Path("/env/bin/python"))
        self.assertEqual(args.vosk_model_path, Path("/env/model"))
        self.assertEqual(
            args.vosk_decoding_method, benchmark_whisper.VOSK_DEFAULT_DECODING_METHOD
        )

        with mock.patch.dict("os.environ", {}, clear=True):
            fallback_args = benchmark_whisper.parse_args([])

        self.assertEqual(fallback_args.vosk_python, Path(".venvs/vosk/bin/python"))
        self.assertIsNone(fallback_args.vosk_model_path)
        self.assertEqual(
            fallback_args.vosk_decoding_method,
            benchmark_whisper.VOSK_DEFAULT_DECODING_METHOD,
        )

    def test_resolve_vosk_model_path_uses_pinned_snapshot_and_required_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--alphacep--vosk-model-ru"
            revision = "c" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            (snapshot / "am-onnx").mkdir(parents=True)
            (snapshot / "lang").mkdir()
            (cache_root / "refs" / "main").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            for relative_path in benchmark_whisper.VOSK_REQUIRED_MODEL_FILES:
                (snapshot / relative_path).write_bytes(b"model")

            with mock.patch.object(
                benchmark_whisper, "VOSK_MODEL_CACHE_ROOT", cache_root
            ):
                resolved = benchmark_whisper.resolve_vosk_model_path()

        self.assertEqual(resolved, snapshot.resolve())

    def test_resolve_vosk_model_path_selects_cache_root_and_small_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--alphacep--vosk-model-small-ru"
            revision = "d" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            (snapshot / "am").mkdir(parents=True)
            (snapshot / "lang").mkdir()
            (cache_root / "refs" / "main").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            for relative_path in benchmark_whisper.VOSK_REQUIRED_MODEL_FILES_BY_VARIANT[
                "vosk-small-ru"
            ]:
                (snapshot / relative_path).write_bytes(b"model")

            with mock.patch.object(
                benchmark_whisper,
                "VOSK_MODEL_CACHE_ROOTS",
                {
                    "vosk-ru": Path(tmpdir) / "unused-full",
                    "vosk-small-ru": cache_root,
                },
            ):
                resolved = benchmark_whisper.resolve_vosk_model_path("vosk-small-ru")

        self.assertEqual(resolved, snapshot.resolve())

    def test_resolve_vosk_model_path_requires_selected_small_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            (model_path / "am-onnx").mkdir(parents=True)
            (model_path / "lang").mkdir()
            for relative_path in benchmark_whisper.VOSK_REQUIRED_MODEL_FILES:
                (model_path / relative_path).write_bytes(b"model")

            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_vosk_model_path("vosk-small-ru", model_path)

    def test_resolve_vosk_model_path_requires_big_fp32_files_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            (model_path / "am-onnx").mkdir(parents=True)
            (model_path / "lang").mkdir()

            for relative_path in benchmark_whisper.VOSK_REQUIRED_MODEL_FILES[:-1]:
                (model_path / relative_path).write_bytes(b"model")

            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_vosk_model_path(model_path)

            (model_path / "lang" / "tokens.txt").write_bytes(b"tokens")
            (model_path / "am-onnx" / "encoder.onnx").unlink()
            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_vosk_model_path(model_path)

    def test_load_vosk_session_resolves_and_stores_local_model_path(self) -> None:
        model_path = Path("/models/vosk")
        args = argparse.Namespace(vosk_model_path=None)
        with mock.patch.object(
            benchmark_whisper, "resolve_vosk_model_path", return_value=model_path
        ) as resolve:
            session = benchmark_whisper.load_backend_session(
                "vosk", "vosk-ru", args
            )

        resolve.assert_called_once_with("vosk-ru", None)
        self.assertEqual(session.backend, "vosk")
        self.assertEqual(session.model, "vosk-ru")
        self.assertEqual(session.device, "subprocess")
        self.assertEqual(session.session, {"model_path": model_path})
        self.assertIsNone(session.load_seconds)

    def test_run_vosk_maps_success_ignores_timestamps_and_peak_rss(self) -> None:
        args = argparse.Namespace(
            language="ru",
            reference_transcript_text="привет мир",
            sample_label="audio",
            audio_duration_seconds=3.0,
            vosk_python=Path("/venv/bin/python"),
            vosk_decoding_method="greedy_search",
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "Привет, мир!",
                "timestamps": [
                    {"text": "Привет", "start_time": 0.0, "end_time": 0.5},
                ],
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
                "language": "ru",
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=128.5,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_vosk(
                Path("audio.mp3"),
                "vosk-ru",
                2,
                args,
                Path("/models/vosk"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "vosk")
        self.assertEqual(result.model, "vosk-ru")
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.transcript, "Привет, мир!")
        self.assertEqual(result.load_seconds, 1.25)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 128.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            "stt_benchmark.workers.vosk",
            {
                "model_path": "/models/vosk",
                "audio_path": "audio.mp3",
                "decoding_method": "greedy_search",
                "quantization": "fp32",
                "streaming": False,
            },
            12.5,
            env=benchmark_whisper.VOSK_OFFLINE_ENV,
        )

    def test_run_vosk_requests_selected_small_model_path(self) -> None:
        args = argparse.Namespace(
            language="ru",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
            vosk_python=Path("/venv/bin/python"),
            vosk_decoding_method="greedy_search",
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "text",
                "transcribe_seconds": 0.5,
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=0.75,
            peak_rss_mb=None,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_vosk(
                Path("audio.mp3"),
                "vosk-small-ru",
                1,
                args,
                Path("/models/vosk-small-ru"),
            )

        self.assertEqual(result.status, "ok")
        request = run_worker.call_args.args[2]
        self.assertEqual(request["model_path"], "/models/vosk-small-ru")

    def test_run_vosk_maps_timeout_and_worker_errors(self) -> None:
        args = argparse.Namespace(
            language="ru",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
        )
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=None,
                error="worker timed out after 1.0 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "load failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", side_effect=executions
        ):
            timeout_result = benchmark_whisper.run_vosk(
                Path("audio.mp3"),
                "vosk-ru",
                1,
                args,
                Path("/models/vosk"),
            )
            worker_error_result = benchmark_whisper.run_vosk(
                Path("audio.mp3"),
                "vosk-ru",
                2,
                args,
                Path("/models/vosk"),
            )

        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("load failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

    def test_boolean_optional_flags_parse(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "benchmark_whisper.py",
                "--audio",
                "en",
                "--no-faster-whisper-vad-filter",
                "--no-condition-on-previous-text",
                "--no-openai-whisper-temperature-fallback",
            ],
        ):
            args = benchmark_whisper.parse_args()
        self.assertFalse(args.faster_whisper_vad_filter)
        self.assertFalse(args.condition_on_previous_text)
        self.assertFalse(args.openai_whisper_temperature_fallback)

    def test_benchmark_parse_args_accepts_explicit_argv(self) -> None:
        args = benchmark_whisper.parse_args(
            [
                "--audio",
                "en",
                "--no-faster-whisper-vad-filter",
                "--no-condition-on-previous-text",
                "--no-openai-whisper-temperature-fallback",
            ]
        )
        self.assertFalse(args.faster_whisper_vad_filter)
        self.assertFalse(args.condition_on_previous_text)
        self.assertFalse(args.openai_whisper_temperature_fallback)

    def test_show_full_table_flag_parses(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "benchmark_whisper.py",
                "--audio",
                "en",
                "--show-full-table",
            ],
        ):
            args = benchmark_whisper.parse_args()
        self.assertTrue(args.show_full_table)

    def test_resolve_audio_inputs_defaults_to_all_bundled_samples(self) -> None:
        args = argparse.Namespace(audios=[])
        with (
            mock.patch.object(
                benchmark_whisper,
                "ensure_audio_file",
                side_effect=lambda path: path.resolve(),
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_reference_transcript",
                side_effect=lambda path: f"normalized:{path.name}",
            ),
            mock.patch.object(
                benchmark_whisper,
                "get_audio_duration_seconds",
                side_effect=lambda _path: 10.0,
            ),
        ):
            resolved = benchmark_whisper.resolve_audio_inputs(args)

        self.assertEqual([item.selector_language for item in resolved], ["en", "ru"])
        self.assertEqual([item.forced_language for item in resolved], ["en", "ru"])

    def test_resolve_audio_inputs_auto_applies_to_all_defaults(self) -> None:
        args = argparse.Namespace(audios=["auto"])
        with (
            mock.patch.object(
                benchmark_whisper,
                "ensure_audio_file",
                side_effect=lambda path: path.resolve(),
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_reference_transcript",
                side_effect=lambda path: f"normalized:{path.name}",
            ),
            mock.patch.object(
                benchmark_whisper,
                "get_audio_duration_seconds",
                side_effect=lambda _path: 10.0,
            ),
        ):
            resolved = benchmark_whisper.resolve_audio_inputs(args)

        self.assertEqual([item.forced_language for item in resolved], [None, None])

    def test_resolve_audio_inputs_specific_forced_language_beats_auto(self) -> None:
        args = argparse.Namespace(audios=["auto", "ru"])
        with (
            mock.patch.object(
                benchmark_whisper,
                "ensure_audio_file",
                side_effect=lambda path: path.resolve(),
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_reference_transcript",
                side_effect=lambda path: f"normalized:{path.name}",
            ),
            mock.patch.object(
                benchmark_whisper,
                "get_audio_duration_seconds",
                side_effect=lambda _path: 10.0,
            ),
        ):
            resolved = benchmark_whisper.resolve_audio_inputs(args)

        by_language = {item.selector_language: item for item in resolved}
        self.assertIsNone(by_language["en"].forced_language)
        self.assertEqual(by_language["ru"].forced_language, "ru")

    def test_main_skips_unsupported_backend_model_combo(self) -> None:
        output_path = Path("/tmp/test-output.json")
        written_payload: dict[str, object] = {}
        fake_args = argparse.Namespace(
            audios=["en"],
            models=["large-v3-turbo"],
            backends=["lightning-whisper-mlx"],
            runs=1,
            task="transcribe",
            beam_size=5,
            compute_type="default",
            faster_whisper_vad_filter=True,
            condition_on_previous_text=True,
            openai_whisper_temperature_fallback=True,
            hallucination_silence_threshold=2.0,
            device="auto",
            output=output_path,
            warmup=False,
            insanely_fast_whisper_device_id="mps",
            insanely_fast_whisper_batch_size=1,
            insanely_fast_whisper_flash=False,
            lightning_whisper_mlx_batch_size=12,
            show_full_table=False,
        )

        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            mock.patch.object(benchmark_whisper, "parse_args", return_value=fake_args),
            mock.patch.object(
                benchmark_whisper,
                "resolve_output_paths",
                return_value=output_path,
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_audio_inputs",
                return_value=[
                    benchmark_whisper.ResolvedAudioInput(
                        audio_path=Path("audio.mp3"),
                        reference_transcript_path=None,
                        reference_transcript_text=None,
                        forced_language="en",
                        selector_language="en",
                        sample_label="audio",
                        source="default-language",
                        audio_duration_seconds=1.0,
                    )
                ],
            ),
            mock.patch.object(
                benchmark_whisper,
                "BACKEND_CAPABILITIES",
                {
                    "lightning-whisper-mlx": benchmark_whisper.BackendCapabilities(
                        supported_models={"tiny"},
                        supports_hallucination_silence_threshold=True,
                    )
                },
            ),
            mock.patch.object(
                benchmark_whisper,
                "write_json",
                side_effect=lambda _path, payload: written_payload.update(payload),
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = benchmark_whisper.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            list(written_payload.keys()),
            ["metadata", "skipped", "summary", "runs"],
        )
        self.assertIn(
            "Skipping lightning-whisper-mlx on sample audio model large-v3-turbo (not supported).",
            stderr.getvalue(),
        )
        self.assertEqual(
            written_payload["skipped"],
            [
                {
                    "audio": "audio.mp3",
                    "sample_label": "audio",
                    "forced_language": "en",
                    "backend": "lightning-whisper-mlx",
                    "model": "large-v3-turbo",
                    "reason": "not supported",
                }
            ],
        )
        self.assertEqual(written_payload["summary"], [])
        self.assertEqual(written_payload["runs"], [])
        self.assertIn("Skipped:", stdout.getvalue())
        self.assertIn(
            "lightning-whisper-mlx large-v3-turbo: not supported",
            stdout.getvalue(),
        )

    def test_main_prints_runs_table_before_summary_when_enabled(self) -> None:
        output_path = Path("/tmp/test-output.json")
        fake_args = argparse.Namespace(
            audios=["en"],
            models=["tiny"],
            backends=["mlx-whisper"],
            runs=1,
            task="transcribe",
            beam_size=5,
            compute_type="default",
            faster_whisper_vad_filter=True,
            condition_on_previous_text=True,
            openai_whisper_temperature_fallback=True,
            hallucination_silence_threshold=2.0,
            device="auto",
            output=output_path,
            warmup=False,
            insanely_fast_whisper_device_id="mps",
            insanely_fast_whisper_batch_size=1,
            insanely_fast_whisper_flash=False,
            lightning_whisper_mlx_batch_size=12,
            show_full_table=True,
        )

        fake_result = benchmark_whisper.RunResult(
            audio="audio.mp3",
            sample_label="audio",
            audio_duration_seconds=10.0,
            forced_language="en",
            backend="mlx-whisper",
            model="tiny",
            backend_device="mlx",
            run_index=1,
            load_seconds=1.25,
            transcribe_seconds=2.5,
            total_seconds=3.75,
            transcript="hello world",
            transcript_chars=11,
            transcript_words=2,
            wer=None,
            cer=None,
            detected_language="en",
            detected_language_probability=None,
            status="ok",
            error=None,
            peak_rss_mb=None,
        )

        stdout = io.StringIO()
        with (
            mock.patch.object(benchmark_whisper, "parse_args", return_value=fake_args),
            mock.patch.object(
                benchmark_whisper,
                "resolve_output_paths",
                return_value=output_path,
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_audio_inputs",
                return_value=[
                    benchmark_whisper.ResolvedAudioInput(
                        audio_path=Path("audio.mp3"),
                        reference_transcript_path=None,
                        reference_transcript_text=None,
                        forced_language="en",
                        selector_language="en",
                        sample_label="audio",
                        source="default-language",
                        audio_duration_seconds=10.0,
                    )
                ],
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_backend_session",
                return_value=benchmark_whisper.BackendSession(
                    backend="mlx-whisper",
                    model="tiny",
                    device="mlx",
                    session=object(),
                    load_seconds=1.25,
                ),
            ),
            mock.patch.object(benchmark_whisper, "maybe_warmup"),
            mock.patch.object(
                benchmark_whisper, "run_single_backend", return_value=fake_result
            ),
            mock.patch.object(benchmark_whisper, "write_json"),
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = benchmark_whisper.main()

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("Runs:", output)
        self.assertIn("\nColumns:\n", output)
        self.assertLess(output.index("Runs:"), output.index("\nColumns:\n"))


class BackendInvocationTests(unittest.TestCase):
    def test_lightning_whisper_mlx_load_is_local_only(self) -> None:
        mlx_module = types.ModuleType("mlx")
        mlx_core_module = types.ModuleType("mlx.core")
        mlx_core_module.float16 = object()
        mlx_module.core = mlx_core_module

        transcribe_module = types.ModuleType("lightning_whisper_mlx.transcribe")
        model_holder = mock.Mock()
        transcribe_module.ModelHolder = model_holder
        lightning_module = types.ModuleType("lightning_whisper_mlx")
        lightning_module.transcribe = transcribe_module

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "mlx": mlx_module,
                    "mlx.core": mlx_core_module,
                    "lightning_whisper_mlx": lightning_module,
                    "lightning_whisper_mlx.transcribe": transcribe_module,
                },
            ),
            mock.patch.object(
                benchmark_whisper,
                "snapshot_download",
                return_value="/local/lightning-whisper",
            ) as snapshot_download,
        ):
            session = benchmark_whisper.load_backend_session(
                "lightning-whisper-mlx",
                "tiny",
                argparse.Namespace(),
            )

        snapshot_download.assert_called_once_with(
            repo_id=benchmark_whisper.LIGHTNING_WHISPER_MLX_REPOS["tiny"],
            allow_patterns=["config.json", "weights.npz"],
            local_files_only=True,
        )
        model_holder.get_model.assert_called_once_with(
            "/local/lightning-whisper",
            dtype=mlx_core_module.float16,
        )
        self.assertEqual(session.session["model_path"], "/local/lightning-whisper")

    def test_mlx_audio_redirects_output_path_and_keeps_transcript(self) -> None:
        args = argparse.Namespace(
            language="en",
            task="transcribe",
            beam_size=5,
            condition_on_previous_text=True,
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=10.0,
        )

        captured_kwargs: dict[str, object] = {}

        def fake_generate_transcription(**kwargs):
            captured_kwargs.update(kwargs)
            return {"text": "hello world", "language": "en"}

        with mock.patch(
            "mlx_audio.stt.generate.generate_transcription",
            side_effect=fake_generate_transcription,
        ):
            result = benchmark_whisper.run_mlx_audio(
                audio_path=Path("audio.mp3"),
                model_name="tiny",
                run_index=1,
                args=args,
                session=object(),
                load_seconds=0.5,
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.transcript, "hello world")
        self.assertIn("output_path", captured_kwargs)
        self.assertTrue(str(captured_kwargs["output_path"]).endswith("/transcript"))

    def test_mlx_whisper_does_not_pass_beam_size(self) -> None:
        args = argparse.Namespace(
            language="en",
            task="transcribe",
            beam_size=5,
            condition_on_previous_text=True,
            hallucination_silence_threshold=2.0,
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=10.0,
        )

        with mock.patch(
            "mlx_whisper.transcribe",
            return_value={"text": "hello", "language": "en"},
        ) as transcribe:
            result = benchmark_whisper.run_mlx_whisper(
                audio_path=Path("audio.mp3"),
                model_name="tiny",
                run_index=1,
                args=args,
                session={"model_repo": "mlx-community/whisper-tiny-mlx"},
                load_seconds=0.5,
            )

        self.assertEqual(result.status, "ok")
        _, kwargs = transcribe.call_args
        self.assertNotIn("beam_size", kwargs)

    def test_lightning_whisper_mlx_does_not_pass_beam_size(self) -> None:
        args = argparse.Namespace(
            language="en",
            task="transcribe",
            beam_size=5,
            condition_on_previous_text=True,
            lightning_whisper_mlx_batch_size=12,
            hallucination_silence_threshold=2.0,
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=10.0,
        )

        with mock.patch(
            "lightning_whisper_mlx.transcribe.transcribe_audio",
            return_value={"text": "hello", "language": "en"},
        ) as transcribe_audio:
            result = benchmark_whisper.run_lightning_whisper_mlx(
                audio_path=Path("audio.mp3"),
                model_name="tiny",
                run_index=1,
                args=args,
                session={"model_path": "mlx-community/whisper-tiny"},
                load_seconds=0.5,
            )

        self.assertEqual(result.status, "ok")
        _, kwargs = transcribe_audio.call_args
        self.assertNotIn("beam_size", kwargs)

    def test_insanely_fast_whisper_requests_timestamps_without_condition_on_prev_tokens(
        self,
    ) -> None:
        args = argparse.Namespace(
            language="en",
            task="transcribe",
            condition_on_previous_text=True,
            insanely_fast_whisper_batch_size=1,
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=10.0,
        )
        pipe_calls: list[dict[str, object]] = []

        def fake_pipe(audio, **kwargs):
            pipe_calls.append({"audio": audio, **kwargs})
            return {"text": "hello", "language": "en", "chunks": []}

        with mock.patch(
            "insanely_fast_whisper.utils.result.build_result",
            return_value={"text": "hello"},
        ):
            result = benchmark_whisper.run_insanely_fast_whisper(
                audio_path=Path("audio.mp3"),
                model_name="tiny",
                run_index=1,
                args=args,
                session={
                    "pipe": fake_pipe,
                    "generate_kwargs": {
                        "task": "transcribe",
                        "language": "en",
                    },
                },
                load_seconds=0.5,
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(pipe_calls), 1)
        self.assertEqual(pipe_calls[0]["audio"], "audio.mp3")
        self.assertEqual(pipe_calls[0]["return_timestamps"], True)
        self.assertEqual(pipe_calls[0]["return_language"], True)
        self.assertEqual(
            pipe_calls[0]["generate_kwargs"],
            {"task": "transcribe", "language": "en"},
        )


class Qwen3ASRHFBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _execution(
        payload: dict[str, object], *, rss: float | None = 512.5
    ) -> WorkerExecution:
        return WorkerExecution(
            status="ok",
            payload=payload,
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=rss,
            error=None,
        )

    def test_options_use_cli_environment_and_builtin_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "QWEN3_ASR_HF_PYTHON": "/env/python",
                "QWEN3_ASR_HF_MODEL_PATH": "/env/model",
                "QWEN3_ASR_HF_DEVICE": "mps",
                "QWEN3_ASR_HF_MAX_TOKENS": "123",
            },
            clear=True,
        ):
            env_args = benchmark_whisper.parse_args(["--profile", "qwen"])
            cli_args = benchmark_whisper.parse_args(
                [
                    "--profile",
                    "qwen",
                    "--qwen3-asr-hf-python",
                    "/cli/python",
                    "--qwen3-asr-hf-model-path",
                    "/cli/model",
                    "--qwen3-asr-hf-device",
                    "cpu",
                    "--qwen3-asr-hf-max-tokens",
                    "456",
                ]
            )

        self.assertEqual(env_args.qwen3_asr_hf_python, Path("/env/python"))
        self.assertEqual(env_args.qwen3_asr_hf_model_path, Path("/env/model"))
        self.assertEqual(env_args.qwen3_asr_hf_device, "mps")
        self.assertEqual(env_args.qwen3_asr_hf_max_tokens, 123)
        self.assertEqual(cli_args.qwen3_asr_hf_python, Path("/cli/python"))
        self.assertEqual(cli_args.qwen3_asr_hf_model_path, Path("/cli/model"))
        self.assertEqual(cli_args.qwen3_asr_hf_device, "cpu")
        self.assertEqual(cli_args.qwen3_asr_hf_max_tokens, 456)

        with mock.patch.dict("os.environ", {}, clear=True):
            defaults = benchmark_whisper.parse_args([])
        self.assertEqual(
            defaults.qwen3_asr_hf_python,
            benchmark_whisper.DEFAULT_QWEN3_ASR_HF_PYTHON,
        )
        self.assertIsNone(defaults.qwen3_asr_hf_model_path)
        self.assertEqual(
            defaults.qwen3_asr_hf_device,
            benchmark_whisper.DEFAULT_QWEN3_ASR_HF_DEVICE,
        )
        self.assertEqual(
            defaults.qwen3_asr_hf_max_tokens,
            benchmark_whisper.QWEN3_ASR_HF_DEFAULT_MAX_TOKENS,
        )

    def test_resolver_uses_main_snapshot_and_exact_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--Qwen--Qwen3-ASR-1.7B-hf"
            revision = "b" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (cache_root / "refs" / "main").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            for relative_path in benchmark_whisper.QWEN3_ASR_HF_REQUIRED_MODEL_FILES:
                (snapshot / relative_path).write_bytes(b"model")

            with mock.patch.object(
                benchmark_whisper, "QWEN3_ASR_HF_MODEL_CACHE_ROOT", cache_root
            ):
                resolved = benchmark_whisper.resolve_qwen3_asr_hf_model_path()

        self.assertEqual(resolved, snapshot.resolve())

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            for relative_path in benchmark_whisper.QWEN3_ASR_HF_REQUIRED_MODEL_FILES[:-1]:
                (model_path / relative_path).write_bytes(b"model")
            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_qwen3_asr_hf_model_path(model_path)

    def test_session_resolves_and_stores_local_model_path(self) -> None:
        model_path = Path("/models/qwen3-asr-hf")
        with mock.patch.object(
            benchmark_whisper,
            "resolve_qwen3_asr_hf_model_path",
            return_value=model_path,
        ) as resolve:
            session = benchmark_whisper.load_backend_session(
                "qwen3-asr-hf",
                benchmark_whisper.QWEN3_ASR_HF_MODEL_VARIANT,
                argparse.Namespace(qwen3_asr_hf_model_path=None),
            )

        resolve.assert_called_once_with(None)
        self.assertEqual(session.device, "subprocess")
        self.assertEqual(session.session, {"model_path": model_path})
        self.assertIsNone(session.load_seconds)

    def test_run_maps_success_request_timing_rss_and_scores(self) -> None:
        args = argparse.Namespace(
            language="ru",
            task="transcribe",
            qwen3_asr_hf_device="mps",
            qwen3_asr_hf_max_tokens=123,
            qwen3_asr_hf_python=Path("/venv/bin/python"),
            reference_transcript_text="привет мир",
            sample_label="audio",
            audio_duration_seconds=3.0,
            worker_timeout_seconds=12.5,
        )
        execution = self._execution(
            {
                "status": "ok",
                "transcript": "Привет, мир!",
                "segments": [],
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
                "language": "Russian",
                "timestamps_supported": False,
            }
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_qwen3_asr_hf(
                Path("audio.mp3"),
                benchmark_whisper.QWEN3_ASR_HF_MODEL_VARIANT,
                2,
                args,
                Path("/models/qwen3-asr-hf"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "qwen3-asr-hf")
        self.assertEqual(result.model, benchmark_whisper.QWEN3_ASR_HF_MODEL_VARIANT)
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.forced_language, "ru")
        self.assertEqual(result.transcript, "Привет, мир!")
        self.assertEqual(result.load_seconds, 1.25)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 512.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            benchmark_whisper.QWEN3_ASR_HF_WORKER_MODULE,
            {
                "model_path": "/models/qwen3-asr-hf",
                "audio_path": "audio.mp3",
                "device": "mps",
                "max_new_tokens": 123,
                "language": "ru",
            },
            12.5,
            env=benchmark_whisper.QWEN3_ASR_HF_OFFLINE_ENV,
        )

    def test_run_omits_auto_language_rejects_translate_and_maps_worker_errors(self) -> None:
        args = argparse.Namespace(
            language="auto",
            task="transcribe",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
            qwen3_asr_hf_python=Path("/venv/bin/python"),
            worker_timeout_seconds=12.5,
        )
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=64.0,
                error="worker timed out after 12.5 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "load failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", side_effect=executions
        ) as run_worker:
            timeout_result = benchmark_whisper.run_qwen3_asr_hf(
                Path("audio.mp3"),
                benchmark_whisper.QWEN3_ASR_HF_MODEL_VARIANT,
                1,
                args,
                Path("/models/qwen3-asr-hf"),
            )
            worker_error_result = benchmark_whisper.run_qwen3_asr_hf(
                Path("audio.mp3"),
                benchmark_whisper.QWEN3_ASR_HF_MODEL_VARIANT,
                2,
                args,
                Path("/models/qwen3-asr-hf"),
            )

        self.assertNotIn("language", run_worker.call_args_list[0].args[2])
        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(timeout_result.peak_rss_mb, 64.0)
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("load failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

        args.task = "translate"
        with mock.patch.object(benchmark_whisper, "run_json_worker") as run_worker:
            result = benchmark_whisper.run_qwen3_asr_hf(
                Path("audio.mp3"),
                benchmark_whisper.QWEN3_ASR_HF_MODEL_VARIANT,
                3,
                args,
                Path("/models/qwen3-asr-hf"),
            )
        self.assertEqual(result.status, "error")
        self.assertIn("only the transcribe task", result.error or "")
        run_worker.assert_not_called()

    def test_qwen_profile_main_runs_mlx_and_hf_pairs_with_warmup(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "qwen"])
        args.output = Path("/tmp/test-output.json")
        args.runs = 1
        args.warmup = True
        audio_input = benchmark_whisper.ResolvedAudioInput(
            audio_path=Path("audio.mp3"),
            reference_transcript_path=None,
            reference_transcript_text=None,
            forced_language="en",
            selector_language="en",
            sample_label="audio",
            source="default-language",
            audio_duration_seconds=1.0,
        )
        written_payload: dict[str, object] = {}

        def fake_run(
            audio_path: Path,
            model: str,
            run_index: int,
            run_args: argparse.Namespace,
            _model_path: Path,
        ) -> benchmark_whisper.RunResult:
            return benchmark_whisper.build_run_result(
                backend="qwen3-asr",
                model_name=model,
                run_index=run_index,
                load_seconds=0.2,
                transcribe_seconds=0.1,
                transcript="ok",
                detected_language="en",
                detected_language_probability=None,
                reference_transcript=None,
                audio_path=audio_path,
                sample_label=run_args.sample_label,
                audio_duration_seconds=run_args.audio_duration_seconds,
                forced_language=run_args.language,
            )

        def fake_hf_run(
            audio_path: Path,
            model: str,
            run_index: int,
            run_args: argparse.Namespace,
            _model_path: Path,
        ) -> benchmark_whisper.RunResult:
            result = fake_run(audio_path, model, run_index, run_args, _model_path)
            result.backend = "qwen3-asr-hf"
            return result

        with (
            mock.patch.object(benchmark_whisper, "parse_args", return_value=args),
            mock.patch.object(
                benchmark_whisper, "resolve_output_paths", return_value=args.output
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_audio_inputs",
                return_value=[audio_input],
            ),
            mock.patch.object(
                benchmark_whisper,
                "resolve_qwen3_asr_model_path",
                return_value=Path("/models/qwen3-asr-mlx"),
            ) as resolve_mlx,
            mock.patch.object(
                benchmark_whisper,
                "resolve_qwen3_asr_hf_model_path",
                return_value=Path("/models/qwen3-asr-hf"),
            ) as resolve_hf,
            mock.patch.object(
                benchmark_whisper, "run_qwen3_asr", side_effect=fake_run
            ) as run_mlx,
            mock.patch.object(
                benchmark_whisper, "run_qwen3_asr_hf", side_effect=fake_hf_run
            ) as run_hf,
            mock.patch.object(
                benchmark_whisper,
                "write_json",
                side_effect=lambda _path, payload: written_payload.update(payload),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            exit_code = benchmark_whisper.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(resolve_mlx.call_count, 2)
        resolve_hf.assert_called_once_with(None)
        self.assertEqual(run_mlx.call_count, 4)
        self.assertEqual(run_hf.call_count, 2)
        self.assertEqual(
            [(call.args[1], call.args[2]) for call in run_mlx.call_args_list],
            [
                ("qwen3-asr-0.6b-8bit", 0),
                ("qwen3-asr-0.6b-8bit", 1),
                ("qwen3-asr-1.7b-8bit", 0),
                ("qwen3-asr-1.7b-8bit", 1),
            ],
        )
        self.assertEqual(
            [(call.args[1], call.args[2]) for call in run_hf.call_args_list],
            [("qwen3-asr-1.7b-hf", 0), ("qwen3-asr-1.7b-hf", 1)],
        )
        self.assertEqual(written_payload["skipped"], [])

    def test_metadata_records_profile_capabilities_and_offline_options(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "qwen"])
        metadata = benchmark_whisper.build_metadata(args, [])

        self.assertEqual(metadata["profile"], "qwen")
        self.assertEqual(
            metadata["benchmark_pairs"],
            [list(pair) for pair in benchmark_whisper.QWEN_BENCHMARK_PAIRS],
        )
        self.assertEqual(
            metadata["qwen3_asr_hf_python"], ".venvs/qwen3-asr-hf/bin/python"
        )
        self.assertIsNone(metadata["qwen3_asr_hf_model_path"])
        self.assertEqual(
            metadata["qwen3_asr_hf_model_cache_root"],
            "/Volumes/512GB/hf/hub/models--Qwen--Qwen3-ASR-1.7B-hf",
        )
        self.assertEqual(
            metadata["qwen3_asr_hf_required_model_files"],
            [
                "config.json",
                "model.safetensors",
                "processor_config.json",
                "tokenizer.json",
            ],
        )
        self.assertEqual(metadata["qwen3_asr_hf_device"], "auto")
        self.assertEqual(metadata["qwen3_asr_hf_max_tokens"], 4096)
        self.assertTrue(metadata["qwen3_asr_hf_multilingual"])
        self.assertEqual(metadata["qwen3_asr_hf_supported_tasks"], ["transcribe"])
        self.assertFalse(metadata["qwen3_asr_hf_segment_timestamps"])
        self.assertEqual(
            metadata["qwen3_asr_hf_timestamp_semantics"], "unsupported"
        )
        self.assertFalse(metadata["qwen3_asr_hf_forced_aligner"])
        self.assertEqual(
            metadata["qwen3_asr_hf_offline_env"],
            benchmark_whisper.QWEN3_ASR_HF_OFFLINE_ENV,
        )


class GigaamMultilingualMlxBenchmarkTests(unittest.TestCase):
    def test_profile_has_exact_en_ru_pairs_and_main_is_unchanged(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "gigaam-multilingual"])

        self.assertEqual(args.audios, [])
        self.assertEqual(
            args.models,
            [
                "gigaam-multilingual-large-ctc",
                "gigaam-multilingual-mlx-fp16",
            ],
        )
        self.assertEqual(
            args.backends, ["gigaam-multilingual", "gigaam-multilingual-mlx"]
        )
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            benchmark_whisper.GIGAAM_MULTILINGUAL_BENCHMARK_PAIRS,
        )

        with (
            mock.patch.object(
                benchmark_whisper,
                "ensure_audio_file",
                side_effect=lambda path: path.resolve(),
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_reference_transcript",
                side_effect=lambda path: f"normalized:{path.name}",
            ),
            mock.patch.object(
                benchmark_whisper, "get_audio_duration_seconds", return_value=1.0
            ),
        ):
            audio_inputs = benchmark_whisper.resolve_audio_inputs(args)

        self.assertEqual([item.selector_language for item in audio_inputs], ["en", "ru"])
        self.assertEqual([item.forced_language for item in audio_inputs], ["en", "ru"])

        main_args = benchmark_whisper.parse_args([])
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(main_args),
            benchmark_whisper.MAIN_BENCHMARK_PAIRS,
        )
        self.assertNotIn("gigaam-multilingual-mlx", main_args.backends)

    def test_options_use_cli_environment_and_builtin_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "GIGAAM_MULTILINGUAL_MLX_PYTHON": "/env/python",
                "GIGAAM_MULTILINGUAL_MLX_MODEL_PATH": "/env/model",
                "GIGAAM_MULTILINGUAL_MLX_CHUNK_SECONDS": "11.5",
                "GIGAAM_MULTILINGUAL_MLX_OVERLAP_SECONDS": "1.5",
            },
            clear=True,
        ):
            env_args = benchmark_whisper.parse_args(
                ["--profile", "gigaam-multilingual"]
            )
            cli_args = benchmark_whisper.parse_args(
                [
                    "--profile",
                    "gigaam-multilingual",
                    "--gigaam-multilingual-mlx-python",
                    "/cli/python",
                    "--gigaam-multilingual-mlx-model-path",
                    "/cli/model",
                    "--gigaam-multilingual-mlx-chunk-seconds",
                    "13",
                    "--gigaam-multilingual-mlx-overlap-seconds",
                    "2",
                ]
            )

        self.assertEqual(env_args.gigaam_multilingual_mlx_python, Path("/env/python"))
        self.assertEqual(env_args.gigaam_multilingual_mlx_model_path, Path("/env/model"))
        self.assertEqual(env_args.gigaam_multilingual_mlx_chunk_seconds, 11.5)
        self.assertEqual(env_args.gigaam_multilingual_mlx_overlap_seconds, 1.5)
        self.assertEqual(cli_args.gigaam_multilingual_mlx_python, Path("/cli/python"))
        self.assertEqual(cli_args.gigaam_multilingual_mlx_model_path, Path("/cli/model"))
        self.assertEqual(cli_args.gigaam_multilingual_mlx_chunk_seconds, 13.0)
        self.assertEqual(cli_args.gigaam_multilingual_mlx_overlap_seconds, 2.0)

        with mock.patch.dict("os.environ", {}, clear=True):
            defaults = benchmark_whisper.parse_args([])
        self.assertEqual(
            defaults.gigaam_multilingual_mlx_python,
            benchmark_whisper.DEFAULT_GIGAAM_MULTILINGUAL_MLX_PYTHON,
        )
        self.assertIsNone(defaults.gigaam_multilingual_mlx_model_path)
        self.assertEqual(
            defaults.gigaam_multilingual_mlx_chunk_seconds,
            benchmark_whisper.DEFAULT_GIGAAM_MULTILINGUAL_MLX_CHUNK_SECONDS,
        )
        self.assertEqual(
            defaults.gigaam_multilingual_mlx_overlap_seconds,
            benchmark_whisper.DEFAULT_GIGAAM_MULTILINGUAL_MLX_OVERLAP_SECONDS,
        )

    def test_resolver_uses_main_snapshot_and_exact_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--ai-babai--gigaam-multilingual-mlx"
            revision = "a" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (cache_root / "refs" / "main").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            for relative_path in benchmark_whisper.GIGAAM_MULTILINGUAL_MLX_REQUIRED_MODEL_FILES:
                (snapshot / relative_path).write_bytes(b"model")

            with mock.patch.object(
                benchmark_whisper,
                "GIGAAM_MULTILINGUAL_MLX_MODEL_CACHE_ROOT",
                cache_root,
            ):
                resolved = benchmark_whisper.resolve_gigaam_multilingual_mlx_model_path()

        self.assertEqual(resolved, snapshot.resolve())

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            for relative_path in benchmark_whisper.GIGAAM_MULTILINGUAL_MLX_REQUIRED_MODEL_FILES[:-1]:
                (model_path / relative_path).write_bytes(b"model")
            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_gigaam_multilingual_mlx_model_path(model_path)

    def test_session_resolves_and_stores_local_model_path(self) -> None:
        model_path = Path("/models/gigaam-multilingual-mlx")
        with mock.patch.object(
            benchmark_whisper,
            "resolve_gigaam_multilingual_mlx_model_path",
            return_value=model_path,
        ) as resolve:
            session = benchmark_whisper.load_backend_session(
                "gigaam-multilingual-mlx",
                benchmark_whisper.GIGAAM_MULTILINGUAL_MLX_MODEL,
                argparse.Namespace(gigaam_multilingual_mlx_model_path=None),
            )

        resolve.assert_called_once_with(None)
        self.assertEqual(session.device, "subprocess")
        self.assertEqual(session.session, {"model_path": model_path})
        self.assertIsNone(session.load_seconds)

    def test_run_maps_success_request_timing_rss_and_scores(self) -> None:
        args = argparse.Namespace(
            language="ru",
            task="transcribe",
            gigaam_multilingual_mlx_chunk_seconds=12.5,
            gigaam_multilingual_mlx_overlap_seconds=1.5,
            gigaam_multilingual_mlx_python=Path("/venv/bin/python"),
            reference_transcript_text="привет мир",
            sample_label="audio",
            audio_duration_seconds=3.0,
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "Привет, мир!",
                "timestamps": [{"text": "Привет", "start": 0.0, "end": 0.5}],
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
                "language": "ru",
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=512.5,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_gigaam_multilingual_mlx(
                Path("audio.mp3"),
                benchmark_whisper.GIGAAM_MULTILINGUAL_MLX_MODEL,
                2,
                args,
                Path("/models/gigaam-multilingual-mlx"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "gigaam-multilingual-mlx")
        self.assertEqual(result.model, benchmark_whisper.GIGAAM_MULTILINGUAL_MLX_MODEL)
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.load_seconds, 1.25)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 512.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            benchmark_whisper.GIGAAM_MULTILINGUAL_MLX_WORKER_MODULE,
            {
                "model_path": "/models/gigaam-multilingual-mlx",
                "audio_path": "audio.mp3",
                "variant": "fp16",
                "chunk_seconds": 12.5,
                "overlap_seconds": 1.5,
                "language": "ru",
            },
            12.5,
            env=benchmark_whisper.GIGAAM_MULTILINGUAL_MLX_OFFLINE_ENV,
        )

    def test_run_omits_auto_language_and_maps_worker_errors(self) -> None:
        args = argparse.Namespace(
            language="auto",
            task="transcribe",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
        )
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=64.0,
                error="worker timed out after 12.5 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "load failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", side_effect=executions
        ) as run_worker:
            timeout_result = benchmark_whisper.run_gigaam_multilingual_mlx(
                Path("audio.mp3"),
                benchmark_whisper.GIGAAM_MULTILINGUAL_MLX_MODEL,
                1,
                args,
                Path("/models/gigaam-multilingual-mlx"),
            )
            worker_error_result = benchmark_whisper.run_gigaam_multilingual_mlx(
                Path("audio.mp3"),
                benchmark_whisper.GIGAAM_MULTILINGUAL_MLX_MODEL,
                2,
                args,
                Path("/models/gigaam-multilingual-mlx"),
            )

        self.assertNotIn("language", run_worker.call_args_list[0].args[2])
        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(timeout_result.peak_rss_mb, 64.0)
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("load failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

    def test_metadata_records_mlx_capabilities_and_runtime_options(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "gigaam-multilingual"])
        metadata = benchmark_whisper.build_metadata(args, [])

        self.assertEqual(metadata["profile"], "gigaam-multilingual")
        self.assertEqual(
            metadata["benchmark_pairs"],
            [list(pair) for pair in benchmark_whisper.GIGAAM_MULTILINGUAL_BENCHMARK_PAIRS],
        )
        self.assertEqual(
            metadata["gigaam_multilingual_mlx_python"],
            ".venvs/gigaam-multilingual-mlx/bin/python",
        )
        self.assertEqual(metadata["gigaam_multilingual_mlx_model_path"], None)
        self.assertEqual(
            metadata["gigaam_multilingual_mlx_model_cache_root"],
            "/Volumes/512GB/hf/hub/models--ai-babai--gigaam-multilingual-mlx",
        )
        self.assertEqual(metadata["gigaam_multilingual_mlx_chunk_seconds"], 20.0)
        self.assertEqual(metadata["gigaam_multilingual_mlx_overlap_seconds"], 2.0)
        self.assertTrue(metadata["gigaam_multilingual_mlx_multilingual"])
        self.assertEqual(metadata["gigaam_multilingual_mlx_supported_tasks"], ["transcribe"])
        self.assertTrue(metadata["gigaam_multilingual_mlx_word_timestamps"])
        self.assertEqual(
            metadata["gigaam_multilingual_mlx_offline_env"],
            benchmark_whisper.GIGAAM_MULTILINGUAL_MLX_OFFLINE_ENV,
        )

    def test_profile_uses_generic_warmup_and_run_path(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "vibevoice"])
        args.warmup = True
        args.runs = 1
        args.output = Path("/tmp/test-output.json")
        audio_inputs = [
            benchmark_whisper.ResolvedAudioInput(
                audio_path=Path("en.mp3"),
                reference_transcript_path=None,
                reference_transcript_text=None,
                forced_language="en",
                selector_language="en",
                sample_label="en",
                source="default-language",
                audio_duration_seconds=1.0,
            ),
            benchmark_whisper.ResolvedAudioInput(
                audio_path=Path("ru.mp3"),
                reference_transcript_path=None,
                reference_transcript_text=None,
                forced_language="ru",
                selector_language="ru",
                sample_label="ru",
                source="default-language",
                audio_duration_seconds=1.0,
            ),
        ]
        session = benchmark_whisper.BackendSession(
            backend="vibevoice",
            model=benchmark_whisper.VIBEVOICE_MODEL_VARIANT,
            device="subprocess",
            session={"model_path": Path("/models/vibevoice")},
            load_seconds=None,
        )

        def fake_run(
            backend: str,
            audio_path: Path,
            model: str,
            run_index: int,
            run_args: argparse.Namespace,
            _session: benchmark_whisper.BackendSession,
            _load_seconds: float | None,
        ) -> benchmark_whisper.RunResult:
            return benchmark_whisper.build_run_result(
                backend=backend,
                model_name=model,
                run_index=run_index,
                load_seconds=None,
                transcribe_seconds=0.1,
                transcript="ok",
                detected_language=run_args.language,
                detected_language_probability=None,
                reference_transcript=None,
                audio_path=audio_path,
                sample_label=run_args.sample_label,
                audio_duration_seconds=run_args.audio_duration_seconds,
                forced_language=run_args.language,
            )

        written_payload: dict[str, object] = {}
        with (
            mock.patch.object(benchmark_whisper, "parse_args", return_value=args),
            mock.patch.object(
                benchmark_whisper, "resolve_output_paths", return_value=args.output
            ),
            mock.patch.object(
                benchmark_whisper, "resolve_audio_inputs", return_value=audio_inputs
            ),
            mock.patch.object(
                benchmark_whisper, "load_backend_session", return_value=session
            ) as load_session,
            mock.patch.object(benchmark_whisper, "maybe_warmup") as warmup,
            mock.patch.object(
                benchmark_whisper, "run_single_backend", side_effect=fake_run
            ) as run_single,
            mock.patch.object(
                benchmark_whisper,
                "write_json",
                side_effect=lambda _path, payload: written_payload.update(payload),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            exit_code = benchmark_whisper.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(load_session.call_count, 2)
        self.assertEqual(warmup.call_count, 2)
        self.assertEqual(run_single.call_count, 2)
        self.assertEqual(written_payload["skipped"], [])


class VibevoiceBenchmarkTests(unittest.TestCase):
    def test_profile_has_exact_en_ru_pair_and_main_is_unchanged(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "vibevoice"])

        self.assertEqual(args.audios, [])
        self.assertEqual(args.models, [benchmark_whisper.VIBEVOICE_MODEL_VARIANT])
        self.assertEqual(args.backends, ["vibevoice"])
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            benchmark_whisper.VIBEVOICE_BENCHMARK_PAIRS,
        )
        with (
            mock.patch.object(
                benchmark_whisper,
                "ensure_audio_file",
                side_effect=lambda path: path.resolve(),
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_reference_transcript",
                side_effect=lambda path: f"normalized:{path.name}",
            ),
            mock.patch.object(
                benchmark_whisper, "get_audio_duration_seconds", return_value=1.0
            ),
        ):
            audio_inputs = benchmark_whisper.resolve_audio_inputs(args)
        self.assertEqual([item.selector_language for item in audio_inputs], ["en", "ru"])
        self.assertEqual([item.forced_language for item in audio_inputs], ["en", "ru"])

        main_args = benchmark_whisper.parse_args([])
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(main_args),
            benchmark_whisper.MAIN_BENCHMARK_PAIRS,
        )
        self.assertNotIn("vibevoice", main_args.backends)

    def test_options_use_cli_environment_and_builtin_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "VIBEVOICE_PYTHON": "/env/python",
                "VIBEVOICE_MODEL_PATH": "/env/model",
                "VIBEVOICE_DEVICE": "cpu",
                "VIBEVOICE_MODE": "parsed",
                "VIBEVOICE_ACOUSTIC_TOKENIZER_CHUNK_SIZE": "64000",
            },
            clear=True,
        ):
            env_args = benchmark_whisper.parse_args(["--profile", "vibevoice"])
            cli_args = benchmark_whisper.parse_args(
                [
                    "--profile",
                    "vibevoice",
                    "--vibevoice-python",
                    "/cli/python",
                    "--vibevoice-model-path",
                    "/cli/model",
                    "--vibevoice-device",
                    "mps",
                    "--vibevoice-mode",
                    "transcription_only",
                    "--vibevoice-acoustic-tokenizer-chunk-size",
                    "32000",
                ]
            )

        self.assertEqual(env_args.vibevoice_python, Path("/env/python"))
        self.assertEqual(env_args.vibevoice_model_path, Path("/env/model"))
        self.assertEqual(env_args.vibevoice_device, "cpu")
        self.assertEqual(env_args.vibevoice_mode, "parsed")
        self.assertEqual(env_args.vibevoice_acoustic_tokenizer_chunk_size, 64000)
        self.assertEqual(cli_args.vibevoice_python, Path("/cli/python"))
        self.assertEqual(cli_args.vibevoice_model_path, Path("/cli/model"))
        self.assertEqual(cli_args.vibevoice_device, "mps")
        self.assertEqual(cli_args.vibevoice_mode, "transcription_only")
        self.assertEqual(cli_args.vibevoice_acoustic_tokenizer_chunk_size, 32000)

        with mock.patch.dict("os.environ", {}, clear=True):
            defaults = benchmark_whisper.parse_args([])
        self.assertEqual(defaults.vibevoice_python, benchmark_whisper.DEFAULT_VIBEVOICE_PYTHON)
        self.assertIsNone(defaults.vibevoice_model_path)
        self.assertEqual(defaults.vibevoice_device, "auto")
        self.assertEqual(defaults.vibevoice_mode, "transcription_only")
        self.assertIsNone(defaults.vibevoice_acoustic_tokenizer_chunk_size)

    def test_resolver_uses_main_snapshot_and_all_exact_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--microsoft--VibeVoice-ASR-HF"
            revision = "b" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (cache_root / "refs" / "main").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            for relative_path in benchmark_whisper.VIBEVOICE_REQUIRED_MODEL_FILES:
                (snapshot / relative_path).write_bytes(b"model")

            with mock.patch.object(
                benchmark_whisper, "VIBEVOICE_MODEL_CACHE_ROOT", cache_root
            ):
                resolved = benchmark_whisper.resolve_vibevoice_model_path()

        self.assertEqual(resolved, snapshot.resolve())
        self.assertEqual(len(benchmark_whisper.VIBEVOICE_MODEL_SHARD_FILES), 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            for relative_path in benchmark_whisper.VIBEVOICE_REQUIRED_MODEL_FILES[:-1]:
                (model_path / relative_path).write_bytes(b"model")
            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_vibevoice_model_path(model_path)

    def test_session_resolves_and_stores_local_model_path(self) -> None:
        model_path = Path("/models/vibevoice")
        with mock.patch.object(
            benchmark_whisper, "resolve_vibevoice_model_path", return_value=model_path
        ) as resolve:
            session = benchmark_whisper.load_backend_session(
                "vibevoice",
                benchmark_whisper.VIBEVOICE_MODEL_VARIANT,
                argparse.Namespace(vibevoice_model_path=None),
            )

        resolve.assert_called_once_with(None)
        self.assertEqual(session.device, "subprocess")
        self.assertEqual(session.session, {"model_path": model_path})
        self.assertIsNone(session.load_seconds)

    def test_run_maps_parsed_success_request_timing_rss_and_scores(self) -> None:
        args = argparse.Namespace(
            language="en",
            task="transcribe",
            vibevoice_device="mps",
            vibevoice_mode="parsed",
            vibevoice_acoustic_tokenizer_chunk_size=64000,
            vibevoice_python=Path("/venv/bin/python"),
            reference_transcript_text="hello world",
            sample_label="audio",
            audio_duration_seconds=3.0,
            worker_timeout_seconds=12.5,
        )
        execution = WorkerExecution(
            status="ok",
            payload={
                "status": "ok",
                "transcript": "Hello, world!",
                "segments": [{"Speaker": "SPEAKER_00", "Content": "Hello, world!"}],
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
                "mode": "parsed",
                "device": "mps",
            },
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=768.5,
            error=None,
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_vibevoice(
                Path("audio.mp3"),
                benchmark_whisper.VIBEVOICE_MODEL_VARIANT,
                2,
                args,
                Path("/models/vibevoice"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "vibevoice")
        self.assertEqual(result.model, benchmark_whisper.VIBEVOICE_MODEL_VARIANT)
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.load_seconds, 1.25)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 768.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            benchmark_whisper.VIBEVOICE_WORKER_MODULE,
            {
                "model_path": "/models/vibevoice",
                "audio_path": "audio.mp3",
                "device": "mps",
                "mode": "parsed",
                "acoustic_tokenizer_chunk_size": 64000,
            },
            12.5,
            env=benchmark_whisper.VIBEVOICE_OFFLINE_ENV,
        )

    def test_run_uses_transcription_only_by_default_and_maps_worker_errors(self) -> None:
        args = argparse.Namespace(
            language="ru",
            task="transcribe",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
        )
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=64.0,
                error="worker timed out after 12.5 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "load failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", side_effect=executions
        ) as run_worker:
            timeout_result = benchmark_whisper.run_vibevoice(
                Path("audio.mp3"),
                benchmark_whisper.VIBEVOICE_MODEL_VARIANT,
                1,
                args,
                Path("/models/vibevoice"),
            )
            worker_error_result = benchmark_whisper.run_vibevoice(
                Path("audio.mp3"),
                benchmark_whisper.VIBEVOICE_MODEL_VARIANT,
                2,
                args,
                Path("/models/vibevoice"),
            )

        self.assertEqual(run_worker.call_args_list[0].args[2]["mode"], "transcription_only")
        self.assertNotIn("acoustic_tokenizer_chunk_size", run_worker.call_args_list[0].args[2])
        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(timeout_result.peak_rss_mb, 64.0)
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("load failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

    def test_run_rejects_translate(self) -> None:
        args = argparse.Namespace(
            language="en",
            task="translate",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=1.0,
        )
        with mock.patch.object(benchmark_whisper, "run_json_worker") as run_worker:
            result = benchmark_whisper.run_vibevoice(
                Path("audio.mp3"),
                benchmark_whisper.VIBEVOICE_MODEL_VARIANT,
                1,
                args,
                Path("/models/vibevoice"),
            )

        self.assertEqual(result.status, "error")
        self.assertIn("only the transcribe task", result.error or "")
        run_worker.assert_not_called()

    def test_metadata_records_multilingual_transcribe_only_parsed_segments_and_offline_env(
        self,
    ) -> None:
        args = benchmark_whisper.parse_args(["--profile", "vibevoice"])
        metadata = benchmark_whisper.build_metadata(args, [])

        self.assertEqual(metadata["profile"], "vibevoice")
        self.assertEqual(
            metadata["benchmark_pairs"],
            [list(pair) for pair in benchmark_whisper.VIBEVOICE_BENCHMARK_PAIRS],
        )
        self.assertEqual(metadata["vibevoice_python"], ".venvs/vibevoice/bin/python")
        self.assertEqual(metadata["vibevoice_model_path"], None)
        self.assertEqual(
            metadata["vibevoice_model_cache_root"],
            "/Volumes/512GB/hf/hub/models--microsoft--VibeVoice-ASR-HF",
        )
        self.assertEqual(metadata["vibevoice_device"], "auto")
        self.assertEqual(metadata["vibevoice_mode"], "transcription_only")
        self.assertTrue(metadata["vibevoice_multilingual"])
        self.assertEqual(metadata["vibevoice_supported_tasks"], ["transcribe"])
        self.assertTrue(metadata["vibevoice_segment_timestamps"])
        self.assertEqual(
            metadata["vibevoice_offline_env"], benchmark_whisper.VIBEVOICE_OFFLINE_ENV
        )
        self.assertEqual(
            benchmark_whisper.BACKEND_CAPABILITIES["vibevoice"].supported_tasks,
            ("transcribe",),
        )


class ParakeetBenchmarkTests(unittest.TestCase):
    def test_parakeet_profile_has_exact_en_ru_three_pair_matrix(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "parakeet"])

        self.assertEqual(args.audios, [])
        self.assertEqual(
            args.models,
            [
                "parakeet-tdt-0.6b-v3-hf",
                "parakeet-tdt-0.6b-v3-sherpa-fp32",
                "parakeet-tdt-0.6b-v3-sherpa-int8",
            ],
        )
        self.assertEqual(args.backends, ["parakeet-hf", "parakeet-sherpa"])
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            benchmark_whisper.PARAKEET_BENCHMARK_PAIRS,
        )
        with (
            mock.patch.object(
                benchmark_whisper,
                "ensure_audio_file",
                side_effect=lambda path: path.resolve(),
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_reference_transcript",
                side_effect=lambda path: f"normalized:{path.name}",
            ),
            mock.patch.object(
                benchmark_whisper,
                "get_audio_duration_seconds",
                return_value=1.0,
            ),
        ):
            audio_inputs = benchmark_whisper.resolve_audio_inputs(args)
        self.assertEqual([item.selector_language for item in audio_inputs], ["en", "ru"])
        self.assertEqual([item.forced_language for item in audio_inputs], ["en", "ru"])

    def test_main_profile_remains_without_parakeet(self) -> None:
        args = benchmark_whisper.parse_args([])

        self.assertEqual(args.profile, "main")
        self.assertNotIn("parakeet-hf", args.backends)
        self.assertNotIn("parakeet-sherpa", args.backends)
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            benchmark_whisper.MAIN_BENCHMARK_PAIRS,
        )

    def test_parakeet_options_use_cli_values_over_environment(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "PARAKEET_HF_PYTHON": "/env/hf-python",
                "PARAKEET_HF_MODEL_PATH": "/env/hf-model",
                "PARAKEET_HF_DEVICE": "mps",
                "PARAKEET_SHERPA_PYTHON": "/env/sherpa-python",
                "PARAKEET_SHERPA_MODEL_PATH": "/env/sherpa-model",
                "PARAKEET_SHERPA_THREADS": "3",
            },
            clear=True,
        ):
            args = benchmark_whisper.parse_args(
                [
                    "--profile",
                    "parakeet",
                    "--parakeet-hf-python",
                    "/cli/hf-python",
                    "--parakeet-hf-model-path",
                    "/cli/hf-model",
                    "--parakeet-hf-device",
                    "cpu",
                    "--parakeet-sherpa-python",
                    "/cli/sherpa-python",
                    "--parakeet-sherpa-model-path",
                    "/cli/sherpa-model",
                    "--parakeet-sherpa-threads",
                    "7",
                ]
            )

        self.assertEqual(args.parakeet_hf_python, Path("/cli/hf-python"))
        self.assertEqual(args.parakeet_hf_model_path, Path("/cli/hf-model"))
        self.assertEqual(args.parakeet_hf_device, "cpu")
        self.assertEqual(args.parakeet_sherpa_python, Path("/cli/sherpa-python"))
        self.assertEqual(args.parakeet_sherpa_model_path, Path("/cli/sherpa-model"))
        self.assertEqual(args.parakeet_sherpa_threads, 7)

    def test_parakeet_options_use_environment_and_builtin_defaults(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "PARAKEET_HF_PYTHON": "/env/hf-python",
                "PARAKEET_HF_MODEL_PATH": "/env/hf-model",
                "PARAKEET_HF_DEVICE": "cpu",
                "PARAKEET_SHERPA_PYTHON": "/env/sherpa-python",
                "PARAKEET_SHERPA_MODEL_PATH": "/env/sherpa-model",
                "PARAKEET_SHERPA_THREADS": "6",
            },
            clear=True,
        ):
            args = benchmark_whisper.parse_args([])

        self.assertEqual(args.parakeet_hf_python, Path("/env/hf-python"))
        self.assertEqual(args.parakeet_hf_model_path, Path("/env/hf-model"))
        self.assertEqual(args.parakeet_hf_device, "cpu")
        self.assertEqual(args.parakeet_sherpa_python, Path("/env/sherpa-python"))
        self.assertEqual(args.parakeet_sherpa_model_path, Path("/env/sherpa-model"))
        self.assertEqual(args.parakeet_sherpa_threads, 6)

        with mock.patch.dict("os.environ", {}, clear=True):
            fallback_args = benchmark_whisper.parse_args([])

        self.assertEqual(
            fallback_args.parakeet_hf_python,
            benchmark_whisper.DEFAULT_PARAKEET_HF_PYTHON,
        )
        self.assertIsNone(fallback_args.parakeet_hf_model_path)
        self.assertEqual(
            fallback_args.parakeet_hf_device,
            benchmark_whisper.DEFAULT_PARAKEET_HF_DEVICE,
        )
        self.assertEqual(
            fallback_args.parakeet_sherpa_python,
            benchmark_whisper.DEFAULT_PARAKEET_SHERPA_PYTHON,
        )
        self.assertIsNone(fallback_args.parakeet_sherpa_model_path)
        self.assertEqual(
            fallback_args.parakeet_sherpa_threads,
            benchmark_whisper.DEFAULT_PARAKEET_SHERPA_THREADS,
        )

    def test_parakeet_hf_resolver_uses_pinned_main_snapshot_and_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--nvidia--parakeet-tdt-0.6b-v3"
            revision = "a" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (cache_root / "refs" / "main").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            for relative_path in benchmark_whisper.PARAKEET_HF_REQUIRED_MODEL_FILES:
                (snapshot / relative_path).write_bytes(b"model")

            with mock.patch.object(
                benchmark_whisper, "PARAKEET_HF_MODEL_CACHE_ROOT", cache_root
            ):
                resolved = benchmark_whisper.resolve_parakeet_hf_model_path()

        self.assertEqual(resolved, snapshot.resolve())

    def test_parakeet_hf_resolver_requires_official_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            for relative_path in benchmark_whisper.PARAKEET_HF_REQUIRED_MODEL_FILES[:-1]:
                (model_path / relative_path).write_bytes(b"model")

            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_parakeet_hf_model_path(model_path)

    def test_parakeet_sherpa_resolver_validates_selected_exact_variant_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir()
            for relative_path in benchmark_whisper.PARAKEET_SHERPA_REQUIRED_MODEL_FILES[
                "int8"
            ]:
                (model_path / relative_path).write_bytes(b"model")

            resolved = benchmark_whisper.resolve_parakeet_sherpa_model_path(
                benchmark_whisper.PARAKEET_SHERPA_INT8_MODEL_VARIANT, model_path
            )
            self.assertEqual(resolved, model_path.resolve())

            with self.assertRaises(FileNotFoundError):
                benchmark_whisper.resolve_parakeet_sherpa_model_path(
                    benchmark_whisper.PARAKEET_SHERPA_FP32_MODEL_VARIANT, model_path
                )

    def test_parakeet_sessions_resolve_and_store_local_model_paths(self) -> None:
        hf_model_path = Path("/models/parakeet-hf")
        sherpa_model_path = Path("/models/parakeet-sherpa")
        with (
            mock.patch.object(
                benchmark_whisper,
                "resolve_parakeet_hf_model_path",
                return_value=hf_model_path,
            ) as resolve_hf,
            mock.patch.object(
                benchmark_whisper,
                "resolve_parakeet_sherpa_model_path",
                return_value=sherpa_model_path,
            ) as resolve_sherpa,
        ):
            hf_session = benchmark_whisper.load_backend_session(
                "parakeet-hf",
                benchmark_whisper.PARAKEET_HF_MODEL_VARIANT,
                argparse.Namespace(parakeet_hf_model_path=None),
            )
            sherpa_session = benchmark_whisper.load_backend_session(
                "parakeet-sherpa",
                benchmark_whisper.PARAKEET_SHERPA_FP32_MODEL_VARIANT,
                argparse.Namespace(parakeet_sherpa_model_path=None),
            )

        resolve_hf.assert_called_once_with(None)
        resolve_sherpa.assert_called_once_with(
            benchmark_whisper.PARAKEET_SHERPA_FP32_MODEL_VARIANT, None
        )
        self.assertEqual(hf_session.device, "subprocess")
        self.assertEqual(hf_session.session, {"model_path": hf_model_path})
        self.assertEqual(sherpa_session.device, "subprocess")
        self.assertEqual(sherpa_session.session, {"model_path": sherpa_model_path})

    @staticmethod
    def _execution(payload: dict[str, object], *, rss: float | None = 512.5) -> WorkerExecution:
        return WorkerExecution(
            status="ok",
            payload=payload,
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=rss,
            error=None,
        )

    def test_run_parakeet_hf_maps_success_request_language_and_metrics(self) -> None:
        args = argparse.Namespace(
            language="en",
            parakeet_hf_device="cpu",
            parakeet_hf_python=Path("/venv/bin/python"),
            reference_transcript_text="hello world",
            sample_label="audio",
            audio_duration_seconds=3.0,
            worker_timeout_seconds=12.5,
        )
        execution = self._execution(
            {
                "status": "ok",
                "transcript": "Hello, world!",
                "segments": [{"token": "Hello", "start": 0.0, "end": 0.5}],
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
                "device": "cpu",
            }
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_parakeet_hf(
                Path("audio.mp3"),
                benchmark_whisper.PARAKEET_HF_MODEL_VARIANT,
                2,
                args,
                Path("/models/parakeet-hf"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "parakeet-hf")
        self.assertEqual(result.model, benchmark_whisper.PARAKEET_HF_MODEL_VARIANT)
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.forced_language, "en")
        self.assertEqual(result.transcript, "Hello, world!")
        self.assertEqual(result.load_seconds, 1.25)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 512.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            "stt_benchmark.workers.parakeet_hf",
            {
                "model_path": "/models/parakeet-hf",
                "audio_path": "audio.mp3",
                "device": "cpu",
            },
            12.5,
            env=benchmark_whisper.PARAKEET_HF_OFFLINE_ENV,
        )

    def test_run_parakeet_sherpa_maps_success_variant_threads_and_language(self) -> None:
        args = argparse.Namespace(
            language="ru",
            parakeet_sherpa_python=Path("/venv/bin/python"),
            parakeet_sherpa_threads=7,
            reference_transcript_text="привет мир",
            sample_label="audio",
            audio_duration_seconds=3.0,
            worker_timeout_seconds=12.5,
        )
        execution = self._execution(
            {
                "status": "ok",
                "transcript": "Привет, мир!",
                "segments": [{"text": "▁привет", "start": 0.0}],
                "timestamps": [0.0],
                "timestamp_semantics": "token/frame starts",
                "quantization": "int8",
                "load_seconds": 1.25,
                "transcribe_seconds": 2.5,
                "device": "cpu",
            }
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_parakeet_sherpa(
                Path("audio.mp3"),
                benchmark_whisper.PARAKEET_SHERPA_INT8_MODEL_VARIANT,
                2,
                args,
                Path("/models/parakeet-sherpa"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "parakeet-sherpa")
        self.assertEqual(
            result.model, benchmark_whisper.PARAKEET_SHERPA_INT8_MODEL_VARIANT
        )
        self.assertEqual(result.backend_device, "subprocess")
        self.assertEqual(result.forced_language, "ru")
        self.assertEqual(result.transcript, "Привет, мир!")
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.peak_rss_mb, 512.5)
        run_worker.assert_called_once_with(
            Path("/venv/bin/python"),
            "stt_benchmark.workers.parakeet_sherpa",
            {
                "model_path": "/models/parakeet-sherpa",
                "audio_path": "audio.mp3",
                "quantization": "int8",
                "threads": 7,
            },
            12.5,
        )

    def test_parakeet_runs_map_timeout_and_worker_errors(self) -> None:
        args = argparse.Namespace(
            language="en",
            reference_transcript_text=None,
            sample_label="audio",
            audio_duration_seconds=3.0,
        )
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=64.0,
                error="worker timed out after 12.5 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "load failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", side_effect=executions
        ):
            timeout_result = benchmark_whisper.run_parakeet_hf(
                Path("audio.mp3"),
                benchmark_whisper.PARAKEET_HF_MODEL_VARIANT,
                1,
                args,
                Path("/models/parakeet-hf"),
            )
            worker_error_result = benchmark_whisper.run_parakeet_sherpa(
                Path("audio.mp3"),
                benchmark_whisper.PARAKEET_SHERPA_FP32_MODEL_VARIANT,
                2,
                args,
                Path("/models/parakeet-sherpa"),
            )

        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(timeout_result.peak_rss_mb, 64.0)
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("load failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

    def test_parakeet_metadata_records_runtime_paths_and_capabilities(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "parakeet"])
        metadata = benchmark_whisper.build_metadata(args, [])

        self.assertEqual(metadata["profile"], "parakeet")
        self.assertEqual(metadata["benchmark_pairs"], [list(pair) for pair in benchmark_whisper.PARAKEET_BENCHMARK_PAIRS])
        self.assertEqual(
            metadata["parakeet_hf_python"], ".venvs/parakeet-hf/bin/python"
        )
        self.assertEqual(metadata["parakeet_hf_model_path"], None)
        self.assertEqual(metadata["parakeet_hf_device"], "auto")
        self.assertEqual(
            metadata["parakeet_hf_model_cache_root"],
            "/Volumes/512GB/hf/hub/models--nvidia--parakeet-tdt-0.6b-v3",
        )
        self.assertTrue(metadata["parakeet_hf_multilingual"])
        self.assertEqual(metadata["parakeet_hf_supported_tasks"], ["transcribe"])
        self.assertTrue(metadata["parakeet_hf_segment_timestamps"])
        self.assertEqual(metadata["parakeet_hf_timestamp_semantics"], "token timestamps")
        self.assertEqual(
            metadata["parakeet_hf_offline_env"], benchmark_whisper.PARAKEET_HF_OFFLINE_ENV
        )
        self.assertEqual(
            metadata["parakeet_sherpa_python"], ".venvs/parakeet-sherpa/bin/python"
        )
        self.assertEqual(metadata["parakeet_sherpa_model_path"], None)
        self.assertEqual(metadata["parakeet_sherpa_threads"], 4)
        self.assertEqual(
            metadata["parakeet_sherpa_model_variants"],
            [
                "parakeet-tdt-0.6b-v3-sherpa-fp32",
                "parakeet-tdt-0.6b-v3-sherpa-int8",
            ],
        )
        self.assertTrue(metadata["parakeet_sherpa_multilingual"])
        self.assertEqual(metadata["parakeet_sherpa_supported_tasks"], ["transcribe"])
        self.assertTrue(metadata["parakeet_sherpa_segment_timestamps"])
        self.assertEqual(
            metadata["parakeet_sherpa_timestamp_semantics"], "token/frame starts"
        )


class WhisperCppBenchmarkTests(unittest.TestCase):
    def test_whisper_cpp_profile_has_exact_en_ru_six_pair_matrix(self) -> None:
        args = benchmark_whisper.parse_args(["--profile", "whisper-cpp"])

        self.assertEqual(args.audios, [])
        self.assertEqual(args.models, list(benchmark_whisper.WHISPER_CPP_MODEL_FILES))
        self.assertEqual(args.backends, ["whisper-cpp"])
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(args),
            [("whisper-cpp", model) for model in args.models],
        )
        with (
            mock.patch.object(
                benchmark_whisper,
                "ensure_audio_file",
                side_effect=lambda path: path.resolve(),
            ),
            mock.patch.object(
                benchmark_whisper,
                "load_reference_transcript",
                side_effect=lambda path: f"normalized:{path.name}",
            ),
            mock.patch.object(
                benchmark_whisper,
                "get_audio_duration_seconds",
                return_value=1.0,
            ),
        ):
            audio_inputs = benchmark_whisper.resolve_audio_inputs(args)
        self.assertEqual([item.forced_language for item in audio_inputs], ["en", "ru"])

    def test_whisper_cpp_options_use_cli_over_environment_and_default_threads(self) -> None:
        with mock.patch.dict(
            "os.environ", {"WHISPER_CPP_EXECUTABLE": "/env/whisper-cli"}, clear=True
        ):
            env_args = benchmark_whisper.parse_args(["--profile", "whisper-cpp"])
            cli_args = benchmark_whisper.parse_args(
                [
                    "--profile",
                    "whisper-cpp",
                    "--whisper-cpp-executable",
                    "/cli/whisper-cli",
                    "--whisper-cpp-threads",
                    "4",
                ]
            )

        self.assertEqual(env_args.whisper_cpp_executable, Path("/env/whisper-cli"))
        self.assertEqual(env_args.whisper_cpp_threads, 8)
        self.assertEqual(cli_args.whisper_cpp_executable, Path("/cli/whisper-cli"))
        self.assertEqual(cli_args.whisper_cpp_threads, 4)

    def test_whisper_cpp_resolver_uses_main_snapshot_and_selected_nonempty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--ggerganov--whisper.cpp"
            revision = "a" * 40
            snapshot = cache_root / "snapshots" / revision
            (cache_root / "refs").mkdir(parents=True)
            snapshot.mkdir(parents=True)
            (cache_root / "refs" / "main").write_text(
                f"{revision}\n", encoding="utf-8"
            )
            for filename in benchmark_whisper.WHISPER_CPP_MODEL_FILES.values():
                (snapshot / filename).write_bytes(b"model")

            with mock.patch.object(
                benchmark_whisper, "WHISPER_CPP_MODEL_CACHE_ROOT", cache_root
            ):
                resolved = benchmark_whisper.resolve_whisper_cpp_model_path(
                    "whisper-cpp-tiny-q5_1"
                )

        self.assertEqual(
            resolved, (snapshot / "ggml-tiny-q5_1.bin").resolve()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            revision = "b" * 40
            snapshot = cache_root / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "ggml-tiny.bin").touch()
            (cache_root / "refs").mkdir()
            (cache_root / "refs" / "main").write_text(revision)
            with mock.patch.object(
                benchmark_whisper,
                "WHISPER_CPP_MODEL_CACHE_ROOT",
                cache_root,
            ):
                with self.assertRaises(FileNotFoundError):
                    benchmark_whisper.resolve_whisper_cpp_model_path(
                        "whisper-cpp-tiny-fp16"
                    )

    def test_whisper_cpp_session_resolves_one_selected_model_file(self) -> None:
        model_path = Path("/models/ggml-tiny.bin")
        with mock.patch.object(
            benchmark_whisper,
            "resolve_whisper_cpp_model_path",
            return_value=model_path,
        ) as resolve:
            session = benchmark_whisper.load_backend_session(
                "whisper-cpp",
                "whisper-cpp-tiny-fp16",
                argparse.Namespace(),
            )

        resolve.assert_called_once_with("whisper-cpp-tiny-fp16")
        self.assertEqual(session.device, "subprocess/Metal")
        self.assertEqual(session.session, {"model_path": model_path})
        self.assertIsNone(session.load_seconds)

    def _whisper_cpp_args(self, language: str | None = "ru") -> argparse.Namespace:
        return argparse.Namespace(
            language=language,
            task="translate",
            beam_size=3,
            whisper_cpp_executable=Path("/cli/whisper-cli"),
            whisper_cpp_threads=4,
            worker_timeout_seconds=12.5,
            reference_transcript_text="привет мир",
            sample_label="audio",
            audio_duration_seconds=3.0,
        )

    @staticmethod
    def _execution(payload: dict[str, object], *, rss: float | None = 512.5) -> WorkerExecution:
        return WorkerExecution(
            status="ok",
            payload=payload,
            returncode=0,
            stdout="{}",
            stderr="",
            wall_seconds=4.0,
            peak_rss_mb=rss,
            error=None,
        )

    def test_run_whisper_cpp_maps_success_request_timing_rss_and_scores(self) -> None:
        args = self._whisper_cpp_args()
        execution = self._execution(
            {
                "status": "ok",
                "transcript": "Привет, мир!",
                "transcribe_seconds": 2.5,
                "language": "ru",
                "segments": [{"start": 0.0, "end": 0.5, "text": "Привет"}],
            }
        )

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", return_value=execution
        ) as run_worker:
            result = benchmark_whisper.run_whisper_cpp(
                Path("audio.mp3"),
                "whisper-cpp-tiny-q5_1",
                2,
                args,
                Path("/models/ggml-tiny-q5_1.bin"),
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.backend, "whisper-cpp")
        self.assertEqual(result.model, "whisper-cpp-tiny-q5_1")
        self.assertEqual(result.backend_device, "subprocess/Metal")
        self.assertEqual(result.load_seconds, None)
        self.assertEqual(result.transcribe_seconds, 2.5)
        self.assertEqual(result.total_seconds, 4.0)
        self.assertEqual(result.wer, 0.0)
        self.assertEqual(result.cer, 0.0)
        self.assertEqual(result.peak_rss_mb, 512.5)
        run_worker.assert_called_once_with(
            sys.executable,
            benchmark_whisper.WHISPER_CPP_WORKER_MODULE,
            {
                "executable": "/cli/whisper-cli",
                "model_path": "/models/ggml-tiny-q5_1.bin",
                "audio_path": "audio.mp3",
                "task": "translate",
                "beam_size": 3,
                "threads": 4,
                "timeout_seconds": 12.5,
                "language": "ru",
            },
            12.5,
        )

    def test_run_whisper_cpp_omits_auto_language_and_maps_worker_errors(self) -> None:
        args = self._whisper_cpp_args(language="auto")
        executions = [
            WorkerExecution(
                status="timeout",
                payload=None,
                returncode=None,
                stdout="",
                stderr="",
                wall_seconds=1.0,
                peak_rss_mb=64.0,
                error="worker timed out after 12.5 seconds",
            ),
            WorkerExecution(
                status="worker_error",
                payload={"status": "error", "error": "whisper-cli failed"},
                returncode=1,
                stdout="{}",
                stderr="",
                wall_seconds=0.5,
                peak_rss_mb=99.0,
                error="worker exited with return code 1",
            ),
        ]

        with mock.patch.object(
            benchmark_whisper, "run_json_worker", side_effect=executions
        ) as run_worker:
            timeout_result = benchmark_whisper.run_whisper_cpp(
                Path("audio.mp3"),
                "whisper-cpp-tiny-fp16",
                1,
                args,
                Path("/model.bin"),
            )
            worker_error_result = benchmark_whisper.run_whisper_cpp(
                Path("audio.mp3"),
                "whisper-cpp-tiny-fp16",
                2,
                args,
                Path("/model.bin"),
            )

        self.assertNotIn("language", run_worker.call_args_list[0].args[2])
        self.assertEqual(timeout_result.status, "error")
        self.assertIn("timed out", timeout_result.error or "")
        self.assertEqual(timeout_result.peak_rss_mb, 64.0)
        self.assertEqual(worker_error_result.status, "error")
        self.assertIn("whisper-cli failed", worker_error_result.error or "")
        self.assertEqual(worker_error_result.peak_rss_mb, 99.0)

    def test_whisper_cpp_metadata_and_main_profile_remain_separate(self) -> None:
        main_args = benchmark_whisper.parse_args([])
        self.assertEqual(main_args.profile, "main")
        self.assertNotIn("whisper-cpp", main_args.backends)
        self.assertEqual(
            benchmark_whisper.iter_benchmark_pairs(main_args),
            benchmark_whisper.MAIN_BENCHMARK_PAIRS,
        )

        args = benchmark_whisper.parse_args(["--profile", "whisper-cpp"])
        metadata = benchmark_whisper.build_metadata(args, [])
        self.assertEqual(metadata["whisper_cpp_python"], sys.executable)
        self.assertEqual(
            metadata["whisper_cpp_executable"], "/opt/homebrew/bin/whisper-cli"
        )
        self.assertEqual(metadata["whisper_cpp_threads"], 8)
        self.assertEqual(
            metadata["whisper_cpp_model_cache_root"],
            "/Volumes/512GB/hf/hub/models--ggerganov--whisper.cpp",
        )
        self.assertEqual(metadata["whisper_cpp_model_files"], benchmark_whisper.WHISPER_CPP_MODEL_FILES)
        self.assertEqual(metadata["whisper_cpp_multilingual"], True)
        self.assertEqual(
            metadata["whisper_cpp_supported_tasks"], ["transcribe", "translate"]
        )
        self.assertEqual(metadata["whisper_cpp_segment_timestamps"], True)
        self.assertEqual(
            metadata["whisper_cpp_supports_condition_on_previous_text"], False
        )
        self.assertEqual(
            metadata["whisper_cpp_supports_hallucination_silence_threshold"], False
        )
        self.assertEqual(metadata["whisper_cpp_device"], "subprocess/Metal")


class DownloadModelsCliTests(unittest.TestCase):
    def test_download_models_parse_models(self) -> None:
        with mock.patch(
            "sys.argv",
            ["download_models.py", "--mlx-whisper", "--models", "tiny", "large-v3"],
        ):
            args = download_models.parse_args()
        self.assertEqual(args.models, ["tiny", "large-v3"])
        self.assertTrue(args.mlx_whisper)

    def test_download_models_parse_args_accepts_explicit_argv(self) -> None:
        args = download_models.parse_args(["--mlx-whisper", "--models", "tiny"])
        self.assertEqual(args.models, ["tiny"])
        self.assertTrue(args.mlx_whisper)


class PrepareSamplesCliTests(unittest.TestCase):
    def test_prepare_samples_parse_args_accepts_explicit_argv(self) -> None:
        args = prepare_samples.parse_args(["--lang", "ru", "--target-duration", "300"])
        self.assertEqual(args.lang, "ru")
        self.assertEqual(args.target_duration, 300)


class SmokeTestCliTests(unittest.TestCase):
    def test_smoke_test_parse_args_accepts_explicit_argv(self) -> None:
        args = smoke_test.parse_args(
            ["--backend", "openai-whisper", "--model", "small"]
        )
        self.assertEqual(args.backend, "openai-whisper")
        self.assertEqual(args.model, "small")

    def test_smoke_test_delegates_to_benchmark_main(self) -> None:
        with mock.patch.object(
            benchmark_whisper, "main", return_value=0
        ) as benchmark_main:
            exit_code = smoke_test.main(
                [
                    "--audio",
                    "ru",
                    "--backend",
                    "mlx-whisper",
                    "--model",
                    "tiny",
                    "--output",
                    "out.json",
                ]
            )

        self.assertEqual(exit_code, 0)
        benchmark_main.assert_called_once_with(
            [
                "--audio",
                "ru",
                "--backends",
                "mlx-whisper",
                "--models",
                "tiny",
                "--runs",
                "1",
                "--output",
                "out.json",
            ]
        )


class UnifiedCliTests(unittest.TestCase):
    def test_main_without_command_returns_error(self) -> None:
        self.assertEqual(stt_cli.main([]), 2)

    def test_resolve_command_main_returns_current_callable(self) -> None:
        self.assertIs(stt_cli.resolve_command_main("benchmark"), benchmark_whisper.main)

    def test_ensure_workspace_root_first_moves_workspace_ahead_of_site_packages(
        self,
    ) -> None:
        workspace_root = str(stt_cli.WORKSPACE_ROOT)
        fake_sys_path = [
            "/tmp/venv/bin",
            "/tmp/venv/lib/pythonX.Y/site-packages",
            workspace_root,
            "/tmp/other",
        ]

        with mock.patch.object(stt_cli.sys, "path", fake_sys_path):
            stt_cli.ensure_workspace_root_first()

        self.assertEqual(stt_cli.sys.path[0], workspace_root)
        self.assertEqual(stt_cli.sys.path.count(workspace_root), 1)

    def test_cli_dispatches_benchmark_subcommand(self) -> None:
        with mock.patch.object(
            benchmark_whisper, "main", return_value=0
        ) as command_main:
            exit_code = stt_cli.main(["benchmark", "--audio", "en", "--runs", "2"])

        self.assertEqual(exit_code, 0)
        command_main.assert_called_once_with(["--audio", "en", "--runs", "2"])

    def test_cli_forwards_subcommand_help(self) -> None:
        with mock.patch.object(
            benchmark_whisper, "main", return_value=0
        ) as command_main:
            exit_code = stt_cli.main(["benchmark", "--help"])

        self.assertEqual(exit_code, 0)
        command_main.assert_called_once_with(["--help"])

    def test_cli_dispatches_download_models_subcommand(self) -> None:
        with mock.patch.object(download_models, "main", return_value=0) as command_main:
            exit_code = stt_cli.main(["download-models", "--mlx-whisper"])

        self.assertEqual(exit_code, 0)
        command_main.assert_called_once_with(["--mlx-whisper"])

    def test_cli_dispatches_prepare_samples_subcommand(self) -> None:
        with mock.patch.object(prepare_samples, "main", return_value=0) as command_main:
            exit_code = stt_cli.main(["prepare-samples", "--lang", "en"])

        self.assertEqual(exit_code, 0)
        command_main.assert_called_once_with(["--lang", "en"])

    def test_cli_dispatches_smoke_test_subcommand(self) -> None:
        with mock.patch.object(smoke_test, "main", return_value=0) as command_main:
            exit_code = stt_cli.main(["smoke-test", "--backend", "mlx-whisper"])

        self.assertEqual(exit_code, 0)
        command_main.assert_called_once_with(["--backend", "mlx-whisper"])


class ParityTests(unittest.TestCase):
    def test_default_models_match_downloadable_model_sets(self) -> None:
        default_models = set(benchmark_whisper.DEFAULT_MODELS)
        self.assertEqual(default_models, set(download_models.FASTER_WHISPER_REPOS))
        self.assertEqual(default_models, set(download_models.MLX_WHISPER_REPOS))
        self.assertEqual(default_models, set(benchmark_whisper.MLX_AUDIO_WHISPER_REPOS))
        self.assertEqual(
            default_models, set(benchmark_whisper.INSANELY_FAST_WHISPER_REPOS)
        )
        self.assertEqual(default_models, set(download_models.OPENAI_WHISPER_MODELS))

    def test_backend_capabilities_match_repo_maps_for_supported_backends(self) -> None:
        self.assertEqual(
            benchmark_whisper.BACKEND_CAPABILITIES["mlx-whisper"].supported_models,
            set(benchmark_whisper.MLX_WHISPER_REPOS),
        )
        self.assertEqual(
            benchmark_whisper.BACKEND_CAPABILITIES["mlx-audio"].supported_models,
            set(benchmark_whisper.MLX_AUDIO_WHISPER_REPOS),
        )
        self.assertEqual(
            benchmark_whisper.BACKEND_CAPABILITIES[
                "lightning-whisper-mlx"
            ].supported_models,
            set(benchmark_whisper.LIGHTNING_WHISPER_MLX_REPOS),
        )
        self.assertEqual(
            benchmark_whisper.BACKEND_CAPABILITIES[
                "insanely-fast-whisper"
            ].supported_models,
            set(benchmark_whisper.INSANELY_FAST_WHISPER_REPOS),
        )
        self.assertEqual(
            benchmark_whisper.BACKEND_CAPABILITIES["openai-whisper"].supported_models,
            set(benchmark_whisper.OPENAI_WHISPER_REPOS),
        )
        self.assertEqual(
            benchmark_whisper.BACKEND_CAPABILITIES["t-one"].supported_models,
            {"t-one-greedy"},
        )
        self.assertEqual(
            benchmark_whisper.BACKEND_CAPABILITIES["vosk"].supported_models,
            {"vosk-ru", "vosk-small-ru"},
        )
        self.assertEqual(
            benchmark_whisper.BACKEND_CAPABILITIES["qwen3-asr"].supported_models,
            {"qwen3-asr-0.6b-8bit", "qwen3-asr-1.7b-8bit"},
        )

    def test_downloader_imported_repo_maps_match_benchmark_repo_maps(self) -> None:
        self.assertEqual(
            download_models.MLX_WHISPER_REPOS,
            benchmark_whisper.MLX_WHISPER_REPOS,
        )
        self.assertEqual(
            download_models.MLX_AUDIO_WHISPER_REPOS,
            benchmark_whisper.MLX_AUDIO_WHISPER_REPOS,
        )
        self.assertEqual(
            download_models.INSANELY_FAST_WHISPER_REPOS,
            benchmark_whisper.INSANELY_FAST_WHISPER_REPOS,
        )
        self.assertEqual(
            download_models.LIGHTNING_WHISPER_MLX_REPOS,
            benchmark_whisper.LIGHTNING_WHISPER_MLX_REPOS,
        )


if __name__ == "__main__":
    unittest.main()
