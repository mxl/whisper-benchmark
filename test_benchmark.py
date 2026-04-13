import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark_whisper
import download_models
import prepare_samples
import smoke_test
from stt_benchmark import cli as stt_cli


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
            audios=["en", "ru"],
            models=["tiny", "large-v3"],
            backends=["mlx-whisper", "openai-whisper"],
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
        self.assertEqual(len(metadata["audios"]), 2)
        self.assertEqual(metadata["audios"][0]["audio"], "samples/en.mp3")
        self.assertEqual(metadata["audios"][0]["forced_language"], "en")
        self.assertEqual(metadata["models"], ["tiny", "large-v3"])
        self.assertEqual(metadata["backends"], ["mlx-whisper", "openai-whisper"])
        self.assertEqual(metadata["runs"], 2)
        self.assertEqual(metadata["task"], "transcribe")
        self.assertEqual(metadata["condition_on_previous_text"], False)
        self.assertEqual(metadata["hallucination_silence_threshold"], 2.0)
        self.assertEqual(metadata["warmup"], True)
        self.assertEqual(metadata["show_full_table"], False)
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

    def test_insanely_fast_whisper_does_not_request_timestamps(self) -> None:
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
                        "condition_on_prev_tokens": True,
                    },
                },
                load_seconds=0.5,
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(pipe_calls), 1)
        self.assertEqual(pipe_calls[0]["audio"], "audio.mp3")
        self.assertNotIn("return_timestamps", pipe_calls[0])
        self.assertEqual(pipe_calls[0]["return_language"], True)


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
