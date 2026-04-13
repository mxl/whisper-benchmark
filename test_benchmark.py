import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import benchmark_whisper
import download_models


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
        )
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
            models=["tiny", "large-v3"],
            backends=["mlx-whisper", "openai-whisper"],
            runs=2,
            language="en",
            task="transcribe",
            reference_transcript=Path("reference.txt"),
            beam_size=5,
            compute_type="default",
            device="auto",
            faster_whisper_vad_filter=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            openai_whisper_temperature_fallback=True,
            mlx_prefix="mlx-community/whisper-",
            mlx_suffix="-mlx",
            lightning_whisper_mlx_batch_size=12,
            insanely_fast_whisper_device_id="mps",
            insanely_fast_whisper_batch_size=1,
            insanely_fast_whisper_flash=False,
            warmup=True,
        )

        metadata = benchmark_whisper.build_metadata(
            args=args,
            audio_path=Path("audio.mp3"),
            audio_duration_seconds=12.5,
        )

        self.assertEqual(metadata["audio"], "audio.mp3")
        self.assertEqual(metadata["audio_duration_seconds"], 12.5)
        self.assertEqual(metadata["models"], ["tiny", "large-v3"])
        self.assertEqual(metadata["backends"], ["mlx-whisper", "openai-whisper"])
        self.assertEqual(metadata["runs"], 2)
        self.assertEqual(metadata["language"], "en")
        self.assertEqual(metadata["task"], "transcribe")
        self.assertEqual(metadata["reference_transcript"], "reference.txt")
        self.assertEqual(metadata["condition_on_previous_text"], False)
        self.assertEqual(metadata["hallucination_silence_threshold"], 2.0)
        self.assertEqual(metadata["warmup"], True)
        self.assertIn("platform", metadata)
        self.assertIn("python_version", metadata)


class BenchmarkCliTests(unittest.TestCase):
    def test_boolean_optional_flags_parse(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "benchmark_whisper.py",
                "audio.mp3",
                "--no-faster-whisper-vad-filter",
                "--no-condition-on-previous-text",
                "--no-openai-whisper-temperature-fallback",
            ],
        ):
            args = benchmark_whisper.parse_args()
        self.assertFalse(args.faster_whisper_vad_filter)
        self.assertFalse(args.condition_on_previous_text)
        self.assertFalse(args.openai_whisper_temperature_fallback)

    def test_main_skips_unsupported_backend_model_combo(self) -> None:
        output_path = Path("/tmp/test-output.json")
        written_payload: dict[str, object] = {}
        fake_args = argparse.Namespace(
            audio=Path("audio.mp3"),
            models=["large-v3-turbo"],
            backends=["lightning-whisper-mlx"],
            runs=1,
            language="en",
            task="transcribe",
            beam_size=5,
            compute_type="default",
            faster_whisper_vad_filter=True,
            condition_on_previous_text=True,
            openai_whisper_temperature_fallback=True,
            hallucination_silence_threshold=2.0,
            device="auto",
            mlx_prefix="mlx-community/whisper-",
            mlx_suffix="-mlx",
            output=output_path,
            reference_transcript=None,
            warmup=False,
            insanely_fast_whisper_device_id="mps",
            insanely_fast_whisper_batch_size=1,
            insanely_fast_whisper_flash=False,
            lightning_whisper_mlx_batch_size=12,
        )

        stderr = io.StringIO()
        with (
            mock.patch.object(benchmark_whisper, "parse_args", return_value=fake_args),
            mock.patch.object(
                benchmark_whisper,
                "resolve_output_paths",
                return_value=output_path,
            ),
            mock.patch.object(
                benchmark_whisper, "ensure_audio_file", return_value=Path("audio.mp3")
            ),
            mock.patch.object(
                benchmark_whisper, "get_audio_duration_seconds", return_value=1.0
            ),
            mock.patch.object(
                benchmark_whisper, "load_reference_transcript", return_value=None
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
            mock.patch.object(benchmark_whisper, "print_summary"),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = benchmark_whisper.main()

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "Skipping lightning-whisper-mlx on model large-v3-turbo (not supported).",
            stderr.getvalue(),
        )
        self.assertEqual(
            written_payload["skipped"],
            [
                {
                    "backend": "lightning-whisper-mlx",
                    "model": "large-v3-turbo",
                    "reason": "not supported",
                }
            ],
        )
        self.assertEqual(written_payload["summary"], [])
        self.assertEqual(written_payload["runs"], [])


class DownloadModelsCliTests(unittest.TestCase):
    def test_download_models_parse_models(self) -> None:
        with mock.patch(
            "sys.argv",
            ["download_models.py", "--mlx-whisper", "--models", "tiny", "large-v3"],
        ):
            args = download_models.parse_args()
        self.assertEqual(args.models, ["tiny", "large-v3"])
        self.assertTrue(args.mlx_whisper)


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
