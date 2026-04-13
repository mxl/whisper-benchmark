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
            csv_output=None,
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
                return_value=(output_path, None),
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
                "BACKEND_SUPPORTED_MODELS",
                {"lightning-whisper-mlx": {"tiny"}},
            ),
            mock.patch.object(benchmark_whisper, "write_json"),
            mock.patch.object(benchmark_whisper, "print_summary"),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = benchmark_whisper.main()

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "Skipping lightning-whisper-mlx on model large-v3-turbo (not supported).",
            stderr.getvalue(),
        )


class DownloadModelsCliTests(unittest.TestCase):
    def test_download_models_parse_models(self) -> None:
        with mock.patch(
            "sys.argv",
            ["download_models.py", "--mlx-whisper", "--models", "tiny", "large-v3"],
        ):
            args = download_models.parse_args()
        self.assertEqual(args.models, ["tiny", "large-v3"])
        self.assertTrue(args.mlx_whisper)


if __name__ == "__main__":
    unittest.main()
