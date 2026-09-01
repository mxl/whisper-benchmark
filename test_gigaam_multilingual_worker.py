import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stt_benchmark.workers import gigaam_multilingual


class FakeSoundfile:
    def __init__(self, audio, sample_rate=16_000) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.read_calls = []
        self.write_calls = []

    def read(self, path, *, dtype, always_2d):
        self.read_calls.append((path, dtype, always_2d))
        return self.audio, self.sample_rate

    def write(self, path, audio, sample_rate):
        self.write_calls.append((path, audio, sample_rate))


class GigaAMMultilingualWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "model"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text("{}")
        (self.model_path / "modeling_gigaam.py").write_text("# fake")
        (self.model_path / "pytorch_model.bin").write_bytes(b"model")
        self.audio_path = root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_validation_rejects_paths_model_files_variant_language_and_options(self) -> None:
        loader = mock.Mock()
        cases = [
            {"model_path": "https://example.invalid/model"},
            self.request | {"audio_path": "file:///tmp/audio.wav"},
            self.request | {"variant": "ssl"},
            self.request | {"language": "de"},
            self.request | {"language": 7},
            self.request | {"chunk_seconds": 0},
            self.request | {"chunk_seconds": 30},
            self.request | {"chunk_seconds": "25"},
            self.request | {"chunk_seconds": True},
            self.request | {"chunk_seconds": float("nan")},
        ]
        for request in cases:
            with self.subTest(request=request):
                result = gigaam_multilingual.execute(request, model_loader=loader)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error_type"], "validation_error")
        loader.assert_not_called()

        missing_weights = Path(self.tempdir.name) / "missing-weights"
        missing_weights.mkdir()
        (missing_weights / "config.json").write_text("{}")
        result = gigaam_multilingual.execute(
            self.request | {"model_path": str(missing_weights)}, model_loader=loader
        )
        self.assertEqual(result["error_type"], "validation_error")
        self.assertIn("model weights", result["error"])
        loader.assert_not_called()

    def test_validation_happens_before_transformers_import(self) -> None:
        with mock.patch.object(
            gigaam_multilingual,
            "_load_model",
            side_effect=AssertionError("transformers imported during validation"),
        ) as loader:
            result = gigaam_multilingual.execute(self.request | {"language": "de"})

        self.assertEqual(result["error_type"], "validation_error")
        loader.assert_not_called()

    def test_official_loader_uses_local_snapshot_and_no_revision_or_language(self) -> None:
        auto_model = mock.Mock()
        transformers = SimpleNamespace(AutoModel=auto_model)
        with mock.patch.dict("sys.modules", {"transformers": transformers}):
            gigaam_multilingual._load_model(str(self.model_path))

        auto_model.from_pretrained.assert_called_once_with(
            str(self.model_path), trust_remote_code=True, local_files_only=True
        )

    def test_short_transcription_uses_text_and_defaults(self) -> None:
        model = mock.Mock()
        model.transcribe.return_value = SimpleNamespace(text="Привет")
        clock_values = iter([10.0, 12.5, 20.0, 21.25])

        with mock.patch.dict(os.environ, {}, clear=True):
            result = gigaam_multilingual.execute(
                self.request,
                model_loader=lambda path: model,
                clock=lambda: next(clock_values),
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "Привет")
        self.assertEqual(result["load_seconds"], 2.5)
        self.assertEqual(result["transcribe_seconds"], 1.25)
        self.assertEqual(result["language"], "ru")
        self.assertEqual(result["variant"], "large_ctc")
        self.assertEqual(result["transcription_mode"], "short")
        model.transcribe.assert_called_once_with(self.request["audio_path"])
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(
            result["effective_config"]["chunk_seconds"],
            gigaam_multilingual.DEFAULT_CHUNK_SECONDS,
        )
        self.assertEqual(result["effective_config"]["chunk_overlap_seconds"], 0.0)
        self.assertTrue(result["effective_config"]["trust_remote_code"])

    def test_supported_language_and_variant_are_metadata_only(self) -> None:
        model = mock.Mock()
        model.transcribe.return_value = SimpleNamespace(text="hello")
        result = gigaam_multilingual.execute(
            self.request | {"language": "en", "variant": "ctc"},
            model_loader=lambda path: model,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["variant"], "ctc")
        model.transcribe.assert_called_once_with(self.request["audio_path"])
        self.assertNotIn("language", model.transcribe.call_args.kwargs)

    def test_exact_too_long_error_uses_ordered_fixed_chunks_without_longform(self) -> None:
        audio = [0.0] * (16_000 * 2 + 1)
        soundfile = FakeSoundfile(audio)
        audio_path = self.request["audio_path"]

        class Model:
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.transcribe_longform = mock.Mock(
                    side_effect=AssertionError("gated longform must not be called")
                )

            def transcribe(self, path: str):
                self.calls.append(path)
                if path == audio_path:
                    raise ValueError(
                        "Too long wav file, use 'transcribe_longform' method."
                    )
                return SimpleNamespace(
                    text={
                        "chunk_0000.wav": " first ",
                        "chunk_0001.wav": " ",
                        "chunk_0002.wav": "third",
                    }[Path(path).name]
                )

        model = Model()
        result = gigaam_multilingual.execute(
            self.request | {"chunk_seconds": 1},
            model_loader=lambda path: model,
            soundfile_module=soundfile,
            clock=iter([1.0, 2.0, 3.0, 4.0]).__next__,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "first third")
        self.assertEqual(result["transcription_mode"], "offline_fixed_chunks")
        self.assertEqual(result["effective_config"]["chunk_seconds"], 1.0)
        self.assertEqual(result["effective_config"]["chunk_overlap_seconds"], 0.0)
        self.assertEqual(
            [Path(path).name for path in model.calls[1:]],
            ["chunk_0000.wav", "chunk_0001.wav", "chunk_0002.wav"],
        )
        self.assertEqual(
            soundfile.read_calls,
            [(self.request["audio_path"], "float32", False)],
        )
        self.assertEqual(
            [Path(path).name for path, _, _ in soundfile.write_calls],
            ["chunk_0000.wav", "chunk_0001.wav", "chunk_0002.wav"],
        )
        self.assertEqual([rate for _, _, rate in soundfile.write_calls], [16_000] * 3)
        model.transcribe_longform.assert_not_called()

    def test_unrelated_error_does_not_fall_back_or_call_longform(self) -> None:
        model = mock.Mock()
        model.transcribe.side_effect = ValueError("audio decode failed")
        model.transcribe_longform = mock.Mock()

        result = gigaam_multilingual.execute(
            self.request,
            model_loader=lambda path: model,
            soundfile_module=mock.Mock(),
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("audio decode failed", result["error"])
        model.transcribe.assert_called_once_with(self.request["audio_path"])
        model.transcribe_longform.assert_not_called()

    def test_loader_and_transcription_failures_are_error_payloads(self) -> None:
        result = gigaam_multilingual.execute(
            self.request,
            model_loader=lambda path: (_ for _ in ()).throw(RuntimeError("load failed")),
        )
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        model = mock.Mock()
        model.transcribe.side_effect = RuntimeError("transcribe failed")
        result = gigaam_multilingual.execute(
            self.request, model_loader=lambda path: model
        )
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("transcribe failed", result["error"])

    def test_cli_emits_one_json_object_and_returns_status_code(self) -> None:
        success = {
            "status": "ok",
            "transcript": "ok",
            "load_seconds": 0.1,
            "transcribe_seconds": 0.2,
            "language": "ru",
            "variant": "large_ctc",
            "transcription_mode": "short",
            "effective_config": {},
        }
        output = io.StringIO()
        with (
            mock.patch.object(gigaam_multilingual, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(json.dumps(self.request))),
            contextlib.redirect_stdout(output),
        ):
            return_code = gigaam_multilingual.main()

        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = gigaam_multilingual.main()

        self.assertNotEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
