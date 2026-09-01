import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stt_benchmark.workers import gigaam_multilingual_mlx


class FakeMLX:
    def __init__(self) -> None:
        self.array_calls = []
        self.eval_calls = []

    def array(self, value):
        self.array_calls.append(value)
        return FakeArray(value)

    def eval(self, *values):
        self.eval_calls.append(values)


class FakeArray:
    def __init__(self, value) -> None:
        self.value = value

    def __getitem__(self, key):
        return self


class GigaAMMultilingualMLXWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "model"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text("{}")
        (self.model_path / "manifest.json").write_text("{}")
        (self.model_path / "model.safetensors").write_bytes(b"model")
        self.audio_path = root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_validation_rejects_non_local_paths_incomplete_model_and_options(self):
        loader = mock.Mock()
        cases = [
            self.request | {"model_path": "https://example.invalid/model"},
            self.request | {"audio_path": "file:///tmp/audio.wav"},
            self.request | {"variant": "int8"},
            self.request | {"language": "de"},
            self.request | {"chunk_seconds": 0},
            self.request | {"overlap_seconds": 20},
            self.request | {"overlap_seconds": -1},
            self.request | {"chunk_seconds": float("nan")},
        ]
        for request in cases:
            with self.subTest(request=request):
                result = gigaam_multilingual_mlx.execute(request, model_loader=loader)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error_type"], "validation_error")
        loader.assert_not_called()

        incomplete = Path(self.tempdir.name) / "incomplete"
        incomplete.mkdir()
        (incomplete / "config.json").write_text("{}")
        result = gigaam_multilingual_mlx.execute(
            self.request | {"model_path": str(incomplete)}, model_loader=loader
        )
        self.assertEqual(result["error_type"], "validation_error")
        self.assertIn("manifest.json", result["error"])
        loader.assert_not_called()

    def test_official_loader_receives_exact_local_path_and_offline_flag(self):
        factory = mock.Mock(return_value="model")
        fake_package = types.ModuleType("gigaam_multilingual_mlx")
        fake_package.load_model = factory
        with mock.patch.dict(sys.modules, {"gigaam_multilingual_mlx": fake_package}):
            result = gigaam_multilingual_mlx._load_model(str(self.model_path))

        self.assertEqual(result, "model")
        factory.assert_called_once_with(str(self.model_path), local_files_only=True)

    def test_official_audio_loader_requests_16khz(self):
        loader = mock.Mock(return_value=[0.0, 0.1])
        fake_audio = types.ModuleType("gigaam_multilingual_mlx.audio")
        fake_audio.load_audio = loader
        with mock.patch.dict(
            sys.modules,
            {
                "gigaam_multilingual_mlx": types.ModuleType("gigaam_multilingual_mlx"),
                "gigaam_multilingual_mlx.audio": fake_audio,
            },
        ):
            result = gigaam_multilingual_mlx._load_audio(str(self.audio_path))

        self.assertEqual(result, [0.0, 0.1])
        loader.assert_called_once_with(str(self.audio_path), sample_rate=16_000)

    def test_greedy_ctc_transcription_is_deterministic_and_returns_word_timestamps(self):
        class Model:
            config = SimpleNamespace(
                dtype="float16",
                sample_rate=16_000,
                vocabulary=[" ", "a", "b"],
            )

            def __init__(self):
                self.calls = []
                self.decode_calls = []

            def __call__(self, audio, lengths):
                self.calls.append((audio, lengths))
                return "logits", [4]

            def greedy_decode(self, logits, lengths):
                self.decode_calls.append((logits, lengths))
                return [
                    {
                        "text": "ab",
                        "token_ids": [1, 2],
                        "token_frames": [0, 2],
                    }
                ]

        model = Model()
        mlx = FakeMLX()
        audio = [0.0] * 16_000
        result = gigaam_multilingual_mlx.execute(
            self.request | {"language": "kk", "chunk_seconds": 1, "overlap_seconds": 0},
            model_loader=lambda path: model,
            audio_loader=lambda path, sample_rate: audio,
            mlx_module=mlx,
            clock=iter([10.0, 12.5, 20.0, 21.25]).__next__,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "ab")
        self.assertEqual(result["language"], "kk")
        self.assertEqual(result["language_source"], "request_metadata")
        self.assertEqual(result["load_seconds"], 2.5)
        self.assertEqual(result["transcribe_seconds"], 1.25)
        self.assertEqual(
            result["timestamps"], [{"text": "ab", "start": 0.0, "end": 0.75}]
        )
        self.assertEqual(result["decoder"], "greedy_ctc")
        self.assertEqual(
            result["timestamp_semantics"],
            "approximate greedy-CTC word emission times",
        )
        self.assertEqual(result["effective_config"]["audio_dtype"], "float32")
        self.assertEqual(result["effective_config"]["audio_channels"], 1)
        self.assertTrue(result["effective_config"]["deterministic"])
        self.assertTrue(result["effective_config"]["local_files_only"])
        self.assertTrue(result["effective_config"]["offline"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.decode_calls), 1)
        self.assertEqual(len(mlx.eval_calls), 1)

    def test_text_only_decode_result_has_empty_timestamps(self):
        model = mock.Mock()
        model.config = SimpleNamespace(sample_rate=16_000, dtype="float16")
        model.return_value = ("logits", [2])
        model.greedy_decode.return_value = [{"text": "hello"}]

        result = gigaam_multilingual_mlx.execute(
            self.request,
            model_loader=lambda path: model,
            audio_loader=lambda path, sample_rate: [0.0] * 16_000,
            mlx_module=FakeMLX(),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "hello")
        self.assertEqual(result["timestamps"], [])
        self.assertFalse(result["effective_config"]["word_timestamps"])

    def test_explicit_word_timestamps_are_preserved_and_overlap_is_deterministic(self):
        class Model:
            config = SimpleNamespace(sample_rate=16_000, dtype="float16")

            def __call__(self, audio, lengths):
                return "logits", [10]

            def greedy_decode(self, logits, lengths):
                return [
                    {
                        "text": "word",
                        "words": [{"text": "word", "start": 0, "end": 0.5}],
                    }
                ]

        result = gigaam_multilingual_mlx.execute(
            self.request | {"chunk_seconds": 1, "overlap_seconds": 0.5},
            model_loader=lambda path: Model(),
            audio_loader=lambda path, sample_rate: [0.0] * (16_000 * 1 + 8_000),
            mlx_module=FakeMLX(),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "word word")
        self.assertEqual(
            result["timestamps"],
            [
                {"text": "word", "start": 0.0, "end": 0.5},
                {"text": "word", "start": 0.5, "end": 1.0},
            ],
        )
        self.assertEqual(len(result["chunks"]), 2)

    def test_load_and_transcription_failures_are_structured(self):
        result = gigaam_multilingual_mlx.execute(
            self.request,
            model_loader=lambda path: (_ for _ in ()).throw(RuntimeError("load failed")),
        )
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        result = gigaam_multilingual_mlx.execute(
            self.request,
            model_loader=lambda path: object(),
            audio_loader=lambda path, sample_rate: (_ for _ in ()).throw(
                RuntimeError("decode failed")
            ),
        )
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("decode failed", result["error"])

    def test_cli_emits_one_json_object_and_returns_status_code(self):
        success = {
            "status": "ok",
            "transcript": "ok",
            "timestamps": [],
            "chunks": [],
            "duration_seconds": 1.0,
            "language": "auto",
            "load_seconds": 0.1,
            "transcribe_seconds": 0.2,
            "effective_config": {},
        }
        output = io.StringIO()
        with (
            mock.patch.object(gigaam_multilingual_mlx, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(json.dumps(self.request))),
            contextlib.redirect_stdout(output),
        ):
            return_code = gigaam_multilingual_mlx.main()

        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = gigaam_multilingual_mlx.main()
        self.assertNotEqual(return_code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
