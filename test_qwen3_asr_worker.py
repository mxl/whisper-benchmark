import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from stt_benchmark.workers import qwen3_asr


@dataclass
class Segment:
    start: float
    end: float
    text: str
    language: str
    words: list[dict[str, object]] | None = None


class Qwen3ASRWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "model"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text("{}")
        self.audio_path = root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_validation_rejects_uris_missing_paths_empty_model_and_options(self) -> None:
        loader = mock.Mock()
        cases = [
            self.request | {"model_path": "https://example.invalid/model"},
            self.request | {"audio_path": "file:///tmp/audio.wav"},
            self.request | {"audio_path": "missing.wav"},
            self.request | {"language": ""},
            self.request | {"language": 7},
            self.request | {"max_tokens": 0},
            self.request | {"max_tokens": -1},
            self.request | {"max_tokens": 1.0},
            self.request | {"max_tokens": True},
            self.request | {"temperature": -0.01},
            self.request | {"temperature": "0"},
            self.request | {"temperature": float("nan")},
        ]
        for request in cases:
            with self.subTest(request=request):
                result = qwen3_asr.execute(request, model_loader=loader)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error_type"], "validation_error")
        loader.assert_not_called()

        empty_model = Path(self.tempdir.name) / "empty-model"
        empty_model.mkdir()
        result = qwen3_asr.execute(
            self.request | {"model_path": str(empty_model)}, model_loader=loader
        )
        self.assertEqual(result["error_type"], "validation_error")
        self.assertIn("non-empty", result["error"])
        loader.assert_not_called()

    def test_validation_happens_before_lazy_mlx_import(self) -> None:
        with mock.patch.object(
            qwen3_asr,
            "_load_model",
            side_effect=AssertionError("MLX imported during validation"),
        ) as loader:
            result = qwen3_asr.execute(self.request | {"temperature": -1})

        self.assertEqual(result["error_type"], "validation_error")
        loader.assert_not_called()

    def test_object_result_uses_local_paths_options_and_segment_fields(self) -> None:
        model = mock.Mock()
        model.generate.return_value = types.SimpleNamespace(
            text="Привет",
            segments=[
                Segment(
                    0.0,
                    1.25,
                    "Привет",
                    "Russian",
                    words=[{"start": 0.1, "end": 0.2}],
                )
            ],
            language="Russian",
        )
        loader = mock.Mock(return_value=model)
        request = self.request | {
            "language": "Russian",
            "max_tokens": 123,
            "temperature": 0.25,
        }
        result = qwen3_asr.execute(
            request,
            model_loader=loader,
            clock=iter([10.0, 12.5, 20.0, 21.25]).__next__,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "Привет")
        self.assertEqual(
            result["segments"],
            [{"start": 0.0, "end": 1.25, "text": "Привет", "language": "Russian"}],
        )
        self.assertEqual(result["language"], "Russian")
        self.assertEqual(result["load_seconds"], 2.5)
        self.assertEqual(result["transcribe_seconds"], 1.25)
        loader.assert_called_once_with(self.request["model_path"])
        model.generate.assert_called_once_with(
            self.request["audio_path"],
            language="Russian",
            max_tokens=123,
            temperature=0.25,
            verbose=False,
            stream=False,
        )
        self.assertEqual(result["timestamp_semantics"], "chunk/segment offsets")
        self.assertTrue(result["effective_config"]["no_word_timestamps"])
        self.assertEqual(result["effective_config"]["backend"], "mlx")
        self.assertTrue(result["effective_config"]["offline"])
        self.assertTrue(result["effective_config"]["local_files_only"])
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_dict_result_and_default_options(self) -> None:
        model = mock.Mock()
        model.generate.return_value = {
            "text": "hello",
            "segments": [
                {
                    "start": 0,
                    "end": 0.5,
                    "text": "hello",
                    "language": "English",
                    "tokens": [1, 2, 3],
                }
            ],
            "language": "English",
        }
        result = qwen3_asr.execute(
            self.request,
            model_loader=lambda path: model,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "hello")
        self.assertEqual(result["language"], "English")
        self.assertEqual(
            result["segments"],
            [{"start": 0, "end": 0.5, "text": "hello", "language": "English"}],
        )
        self.assertEqual(result["effective_config"]["max_tokens"], 8192)
        self.assertEqual(result["effective_config"]["temperature"], 0.0)
        model.generate.assert_called_once_with(
            self.request["audio_path"],
            language=None,
            max_tokens=8192,
            temperature=0.0,
            verbose=False,
            stream=False,
        )

    def test_auto_language_is_normalized_to_none_for_model_and_metadata(self) -> None:
        model = mock.Mock()
        model.generate.return_value = {
            "text": "hello",
            "segments": [],
            "language": "English",
        }

        result = qwen3_asr.execute(
            self.request | {"language": "auto"},
            model_loader=lambda path: model,
        )

        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["effective_config"]["language"])
        model.generate.assert_called_once_with(
            self.request["audio_path"],
            language=None,
            max_tokens=8192,
            temperature=0.0,
            verbose=False,
            stream=False,
        )

    def test_lazy_official_loader_can_be_injected_as_fake_mlx_audio_module(self) -> None:
        model = mock.Mock()
        model.generate.return_value = {
            "text": "ok",
            "segments": [],
            "language": "en",
        }
        load = mock.Mock(return_value=model)
        fake_stt = types.ModuleType("mlx_audio.stt")
        fake_stt.load = load
        fake_mlx_audio = types.ModuleType("mlx_audio")
        fake_mlx_audio.__path__ = []
        with mock.patch.dict(
            sys.modules,
            {"mlx_audio": fake_mlx_audio, "mlx_audio.stt": fake_stt},
        ):
            result = qwen3_asr.execute(self.request)

        self.assertEqual(result["status"], "ok")
        load.assert_called_once_with(self.request["model_path"])

    def test_load_and_transcribe_failures_are_error_payloads(self) -> None:
        result = qwen3_asr.execute(
            self.request,
            model_loader=lambda path: (_ for _ in ()).throw(
                RuntimeError("load failed")
            ),
        )
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        model = mock.Mock()
        model.generate.side_effect = RuntimeError("transcribe failed")
        result = qwen3_asr.execute(self.request, model_loader=lambda path: model)
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("transcribe failed", result["error"])

    def test_unsupported_result_is_a_transcribe_error(self) -> None:
        model = mock.Mock()
        model.generate.return_value = {"text": None, "segments": [], "language": "en"}
        result = qwen3_asr.execute(self.request, model_loader=lambda path: model)
        self.assertEqual(result["error_type"], "transcribe_error")

    def test_cli_emits_one_json_object_and_returns_status_code(self) -> None:
        success = {
            "status": "ok",
            "transcript": "ok",
            "segments": [],
            "language": "en",
            "load_seconds": 0.1,
            "transcribe_seconds": 0.2,
            "effective_config": {},
        }
        output = io.StringIO()
        with (
            mock.patch.object(qwen3_asr, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(json.dumps(self.request))),
            contextlib.redirect_stdout(output),
        ):
            return_code = qwen3_asr.main()

        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = qwen3_asr.main()

        self.assertNotEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
