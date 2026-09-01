import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stt_benchmark.workers import gigaam


class GigaAMWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "model"
        self.model_path.mkdir()
        self.audio_path = root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_validation_rejects_missing_and_non_local_paths(self) -> None:
        result = gigaam.execute({"model_path": "https://example.invalid/model"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "validation_error")

        result = gigaam.execute(self.request | {"audio_path": "missing.wav"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "validation_error")

    def test_string_transcript_and_effective_config(self) -> None:
        class Model:
            def transcribe(self, path: str) -> str:
                self.path = path
                return "Привет"

        model = Model()
        clock_values = iter([10.0, 12.5, 20.0, 21.25])
        result = gigaam.execute(
            self.request,
            model_loader=lambda path: model,
            clock=lambda: next(clock_values),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "Привет")
        self.assertEqual(result["load_seconds"], 2.5)
        self.assertEqual(result["transcribe_seconds"], 1.25)
        self.assertEqual(result["transcription_mode"], "short")
        self.assertEqual(model.path, self.request["audio_path"])
        self.assertEqual(result["effective_config"]["model_path"], self.request["model_path"])
        self.assertEqual(result["effective_config"]["audio_path"], self.request["audio_path"])
        self.assertTrue(result["effective_config"]["offline"])
        self.assertTrue(result["effective_config"]["local_files_only"])
        self.assertEqual(result["effective_config"]["transcription_mode"], "short")
        self.assertEqual(
            result["effective_config"]["chunk_seconds"],
            gigaam.DEFAULT_CHUNK_SECONDS,
        )
        self.assertEqual(result["effective_config"]["chunk_overlap_seconds"], 0.0)

    def test_list_and_text_object_transcripts(self) -> None:
        class TextResult:
            text = "object transcript"

        for returned, expected in [
            (["list transcript"], "list transcript"),
            ([TextResult()], "object transcript"),
            (TextResult(), "object transcript"),
        ]:
            with self.subTest(returned=returned):
                model = mock.Mock()
                model.transcribe.return_value = returned
                result = gigaam.execute(
                    self.request,
                    model_loader=lambda path, model=model: model,
                    clock=iter([1.0, 2.0, 3.0, 4.0]).__next__,
                )
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["transcript"], expected)

    def test_loader_and_transcribe_errors_are_error_payloads(self) -> None:
        result = gigaam.execute(
            self.request,
            model_loader=lambda path: (_ for _ in ()).throw(RuntimeError("load failed")),
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        model = mock.Mock()
        model.transcribe.side_effect = RuntimeError("transcribe failed")
        result = gigaam.execute(self.request, model_loader=lambda path: model)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("transcribe failed", result["error"])

    def test_exact_too_long_error_uses_ordered_offline_fixed_chunks(self) -> None:
        audio = [0.0] * (16_000 * 2 + 1)
        soundfile = mock.Mock()
        soundfile.read.return_value = (audio, 16_000)
        audio_path = self.request["audio_path"]

        class Model:
            def __init__(self) -> None:
                self.calls: list[str] = []
                self.transcribe_longform = mock.Mock(
                    side_effect=AssertionError("official longform must not be called")
                )

            def transcribe(self, path: str) -> str:
                self.calls.append(path)
                if path == audio_path:
                    raise ValueError(
                        "Too long wav file, use 'transcribe_longform' method."
                    )
                return {
                    "chunk_0000.wav": " first ",
                    "chunk_0001.wav": " ",
                    "chunk_0002.wav": "third",
                }[Path(path).name]

        model = Model()
        with mock.patch.dict(sys.modules, {"soundfile": soundfile}):
            result = gigaam.execute(
                self.request | {"chunk_seconds": 1},
                model_loader=lambda path: model,
                clock=iter([1.0, 2.0, 3.0, 4.0]).__next__,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "first third")
        self.assertEqual(result["transcription_mode"], "offline_fixed_chunks")
        self.assertEqual(
            result["effective_config"]["transcription_mode"],
            "offline_fixed_chunks",
        )
        self.assertEqual(result["effective_config"]["chunk_seconds"], 1.0)
        self.assertEqual(result["effective_config"]["chunk_overlap_seconds"], 0.0)
        self.assertEqual(
            [Path(path).name for path in model.calls[1:]],
            ["chunk_0000.wav", "chunk_0001.wav", "chunk_0002.wav"],
        )
        soundfile.read.assert_called_once_with(
            self.request["audio_path"], dtype="float32", always_2d=False
        )
        self.assertEqual(
            [Path(call.args[0]).name for call in soundfile.write.call_args_list],
            ["chunk_0000.wav", "chunk_0001.wav", "chunk_0002.wav"],
        )
        self.assertEqual(
            [call.args[2] for call in soundfile.write.call_args_list],
            [16_000, 16_000, 16_000],
        )
        model.transcribe_longform.assert_not_called()

    def test_fixed_chunks_downmix_stereo_audio(self) -> None:
        class StereoAudio:
            ndim = 2

            def __init__(self) -> None:
                self.frames = [(1.0, 3.0), (2.0, 4.0)]

            def mean(self, axis: int):
                self.assert_axis = axis
                return [sum(frame) / len(frame) for frame in self.frames]

            def __len__(self) -> int:
                return len(self.frames)

            def __getitem__(self, item):
                return self.frames[item]

        audio = StereoAudio()
        soundfile = mock.Mock()
        soundfile.read.return_value = (audio, 16_000)
        audio_path = self.request["audio_path"]

        class Model:
            def transcribe(self, path: str) -> str:
                if path == audio_path:
                    raise ValueError(
                        "Too long wav file, use 'transcribe_longform' method."
                    )
                return "downmixed"

        model = Model()
        with mock.patch.dict(sys.modules, {"soundfile": soundfile}):
            result = gigaam.execute(
                self.request | {"chunk_seconds": 1},
                model_loader=lambda path: model,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "downmixed")
        self.assertEqual(audio.assert_axis, 1)
        self.assertEqual(soundfile.write.call_args.args[1], [2.0, 3.0])

    def test_fixed_chunks_reject_unsupported_sample_rate(self) -> None:
        soundfile = mock.Mock()
        soundfile.read.return_value = ([0.0], 8_000)
        model = mock.Mock()
        model.transcribe.side_effect = ValueError(
            "Too long wav file, use 'transcribe_longform' method."
        )

        with mock.patch.dict(sys.modules, {"soundfile": soundfile}):
            result = gigaam.execute(self.request, model_loader=lambda path: model)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("unsupported sample rate", result["error"])
        model.transcribe.assert_called_once_with(self.request["audio_path"])

    def test_invalid_chunk_seconds_is_a_validation_error(self) -> None:
        for value in [0, 30, "25", True, float("nan")]:
            with self.subTest(value=value):
                loader = mock.Mock()
                result = gigaam.execute(
                    self.request | {"chunk_seconds": value},
                    model_loader=loader,
                )

                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error_type"], "validation_error")
                loader.assert_not_called()

    def test_unrelated_value_error_does_not_fall_back(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.longform_called = False
                self.transcribe_longform = mock.Mock()

            def transcribe(self, path: str) -> str:
                raise ValueError("audio decode failed")

        model = Model()
        result = gigaam.execute(self.request, model_loader=lambda path: model)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("audio decode failed", result["error"])
        self.assertFalse(model.longform_called)
        model.transcribe_longform.assert_not_called()

    def test_cli_emits_one_json_object_and_returns_status_code(self) -> None:
        output = io.StringIO()
        request = json.dumps(self.request)
        success = {
            "status": "ok",
            "transcript": "ok",
            "load_seconds": 0.1,
            "transcribe_seconds": 0.2,
            "language": "ru",
            "variant": "e2e_rnnt",
            "effective_config": {},
        }
        with (
            mock.patch.object(gigaam, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(request)),
            contextlib.redirect_stdout(output),
        ):
            return_code = gigaam.main()

        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = gigaam.main()
        self.assertNotEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
