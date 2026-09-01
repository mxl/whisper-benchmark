import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stt_benchmark.workers import vosk


class FakeAudio:
    ndim = 1

    def __init__(self, length: int = 0, values=None) -> None:
        self.values = list(values) if values is not None else [0.0] * length

    def astype(self, dtype: str, copy: bool = False):
        self.astype_args = (dtype, copy)
        return self

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return (
            FakeAudio(values=self.values[item])
            if isinstance(item, slice)
            else self.values[item]
        )


class FakeSoundfile:
    def __init__(self, sample_rate: int = 16_000, audio=None) -> None:
        self.sample_rate = sample_rate
        self.audio = FakeAudio() if audio is None else audio
        self.read_calls = []

    def read(self, path, *, dtype, always_2d):
        self.read_calls.append((path, dtype, always_2d))
        return self.audio, self.sample_rate


class FakeRecognizer:
    def __init__(self, text="Привет", timestamps=None, results=None) -> None:
        self.text = text
        self.timestamps = [] if timestamps is None else timestamps
        self.results = results
        self.streams = []
        self.decode_calls = 0

    def create_stream(self):
        stream = SimpleNamespace(
            result=None,
            waveform=None,
            accept_waveform=None,
        )

        def accept_waveform(sample_rate, audio):
            stream.waveform = (sample_rate, audio)

        stream.accept_waveform = accept_waveform
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        self.decode_calls += 1
        index = self.decode_calls - 1
        if self.results is None:
            text = self.text
            timestamps = self.timestamps
        else:
            text, timestamps = self.results[index]
        stream.result = SimpleNamespace(
            text=text,
            timestamps=timestamps,
        )


class FakeSherpa:
    def __init__(self, recognizer=None, error=None) -> None:
        self.recognizer = recognizer or FakeRecognizer()
        self.error = error
        self.calls = []
        self.OfflineRecognizer = self

    def from_transducer(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.recognizer


class VoskWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "model"
        self.audio_path = root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_layout(self, layout="am-onnx", quantization="fp32", tokens=True) -> None:
        model_dir = self.model_path / layout
        model_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".int8.onnx" if quantization == "int8" else ".onnx"
        for name in ("encoder", "decoder", "joiner"):
            (model_dir / f"{name}{suffix}").write_bytes(b"model")
        if tokens:
            (self.model_path / "lang").mkdir(exist_ok=True)
            (self.model_path / "lang" / "tokens.txt").write_text("0 <blk>\n")

    def execute(self, request=None, *, sherpa=None, soundfile=None, clock=None):
        return vosk.execute(
            self.request if request is None else request,
            sherpa_module=sherpa or FakeSherpa(),
            soundfile_module=soundfile or FakeSoundfile(),
            **({"clock": clock} if clock is not None else {}),
        )

    def test_validation_rejects_missing_uri_layout_and_rate(self) -> None:
        result = vosk.execute({"model_path": "https://example.invalid/model"})
        self.assertEqual(result["error_type"], "validation_error")

        self.model_path.mkdir()
        result = self.execute(self.request | {"audio_path": "file:///tmp/audio.wav"})
        self.assertEqual(result["error_type"], "validation_error")

        result = self.execute()
        self.assertEqual(result["error_type"], "validation_error")
        self.assertIn("exact fp32 transducer files", result["error"])

        self.make_layout()
        result = self.execute(self.request | {"quantization": "int4"})
        self.assertEqual(result["error_type"], "validation_error")

        self.make_layout()
        result = self.execute(soundfile=FakeSoundfile(sample_rate=8_000))
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("unsupported sample rate", result["error"])

    def test_default_big_config_uses_modified_beam_and_auto_threads(self) -> None:
        self.make_layout("am-onnx")
        sherpa = FakeSherpa()
        soundfile = FakeSoundfile(audio=FakeAudio(length=1))
        result = self.execute(sherpa=sherpa, soundfile=soundfile)

        self.assertEqual(result["status"], "ok")
        call = sherpa.calls[0]
        self.assertEqual(call["encoder"], str(self.model_path / "am-onnx" / "encoder.onnx"))
        self.assertEqual(call["decoder"], str(self.model_path / "am-onnx" / "decoder.onnx"))
        self.assertEqual(call["joiner"], str(self.model_path / "am-onnx" / "joiner.onnx"))
        self.assertEqual(call["tokens"], str(self.model_path / "lang" / "tokens.txt"))
        self.assertEqual(call["decoding_method"], "modified_beam_search")
        self.assertEqual(call["num_threads"], 0)
        self.assertEqual(call["provider"], "cpu")
        self.assertEqual(call["sample_rate"], 16_000)
        self.assertEqual(result["decoding_method"], "modified_beam_search")
        self.assertEqual(result["quantization"], "fp32")
        self.assertEqual(result["effective_config"]["layout"], "big")
        self.assertEqual(result["effective_config"]["chunk_seconds"], 20.0)
        self.assertEqual(result["effective_config"]["chunk_overlap_seconds"], 0.0)
        self.assertTrue(result["effective_config"]["chunked"])
        self.assertTrue(result["effective_config"]["offline"])
        self.assertFalse(result["effective_config"]["network_access"])

    def test_small_config_and_greedy_decoding(self) -> None:
        self.make_layout("am")
        sherpa = FakeSherpa()
        result = self.execute(
            self.request | {"decoding_method": "greedy_search", "num_threads": 2},
            sherpa=sherpa,
            soundfile=FakeSoundfile(audio=FakeAudio(length=1)),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(sherpa.calls[0]["decoding_method"], "greedy_search")
        self.assertEqual(sherpa.calls[0]["num_threads"], 2)
        self.assertEqual(result["effective_config"]["layout"], "small")

        auto_result = self.execute(
            sherpa=sherpa, soundfile=FakeSoundfile(audio=FakeAudio(length=1))
        )
        self.assertEqual(auto_result["status"], "ok")
        self.assertEqual(sherpa.calls[-1]["num_threads"], 4)

    def test_int8_selects_only_matching_int8_files(self) -> None:
        self.make_layout("am", "int8")
        sherpa = FakeSherpa()
        result = self.execute(
            self.request | {"quantization": "int8"},
            sherpa=sherpa,
            soundfile=FakeSoundfile(audio=FakeAudio(length=1)),
        )

        self.assertEqual(result["status"], "ok")
        self.assertTrue(sherpa.calls[0]["encoder"].endswith("am/encoder.int8.onnx"))
        self.assertTrue(sherpa.calls[0]["decoder"].endswith("am/decoder.int8.onnx"))
        self.assertTrue(sherpa.calls[0]["joiner"].endswith("am/joiner.int8.onnx"))
        self.assertEqual(result["quantization"], "int8")

        fp32_result = self.execute(
            sherpa=sherpa, soundfile=FakeSoundfile(audio=FakeAudio(length=1))
        )
        self.assertEqual(fp32_result["error_type"], "validation_error")

    def test_chunk_order_timestamp_offsets_and_stereo_downmix(self) -> None:
        self.make_layout()
        audio = mock.Mock()
        audio.ndim = 2
        downmixed = FakeAudio(length=32_000 + 1)
        audio.mean.return_value = downmixed
        recognizer = FakeRecognizer(
            results=[
                (" first ", [0, 1.25]),
                (" ", [0.5]),
                ("third", [1.0]),
            ]
        )
        sherpa = FakeSherpa(recognizer)
        soundfile = FakeSoundfile(audio=audio)
        result = self.execute(
            self.request | {"chunk_seconds": 1},
            sherpa=sherpa,
            soundfile=soundfile,
        )

        self.assertEqual(result["transcript"], "first third")
        self.assertEqual(result["timestamps"], [0.0, 1.25, 1.5, 3.0])
        self.assertEqual(result["effective_config"]["timestamp_semantics"], "token/frame starts")
        self.assertFalse(result["effective_config"]["word_timestamps"])
        self.assertEqual(result["effective_config"]["chunk_seconds"], 1.0)
        self.assertEqual(soundfile.read_calls, [(self.request["audio_path"], "float32", False)])
        audio.mean.assert_called_once_with(axis=1)
        self.assertEqual(downmixed.astype_args, ("float32", False))
        self.assertEqual(recognizer.decode_calls, 3)
        self.assertEqual(len(recognizer.streams), 3)
        self.assertEqual(
            [stream.waveform[0] for stream in recognizer.streams],
            [16_000, 16_000, 16_000],
        )
        self.assertEqual(
            [len(stream.waveform[1]) for stream in recognizer.streams],
            [16_000, 16_000, 1],
        )
        self.assertEqual(
            recognizer.streams[0].waveform[1].values,
            downmixed.values[:16_000],
        )

    def test_load_and_transcribe_failures(self) -> None:
        self.make_layout()
        result = self.execute(sherpa=FakeSherpa(error=RuntimeError("load failed")))
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        recognizer = mock.Mock()
        recognizer.create_stream.side_effect = RuntimeError("transcribe failed")
        result = self.execute(
            sherpa=FakeSherpa(recognizer),
            soundfile=FakeSoundfile(audio=FakeAudio(length=1)),
        )
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("transcribe failed", result["error"])

    def test_long_audio_is_decoded_in_fixed_chunks(self) -> None:
        self.make_layout()
        audio = FakeAudio(length=16_000 * 45 + 123)
        recognizer = FakeRecognizer(
            results=[("one", [0.25]), ("two", [0.5]), ("three", [0.75])]
        )
        result = self.execute(
            self.request | {"chunk_seconds": 20},
            sherpa=FakeSherpa(recognizer),
            soundfile=FakeSoundfile(audio=audio),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "one two three")
        self.assertEqual(result["timestamps"], [0.25, 20.5, 40.75])
        self.assertEqual(recognizer.decode_calls, 3)
        self.assertEqual(
            [len(stream.waveform[1]) for stream in recognizer.streams],
            [320_000, 320_000, 80_123],
        )

    def test_invalid_chunk_seconds_is_a_validation_error(self) -> None:
        self.model_path.mkdir()
        with mock.patch.object(vosk, "_import_sherpa_onnx") as importer:
            for value in [0, -1, 29.1, "20", True, float("nan")]:
                with self.subTest(value=value):
                    result = vosk.execute(
                        self.request | {"chunk_seconds": value},
                        sherpa_module=None,
                    )
                    self.assertEqual(result["error_type"], "validation_error")
            importer.assert_not_called()

    def test_validation_happens_before_foreign_import_and_bad_threads_rejected(self) -> None:
        self.model_path.mkdir()
        with mock.patch.object(vosk, "_import_sherpa_onnx") as importer:
            for value in [0, -1, 1.0, True, "2"]:
                with self.subTest(value=value):
                    result = vosk.execute(
                        self.request | {"num_threads": value},
                        sherpa_module=None,
                    )
                    self.assertEqual(result["error_type"], "validation_error")
            importer.assert_not_called()

    def test_cli_emits_one_json_object_and_returns_status_code(self) -> None:
        self.make_layout()
        request = json.dumps(self.request)
        success = {
            "status": "ok",
            "transcript": "ok",
            "timestamps": [],
            "load_seconds": 0.1,
            "transcribe_seconds": 0.2,
            "decoding_method": "modified_beam_search",
            "quantization": "fp32",
            "effective_config": {},
        }
        output = io.StringIO()
        with (
            mock.patch.object(vosk, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(request)),
            contextlib.redirect_stdout(output),
        ):
            return_code = vosk.main()
        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = vosk.main()
        self.assertNotEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
