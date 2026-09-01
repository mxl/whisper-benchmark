import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stt_benchmark.workers import parakeet_sherpa


class FakeAudio:
    ndim = 1

    def __init__(self, values=None) -> None:
        self.values = list(values or [])

    def astype(self, dtype: str, copy: bool = False):
        self.astype_args = (dtype, copy)
        return self

    def __len__(self):
        return len(self.values)


class FakeSoundfile:
    def __init__(self, audio, sample_rate=16_000) -> None:
        self.audio = audio
        self.sample_rate = sample_rate
        self.read_calls = []

    def read(self, path, *, dtype, always_2d):
        self.read_calls.append((path, dtype, always_2d))
        return self.audio, self.sample_rate


class FakeStream:
    def __init__(self, result) -> None:
        self.result = result
        self.waveform = None

    def accept_waveform(self, sample_rate, audio):
        self.waveform = (sample_rate, audio)


class FakeRecognizer:
    def __init__(self, result):
        self.result = result
        self.streams = []
        self.decode_calls = 0

    def create_stream(self):
        stream = FakeStream(self.result)
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream):
        self.decode_calls += 1


class FakeSherpa:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.calls = []
        self.OfflineRecognizer = self

    def from_transducer(self, **kwargs):
        self.calls.append(kwargs)
        return self.recognizer


class ParakeetSherpaWorkerTests(unittest.TestCase):
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

    def make_artifacts(self, quantization="fp32") -> None:
        names = ["decoder.onnx", "joiner.onnx", "tokens.txt", "bpe.vocab"]
        if quantization == "fp32":
            names.extend(["encoder.onnx", "encoder.weights"])
        else:
            names.extend(
                ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"]
            )
        for name in names:
            (self.model_path / name).write_bytes(b"model")

    def execute(self, request=None, *, result=None, soundfile=None, resample_fn=None):
        result = SimpleNamespace(text=" transcript ") if result is None else result
        recognizer = FakeRecognizer(result)
        sherpa = FakeSherpa(recognizer)
        return parakeet_sherpa.execute(
            self.request if request is None else request,
            sherpa_module=sherpa,
            soundfile_module=soundfile
            or FakeSoundfile(FakeAudio([0.0, 1.0, 2.0])),
            resample_fn=resample_fn,
            clock=iter([10.0, 12.5, 20.0, 21.25]).__next__,
        ), sherpa

    def test_validation_rejects_nonlocal_paths_bad_options_and_fp16(self) -> None:
        self.make_artifacts()
        with mock.patch.object(
            parakeet_sherpa,
            "_import_sherpa_onnx",
            side_effect=AssertionError("sherpa imported"),
        ):
            cases = [
                {"model_path": "https://example.invalid/model"},
                self.request | {"audio_path": "file:///tmp/audio.wav"},
                self.request | {"quantization": "fp16"},
                self.request | {"quantization": "int4"},
                self.request | {"threads": 0},
                self.request | {"threads": 1.0},
                self.request | {"threads": True},
            ]
            for request in cases:
                with self.subTest(request=request):
                    result = parakeet_sherpa.execute(request)
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["error_type"], "validation_error")

    def test_exact_artifacts_are_required_without_cross_quantization_fallback(self) -> None:
        self.make_artifacts("int8")
        result = parakeet_sherpa.execute(
            self.request,
            sherpa_module=mock.Mock(),
            soundfile_module=mock.Mock(),
        )
        self.assertEqual(result["error_type"], "validation_error")
        self.assertIn("encoder.onnx", result["error"])

        result = parakeet_sherpa.execute(
            self.request | {"quantization": "int8"},
            sherpa_module=FakeSherpa(FakeRecognizer(SimpleNamespace(text="ok"))),
            soundfile_module=FakeSoundfile(FakeAudio()),
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["files"]["encoder"].endswith("encoder.int8.onnx"))

    def test_default_fp32_loads_exact_cpu_nemo_transducer_configuration(self) -> None:
        self.make_artifacts()
        result, sherpa = self.execute()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "transcript")
        self.assertEqual(result["load_seconds"], 2.5)
        self.assertEqual(result["transcribe_seconds"], 1.25)
        call = sherpa.calls[0]
        self.assertEqual(call["encoder"], str(self.model_path / "encoder.onnx"))
        self.assertEqual(
            call["encoder"].replace("encoder.onnx", "encoder.weights"),
            str(self.model_path / "encoder.weights"),
        )
        self.assertEqual(call["decoder"], str(self.model_path / "decoder.onnx"))
        self.assertEqual(call["joiner"], str(self.model_path / "joiner.onnx"))
        self.assertEqual(call["tokens"], str(self.model_path / "tokens.txt"))
        self.assertEqual(call["bpe_vocab"], str(self.model_path / "bpe.vocab"))
        self.assertEqual(call["provider"], "cpu")
        self.assertEqual(call["num_threads"], 4)
        self.assertEqual(call["sample_rate"], 16_000)
        self.assertEqual(call["decoding_method"], "greedy_search")
        self.assertEqual(call["max_active_paths"], 4)
        self.assertEqual(call["model_type"], "nemo_transducer")
        self.assertEqual(result["quantization"], "fp32")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["effective_config"]["threads"], 4)
        self.assertTrue(result["effective_config"]["offline"])
        self.assertFalse(result["effective_config"]["network_access"])

    def test_audio_is_downmixed_resampled_and_sent_in_one_offline_stream(self) -> None:
        self.make_artifacts()
        stereo = mock.Mock()
        stereo.ndim = 2
        mono = FakeAudio([1.0, 2.0])
        stereo.mean.return_value = mono
        soundfile = FakeSoundfile(stereo, sample_rate=8_000)
        resampled = FakeAudio([1.0, 2.0, 3.0, 4.0])
        resample_calls = []

        def resample(audio, up, down):
            resample_calls.append((audio, up, down))
            return resampled

        result, sherpa = self.execute(
            soundfile=soundfile,
            resample_fn=resample,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(soundfile.read_calls, [(self.request["audio_path"], "float32", False)])
        stereo.mean.assert_called_once_with(axis=1)
        self.assertEqual(resample_calls, [(mono, 2, 1)])
        self.assertEqual(len(sherpa.recognizer.streams), 1)
        self.assertEqual(sherpa.recognizer.decode_calls, 1)
        self.assertEqual(sherpa.recognizer.streams[0].waveform, (16_000, resampled))
        self.assertTrue(result["effective_config"]["resampled_to_sample_rate"])

    def test_token_timestamps_are_normalized_and_missing_timestamps_are_unsupported(self) -> None:
        self.make_artifacts()
        result, _ = self.execute(
            result=SimpleNamespace(
                text=" hello ",
                tokens=["▁hello", "world"],
                timestamps=[0, 1.25],
            )
        )
        self.assertEqual(result["transcript"], "hello")
        self.assertEqual(result["timestamps"], [0.0, 1.25])
        self.assertEqual(
            result["segments"],
            [
                {"text": "▁hello", "start": 0.0},
                {"text": "world", "start": 1.25},
            ],
        )
        self.assertEqual(result["timestamp_semantics"], "token/frame starts")

        result, _ = self.execute(result=SimpleNamespace(text="no timing"))
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["timestamps"], [])
        self.assertEqual(result["timestamp_semantics"], "unsupported")

    def test_load_and_transcribe_failures_are_structured(self) -> None:
        self.make_artifacts()
        result = parakeet_sherpa.execute(
            self.request,
            sherpa_module=SimpleNamespace(
                OfflineRecognizer=SimpleNamespace(
                    from_transducer=mock.Mock(side_effect=RuntimeError("load failed"))
                )
            ),
            soundfile_module=FakeSoundfile(FakeAudio()),
        )
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        recognizer = mock.Mock()
        recognizer.create_stream.side_effect = RuntimeError("decode failed")
        result = parakeet_sherpa.execute(
            self.request,
            sherpa_module=FakeSherpa(recognizer),
            soundfile_module=FakeSoundfile(FakeAudio()),
        )
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("decode failed", result["error"])

    def test_cli_emits_one_json_object_and_matching_status(self) -> None:
        self.make_artifacts()
        output = io.StringIO()
        success = {"status": "ok", "transcript": "ok"}
        with (
            mock.patch.object(parakeet_sherpa, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(json.dumps(self.request))),
            contextlib.redirect_stdout(output),
        ):
            return_code = parakeet_sherpa.main()
        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = parakeet_sherpa.main()
        self.assertNotEqual(return_code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
