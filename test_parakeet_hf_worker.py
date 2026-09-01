import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stt_benchmark.workers import parakeet_hf


class FakeAudio:
    ndim = 2

    def mean(self, *, axis):
        if axis != 1:
            raise AssertionError("audio must be downmixed across channels")
        return [0.25, 0.5, 0.75]


class FakeSoundfile:
    def __init__(self, sample_rate=8_000):
        self.sample_rate = sample_rate
        self.read_calls = []

    def read(self, path, *, dtype, always_2d):
        self.read_calls.append((path, dtype, always_2d))
        return FakeAudio(), self.sample_rate


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_values=FakeTensor())
        self.to_calls = []

    def to(self, device, *, dtype):
        self.to_calls.append((device, dtype))
        return self


class FakeTensor:
    pass


class FakeProcessor:
    feature_extractor = SimpleNamespace(sampling_rate=16_000)

    def __init__(self, inputs=None):
        self.inputs = inputs or FakeInputs()
        self.process_calls = []
        self.decode_calls = []

    def __call__(self, audio, *, sampling_rate, return_tensors):
        self.process_calls.append((audio, sampling_rate, return_tensors))
        return self.inputs

    def decode(self, sequences, *, durations, skip_special_tokens):
        self.decode_calls.append((sequences, durations, skip_special_tokens))
        return ["  hello world  "], [
            [
                {"token": "hello", "start": 0, "end": 0.5},
                {"token": " world", "start": 0.5, "end": 1.0},
            ]
        ]


class FakeModel:
    dtype = "float32"

    def __init__(self):
        self.device = None
        self.to_calls = []
        self.eval_calls = 0
        self.generate_calls = []

    def to(self, device):
        self.to_calls.append(device)
        self.device = device

    def eval(self):
        self.eval_calls += 1

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return SimpleNamespace(sequences=[1, 2, 3], durations=[4, 5, 6])


class ParakeetHFWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "model"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text("{}")
        (self.model_path / "model.safetensors").write_bytes(b"model")
        (self.model_path / "tokenizer.json").write_bytes(b"tokenizer")
        (self.model_path / "processor_config.json").write_text(
            '{"processor_class":"ParakeetProcessor"}'
        )
        self.audio_path = root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_validation_rejects_non_local_paths_devices_and_incomplete_snapshots(self):
        loader = mock.Mock()
        cases = [
            {"model_path": "https://example.invalid/model", "audio_path": "x"},
            self.request | {"audio_path": "file:///tmp/audio.wav"},
            self.request | {"device": "cuda"},
        ]
        for request in cases:
            with self.subTest(request=request):
                result = parakeet_hf.execute(request, model_loader=loader)
                self.assertEqual(result["error_type"], "validation_error")
        loader.assert_not_called()

        missing_processor = Path(self.tempdir.name) / "missing-processor"
        missing_processor.mkdir()
        (missing_processor / "config.json").write_text("{}")
        (missing_processor / "model.safetensors").write_bytes(b"model")
        (missing_processor / "tokenizer.json").write_bytes(b"tokenizer")
        result = parakeet_hf.execute(
            self.request | {"model_path": str(missing_processor)},
            model_loader=loader,
        )
        self.assertEqual(result["error_type"], "validation_error")
        self.assertIn("processor config", result["error"])
        loader.assert_not_called()

        missing_tokenizer = Path(self.tempdir.name) / "missing-tokenizer"
        missing_tokenizer.mkdir()
        (missing_tokenizer / "config.json").write_text("{}")
        (missing_tokenizer / "model.safetensors").write_bytes(b"model")
        result = parakeet_hf.execute(
            self.request | {"model_path": str(missing_tokenizer)},
            model_loader=loader,
        )
        self.assertEqual(result["error_type"], "validation_error")
        self.assertIn("tokenizer", result["error"])
        loader.assert_not_called()

    def test_validation_accepts_a_safetensors_index(self):
        sharded_model = Path(self.tempdir.name) / "sharded-model"
        sharded_model.mkdir()
        (sharded_model / "config.json").write_text("{}")
        (sharded_model / "model.safetensors.index.json").write_text("{}")
        (sharded_model / "tokenizer.json").write_bytes(b"tokenizer")
        (sharded_model / "processor_config.json").write_text("{}")

        parakeet_hf._validate_model_files(str(sharded_model))

    def test_official_loader_uses_only_exact_local_path(self):
        processor_factory = mock.Mock(return_value="processor")
        model_factory = mock.Mock(return_value="model")
        transformers = SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=processor_factory),
            AutoModelForTDT=SimpleNamespace(from_pretrained=model_factory),
        )
        with mock.patch.dict("sys.modules", {"transformers": transformers}):
            result = parakeet_hf._load_model(str(self.model_path))

        self.assertEqual(result, ("processor", "model"))
        processor_factory.assert_called_once_with(
            str(self.model_path), local_files_only=True
        )
        model_factory.assert_called_once_with(
            str(self.model_path), dtype="auto", local_files_only=True
        )

    def test_transcription_reads_downmixes_resamples_moves_and_decodes_timestamps(self):
        processor = FakeProcessor()
        model = FakeModel()
        soundfile = FakeSoundfile(sample_rate=8_000)
        resample_calls = []

        def resample(audio, up, down):
            resample_calls.append((audio, up, down))
            return [0.1, 0.2, 0.3]

        torch = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True),
            )
        )
        clock_values = iter([10.0, 12.5, 20.0, 21.25])
        with mock.patch.dict(os.environ, {}, clear=True):
            result = parakeet_hf.execute(
                self.request,
                model_loader=lambda path: (processor, model),
                torch_module=torch,
                soundfile_module=soundfile,
                resample_fn=resample,
                clock=lambda: next(clock_values),
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "hello world")
        self.assertEqual(
            result["segments"],
            [
                {"token": "hello", "start": 0.0, "end": 0.5},
                {"token": " world", "start": 0.5, "end": 1.0},
            ],
        )
        self.assertEqual(result["load_seconds"], 2.5)
        self.assertEqual(result["transcribe_seconds"], 1.25)
        self.assertEqual(result["device"], "mps")
        self.assertEqual(result["model_class"], "FakeModel")
        self.assertEqual(model.to_calls, ["mps"])
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(soundfile.read_calls, [(self.request["audio_path"], "float32", False)])
        self.assertEqual(resample_calls[0][1:], (2, 1))
        self.assertEqual(processor.process_calls[0][1:], (16_000, "pt"))
        self.assertEqual(processor.process_calls[0][0], [0.1, 0.2, 0.3])
        self.assertEqual(processor.inputs.to_calls, [("mps", "float32")])
        self.assertEqual(model.generate_calls[0]["return_dict_in_generate"], True)
        self.assertEqual(len(processor.decode_calls), 1)
        self.assertTrue(result["effective_config"]["offline"])
        self.assertTrue(result["effective_config"]["local_files_only"])
        self.assertFalse(result["effective_config"]["torchcodec"])
        self.assertTrue(result["effective_config"]["no_torchcodec"])
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_auto_device_falls_back_to_cpu_and_decode_falls_back_without_durations(self):
        class ProcessorWithoutDurations(FakeProcessor):
            def decode(self, sequences, *, skip_special_tokens):
                self.decode_calls.append((sequences, skip_special_tokens))
                return ["fallback"]

        processor = ProcessorWithoutDurations()
        model = FakeModel()
        torch = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False),
            )
        )
        result = parakeet_hf.execute(
            self.request,
            model_loader=lambda path: (processor, model),
            torch_module=torch,
            soundfile_module=FakeSoundfile(sample_rate=16_000),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "fallback")
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(model.to_calls, ["cpu"])

    def test_load_and_transcribe_failures_are_structured(self):
        result = parakeet_hf.execute(
            self.request,
            model_loader=lambda path: (_ for _ in ()).throw(RuntimeError("load failed")),
        )
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        processor = FakeProcessor()
        model = FakeModel()
        model.generate = mock.Mock(side_effect=RuntimeError("transcribe failed"))
        result = parakeet_hf.execute(
            self.request,
            model_loader=lambda path: (processor, model),
            torch_module=SimpleNamespace(
                backends=SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: False),
                )
            ),
            soundfile_module=FakeSoundfile(sample_rate=16_000),
        )
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("transcribe failed", result["error"])

    def test_cli_emits_one_json_object_and_returns_status_code(self):
        success = {
            "status": "ok",
            "transcript": "ok",
            "segments": [],
            "load_seconds": 0.1,
            "transcribe_seconds": 0.2,
            "effective_config": {},
        }
        output = io.StringIO()
        with (
            mock.patch.object(parakeet_hf, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(json.dumps(self.request))),
            contextlib.redirect_stdout(output),
        ):
            return_code = parakeet_hf.main()

        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = parakeet_hf.main()
        self.assertNotEqual(return_code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
