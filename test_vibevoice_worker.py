import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stt_benchmark.workers import vibevoice


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


class FakeTensor:
    def __init__(self, shape=None):
        self.shape = shape
        self.to_calls = []

    def to(self, device, *, dtype=None):
        self.to_calls.append((device, dtype))
        return self


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=FakeTensor((1, 4)), input_values=FakeTensor())
        self.to_calls = []

    def to(self, device, dtype=None):
        self.to_calls.append((device, dtype))
        return self


class FakeGenerated:
    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise AssertionError("generated IDs must be sliced as a batch")
        return (key[0], key[1])


class FakeProcessor:
    feature_extractor = SimpleNamespace(sampling_rate=24_000)

    def __init__(self, decoded):
        self.decoded = decoded
        self.inputs = FakeInputs()
        self.process_calls = []
        self.decode_calls = []

    def apply_transcription_request(self, *, audio):
        self.process_calls.append(audio)
        return self.inputs

    def decode(self, generated_ids, *, return_format):
        self.decode_calls.append((generated_ids, return_format))
        return self.decoded


class FakeModel:
    dtype = "bfloat16"

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
        return FakeGenerated()


class FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class VibeVoiceWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "model"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text("{}")
        (self.model_path / "model.safetensors.index.json").write_text("{}")
        for shard in range(1, 9):
            (self.model_path / f"model-{shard:05d}-of-00008.safetensors").write_bytes(
                b"model"
            )
        (self.model_path / "tokenizer.json").write_bytes(b"tokenizer")
        (self.model_path / "tokenizer_config.json").write_text("{}")
        (self.model_path / "processor_config.json").write_text("{}")
        (self.model_path / "chat_template.jinja").write_text("template")
        self.audio_path = root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_validation_rejects_nonlocal_paths_modes_devices_and_incomplete_shards(self):
        loader = mock.Mock()
        cases = [
            self.request | {"model_path": "https://example.invalid/model"},
            self.request | {"audio_path": "file:///tmp/audio.wav"},
            self.request | {"mode": "raw"},
            self.request | {"device": "cuda"},
            self.request | {"acoustic_tokenizer_chunk_size": 1},
            self.request | {"acoustic_tokenizer_chunk_size": True},
        ]
        for request in cases:
            with self.subTest(request=request):
                result = vibevoice.execute(request, model_loader=loader)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error_type"], "validation_error")
        loader.assert_not_called()

        (self.model_path / "model-00008-of-00008.safetensors").unlink()
        result = vibevoice.execute(self.request, model_loader=loader)
        self.assertEqual(result["error_type"], "validation_error")
        self.assertIn("model-00008-of-00008.safetensors", result["error"])
        loader.assert_not_called()

    def test_validation_checks_index_references_and_metadata(self):
        index = {"weight_map": {"weight": "model-00008-of-00008.safetensors"}}
        (self.model_path / "model.safetensors.index.json").write_text(
            json.dumps(index)
        )
        vibevoice._validate_model_files(str(self.model_path))

        (self.model_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "missing.safetensors"}})
        )
        with self.assertRaisesRegex(ValueError, "references missing shards"):
            vibevoice._validate_model_files(str(self.model_path))

    def test_official_loader_uses_exact_local_path_and_local_only_transformers_api(self):
        processor_factory = mock.Mock(return_value="processor")
        model_factory = mock.Mock(return_value="model")
        transformers = SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=processor_factory),
            VibeVoiceAsrForConditionalGeneration=SimpleNamespace(
                from_pretrained=model_factory
            ),
        )
        with mock.patch.dict("sys.modules", {"transformers": transformers}):
            result = vibevoice._load_model(str(self.model_path))

        self.assertEqual(result, ("processor", "model"))
        processor_factory.assert_called_once_with(
            str(self.model_path), local_files_only=True
        )
        model_factory.assert_called_once_with(
            str(self.model_path), dtype="auto", local_files_only=True
        )

    def test_transcription_only_reads_audio_resamples_moves_and_decodes_plain_text(self):
        processor = FakeProcessor(["  hello world  "])
        model = FakeModel()
        soundfile = FakeSoundfile(sample_rate=8_000)
        resample_calls = []

        def resample(audio, up, down):
            resample_calls.append((audio, up, down))
            return [0.1, 0.2, 0.3]

        torch = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True),
            ),
            no_grad=FakeNoGrad,
        )
        clock_values = iter([10.0, 12.5, 20.0, 21.25])
        with mock.patch.dict(os.environ, {}, clear=True):
            result = vibevoice.execute(
                self.request,
                model_loader=lambda path: (processor, model),
                torch_module=torch,
                soundfile_module=soundfile,
                resample_fn=resample,
                clock=lambda: next(clock_values),
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "hello world")
        self.assertEqual(result["segments"], [])
        self.assertEqual(result["mode"], "transcription_only")
        self.assertEqual(result["load_seconds"], 2.5)
        self.assertEqual(result["transcribe_seconds"], 1.25)
        self.assertEqual(result["device"], "mps")
        self.assertEqual(result["model_class"], "FakeModel")
        self.assertEqual(model.to_calls, ["mps"])
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(
            soundfile.read_calls,
            [(self.request["audio_path"], "float32", False)],
        )
        self.assertEqual(resample_calls[0][1:], (3, 1))
        self.assertEqual(processor.process_calls[0], [0.1, 0.2, 0.3])
        self.assertEqual(processor.inputs.to_calls, [("mps", "bfloat16")])
        self.assertEqual(model.generate_calls[0]["do_sample"], False)
        self.assertEqual(processor.decode_calls[0][1], "transcription_only")
        self.assertTrue(result["effective_config"]["offline"])
        self.assertTrue(result["effective_config"]["local_files_only"])
        self.assertFalse(result["effective_config"]["pipeline"])
        self.assertFalse(result["effective_config"]["torchcodec"])
        self.assertEqual(result["effective_config"]["sample_rate"], 24_000)
        self.assertTrue(result["effective_config"]["resampled"])
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_parsed_mode_preserves_who_when_what_segments_and_optional_chunk(self):
        parsed = [
            [
                {
                    "Start": 0,
                    "End": 1.25,
                    "Speaker": 0,
                    "Content": " hello ",
                    "Extra": "preserved",
                },
                {
                    "Start": 1.25,
                    "End": 2.5,
                    "Speaker": 1,
                    "Content": "world",
                },
            ]
        ]
        processor = FakeProcessor(parsed)
        model = FakeModel()
        torch = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False),
            ),
            no_grad=FakeNoGrad,
        )

        result = vibevoice.execute(
            self.request
            | {"mode": "parsed", "acoustic_tokenizer_chunk_size": 64_000},
            model_loader=lambda path: (processor, model),
            torch_module=torch,
            soundfile_module=FakeSoundfile(sample_rate=24_000),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(result["transcript"], "hello  world")
        self.assertEqual(
            result["segments"],
            [
                {
                    "Start": 0,
                    "End": 1.25,
                    "Speaker": 0,
                    "Content": " hello ",
                    "Extra": "preserved",
                },
                {"Start": 1.25, "End": 2.5, "Speaker": 1, "Content": "world"},
            ],
        )
        self.assertEqual(
            model.generate_calls[0]["acoustic_tokenizer_chunk_size"], 64_000
        )
        self.assertEqual(processor.decode_calls[0][1], "parsed")
        self.assertEqual(result["effective_config"]["mode"], "parsed")

    def test_auto_device_falls_back_to_cpu_and_failures_are_structured(self):
        processor = FakeProcessor(["ok"])
        model = FakeModel()
        torch = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False),
            ),
            no_grad=FakeNoGrad,
        )
        result = vibevoice.execute(
            self.request,
            model_loader=lambda path: (processor, model),
            torch_module=torch,
            soundfile_module=FakeSoundfile(sample_rate=24_000),
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["device"], "cpu")

        result = vibevoice.execute(
            self.request,
            model_loader=lambda path: (_ for _ in ()).throw(
                RuntimeError("load failed")
            ),
        )
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        model.generate = mock.Mock(side_effect=RuntimeError("transcribe failed"))
        result = vibevoice.execute(
            self.request,
            model_loader=lambda path: (processor, model),
            torch_module=torch,
            soundfile_module=FakeSoundfile(sample_rate=24_000),
        )
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("transcribe failed", result["error"])

    def test_lazy_transformers_loader_can_be_injected_without_real_model_load(self):
        processor = FakeProcessor(["ok"])
        model = FakeModel()
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoProcessor = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=processor)
        )
        fake_transformers.VibeVoiceAsrForConditionalGeneration = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=model)
        )
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            result = vibevoice.execute(
                self.request,
                torch_module=SimpleNamespace(
                    backends=SimpleNamespace(
                        mps=SimpleNamespace(is_available=lambda: False),
                    ),
                    no_grad=FakeNoGrad,
                ),
                soundfile_module=FakeSoundfile(sample_rate=24_000),
            )

        self.assertEqual(result["status"], "ok")
        fake_transformers.AutoProcessor.from_pretrained.assert_called_once()
        fake_transformers.VibeVoiceAsrForConditionalGeneration.from_pretrained.assert_called_once()

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
            mock.patch.object(vibevoice, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(json.dumps(self.request))),
            contextlib.redirect_stdout(output),
        ):
            return_code = vibevoice.main()

        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = vibevoice.main()
        self.assertNotEqual(return_code, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
