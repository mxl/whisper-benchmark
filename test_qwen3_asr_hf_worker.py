import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stt_benchmark.workers import qwen3_asr_hf


class FakeTensor:
    def __init__(self, shape=None):
        self.shape = shape
        self.to_calls = []

    def to(self, device, dtype=None):
        self.to_calls.append((device, dtype))
        return self


class FakeArray:
    def __init__(self, values, dtype="float32"):
        self.values = values
        self.dtype = dtype
        self.ndim = 1
        self.size = len(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        value = self.values[item]
        return FakeArray(value, dtype=self.dtype) if isinstance(item, slice) else value


class FakeNumpy:
    float32 = "float32"

    @staticmethod
    def asarray(values, *, dtype):
        if isinstance(values, FakeArray):
            return FakeArray(values.values, dtype=dtype)
        return FakeArray(list(values), dtype=dtype)


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=FakeTensor((1, 4)), input_features=FakeTensor())
        self.to_calls = []

    def to(self, device, dtype=None):
        self.to_calls.append((device, dtype))
        return self


class FakeGenerated:
    def __init__(self, generated_ids="generated-after-prompt"):
        self.generated_ids = generated_ids

    def __getitem__(self, key):
        if key != (slice(None), slice(4, None)):
            raise AssertionError(f"unexpected generated-id slice: {key!r}")
        return self.generated_ids


class FakeProcessor:
    feature_extractor = SimpleNamespace(sampling_rate=16_000)

    def __init__(self, decoded):
        self.decoded = decoded
        self.inputs = FakeInputs()
        self.process_calls = []
        self.decode_calls = []

    def apply_transcription_request(self, **kwargs):
        self.process_calls.append(kwargs)
        return self.inputs

    def decode(self, generated_ids, *, return_format):
        self.decode_calls.append((generated_ids, return_format))
        if isinstance(self.decoded, Exception):
            raise self.decoded
        if return_format == "parsed" and isinstance(self.decoded, dict):
            return [self.decoded]
        if return_format == "transcription_only":
            return [self.decoded]
        raise TypeError("parsed format unavailable")


class FakeModel:
    dtype = "bfloat16"

    def __init__(self, output=None):
        self.device = None
        self.to_calls = []
        self.eval_calls = 0
        self.generate_calls = []
        self.output = output if output is not None else FakeGenerated()

    def to(self, device):
        self.to_calls.append(device)
        self.device = device

    def eval(self):
        self.eval_calls += 1

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self.output


class FakeSoundfile:
    def __init__(self, sample_rate=8_000, audio=None):
        self.sample_rate = sample_rate
        self.audio = audio
        self.read_calls = []

    def read(self, path, *, dtype, always_2d):
        self.read_calls.append((path, dtype, always_2d))
        return (
            FakeStereoAudio() if self.audio is None else self.audio,
            self.sample_rate,
        )


class FakeStereoAudio:
    ndim = 2

    def mean(self, *, axis):
        if axis != 1:
            raise AssertionError("audio must be downmixed across channels")
        return [0.375, 0.625]


class FakeInferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeChunkProcessor:
    feature_extractor = SimpleNamespace(sampling_rate=16_000)

    def __init__(self, decoded):
        self.decoded = decoded
        self.process_calls = []
        self.decode_calls = []
        self.current_chunk = None

    def apply_transcription_request(self, **kwargs):
        self.current_chunk = len(self.process_calls)
        self.process_calls.append(kwargs)
        return FakeInputs()

    def decode(self, generated_ids, *, return_format):
        self.decode_calls.append((generated_ids, return_format))
        if return_format != "parsed":
            raise TypeError("parsed format is expected in this test")
        return [self.decoded[self.current_chunk]]


class Qwen3ASRHFWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "model"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text("{}")
        (self.model_path / "processor_config.json").write_text("{}")
        (self.model_path / "model.safetensors").write_bytes(b"model")
        (self.model_path / "tokenizer.json").write_bytes(b"tokenizer")
        self.audio_path = root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }
        self.torch = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True),
            ),
            inference_mode=FakeInferenceMode,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_request_validation_rejects_uris_devices_and_incomplete_snapshots(self):
        loader = mock.Mock()
        cases = [
            self.request | {"model_path": "https://example.invalid/model"},
            self.request | {"audio_path": "file:///tmp/audio.wav"},
            self.request | {"device": "cuda"},
            self.request | {"max_new_tokens": 0},
            self.request | {"max_new_tokens": True},
            self.request | {"language": ""},
            self.request | {"chunk_seconds": 0},
            self.request | {"chunk_seconds": -1},
            self.request | {"chunk_seconds": 300.1},
            self.request | {"chunk_seconds": "30"},
            self.request | {"chunk_seconds": True},
            self.request | {"chunk_seconds": float("nan")},
        ]
        for request in cases:
            with self.subTest(request=request):
                result = qwen3_asr_hf.execute(request, model_loader=loader)
                self.assertEqual(result["error_type"], "validation_error")
        loader.assert_not_called()
        self.assertEqual(
            qwen3_asr_hf._validate_request(self.request | {"chunk_seconds": 300})[
                "chunk_seconds"
            ],
            300.0,
        )

        missing_weights = Path(self.tempdir.name) / "missing-weights"
        missing_weights.mkdir()
        (missing_weights / "config.json").write_text("{}")
        (missing_weights / "processor_config.json").write_text("{}")
        (missing_weights / "tokenizer.json").write_bytes(b"tokenizer")
        result = qwen3_asr_hf.execute(
            self.request | {"model_path": str(missing_weights)},
            model_loader=loader,
        )
        self.assertEqual(result["error_type"], "validation_error")
        self.assertIn("model weights", result["error"])
        loader.assert_not_called()

    def test_official_loader_uses_exact_local_path_and_local_only_args(self):
        processor_factory = mock.Mock(return_value="processor")
        model_factory = mock.Mock(return_value="model")
        transformers = SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=processor_factory),
            AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=model_factory),
        )
        with mock.patch.dict("sys.modules", {"transformers": transformers}):
            result = qwen3_asr_hf._load_model(str(self.model_path))

        self.assertEqual(result, ("processor", "model"))
        processor_factory.assert_called_once_with(
            str(self.model_path), local_files_only=True
        )
        model_factory.assert_called_once_with(
            str(self.model_path), dtype="auto", local_files_only=True
        )

    def test_auto_language_omits_language_and_passes_raw_mono_float32_array(self):
        processor = FakeProcessor(
            {"language": "English", "transcription": " hello world "}
        )
        model = FakeModel()
        soundfile = FakeSoundfile(sample_rate=8_000)
        resample_calls = []

        def resample(audio, up, down):
            resample_calls.append((audio, up, down))
            return FakeArray([0.1, 0.2, 0.3], dtype="float64")

        clock_values = iter([10.0, 12.5, 20.0, 21.25])
        with mock.patch.dict(os.environ, {}, clear=True):
            result = qwen3_asr_hf.execute(
                self.request | {"language": "auto", "max_new_tokens": 123},
                model_loader=lambda path: (processor, model),
                torch_module=self.torch,
                soundfile_module=soundfile,
                resample_fn=resample,
                numpy_module=FakeNumpy,
                clock=lambda: next(clock_values),
            )
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "hello world")
        self.assertEqual(result["language"], "English")
        self.assertEqual(
            result["segments"],
            [{"start": 0.0, "end": 3 / 16_000, "text": "hello world"}],
        )
        self.assertEqual(result["device"], "mps")
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(result["chunk_seconds"], 30.0)
        self.assertEqual(result["timestamp_semantics"], "chunk/segment offsets")
        self.assertEqual(result["load_seconds"], 2.5)
        self.assertEqual(result["transcribe_seconds"], 1.25)
        self.assertEqual(soundfile.read_calls[0][1:], ("float32", False))
        self.assertEqual(resample_calls[0][1:], (2, 1))
        audio_kwargs = processor.process_calls[0]
        self.assertEqual(set(audio_kwargs), {"audio"})
        self.assertIsInstance(audio_kwargs["audio"], FakeArray)
        self.assertEqual(audio_kwargs["audio"].dtype, "float32")
        self.assertEqual(audio_kwargs["audio"].ndim, 1)
        self.assertEqual(processor.decode_calls, [("generated-after-prompt", "parsed")])
        self.assertEqual(model.to_calls, ["mps"])
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 123)
        self.assertFalse(model.generate_calls[0]["do_sample"])
        self.assertTrue(result["effective_config"]["local_files_only"])
        self.assertFalse(result["effective_config"]["forced_aligner"])
        self.assertEqual(result["effective_config"]["chunk_seconds"], 30.0)
        self.assertEqual(result["effective_config"]["chunk_count"], 1)
        self.assertEqual(result["effective_config"]["chunk_overlap_seconds"], 0.0)
        self.assertTrue(result["effective_config"]["chunked"])
        self.assertFalse(result["timestamps_supported"])

    def test_forced_language_is_passed_and_default_generation_is_long_enough(self):
        processor = FakeProcessor(
            {"language": "Russian", "transcription": "привет"}
        )
        model = FakeModel()
        result = qwen3_asr_hf.execute(
            self.request | {"language": "ru"},
            model_loader=lambda path: (processor, model),
            torch_module=SimpleNamespace(
                backends=SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: False),
                )
            ),
            soundfile_module=FakeSoundfile(sample_rate=16_000),
            numpy_module=FakeNumpy,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(processor.process_calls[0]["language"], "ru")
        self.assertEqual(model.generate_calls[0]["max_new_tokens"], 4096)
        self.assertEqual(result["device"], "cpu")

    def test_fixed_chunks_join_nonempty_transcripts_and_return_offsets(self):
        processor = FakeChunkProcessor(
            [
                {"language": "English", "transcription": " first "},
                {"language": "English", "transcription": " "},
                {"language": "English", "transcription": "third"},
            ]
        )
        model = FakeModel()
        loader = mock.Mock(return_value=(processor, model))
        soundfile = FakeSoundfile(
            sample_rate=16_000,
            audio=FakeArray([0.0] * 32_001),
        )
        result = qwen3_asr_hf.execute(
            self.request | {"language": "auto", "chunk_seconds": 1},
            model_loader=loader,
            torch_module=self.torch,
            soundfile_module=soundfile,
            numpy_module=FakeNumpy,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "first third")
        self.assertEqual(result["language"], "English")
        self.assertEqual(
            result["segments"],
            [
                {"start": 0.0, "end": 1.0, "text": "first"},
                {"start": 1.0, "end": 2.0, "text": ""},
                {"start": 2.0, "end": 2.0000625, "text": "third"},
            ],
        )
        self.assertEqual(result["chunk_count"], 3)
        self.assertEqual(result["effective_config"]["chunk_count"], 3)
        self.assertEqual(result["effective_config"]["chunk_seconds"], 1.0)
        self.assertEqual(result["effective_config"]["chunk_overlap_seconds"], 0.0)
        self.assertEqual(result["effective_config"]["timestamp_semantics"], "chunk/segment offsets")
        self.assertFalse(result["timestamps_supported"])
        self.assertEqual(len(processor.process_calls), 3)
        self.assertTrue(all(set(call) == {"audio"} for call in processor.process_calls))
        self.assertEqual(len(model.generate_calls), 3)
        self.assertEqual(len(processor.decode_calls), 3)
        loader.assert_called_once_with(self.request["model_path"])
        self.assertEqual(
            [call["audio"].values for call in processor.process_calls],
            [[0.0] * 16_000, [0.0] * 16_000, [0.0]],
        )

    def test_fixed_chunks_repeat_forced_language_for_each_chunk(self):
        processor = FakeChunkProcessor(
            [
                {"language": "Russian", "transcription": "раз"},
                {"language": "Russian", "transcription": "два"},
            ]
        )
        model = FakeModel()
        soundfile = FakeSoundfile(
            sample_rate=16_000,
            audio=FakeArray([0.0] * 32_000),
        )
        result = qwen3_asr_hf.execute(
            self.request | {"language": "ru", "chunk_seconds": 1},
            model_loader=lambda path: (processor, model),
            torch_module=self.torch,
            soundfile_module=soundfile,
            numpy_module=FakeNumpy,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "раз два")
        self.assertEqual(result["segments"], [
            {"start": 0.0, "end": 1.0, "text": "раз"},
            {"start": 1.0, "end": 2.0, "text": "два"},
        ])
        self.assertEqual([call["language"] for call in processor.process_calls], ["ru", "ru"])
        self.assertEqual(result["chunk_count"], 2)

    def test_list_stereo_audio_is_downmixed_before_numpy_conversion(self):
        processor = FakeProcessor(
            {"language": "English", "transcription": "ok"}
        )
        model = FakeModel()

        class ListSoundfile:
            def read(self, path, *, dtype, always_2d):
                return [[0.0, 1.0], [0.5, 0.5]], 16_000

        result = qwen3_asr_hf.execute(
            self.request,
            model_loader=lambda path: (processor, model),
            torch_module=SimpleNamespace(
                backends=SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: False),
                )
            ),
            soundfile_module=ListSoundfile(),
            numpy_module=FakeNumpy,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(processor.process_calls[0]["audio"].values, [0.5, 0.5])

    def test_parsed_decode_falls_back_to_transcription_only_after_prompt_slice(self):
        processor = FakeProcessor("  fallback transcript  ")
        model = FakeModel()
        result = qwen3_asr_hf.execute(
            self.request,
            model_loader=lambda path: (processor, model),
            torch_module=self.torch,
            soundfile_module=FakeSoundfile(sample_rate=16_000),
            numpy_module=FakeNumpy,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "fallback transcript")
        self.assertIsNone(result["language"])
        self.assertEqual(
            processor.decode_calls,
            [
                ("generated-after-prompt", "parsed"),
                ("generated-after-prompt", "transcription_only"),
            ],
        )
        self.assertEqual(result["effective_config"]["decode_format"], "transcription_only")

    def test_load_and_transcribe_failures_are_structured(self):
        result = qwen3_asr_hf.execute(
            self.request,
            model_loader=lambda path: (_ for _ in ()).throw(
                RuntimeError("load failed")
            ),
        )
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        processor = FakeProcessor({"language": "English", "transcription": "ok"})
        model = FakeModel()
        model.generate = mock.Mock(side_effect=RuntimeError("transcribe failed"))
        result = qwen3_asr_hf.execute(
            self.request,
            model_loader=lambda path: (processor, model),
            torch_module=self.torch,
            soundfile_module=FakeSoundfile(sample_rate=16_000),
            numpy_module=FakeNumpy,
        )
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("transcribe failed", result["error"])

    def test_cli_emits_one_json_object_and_returns_status_code(self):
        success = {
            "status": "ok",
            "transcript": "ok",
            "segments": [],
            "language": "English",
            "load_seconds": 0.1,
            "transcribe_seconds": 0.2,
            "effective_config": {},
        }
        output = io.StringIO()
        with (
            mock.patch.object(qwen3_asr_hf, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(json.dumps(self.request))),
            contextlib.redirect_stdout(output),
        ):
            return_code = qwen3_asr_hf.main()

        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = qwen3_asr_hf.main()
        self.assertNotEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
