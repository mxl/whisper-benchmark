import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from stt_benchmark.workers import podlodka


class PodlodkaWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "model"
        self.model_path.mkdir()
        (self.model_path / "config.json").write_text("{}")
        (self.model_path / "model.safetensors").write_bytes(b"model")
        self.audio_path = root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.audio = np.array([0.0, 0.25, -0.5], dtype=np.float32)
        self.soundfile = mock.Mock()
        self.soundfile.read.return_value = (self.audio, 16_000)
        self.soundfile_patch = mock.patch.dict(
            sys.modules, {"soundfile": self.soundfile}
        )
        self.soundfile_patch.start()
        self.addCleanup(self.soundfile_patch.stop)
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def fake_torch(*, mps_available: bool) -> types.ModuleType:
        torch = types.ModuleType("torch")
        torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available)
        )
        torch.inference_mode = contextlib.nullcontext
        return torch

    @staticmethod
    def fake_bundle(
        texts: list[str] | None = None,
    ) -> tuple[mock.Mock, mock.Mock, mock.Mock]:
        processor = mock.Mock()
        input_tensor = mock.Mock(name="input_features")
        input_tensor.to.return_value = input_tensor
        processor.return_value = {"input_features": input_tensor}

        model = mock.Mock()
        model.to.return_value = model
        model.eval.return_value = model
        generated = [types.SimpleNamespace(sequences=["token-0"])]
        if texts is not None:
            generated = [
                types.SimpleNamespace(sequences=[f"token-{index}"])
                for index in range(len(texts))
            ]
            processor.batch_decode.side_effect = [[text] for text in texts]
        else:
            processor.batch_decode.return_value = ["ok"]
        model.generate.side_effect = generated
        return processor, model, input_tensor

    def test_validation_rejects_paths_required_files_and_options_before_loading(self) -> None:
        loader = mock.Mock()
        cases = [
            self.request | {"model_path": "https://example.invalid/model"},
            self.request | {"audio_path": "file:///tmp/audio.wav"},
            self.request | {"audio_path": "missing.wav"},
            self.request | {"language": ""},
            self.request | {"language": 7},
            self.request | {"max_new_tokens": 0},
            self.request | {"max_new_tokens": -1},
            self.request | {"max_new_tokens": 1.0},
            self.request | {"max_new_tokens": True},
            self.request | {"device": "cuda"},
            self.request | {"device": 3},
            self.request | {"chunk_seconds": 0},
            self.request | {"chunk_seconds": -1},
            self.request | {"chunk_seconds": 30.01},
            self.request | {"chunk_seconds": "30"},
            self.request | {"chunk_seconds": True},
            self.request | {"chunk_seconds": float("nan")},
        ]
        for request in cases:
            with self.subTest(request=request):
                result = podlodka.execute(
                    request,
                    model_loader=loader,
                    torch_module=self.fake_torch(mps_available=False),
                )
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error_type"], "validation_error")
        loader.assert_not_called()

        missing_model = Path(self.tempdir.name) / "missing-model-files"
        missing_model.mkdir()
        (missing_model / "config.json").write_text("{}")
        result = podlodka.execute(
            self.request | {"model_path": str(missing_model)},
            model_loader=loader,
            torch_module=self.fake_torch(mps_available=False),
        )
        self.assertEqual(result["error_type"], "validation_error")
        self.assertIn("model weights", result["error"])
        loader.assert_not_called()

    def test_sharded_safetensors_and_bin_layouts_are_accepted(self) -> None:
        for suffix in ("safetensors", "bin"):
            with self.subTest(suffix=suffix):
                processor, model, _input_tensor = self.fake_bundle()
                model_path = Path(self.tempdir.name) / f"sharded-{suffix}"
                model_path.mkdir()
                (model_path / "config.json").write_text("{}")
                (model_path / f"model-00001-of-00002.{suffix}").write_bytes(b"model")
                result = podlodka.execute(
                    self.request | {"model_path": str(model_path)},
                    model_loader=lambda _path: (processor, model),
                    torch_module=self.fake_torch(mps_available=False),
                )
                self.assertEqual(result["status"], "ok")

    def test_direct_generate_uses_local_snapshot_and_auto_language(self) -> None:
        processor, model, input_tensor = self.fake_bundle()
        loader = mock.Mock(return_value=(processor, model))
        clock_values = iter([10.0, 12.5, 20.0, 21.25])

        result = podlodka.execute(
            self.request | {"language": "AUTO", "max_new_tokens": 123},
            model_loader=loader,
            torch_module=self.fake_torch(mps_available=False),
            clock=lambda: next(clock_values),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "ok")
        self.assertEqual(
            result["segments"],
            [{"start": 0.0, "end": 3 / 16_000, "text": "ok"}],
        )
        self.assertIsNone(result["language"])
        self.assertEqual(result["load_seconds"], 2.5)
        self.assertEqual(result["transcribe_seconds"], 1.25)
        loader.assert_called_once_with(self.request["model_path"])
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        processor.assert_called_once()
        np.testing.assert_array_equal(processor.call_args.args[0], self.audio)
        self.assertEqual(
            processor.call_args.kwargs,
            {"sampling_rate": 16_000, "return_tensors": "pt"},
        )
        input_tensor.to.assert_called_once_with("cpu")
        model.generate.assert_called_once_with(
            input_features=input_tensor,
            task="transcribe",
            max_new_tokens=123,
        )
        processor.batch_decode.assert_called_once_with(
            ["token-0"],
            skip_special_tokens=True,
        )
        effective = result["effective_config"]
        self.assertIsNone(effective["language"])
        self.assertTrue(effective["offline"])
        self.assertTrue(effective["local_files_only"])
        self.assertEqual(effective["device"], "cpu")
        self.assertEqual(effective["input_format"], "raw_float32")
        self.assertEqual(effective["sampling_rate"], 16_000)
        self.assertEqual(effective["model_class"], "WhisperForConditionalGeneration")
        self.assertEqual(effective["processor"], "AutoProcessor")
        self.assertTrue(effective["direct_generate"])
        self.assertEqual(effective["chunk_seconds"], 30.0)
        self.assertTrue(effective["no_torchcodec_path"])
        for pipeline_field in (
            "pipeline_task",
            "return_timestamps",
            "word_timestamps",
            "no_word_timestamps",
        ):
            self.assertNotIn(pipeline_field, effective)
        self.assertEqual(result["weights"], ["model.safetensors"])
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        self.soundfile.read.assert_called_once_with(
            self.request["audio_path"], dtype="float32", always_2d=False
        )

    def test_fixed_chunks_use_chunk_boundaries_and_structured_sequences(self) -> None:
        self.soundfile.read.return_value = (
            np.arange(32_000, dtype=np.float32),
            16_000,
        )
        processor, model, input_tensor = self.fake_bundle(["first", "second"])
        model.generate.side_effect = [
            types.SimpleNamespace(sequences=["first-ids"]),
            types.SimpleNamespace(sequences=["second-ids"]),
        ]
        result = podlodka.execute(
            self.request | {"chunk_seconds": 1.0, "language": "en", "device": "mps"},
            model_loader=lambda _path: (processor, model),
            torch_module=self.fake_torch(mps_available=True),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "first second")
        self.assertEqual(
            result["segments"],
            [
                {"start": 0.0, "end": 1.0, "text": "first"},
                {"start": 1.0, "end": 2.0, "text": "second"},
            ],
        )
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["effective_config"]["device"], "mps")
        model.to.assert_called_once_with("mps")
        self.assertEqual(processor.call_count, 2)
        np.testing.assert_array_equal(
            processor.call_args_list[0].args[0], np.arange(16_000, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            processor.call_args_list[1].args[0],
            np.arange(16_000, 32_000, dtype=np.float32),
        )
        self.assertEqual(model.generate.call_count, 2)
        for call in model.generate.call_args_list:
            self.assertIs(call.kwargs["input_features"], input_tensor)
            self.assertEqual(call.kwargs["task"], "transcribe")
            self.assertEqual(call.kwargs["language"], "en")
            self.assertEqual(
                call.kwargs["max_new_tokens"], podlodka.DEFAULT_MAX_NEW_TOKENS
            )

    def test_stereo_audio_is_downmixed_to_float32_before_direct_inference(self) -> None:
        stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        self.soundfile.read.return_value = (stereo, 16_000)
        processor, model, _input_tensor = self.fake_bundle()

        result = podlodka.execute(
            self.request,
            model_loader=lambda _path: (processor, model),
            torch_module=self.fake_torch(mps_available=False),
        )

        self.assertEqual(result["status"], "ok")
        direct_input = processor.call_args.args[0]
        np.testing.assert_array_equal(direct_input, [2.0, 3.0])
        self.assertEqual(direct_input.dtype, np.float32)

    def test_unsupported_sample_rate_is_a_transcribe_error_without_scipy(self) -> None:
        self.soundfile.read.return_value = (self.audio, 8_000)
        processor, model, _input_tensor = self.fake_bundle()

        with mock.patch.dict(sys.modules, {"scipy": None, "scipy.signal": None}):
            result = podlodka.execute(
                self.request,
                model_loader=lambda _path: (processor, model),
                torch_module=self.fake_torch(mps_available=False),
            )

        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("scipy is required", result["error"])
        model.generate.assert_not_called()

    def test_default_loader_uses_direct_transformers_classes_and_not_pipeline(self) -> None:
        processor, model, _input_tensor = self.fake_bundle()
        processor_loader = mock.Mock(return_value=processor)
        model_loader = mock.Mock(return_value=model)
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = types.SimpleNamespace(
            from_pretrained=processor_loader
        )
        transformers.WhisperForConditionalGeneration = types.SimpleNamespace(
            from_pretrained=model_loader
        )

        with mock.patch.dict(
            sys.modules,
            {
                "torch": self.fake_torch(mps_available=True),
                "transformers": transformers,
            },
        ):
            result = podlodka.execute(self.request)

        self.assertEqual(result["status"], "ok")
        processor_loader.assert_called_once_with(
            self.request["model_path"], local_files_only=True
        )
        model_loader.assert_called_once_with(
            self.request["model_path"],
            local_files_only=True,
            dtype="auto",
        )
        self.assertEqual(result["effective_config"]["device"], "mps")

    def test_default_loader_falls_back_to_legacy_torch_dtype(self) -> None:
        processor, model, _input_tensor = self.fake_bundle()
        processor_loader = mock.Mock(return_value=processor)
        model_loader = mock.Mock(
            side_effect=[TypeError("unexpected keyword argument 'dtype'"), model]
        )
        transformers = types.ModuleType("transformers")
        transformers.AutoProcessor = types.SimpleNamespace(
            from_pretrained=processor_loader
        )
        transformers.WhisperForConditionalGeneration = types.SimpleNamespace(
            from_pretrained=model_loader
        )

        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            loaded_processor, loaded_model = podlodka._load_model(
                self.request["model_path"]
            )

        self.assertIs(loaded_processor, processor)
        self.assertIs(loaded_model, model)
        self.assertEqual(
            model_loader.call_args_list[1].kwargs,
            {"local_files_only": True, "torch_dtype": "auto"},
        )

    def test_load_and_transcribe_failures_are_error_payloads(self) -> None:
        result = podlodka.execute(
            self.request,
            model_loader=lambda _path: (_ for _ in ()).throw(
                RuntimeError("load failed")
            ),
            torch_module=self.fake_torch(mps_available=False),
        )
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        processor, model, _input_tensor = self.fake_bundle()
        model.generate.side_effect = RuntimeError("transcribe failed")
        result = podlodka.execute(
            self.request,
            model_loader=lambda _path: (processor, model),
            torch_module=self.fake_torch(mps_available=False),
        )
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("transcribe failed", result["error"])

    def test_unsupported_decode_result_is_a_transcribe_error(self) -> None:
        processor, model, _input_tensor = self.fake_bundle()
        processor.batch_decode.return_value = []
        result = podlodka.execute(
            self.request,
            model_loader=lambda _path: (processor, model),
            torch_module=self.fake_torch(mps_available=False),
        )
        self.assertEqual(result["error_type"], "transcribe_error")

    def test_cli_emits_one_json_object_and_returns_status_code(self) -> None:
        success = {
            "status": "ok",
            "transcript": "ok",
            "segments": [],
            "language": None,
            "load_seconds": 0.1,
            "transcribe_seconds": 0.2,
            "effective_config": {},
        }
        output = io.StringIO()
        with (
            mock.patch.object(podlodka, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(json.dumps(self.request))),
            contextlib.redirect_stdout(output),
        ):
            return_code = podlodka.main()

        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = podlodka.main()

        self.assertNotEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
