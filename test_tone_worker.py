import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stt_benchmark.workers import tone


class DecoderType(Enum):
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"


@dataclass
class Phrase:
    text: str
    start_time: float
    end_time: float


class ToneWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "model"
        self.model_path.mkdir()
        (self.model_path / "model.onnx").write_bytes(b"model")
        self.audio_path = root / "audio.wav"
        self.audio_path.write_bytes(b"audio")
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_validation_rejects_invalid_paths_decoder_and_streaming_before_import(self) -> None:
        factory = mock.Mock()
        reader = mock.Mock()
        with mock.patch.object(
            tone, "_import_tone", side_effect=AssertionError("tone imported")
        ):
            cases = [
                {"model_path": "https://example.invalid/model"},
                self.request | {"audio_path": "missing.wav"},
                self.request | {"decoder": "unknown"},
                self.request | {"streaming": "yes"},
            ]
            for request in cases:
                with self.subTest(request=request):
                    result = tone.execute(
                        request,
                        pipeline_factory=factory,
                        read_audio_fn=reader,
                    )
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(result["error_type"], "validation_error")

        factory.assert_not_called()
        reader.assert_not_called()

    def test_model_artifacts_are_validated_and_beam_requires_kenlm(self) -> None:
        model_path = Path(self.request["model_path"])
        (model_path / "model.onnx").unlink()
        result = tone.execute(
            self.request,
            pipeline_factory=mock.Mock(),
            read_audio_fn=mock.Mock(),
        )
        self.assertEqual(result["error_type"], "validation_error")

        (model_path / "model.onnx").write_bytes(b"model")
        result = tone.execute(
            self.request | {"decoder": "beam"},
            pipeline_factory=mock.Mock(),
            read_audio_fn=mock.Mock(),
        )
        self.assertEqual(result["error_type"], "validation_error")

    def test_offline_greedy_uses_local_pipeline_and_audio(self) -> None:
        calls: list[tuple[str, DecoderType]] = []
        pipeline = mock.Mock()
        pipeline.forward_offline.return_value = [
            Phrase("Привет", 0.1, 0.8),
        ]

        class StreamingCTCPipeline:
            @classmethod
            def from_local(cls, path: str, *, decoder_type: DecoderType):
                calls.append((path, decoder_type))
                return pipeline

        audio = [1, 2, 3]
        reader = mock.Mock(return_value=audio)
        fake_tone = SimpleNamespace(
            DecoderType=DecoderType,
            StreamingCTCPipeline=StreamingCTCPipeline,
            read_audio=reader,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            result = tone.execute(
                self.request,
                tone_module=fake_tone,
                clock=iter([10.0, 12.5, 20.0, 21.25]).__next__,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["transcript"], "Привет")
        self.assertEqual(result["load_seconds"], 2.5)
        self.assertEqual(result["transcribe_seconds"], 1.25)
        self.assertEqual(result["mode"], "offline")
        self.assertEqual(result["decoder"], "greedy")
        self.assertEqual(calls, [(self.request["model_path"], DecoderType.GREEDY)])
        reader.assert_called_once_with(self.request["audio_path"])
        pipeline.forward_offline.assert_called_once_with(audio)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(
            result["effective_config"],
            {
                "local_files_only": True,
                "offline": True,
                "model_path": self.request["model_path"],
                "audio_path": self.request["audio_path"],
                "sample_rate": 8000,
                "chunk_samples": 2400,
                "streaming": False,
                "decoder": "greedy",
                "offline_environment": {"HF_HUB_OFFLINE": "1"},
            },
        )

    def test_beam_maps_to_beam_search(self) -> None:
        (Path(self.request["model_path"]) / "kenlm.bin").write_bytes(b"kenlm")
        factory = mock.Mock(return_value=mock.Mock(forward_offline=lambda audio: []))
        reader = mock.Mock(return_value=[])

        result = tone.execute(
            self.request | {"decoder": "beam"},
            pipeline_factory=factory,
            read_audio_fn=reader,
            tone_module=SimpleNamespace(DecoderType=DecoderType),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(factory.call_args.args[0], self.request["model_path"])
        self.assertIs(factory.call_args.args[1], DecoderType.BEAM_SEARCH)

    def test_offline_phrase_order_and_timestamps_skip_empty_text(self) -> None:
        pipeline = SimpleNamespace(
            forward_offline=lambda audio: [
                Phrase(" first ", 0.0, 0.5),
                Phrase("", 0.5, 0.6),
                Phrase("second", 0.7, 1.1),
            ]
        )
        result = tone.execute(
            self.request,
            pipeline_factory=mock.Mock(return_value=pipeline),
            read_audio_fn=mock.Mock(return_value=[]),
            tone_module=SimpleNamespace(DecoderType=DecoderType),
        )

        self.assertEqual(result["transcript"], "first second")
        self.assertEqual(
            result["timestamps"],
            [
                {"text": "first", "start_time": 0.0, "end_time": 0.5},
                {"text": "second", "start_time": 0.7, "end_time": 1.1},
            ],
        )

    def test_streaming_uses_exact_chunks_then_finalize_in_order(self) -> None:
        audio = list(range(4_800))
        forward_calls: list[tuple[list[int], Any]] = []
        final_states: list[Any] = []

        class Pipeline:
            def forward(self, chunk, state):
                forward_calls.append((list(chunk), state))
                index = len(forward_calls)
                return [Phrase(f"part-{index}", index, index + 0.5)], f"state-{index}"

            def finalize(self, state):
                final_states.append(state)
                return [Phrase("final", 3.0, 3.5)], "done"

        result = tone.execute(
            self.request | {"streaming": True},
            pipeline_factory=mock.Mock(return_value=Pipeline()),
            read_audio_fn=mock.Mock(return_value=audio),
            tone_module=SimpleNamespace(DecoderType=DecoderType),
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["mode"], "streaming")
        self.assertEqual(result["transcript"], "part-1 part-2 final")
        self.assertEqual([len(chunk) for chunk, _ in forward_calls], [2400, 2400])
        self.assertEqual(forward_calls[0][0], list(range(2400)))
        self.assertEqual(forward_calls[1][0], list(range(2400, 4800)))
        self.assertEqual([state for _, state in forward_calls], [None, "state-1"])
        self.assertEqual(final_states, ["state-2"])
        self.assertEqual(
            result["timestamps"],
            [
                {"text": "part-1", "start_time": 1.0, "end_time": 1.5},
                {"text": "part-2", "start_time": 2.0, "end_time": 2.5},
                {"text": "final", "start_time": 3.0, "end_time": 3.5},
            ],
        )

    def test_load_and_transcription_failures_are_error_payloads(self) -> None:
        result = tone.execute(
            self.request,
            pipeline_factory=mock.Mock(side_effect=RuntimeError("load failed")),
            read_audio_fn=mock.Mock(),
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "load_error")
        self.assertIn("load failed", result["error"])

        pipeline = mock.Mock()
        reader = mock.Mock(side_effect=RuntimeError("audio failed"))
        result = tone.execute(
            self.request,
            pipeline_factory=mock.Mock(return_value=pipeline),
            read_audio_fn=reader,
        )
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("audio failed", result["error"])
        pipeline.forward_offline.assert_not_called()

        pipeline.forward_offline.side_effect = RuntimeError("forward failed")
        result = tone.execute(
            self.request,
            pipeline_factory=mock.Mock(return_value=pipeline),
            read_audio_fn=mock.Mock(return_value=[]),
        )
        self.assertEqual(result["error_type"], "transcribe_error")
        self.assertIn("forward failed", result["error"])

    def test_cli_emits_one_json_object_and_returns_status_code(self) -> None:
        success = {
            "status": "ok",
            "transcript": "ok",
            "timestamps": [],
            "load_seconds": 0.1,
            "transcribe_seconds": 0.2,
            "mode": "offline",
            "decoder": "greedy",
            "effective_config": {},
        }
        output = io.StringIO()
        with (
            mock.patch.object(tone, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(json.dumps(self.request))),
            contextlib.redirect_stdout(output),
        ):
            return_code = tone.main()

        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = tone.main()

        self.assertNotEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
