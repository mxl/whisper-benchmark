import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stt_benchmark.workers import whisper_cpp


class WhisperCppWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.model_path = root / "ggml-tiny-q5_1.bin"
        self.audio_path = root / "sample.mp3"
        self.model_path.write_bytes(b"model-data")
        self.audio_path.write_bytes(b"audio-data")
        self.request = {
            "model_path": str(self.model_path),
            "audio_path": str(self.audio_path),
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def _version_result() -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="whisper.cpp version: 1.9.2\n", stderr=""
        )

    @staticmethod
    def _cli_document() -> dict:
        return {
            "systeminfo": "test",
            "model": {"type": "tiny"},
            "params": {"language": "ru", "translate": False},
            "result": {"language": "ru"},
            "transcription": [
                {
                    "timestamps": {"from": "00:00:00,000", "to": "00:00:01,250"},
                    "offsets": {"from": 0, "to": 1250},
                    "text": " first",
                    "tokens": [],
                },
                {
                    "timestamps": {"from": "00:00:01.250", "to": "00:00:02.500"},
                    "offsets": {"from": 1250, "to": 2500},
                    "text": " second ",
                },
            ],
        }

    def _runner(self, calls, document=None):
        document = self._cli_document() if document is None else document

        def run(command, **kwargs):
            calls.append((command, kwargs))
            if command[-1] == "--version":
                return self._version_result()
            output_base = Path(command[command.index("--output-file") + 1])
            output_base.with_name(output_base.name + ".json").write_text(
                json.dumps(document), encoding="utf-8"
            )
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="", stderr=""
            )

        return run

    def test_validation_happens_before_subprocess(self) -> None:
        runner = mock.Mock()
        invalid_requests = [
            {"model_path": "https://example.invalid/model", "audio_path": str(self.audio_path)},
            self.request | {"audio_path": "org/repo"},
            self.request | {"beam_size": 0},
            self.request | {"threads": True},
            self.request | {"task": "summarize"},
        ]
        for request in invalid_requests:
            with self.subTest(request=request):
                result = whisper_cpp.execute(request, run_fn=runner)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error_type"], "validation_error")
        runner.assert_not_called()

    def test_exact_command_defaults_and_translate_options(self) -> None:
        calls = []
        result = whisper_cpp.execute(
            self.request
            | {"language": "ru", "task": "translate", "beam_size": 3},
            run_fn=self._runner(calls),
            clock=iter([10.0, 10.25]).__next__,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(calls), 2)
        command, kwargs = calls[1]
        self.assertEqual(
            command[:10],
            [
                "/opt/homebrew/bin/whisper-cli",
                "--model",
                str(self.model_path),
                str(self.audio_path),
                "--threads",
                "8",
                "--beam-size",
                "3",
                "--output-json-full",
                "--output-file",
            ],
        )
        self.assertEqual(command[-3:], ["--language", "ru", "--translate"])
        self.assertEqual(kwargs["cwd"].startswith("/"), True)
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(result["load_seconds"], None)
        self.assertEqual(result["transcribe_seconds"], 0.25)

        options = result["effective_config"]["command_options"]
        self.assertEqual(options["threads"], 8)
        self.assertEqual(options["beam_size"], 3)
        self.assertTrue(options["translate"])
        self.assertEqual(result["quantization"], "q5_1")
        self.assertEqual(result["model_size_bytes"], len(b"model-data"))

    def test_current_json_full_output_is_normalized(self) -> None:
        calls = []
        result = whisper_cpp.execute(
            self.request,
            run_fn=self._runner(calls),
            clock=iter([0.0, 1.5]).__next__,
        )

        self.assertEqual(result["transcript"], "first second")
        self.assertEqual(
            result["segments"],
            [
                {"start": 0.0, "end": 1.25, "text": "first"},
                {"start": 1.25, "end": 2.5, "text": "second"},
            ],
        )
        self.assertEqual(result["language"], "ru")
        self.assertEqual(result["cli_version"], "1.9.2")

    def test_falls_back_to_json_when_full_json_is_unsupported(self) -> None:
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if command[-1] == "--version":
                return self._version_result()
            if "--output-json-full" in command:
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=2,
                    stdout="",
                    stderr="error: unknown argument --output-json-full",
                )
            output_base = Path(command[command.index("--output-file") + 1])
            output_base.with_suffix(".json").write_text(
                json.dumps(self._cli_document()), encoding="utf-8"
            )
            return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

        result = whisper_cpp.execute(self.request, run_fn=run)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(calls[2][calls[2].index("--output-json")], "--output-json")
        self.assertNotIn("--output-json-full", result["effective_config"]["command"])

    def test_nonzero_timeout_and_missing_output_are_structured(self) -> None:
        def spawn_error(command, **kwargs):
            raise FileNotFoundError("whisper-cli not found")

        result = whisper_cpp.execute(self.request, run_fn=spawn_error)
        self.assertEqual(result["error_type"], "spawn_error")

        def nonzero(command, **kwargs):
            if command[-1] == "--version":
                return self._version_result()
            return subprocess.CompletedProcess(
                args=command, returncode=7, stdout="partial", stderr="failed"
            )

        result = whisper_cpp.execute(self.request, run_fn=nonzero)
        self.assertEqual(result["error_type"], "cli_error")
        self.assertEqual(result["returncode"], 7)
        self.assertEqual(result["stdout"], "partial")

        def timeout(command, **kwargs):
            if command[-1] == "--version":
                return self._version_result()
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="partial")

        result = whisper_cpp.execute(
            self.request,
            run_fn=timeout,
            clock=iter([2.0, 3.0]).__next__,
        )
        self.assertEqual(result["error_type"], "timeout_error")
        self.assertEqual(result["transcribe_seconds"], 1.0)

        result = whisper_cpp.execute(
            self.request,
            run_fn=lambda command, **kwargs: (
                self._version_result()
                if command[-1] == "--version"
                else subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")
            ),
        )
        self.assertEqual(result["error_type"], "protocol_error")
        self.assertIn("did not produce", result["error"])

    def test_cli_protocol_is_one_json_object_and_exit_status_matches(self) -> None:
        success = {"status": "ok", "transcript": "ok"}
        output = io.StringIO()
        with (
            mock.patch.object(whisper_cpp, "execute", return_value=success),
            mock.patch("sys.stdin", io.StringIO(json.dumps(self.request))),
            contextlib.redirect_stdout(output),
        ):
            return_code = whisper_cpp.main()
        self.assertEqual(return_code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue()), success)

        output = io.StringIO()
        with (
            mock.patch("sys.stdin", io.StringIO("not json")),
            contextlib.redirect_stdout(output),
        ):
            return_code = whisper_cpp.main()
        self.assertEqual(return_code, 1)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
