import os
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from stt_benchmark import subprocess_runner


class RunJsonWorkerTests(unittest.TestCase):
    def test_success_wraps_macos_time_and_parses_rss(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"text": "hello"}\n',
            stderr="worker log\nmaximum resident set size: 4096 KB\n",
        )
        with (
            mock.patch.object(subprocess_runner.platform, "system", return_value="Darwin"),
            mock.patch.object(
                subprocess_runner.subprocess, "run", return_value=completed
            ) as run,
            mock.patch.object(
                subprocess_runner.time, "perf_counter", side_effect=[10.0, 10.25]
            ),
        ):
            result = subprocess_runner.run_json_worker(
                Path("/venv/bin/python"),
                "worker_module",
                {"audio": "sample.wav"},
                5.0,
                {"WORKER_MODE": "test"},
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload, {"text": "hello"})
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '{"text": "hello"}\n')
        self.assertEqual(result.stderr, completed.stderr)
        self.assertEqual(result.wall_seconds, 0.25)
        self.assertEqual(result.peak_rss_mb, 4.0)
        self.assertIsNone(result.error)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["/usr/bin/time", "-l", "/venv/bin/python", "-m", "worker_module"])
        self.assertEqual(kwargs["input"], '{"audio": "sample.wav"}\n')
        self.assertEqual(kwargs["env"]["WORKER_MODE"], "test")
        self.assertEqual(kwargs["env"]["PATH"], os.environ["PATH"])
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["timeout"], 5.0)

    def test_timeout_returns_partial_output(self) -> None:
        timeout = subprocess.TimeoutExpired(
            cmd=["python"], timeout=1.0, output="partial", stderr="still running"
        )
        with (
            mock.patch.object(subprocess_runner.platform, "system", return_value="Linux"),
            mock.patch.object(subprocess_runner.subprocess, "run", side_effect=timeout),
            mock.patch.object(
                subprocess_runner.time, "perf_counter", side_effect=[2.0, 3.5]
            ),
        ):
            result = subprocess_runner.run_json_worker("python", "worker", {}, 1.0)

        self.assertEqual(result.status, "timeout")
        self.assertIsNone(result.payload)
        self.assertIsNone(result.returncode)
        self.assertEqual(result.stdout, "partial")
        self.assertEqual(result.stderr, "still running")
        self.assertEqual(result.wall_seconds, 1.5)
        self.assertIsNone(result.peak_rss_mb)

    def test_nonzero_exit_preserves_valid_payload(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=7, stdout='{"partial": true}', stderr="failed"
        )
        with mock.patch.object(
            subprocess_runner.subprocess, "run", return_value=completed
        ):
            result = subprocess_runner.run_json_worker("python", "worker", {}, 1.0)

        self.assertEqual(result.status, "worker_error")
        self.assertEqual(result.payload, {"partial": True})
        self.assertEqual(result.returncode, 7)

    def test_invalid_json_is_protocol_error(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json", stderr="log"
        )
        with mock.patch.object(
            subprocess_runner.subprocess, "run", return_value=completed
        ):
            result = subprocess_runner.run_json_worker("python", "worker", {}, 1.0)

        self.assertEqual(result.status, "protocol_error")
        self.assertIsNone(result.payload)
        self.assertEqual(result.stdout, "not json")
        self.assertEqual(result.stderr, "log")

    def test_spawn_error_is_returned(self) -> None:
        with mock.patch.object(
            subprocess_runner.subprocess,
            "run",
            side_effect=FileNotFoundError("python not found"),
        ):
            result = subprocess_runner.run_json_worker("missing-python", "worker", {}, 1.0)

        self.assertEqual(result.status, "spawn_error")
        self.assertIsNone(result.payload)
        self.assertIsNone(result.returncode)
        self.assertEqual(result.error, "python not found")


if __name__ == "__main__":
    unittest.main()
