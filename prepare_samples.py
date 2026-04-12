#!/usr/bin/env python3
"""
Download and prepare copyright-safe audio samples for ASR benchmarking.

English: LibriSpeech test-clean (CC BY 4.0)
  Speaker 1089, chapter 134686 — ~3 min of clean audiobook read speech.
  Reference: https://openslr.org/12/

Russian: Russian LibriSpeech SLR96 (Public Domain)
  Automatically selects the longest available chapter from any speaker.
  Reference: https://openslr.org/96/

Both clips are concatenated from consecutive utterances, trimmed to a target
duration, and saved as a single MP3 + matching .txt reference transcript.

Usage:
    python3 prepare_samples.py                     # prepare both
    python3 prepare_samples.py --lang en           # English only
    python3 prepare_samples.py --lang ru           # Russian only
    python3 prepare_samples.py --output-dir /path  # custom output dir
    python3 prepare_samples.py --target-duration 180  # ~3 min clips
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

EN_ARCHIVE_URL = "https://openslr.trmal.net/resources/12/test-clean.tar.gz"
EN_SPEAKER = "1089"
EN_CHAPTER = "134686"
EN_OUTPUT_STEM = "librispeech_1089_134686"
EN_ATTRIBUTION = (
    "LibriSpeech test-clean, speaker 1089 chapter 134686. "
    "V. Panayotov et al., ICASSP 2015. License: CC BY 4.0. "
    "https://openslr.org/12/"
)

RU_ARCHIVE_URL = "https://openslr.trmal.net/resources/96/ruls_data.tar.gz"
RU_OUTPUT_STEM = "ruls_sample"
RU_ATTRIBUTION = (
    "Russian LibriSpeech (SLR96), LibriVox public-domain audiobook recordings. "
    "License: Public Domain. https://openslr.org/96/"
)

DEFAULT_TARGET_DURATION = 180  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required binary not found: {name}. Install it and retry.")


def download_with_progress(url: str, dest: Path) -> None:
    log(f"Downloading {url} -> {dest.name} ...")

    tmp_dest = dest.with_name(dest.name + ".part")
    if tmp_dest.exists():
        tmp_dest.unlink()

    def reporthook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  {pct:3d}%", end="", flush=True, file=sys.stderr)

    try:
        urllib.request.urlretrieve(url, tmp_dest, reporthook=reporthook)
        print(file=sys.stderr)
        tmp_dest.replace(dest)
        log(f"  Saved to {dest}")
    except (urllib.error.URLError, OSError):
        print(file=sys.stderr)
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise


def validate_tar_gz(path: Path) -> bool:
    try:
        with tarfile.open(path, "r:gz") as tar:
            for _ in tar:
                pass
        return True
    except (tarfile.TarError, EOFError, OSError):
        return False


def ensure_archive(url: str, dest: Path) -> None:
    if dest.exists():
        log(f"  Archive already present: {dest}")
        if validate_tar_gz(dest):
            return

        log("  Existing archive is corrupted or incomplete; re-downloading.")
        dest.unlink()

    download_with_progress(url, dest)
    if not validate_tar_gz(dest):
        dest.unlink(missing_ok=True)
        raise SystemExit(
            f"Downloaded archive is corrupted or incomplete: {dest}. Please retry."
        )


def audio_duration_s(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def concat_audio_files(inputs: list[Path], output: Path) -> None:
    """Concatenate FLAC/WAV files into a single MP3 via ffmpeg concat demuxer."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as flist:
        for p in inputs:
            flist.write(f"file '{p.resolve()}'\n")
        flist_path = Path(flist.name)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(flist_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-b:a",
                "128k",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        flist_path.unlink()


# ---------------------------------------------------------------------------
# LibriSpeech (English, CC BY 4.0)
# ---------------------------------------------------------------------------


def prepare_english(
    output_dir: Path,
    target_duration: int,
    keep_archive: bool,
) -> None:
    log("\n=== English: LibriSpeech test-clean ===")
    require_binary("ffmpeg")
    require_binary("ffprobe")

    archive_path = output_dir / "librispeech_test_clean.tar.gz"
    ensure_archive(EN_ARCHIVE_URL, archive_path)

    log("  Extracting speaker 1089 chapter 134686 ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Extract only the files we need (fast: skips the rest of test-clean)
        chapter_prefix = f"LibriSpeech/test-clean/{EN_SPEAKER}/{EN_CHAPTER}/"
        with tarfile.open(archive_path, "r:gz") as tar:
            members = [m for m in tar if m.name.startswith(chapter_prefix)]
            if not members:
                raise SystemExit(
                    f"Chapter {EN_CHAPTER} not found in archive. "
                    "The archive structure may have changed."
                )
            tar.extractall(path=tmp, members=members)

        chapter_dir = tmp / "LibriSpeech" / "test-clean" / EN_SPEAKER / EN_CHAPTER
        flac_files = sorted(chapter_dir.glob("*.flac"))
        trans_file = chapter_dir / f"{EN_SPEAKER}-{EN_CHAPTER}.trans.txt"

        if not flac_files:
            raise SystemExit(f"No FLAC files found in {chapter_dir}")
        if not trans_file.exists():
            raise SystemExit(f"Transcript file not found: {trans_file}")

        # Read transcripts keyed by utterance id
        transcripts: dict[str, str] = {}
        for line in trans_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, _, text = line.partition(" ")
            transcripts[utt_id] = text.strip()

        # Select consecutive utterances up to target duration
        selected_flac: list[Path] = []
        selected_text: list[str] = []
        total_duration = 0.0

        for flac in flac_files:
            utt_id = flac.stem
            if utt_id not in transcripts:
                continue
            dur = audio_duration_s(flac)
            if total_duration + dur > target_duration + 10:
                break
            selected_flac.append(flac)
            selected_text.append(transcripts[utt_id])
            total_duration += dur
            if total_duration >= target_duration:
                break

        if not selected_flac:
            raise SystemExit("No utterances selected — check target duration.")

        log(
            f"  Selected {len(selected_flac)} utterances, "
            f"total duration ~{total_duration:.1f}s"
        )

        # Build output
        output_mp3 = output_dir / f"{EN_OUTPUT_STEM}.mp3"
        output_txt = output_dir / f"{EN_OUTPUT_STEM}.txt"
        output_attr = output_dir / f"{EN_OUTPUT_STEM}.attribution.txt"

        log(f"  Concatenating -> {output_mp3.name} ...")
        concat_audio_files(selected_flac, output_mp3)

        transcript = " ".join(selected_text)
        output_txt.write_text(transcript + "\n", encoding="utf-8")
        output_attr.write_text(EN_ATTRIBUTION + "\n", encoding="utf-8")

        actual_dur = audio_duration_s(output_mp3)
        log(f"  Audio duration: {actual_dur:.1f}s")
        log(f"  Transcript words: {len(transcript.split())}")
        log(f"  Written: {output_mp3.name}, {output_txt.name}")

    if not keep_archive:
        archive_path.unlink()
        log("  Removed archive.")


# ---------------------------------------------------------------------------
# Russian LibriSpeech / SLR96 (Public Domain)
# ---------------------------------------------------------------------------


def find_longest_chapter(ruls_root: Path) -> tuple[str, str, list[Path]]:
    """Return (speaker_id, chapter_id, sorted_flac_list) for longest chapter."""
    best: tuple[str, str, list[Path]] | None = None
    best_count = 0

    for speaker_dir in sorted(ruls_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            flacs = sorted(chapter_dir.glob("*.flac"))
            if len(flacs) > best_count:
                best_count = len(flacs)
                best = (speaker_dir.name, chapter_dir.name, flacs)

    if best is None:
        raise SystemExit("No FLAC files found in extracted RuLS archive.")
    return best


def prepare_russian(
    output_dir: Path,
    target_duration: int,
    keep_archive: bool,
) -> None:
    log("\n=== Russian: Russian LibriSpeech SLR96 ===")
    require_binary("ffmpeg")
    require_binary("ffprobe")

    archive_path = output_dir / "ruls_data.tar.gz"
    ensure_archive(RU_ARCHIVE_URL, archive_path)

    log("  Scanning archive for longest chapter (this may take a moment) ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # First pass: find which speaker/chapter has the most files
        # without extracting everything
        chapter_counts: dict[str, int] = {}
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                parts = Path(member.name).parts
                # Expected: ruls_data/<speaker>/<chapter>/<file>
                if len(parts) == 4 and parts[-1].endswith(".flac"):
                    key = f"{parts[1]}/{parts[2]}"
                    chapter_counts[key] = chapter_counts.get(key, 0) + 1

        if not chapter_counts:
            raise SystemExit("No FLAC files found in archive.")

        best_chapter = max(chapter_counts, key=lambda k: chapter_counts[k])
        speaker_id, chapter_id = best_chapter.split("/")
        log(
            f"  Selected speaker={speaker_id} chapter={chapter_id} "
            f"({chapter_counts[best_chapter]} utterances)"
        )

        # Second pass: extract only the chosen chapter
        chapter_prefix = f"ruls_data/{speaker_id}/{chapter_id}/"
        with tarfile.open(archive_path, "r:gz") as tar:
            members = [m for m in tar if m.name.startswith(chapter_prefix)]
            tar.extractall(path=tmp, members=members)

        chapter_dir = tmp / "ruls_data" / speaker_id / chapter_id
        flac_files = sorted(chapter_dir.glob("*.flac"))
        trans_files = list(chapter_dir.glob("*.trans.txt"))

        if not flac_files:
            raise SystemExit(f"No FLAC files extracted to {chapter_dir}")
        if not trans_files:
            raise SystemExit(f"No transcript file found in {chapter_dir}")

        transcripts: dict[str, str] = {}
        for line in trans_files[0].read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt_id, _, text = line.partition(" ")
            transcripts[utt_id] = text.strip()

        # Select consecutive utterances up to target duration
        selected_flac: list[Path] = []
        selected_text: list[str] = []
        total_duration = 0.0

        for flac in flac_files:
            utt_id = flac.stem
            if utt_id not in transcripts:
                continue
            dur = audio_duration_s(flac)
            if total_duration + dur > target_duration + 10:
                break
            selected_flac.append(flac)
            selected_text.append(transcripts[utt_id])
            total_duration += dur
            if total_duration >= target_duration:
                break

        if not selected_flac:
            raise SystemExit("No utterances selected — check target duration.")

        log(
            f"  Selected {len(selected_flac)} utterances, "
            f"total duration ~{total_duration:.1f}s"
        )

        output_stem = f"{RU_OUTPUT_STEM}_{speaker_id}_{chapter_id}"
        output_mp3 = output_dir / f"{output_stem}.mp3"
        output_txt = output_dir / f"{output_stem}.txt"
        output_attr = output_dir / f"{output_stem}.attribution.txt"

        log(f"  Concatenating -> {output_mp3.name} ...")
        concat_audio_files(selected_flac, output_mp3)

        transcript = " ".join(selected_text)
        output_txt.write_text(transcript + "\n", encoding="utf-8")
        output_attr.write_text(RU_ATTRIBUTION + "\n", encoding="utf-8")

        actual_dur = audio_duration_s(output_mp3)
        log(f"  Audio duration: {actual_dur:.1f}s")
        log(f"  Transcript words: {len(transcript.split())}")
        log(f"  Written: {output_mp3.name}, {output_txt.name}")

    if not keep_archive:
        archive_path.unlink()
        log("  Removed archive.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare LibriSpeech (EN) and RuLS (RU) benchmark clips."
    )
    parser.add_argument(
        "--lang",
        choices=["en", "ru", "both"],
        default="both",
        help="Which language(s) to prepare. (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("samples"),
        help="Directory to write output files. (default: samples/)",
    )
    parser.add_argument(
        "--target-duration",
        type=int,
        default=DEFAULT_TARGET_DURATION,
        metavar="SECONDS",
        help=f"Target clip duration in seconds. (default: {DEFAULT_TARGET_DURATION})",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep downloaded .tar.gz archives after extraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.lang in ("en", "both"):
        prepare_english(args.output_dir, args.target_duration, args.keep_archive)

    if args.lang in ("ru", "both"):
        prepare_russian(args.output_dir, args.target_duration, args.keep_archive)

    log("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
