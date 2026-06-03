"""Transcribe a video with AssemblyAI.

Extracts mono 16kHz audio via ffmpeg, uploads to AssemblyAI with
speaker diarization + word-level timestamps + disfluencies (um/uh/restarts
kept as words so they remain cuttable), normalizes the response to the same
JSON shape the rest of the pipeline expects, and writes it to
<edit_dir>/transcripts/<video_stem>.json.

Output JSON shape (compatible with pack_transcripts.py and render.py):
  {
    "words": [
      { "type": "word", "text": "Hello", "start": 0.12, "end": 0.45,
        "speaker_id": "speaker_0" },
      ...
    ]
  }

Note: AssemblyAI does not produce "spacing" or "audio_event" entries.
Silence gaps are still detected by pack_transcripts.py via timing.

Cached: if the output file already exists, the upload is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --num-speakers 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests


ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
POLL_INTERVAL = 3  # seconds between status checks


def load_api_key() -> str:
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "ASSEMBLYAI_API_KEY":
                    return v.strip().strip('"').strip("'")
    v = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not v:
        sys.exit("ASSEMBLYAI_API_KEY not found in .env or environment")
    return v


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _upload(audio_path: Path, api_key: str) -> str:
    """Upload audio to AssemblyAI and return the hosted upload URL."""
    headers = {"authorization": api_key}
    with open(audio_path, "rb") as f:
        resp = requests.post(ASSEMBLYAI_UPLOAD_URL, headers=headers, data=f, timeout=1800)
    if resp.status_code != 200:
        raise RuntimeError(f"AssemblyAI upload failed {resp.status_code}: {resp.text[:500]}")
    return resp.json()["upload_url"]


def _submit(
    upload_url: str,
    api_key: str,
    language: str | None,
    num_speakers: int | None,
) -> str:
    """Submit a transcription job and return the job ID."""
    headers = {"authorization": api_key, "content-type": "application/json"}
    body: dict = {
        "audio_url": upload_url,
        "speech_models": ["universal-3-pro"],
        "speaker_labels": True,
        "disfluencies": True,  # keep um/uh/restarts as words so they're cuttable
    }
    if language:
        body["language_code"] = language
    if num_speakers:
        body["speakers_expected"] = num_speakers

    resp = requests.post(ASSEMBLYAI_TRANSCRIPT_URL, headers=headers, json=body, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"AssemblyAI submit failed {resp.status_code}: {resp.text[:500]}")
    return resp.json()["id"]


def _poll(job_id: str, api_key: str, verbose: bool) -> dict:
    """Poll until complete. Returns the full API response."""
    headers = {"authorization": api_key}
    url = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{job_id}"
    while True:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"AssemblyAI poll failed {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        status = data.get("status")
        if status == "completed":
            return data
        if status == "error":
            raise RuntimeError(f"AssemblyAI transcription error: {data.get('error')}")
        if verbose:
            print(f"  status: {status} …", flush=True)
        time.sleep(POLL_INTERVAL)


def _normalize(data: dict) -> dict:
    """Convert AssemblyAI response to the pipeline-expected JSON shape.

    Differences handled:
      - timestamps: milliseconds → seconds
      - speaker field: "A" → speaker_id "speaker_0" (stable index mapping)
      - adds type = "word" to every entry (no spacing/audio_event equivalents)
    """
    speaker_index: dict[str, int] = {}

    def speaker_id(label: str | None) -> str | None:
        if label is None:
            return None
        if label not in speaker_index:
            speaker_index[label] = len(speaker_index)
        return f"speaker_{speaker_index[label]}"

    normalized: list[dict] = []
    for w in data.get("words") or []:
        normalized.append({
            "type": "word",
            "text": w.get("text", ""),
            "start": w["start"] / 1000.0 if w.get("start") is not None else None,
            "end": w["end"] / 1000.0 if w.get("end") is not None else None,
            "speaker_id": speaker_id(w.get("speaker")),
        })

    return {"words": normalized}


def call_assemblyai(
    audio_path: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"  uploading {audio_path.name}", flush=True)
    upload_url = _upload(audio_path, api_key)

    if verbose:
        print("  submitting transcription job", flush=True)
    job_id = _submit(upload_url, api_key, language, num_speakers)

    if verbose:
        print(f"  job {job_id} — waiting for completion", flush=True)
    raw = _poll(job_id, api_key, verbose=verbose)

    return _normalize(raw)


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
    verbose: bool = True,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists():
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        audio = Path(tmp) / f"{video.stem}.wav"
        extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  audio: {video.stem}.wav ({size_mb:.1f} MB)", flush=True)
        payload = call_assemblyai(audio, api_key, language, num_speakers, verbose=verbose)

    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video with AssemblyAI")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy.",
    )
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    api_key = load_api_key()

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=api_key,
        language=args.language,
        num_speakers=args.num_speakers,
    )


if __name__ == "__main__":
    main()
