"""Audio-energy check for EDL cut edges — the verification "ear".

For each cut edge (start and end of every range in an EDL), measures the
audio RMS in a small window centered on the edge in the SOURCE file and
compares it to an adaptive per-source noise floor. An edge whose energy is
above the floor sits in continuous speech — a HARD SPLICE that risks audible
residue of a removed word or a clipped onset.

The agent cannot literally hear audio; this is the objective proxy that
catches what a waveform PNG cannot. The final subjective listen is the human's.

Usage:
    python helpers/check_cuts.py <edl.json>
    python helpers/check_cuts.py <edl.json> --half 0.025 --margin 0.15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np


def load_pcm(path: Path, sr: int = 16000) -> np.ndarray:
    """Extract whole-file mono PCM as float32 in [-1, 1]."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = Path(f.name)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(wav),
        ]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0 or not wav.exists() or wav.stat().st_size == 0:
            return np.zeros(0, dtype=np.float32)
        with wave.open(str(wav), "rb") as w:
            frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        wav.unlink(missing_ok=True)


def noise_floor(pcm: np.ndarray, sr: int, margin: float, hop: float = 0.02) -> float:
    """Adaptive threshold: `margin` of the way up from the 20th to the 80th
    percentile of frame RMS. Below it = effectively silence for this source."""
    n = max(1, int(sr * hop))
    usable = (pcm.size // n) * n
    if usable == 0:
        return 0.02
    rms = np.sqrt(np.mean(pcm[:usable].reshape(-1, n) ** 2, axis=1))
    lo, hi = float(np.percentile(rms, 20)), float(np.percentile(rms, 80))
    return lo + margin * (hi - lo)


def edge_rms(pcm: np.ndarray, sr: int, t: float, half: float) -> float:
    a = max(0, int((t - half) * sr))
    b = min(pcm.size, int((t + half) * sr))
    seg = pcm[a:b]
    return float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Flag EDL cut edges that land in speech")
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("--half", type=float, default=0.025, help="Half-window (s) around each edge")
    ap.add_argument("--margin", type=float, default=0.15, help="Floor position 0..1 between p20 and p80 RMS")
    args = ap.parse_args()

    if not args.edl.exists():
        sys.exit(f"edl not found: {args.edl}")
    edl = json.loads(args.edl.read_text())
    sources = edl.get("sources", {})
    ranges = edl.get("ranges", [])
    if not ranges:
        sys.exit("edl has no ranges")

    cache: dict[str, tuple[np.ndarray, int, float]] = {}
    flags: list[tuple[str, str, float, float, float]] = []
    checked = 0

    for r in ranges:
        src_key = r["source"]
        src = Path(sources[src_key])
        if src_key not in cache:
            pcm = load_pcm(src)
            sr = 16000
            cache[src_key] = (pcm, sr, noise_floor(pcm, sr, args.margin))
        pcm, sr, thr = cache[src_key]
        if pcm.size == 0:
            continue
        for label, t in (("start", float(r["start"])), ("end", float(r["end"]))):
            checked += 1
            e = edge_rms(pcm, sr, t, args.half)
            if e > thr:
                flags.append((src_key, label, t, round(e, 4), round(thr, 4)))

    print(f"checked {checked} cut edge(s) across {len(cache)} source(s)")
    if not flags:
        print("OK — all cut edges land in silence.")
        return
    print(f"\n{len(flags)} HARD-SPLICE edge(s) — edge sits in speech (residue/clip risk):")
    for s, lab, t, e, thr in flags:
        print(f"  {s} {lab}={t:.3f}s  rms={e} > floor {thr}"
              f"   → snap to nearest trough or add ≥60ms crossfade, then re-check")
    sys.exit(1)


if __name__ == "__main__":
    main()
