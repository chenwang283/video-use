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
    # CLIP = a *clean* cut (small/no fade) that landed in speech → a kept word is
    #        being cut mid-sound (the 's' off "tasks", "deadline" cut short).
    # SPLICE = a cut the EDL marked hard (larger fade) that sits in speech →
    #          intended continuous-speech rejoin; just verify it sounds clean.
    # (We classify by the EDL fade, NOT by ASR word times — those are unreliable.)
    clips: list[tuple[str, str, float, float]] = []
    splices: list[tuple[str, str, float, float]] = []
    checked = 0
    HARD_FADE_MIN = 0.02   # fades >= this mark an intentional hard splice

    for r in ranges:
        src_key = r["source"]
        src = Path(sources[src_key])
        if src_key not in cache:
            pcm = load_pcm(src)
            cache[src_key] = (pcm, 16000, noise_floor(pcm, 16000, args.margin))
        pcm, sr, thr = cache[src_key]
        if pcm.size == 0:
            continue
        fin = float(r.get("fade_in", 0.0)); fout = float(r.get("fade_out", 0.0))
        for label, t, fade in (("start", float(r["start"]), fin),
                               ("end", float(r["end"]), fout)):
            checked += 1
            e = edge_rms(pcm, sr, t, args.half)
            if e > thr:
                (splices if fade >= HARD_FADE_MIN else clips).append(
                    (src_key, label, t, round(e, 4)))

    print(f"checked {checked} cut edge(s) across {len(cache)} source(s)")
    if clips:
        print(f"\n{len(clips)} WORD-CLIP edge(s) — a CLEAN cut landed in speech "
              f"(a kept word is cut mid-sound, e.g. an 's' or the word's tail):")
        for s, lab, t, e in clips:
            print(f"  {s} {lab}={t:.3f}s  rms={e}"
                  f"   → re-place this edge in the surrounding silence (place_edges.py"
                  f" extends to the word's audio extent + pad); re-check")
    if splices:
        print(f"\n{len(splices)} HARD-SPLICE edge(s) — intended mid-speech rejoin; "
              f"confirm it sounds clean (≥60ms crossfade if it pops):")
        for s, lab, t, e in splices:
            print(f"  {s} {lab}={t:.3f}s  rms={e}")
    if not clips and not splices:
        print("OK — all cut edges land in silence.")
        return
    sys.exit(1 if clips else 0)   # only fail the check on real clips


if __name__ == "__main__":
    main()
