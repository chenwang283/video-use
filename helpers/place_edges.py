"""Deterministically place clean cut edges from a coarse EDL + transcript.

The LLM decides WHAT to keep (coarse blocks, in output order, with hard-splice
flags). This helper decides exactly WHERE each edge lands, from the audio energy
and the transcript word boundaries — so edges never clip a word, never drop a
near-silent word, land in silence, and tighten dead air to ~0.3s. Output is a
render-ready EDL with frame-aligned ranges and per-edge fades.

Why this exists: ASR word timestamps (AssemblyAI especially) are unreliable —
durations are inflated, onsets reported early, offsets late, and words sometimes
collapse to a point. Trusting them directly clips word edges ("s" off "tasks",
"deadline" cut short) and drops soft words (a near-silent final "it."). Energy
detection alone clips soft words too. Combining BOTH — energy for the silence
pad, the transcript word extent as a hard floor/ceiling — is what's robust.

INPUT EDL (coarse) — each range is one kept block:
  {"source","start","end","hard_start":bool,"hard_end":bool,"beat"?}
    hard_start/hard_end = True when a removed word sits immediately adjacent
    (a mid-speech splice with no silence on that side; e.g. removing a stutter).

OUTPUT EDL — many ranges (blocks split at internal silences), each with refined
  frame-aligned start/end + fade_in/fade_out, ready for render.py.

Usage:
    python helpers/place_edges.py <coarse_edl.json> -o <edl.json>
    python helpers/place_edges.py <coarse_edl.json> -o <edl.json> --fps 30 --target-sil 0.30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

QUIET = 0.020          # below this RMS = room tone / silence
SPEECH = 0.026         # above this RMS = speech
SPEECH_SOFT = 0.012    # above this = even soft speech (between pure silence ~0.005 and speech)
PAD = 0.15             # room tone kept on a clean edge (TARGET_SIL split across two edges)
DECLICK = 0.006        # tiny fade at a clean cut (inaudible; just prevents clicks)
HARD_FADE = 0.03       # fade at a hard splice (continuous-speech rejoin)


def load_pcm(path: str, sr: int = 16000) -> np.ndarray:
    wav = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", str(sr),
                    "-c:a", "pcm_s16le", str(wav)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data = np.frombuffer(wave.open(str(wav), "rb").readframes(10 ** 9),
                         dtype=np.int16).astype(np.float32) / 32768.0
    wav.unlink(missing_ok=True)
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Place clean cut edges (energy + word boundaries)")
    ap.add_argument("edl", type=Path, help="coarse EDL (blocks with hard_start/hard_end)")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30, help="frame grid to align cuts to")
    ap.add_argument("--target-sil", type=float, default=0.30, help="silence to keep at each cut")
    ap.add_argument("--split-sil", type=float, default=0.40,
                    help="tighten any pause longer than this down to --target-sil")
    args = ap.parse_args()

    edl = json.loads(args.edl.read_text())
    edit_dir = args.edl.resolve().parent
    fps, TARGET, SPLIT = args.fps, args.target_sil, args.split_sil
    HALF = TARGET / 2

    # one transcript + one pcm per source (cached)
    pcms: dict[str, np.ndarray] = {}
    words_by_src: dict[str, list] = {}
    for name, p in edl["sources"].items():
        sp = p if Path(p).is_absolute() else str((edit_dir / p).resolve())
        pcms[name] = load_pcm(sp)
        tj = edit_dir / "transcripts" / f"{name}.json"
        words_by_src[name] = json.loads(tj.read_text())["words"] if tj.exists() else []
    SR = 16000

    def rms(pcm, t, half=0.01):
        a = max(0, int((t - half) * SR)); b = int((t + half) * SR)
        seg = pcm[a:b]
        return float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0

    def win(pcm, t, w):
        a = max(0, int(t * SR)); b = int((t + w) * SR)
        seg = pcm[a:b]
        return float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0

    def find_silences(pcm, lo, hi, min_sil):
        sils, cur, t = [], None, lo
        while t < hi:
            if win(pcm, t, 0.04) < QUIET:
                if cur is None:
                    cur = t
            else:
                if cur is not None and t - cur >= min_sil:
                    sils.append((cur, t))
                cur = None
            t += 0.02
        if cur is not None and hi - cur >= min_sil:
            sils.append((cur, hi))
        return sils

    def first_speech(pcm, near):
        t, hi = near - 0.25, near + 0.70
        while t < hi:
            if rms(pcm, t) > SPEECH:
                return t
            t += 0.008
        return near

    def last_speech(pcm, near, scan_upper):
        last, t = None, near - 0.50
        while t < scan_upper:
            if rms(pcm, t) > SPEECH:
                last = t
            t += 0.008
        return last if last is not None else near

    def align(t):
        return round(round(t * fps) / fps, 4)

    out_ranges = []
    for blk in edl["ranges"]:
        src = blk["source"]
        pcm, words = pcms[src], words_by_src[src]
        bs, be = float(blk["start"]), float(blk["end"])
        hs, he = bool(blk.get("hard_start")), bool(blk.get("hard_end"))
        beat = blk.get("beat", "")
        ws = [w for w in words if w["end"] > bs and w["start"] < be]
        if not ws:
            # no transcript words; keep the block as-is (frame-aligned)
            out_ranges.append({"source": src, "start": align(bs), "end": align(be),
                               "fade_in": HARD_FADE if hs else DECLICK,
                               "fade_out": HARD_FADE if he else DECLICK, "beat": beat})
            continue

        span_lo = ws[0]["start"]
        span_hi = max(ws[-1]["end"], be)
        sils = find_silences(pcm, span_lo + 0.05, span_hi - 0.03, SPLIT)
        runs, rstart = [], span_lo
        for (a, b) in sils:
            runs.append((rstart, a)); rstart = b
        runs.append((rstart, span_hi))
        # drop runs with no real speech (ASR inflates the last word into silence)
        runs = [r for r in runs
                if any(rms(pcm, t) > SPEECH_SOFT for t in np.arange(r[0], r[1], 0.02))]
        if not runs:
            continue

        for j, (rs, re_) in enumerate(runs):
            first, last = j == 0, j == len(runs) - 1
            head_hard, tail_hard = first and hs, last and he

            if head_hard:
                seg_start = rs                                   # exact onset (removed word precedes)
            elif first:
                # block-outer clean start: never clip the block's FIRST word, even
                # if it's a soft sibilant onset the energy scan would miss. Floor =
                # the block's first transcript word start.
                e_on = first_speech(pcm, rs)
                seg_start = min(e_on - PAD, ws[0]["start"] - 0.03)
            else:
                seg_start = rs - HALF                            # rs = silence end; keep HALF

            if tail_hard:
                seg_end = re_                                    # exact offset (removed word follows)
            elif last:
                # block-outer clean end: never clip / drop the block's LAST word,
                # even if it's near-silent (e.g. a trailing "it."). Ceiling = the
                # block's last transcript word end. This also recovers a soft
                # tail word that run-splitting dropped as "all-silent".
                e_off = last_speech(pcm, re_, be + 0.30)
                seg_end = max(e_off + PAD, ws[-1]["end"] + 0.03)
            else:
                seg_end = re_ + HALF                            # re_ = silence start; keep HALF

            out_ranges.append({
                "source": src, "start": align(seg_start), "end": align(seg_end),
                "fade_in": HARD_FADE if head_hard else DECLICK,
                "fade_out": HARD_FADE if tail_hard else DECLICK,
                "beat": beat + (f" [{j+1}/{len(runs)}]" if len(runs) > 1 else ""),
            })

    out = {k: v for k, v in edl.items() if k != "ranges"}
    out["ranges"] = out_ranges
    out["total_duration_s"] = round(sum(r["end"] - r["start"] for r in out_ranges), 2)
    args.output.write_text(json.dumps(out, indent=2))
    total = out["total_duration_s"]
    print(f"placed edges: {len(edl['ranges'])} blocks → {len(out_ranges)} segments")
    print(f"total: {total:.1f}s  ({int(total//60)}m {total%60:04.1f}s)  → {args.output.name}")
    hard = sum(1 for r in out_ranges if r["fade_in"] == HARD_FADE or r["fade_out"] == HARD_FADE)
    print(f"hard-splice edges: {hard} (verify with check_cuts.py)")


if __name__ == "__main__":
    main()
