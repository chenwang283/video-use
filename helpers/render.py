"""Render a video from an EDL — FRAME-LOCKED A/V (no desync, ever).

Why this is built the way it is (hard-won; do not "simplify" back):
A per-segment extract → `-c copy` concat drifts audio against video because
(1) `-ss` before `-i` + a forced `-r` shifts the first frame, (2) AAC per-segment
encoder priming + `-t` frame-rounding makes each clip's audio and video durations
disagree by a few ms, (3) `-frames:v N` with `-t adur` drops the last frame, and
(4) muxing audio+video per segment, then concatenating, leaves ~10ms timestamp
gaps at every boundary so the video ends up at e.g. 29.9fps over a longer span
than the audio → the picture lags more and more.

The fix, implemented here:
  • frame-align every cut to the source fps grid
  • extract VIDEO and AUDIO as SEPARATE files per segment, single `-ss` before -i
      - video: exactly N = round(dur*fps) frames (`-frames:v N`)
      - audio: exactly round(N*SR/fps) samples (PCM, `atrim=end_sample=...`)
  • concat each stream losslessly (`-c copy`)
  • ONE final encode that renumbers frames onto the exact fps grid
    (`setpts=N/(FPS*TB)`, `-fps_mode cfr`) — collapsing the concat gaps — then
    applies overlays (PTS-shifted) and subtitles LAST, and encodes audio once.
Result: true CFR, video duration == audio duration, locked sync.

Fades: applied only where they're needed (hard splices in continuous speech).
A cut that lands in silence does NOT get a fade — fading silence is pointless and
fading a clipped word fragment makes a "weird" swallowed sound. Per-range
`fade_in` / `fade_out` (seconds) in the EDL override; default is a tiny declick.

Usage:
    python helpers/render.py <edl.json> -o final.mp4
    python helpers/render.py <edl.json> -o preview.mp4 --preview
    python helpers/render.py <edl.json> -o final.mp4 --build-subtitles
    python helpers/render.py <edl.json> -o final.mp4 --no-subtitles
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    from grade import get_preset, auto_grade_for_clip  # same directory
except Exception:
    def get_preset(name: str) -> str:
        return ""

    def auto_grade_for_clip(video, start=0.0, duration=None, verbose=False):  # type: ignore
        return "eq=contrast=1.03:saturation=0.98", {}


SR = 48000                       # output audio sample rate
DEFAULT_DECLICK = 0.006          # 6ms — inaudible, only prevents sample-discontinuity clicks
DEFAULT_HARD_FADE = 0.03         # used at a hard splice if the EDL doesn't specify

# -------- Subtitle style (bold-overlay, proven at 1920×1080 and 1080×1920) --
SUB_FORCE_STYLE = (
    "FontName=Helvetica,FontSize=18,Bold=1,"
    "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H00000000,"
    "BorderStyle=1,Outline=2,Shadow=0,"
    "Alignment=2,MarginV=90"
)

# -------- Helpers ------------------------------------------------------------


def run(cmd: list[str], quiet: bool = False) -> None:
    if not quiet:
        print(f"  $ {' '.join(str(c) for c in cmd[:6])}{' …' if len(cmd) > 6 else ''}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def ffprobe(args: list[str]) -> str:
    return subprocess.run(["ffprobe", "-v", "error", *args],
                          capture_output=True, text=True).stdout.strip()


def source_fps(video: Path) -> int:
    """Source frame rate, rounded to the nearest integer (the CFR grid we lock to).
    Screen / talking-head sources are virtually always integer (30, 60, 25, 24)."""
    val = ffprobe(["-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                   "-of", "default=nk=1:nw=1", str(video)])
    try:
        num, den = val.split("/")
        return max(1, round(float(num) / float(den)))
    except Exception:
        return 30


def resolve_grade_filter(grade_field: str | None) -> str:
    if not grade_field:
        return ""
    if grade_field == "auto":
        return "__AUTO__"
    if re.fullmatch(r"[a-zA-Z0-9_\-]+", grade_field):
        try:
            return get_preset(grade_field)
        except KeyError:
            print(f"warning: unknown preset '{grade_field}', using as raw filter")
            return grade_field
    return grade_field


def resolve_path(maybe_path: str, base: Path) -> Path:
    p = Path(maybe_path)
    return p if p.is_absolute() else (base / p).resolve()


# -------- HDR → SDR tone mapping (HLG / PQ sources) --------------------------

HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
TONEMAP_CHAIN = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)


def is_hdr_source(video: Path) -> bool:
    out = ffprobe(["-select_streams", "v:0", "-show_entries", "stream=color_transfer",
                   "-of", "default=noprint_wrappers=1:nokey=1", str(video)])
    return out in HDR_TRANSFERS


def is_portrait_source(video: Path) -> bool:
    out = ffprobe(["-select_streams", "v:0", "-show_entries", "stream=width,height",
                   "-of", "csv=p=0", str(video)])
    try:
        w, h = map(int, out.split(","))
        return h > w
    except Exception:
        return False


# -------- Per-segment extraction (frame-locked, dual-stream) -----------------


def extract_segment(
    source: Path, fps: int, seg_start: float, seg_end: float,
    grade_filter: str, vout: Path, aout: Path,
    fade_in: float, fade_out: float, preview: bool, draft: bool,
    portrait: bool, hdr: bool,
) -> int:
    """Extract ONE segment as separate video + audio files, frame-locked.

    Returns N = the exact video frame count for this segment.
    Cuts are frame-aligned by the caller; here we extract exactly N video frames
    and round(N*SR/fps) audio samples so the streams have identical duration.
    """
    vout.parent.mkdir(parents=True, exist_ok=True)
    N = max(1, round((seg_end - seg_start) * fps))
    nsamp = round(N * SR / fps)
    adur = N / fps

    scale = (f"scale=-2:{1280 if draft else 1920}" if portrait
             else f"scale={1280 if draft else 1920}:-2")
    vf_parts: list[str] = []
    if hdr:
        vf_parts.append(TONEMAP_CHAIN)
    vf_parts.append(scale)
    if grade_filter:
        vf_parts.append(grade_filter)
    vf = ",".join(vf_parts)

    if draft:
        preset, crf = "ultrafast", "30"
    elif preview:
        preset, crf = "veryfast", "21"
    else:
        preset, crf = "fast", "16"   # high-quality intermediate; final re-encode picks the delivery CRF

    # --- video: exactly N frames, no audio ---
    run(["ffmpeg", "-y", "-ss", f"{seg_start:.4f}", "-i", str(source),
         "-t", f"{adur + 0.3:.4f}", "-frames:v", str(N), "-an",
         "-vf", vf, "-c:v", "libx264", "-preset", preset, "-crf", crf,
         "-pix_fmt", "yuv420p", str(vout)], quiet=True)

    # --- audio: exactly nsamp samples (PCM, no encoder priming) + fades ---
    af_parts = [f"atrim=end_sample={nsamp}"]
    if fade_in > 0:
        af_parts.append(f"afade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        af_parts.append(f"afade=t=out:st={max(0.0, adur - fade_out):.4f}:d={fade_out:.3f}")
    af = ",".join(af_parts)
    run(["ffmpeg", "-y", "-ss", f"{seg_start:.4f}", "-i", str(source),
         "-t", f"{adur + 0.3:.4f}", "-vn", "-af", af,
         "-c:a", "pcm_s16le", "-ar", str(SR), "-ac", "2", str(aout)], quiet=True)
    return N


def extract_all_segments(edl: dict, fps: int, edit_dir: Path, source: Path,
                         preview: bool, draft: bool) -> tuple[list[Path], list[Path], int]:
    resolved = resolve_grade_filter(edl.get("grade"))
    is_auto = resolved == "__AUTO__"
    tag = "clips_draft" if draft else ("clips_preview" if preview else "clips_graded")
    clips_dir = edit_dir / tag
    clips_dir.mkdir(parents=True, exist_ok=True)

    ranges = edl["ranges"]
    sources = edl["sources"]
    # probe HDR / orientation once per source (not per segment)
    src_meta: dict[str, tuple[bool, bool]] = {}
    for name, p in sources.items():
        sp = resolve_path(p, edit_dir)
        src_meta[name] = (is_portrait_source(sp), is_hdr_source(sp))
    vpaths, apaths, total_frames = [], [], 0
    print(f"extracting {len(ranges)} segment(s) → {clips_dir.name}/ (frame-locked @ {fps}fps)")
    for i, r in enumerate(ranges):
        src_path = resolve_path(sources[r["source"]], edit_dir)
        portrait, hdr = src_meta[r["source"]]
        s = align(float(r["start"]), fps)
        e = align(float(r["end"]), fps)
        if is_auto:
            seg_filter, _ = auto_grade_for_clip(src_path, start=s, duration=e - s, verbose=False)
        else:
            seg_filter = resolved
        fi = float(r.get("fade_in", DEFAULT_DECLICK))
        fo = float(r.get("fade_out", DEFAULT_DECLICK))
        vout = clips_dir / f"v_{i:04d}.mkv"
        aout = clips_dir / f"a_{i:04d}.mkv"
        N = extract_segment(src_path, fps, s, e, seg_filter, vout, aout, fi, fo,
                            preview, draft, portrait, hdr)
        vpaths.append(vout); apaths.append(aout); total_frames += N
        note = r.get("beat") or r.get("note") or ""
        print(f"  [{i:03d}] {r['source']}  {s:7.2f}-{e:7.2f}  ({N:4d}f)  {note}")
    return vpaths, apaths, total_frames


def align(t: float, fps: int) -> float:
    """Snap a time to the fps frame grid (so segments are whole frames)."""
    return round(round(t * fps) / fps, 4)


# -------- Lossless concat ----------------------------------------------------


def concat_streams(paths: list[Path], out_path: Path, edit_dir: Path, name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lst = edit_dir / f"_concat_{name}.txt"
    lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in paths))
    print(f"concat {name} → {out_path.name}")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out_path)],
        quiet=True)
    lst.unlink(missing_ok=True)


# -------- Master SRT (output-timeline offsets) ------------------------------

PUNCT_BREAK = set(".,!?;:")


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _words_in_range(transcript: dict, t_start: float, t_end: float) -> list[dict]:
    out: list[dict] = []
    for w in transcript.get("words", []):
        if w.get("type") != "word":
            continue
        ws, we = w.get("start"), w.get("end")
        if ws is None or we is None or we <= t_start or ws >= t_end:
            continue
        out.append(w)
    return out


def build_master_srt(edl: dict, fps: int, edit_dir: Path, out_path: Path) -> None:
    transcripts_dir = edit_dir / "transcripts"
    entries: list[tuple[float, float, str]] = []
    seg_offset = 0.0
    for r in edl["ranges"]:
        seg_start = align(float(r["start"]), fps)
        seg_end = align(float(r["end"]), fps)
        seg_duration = seg_end - seg_start
        tr_path = transcripts_dir / f"{r['source']}.json"
        if not tr_path.exists():
            seg_offset += seg_duration
            continue
        transcript = json.loads(tr_path.read_text())
        words_in_seg = _words_in_range(transcript, seg_start, seg_end)
        chunks: list[list[dict]] = []
        current: list[dict] = []
        for w in words_in_seg:
            text = (w.get("text") or "").strip()
            if not text:
                continue
            current.append(w)
            if len(current) >= 2 or text[-1] in PUNCT_BREAK:
                chunks.append(current); current = []
        if current:
            chunks.append(current)
        for chunk in chunks:
            local_start = max(seg_start, chunk[0].get("start", seg_start))
            local_end = min(seg_end, chunk[-1].get("end", seg_end))
            out_start = max(0.0, local_start - seg_start) + seg_offset
            out_end = max(0.0, local_end - seg_start) + seg_offset
            if out_end <= out_start:
                out_end = out_start + 0.4
            text = " ".join((w.get("text") or "").strip() for w in chunk)
            text = re.sub(r"\s+", " ", text).strip().rstrip(",;:").upper()
            entries.append((out_start, out_end, text))
        seg_offset += seg_duration

    entries.sort(key=lambda e: e[0])
    lines: list[str] = []
    for i, (a, b, t) in enumerate(entries, start=1):
        lines += [str(i), f"{_srt_timestamp(a)} --> {_srt_timestamp(b)}", t, ""]
    out_path.write_text("\n".join(lines))
    print(f"master SRT → {out_path.name} ({len(entries)} cues)")


# -------- Final composite: CFR renumber → overlays → subtitles LAST ----------


def build_final(vbase: Path, abase: Path, fps: int, overlays: list[dict],
                subtitles_path: Path | None, out_path: Path, edit_dir: Path,
                preview: bool, draft: bool, loudnorm: bool) -> None:
    """One encode: renumber video to the exact fps grid (collapsing concat gaps),
    composite overlays (PTS-shifted) then subtitles LAST, take audio from the
    sample-exact PCM base. This is the only place the final video is encoded."""
    inputs = ["-i", str(vbase), "-i", str(abase)]
    for ov in overlays:
        inputs += ["-i", str(resolve_path(ov["file"], edit_dir))]

    parts = [f"[0:v]setpts=N/({fps}*TB),fps={fps}[base]"]
    cur = "[base]"
    ov_in = 2
    for idx, ov in enumerate(overlays, start=1):
        t = float(ov["start_in_output"]); dur = float(ov["duration"]); end = t + dur
        parts.append(f"[{ov_in}:v]setpts=PTS-STARTPTS+{t}/TB[o{idx}]")
        parts.append(f"{cur}[o{idx}]overlay=enable='between(t,{t:.3f},{end:.3f})'[v{idx}]")
        cur = f"[v{idx}]"; ov_in += 1
    if subtitles_path and subtitles_path.exists():
        subs = str(subtitles_path.resolve()).replace(":", r"\:").replace("'", r"\'")
        parts.append(f"{cur}subtitles='{subs}':force_style='{SUB_FORCE_STYLE}'[outv]")
        out_label = "[outv]"
    else:
        parts.append(f"{cur}null[outv]")
        out_label = "[outv]"

    if draft:
        preset, crf = "ultrafast", "30"
    elif preview:
        preset, crf = "veryfast", "23"
    else:
        preset, crf = "medium", "19"

    a_args = (["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"] if loudnorm else [])
    cmd = ["ffmpeg", "-y", *inputs,
           "-filter_complex", ";".join(parts),
           "-map", out_label, "-map", "1:a:0",
           "-r", str(fps), "-fps_mode", "cfr",
           "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p",
           *a_args, "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
           "-movflags", "+faststart", str(out_path)]
    print(f"final encode @ {fps}fps CFR  (overlays: {len(overlays)}, "
          f"subs: {'yes' if subtitles_path and subtitles_path.exists() else 'no'}, "
          f"loudnorm: {'yes' if loudnorm else 'no'})")
    run(cmd, quiet=True)


# -------- Main ---------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a video from an EDL (frame-locked A/V)")
    ap.add_argument("edl", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--preview", action="store_true", help="1080p, faster encode for QC")
    ap.add_argument("--draft", action="store_true", help="720p, ultrafast — cut-check only")
    ap.add_argument("--build-subtitles", action="store_true",
                    help="Build master.srt from transcripts + EDL offsets before compositing")
    ap.add_argument("--no-subtitles", action="store_true")
    ap.add_argument("--no-loudnorm", action="store_true")
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"edl not found: {edl_path}")
    edl = json.loads(edl_path.read_text())
    edit_dir = edl_path.parent
    out_path = args.output.resolve()

    # fps grid = source fps (all sources assumed same rate; uses the first).
    first_src = resolve_path(next(iter(edl["sources"].values())), edit_dir)
    fps = source_fps(first_src)

    vpaths, apaths, total_frames = extract_all_segments(
        edl, fps, edit_dir, first_src, preview=args.preview, draft=args.draft)

    vbase = edit_dir / ("_vbase_draft.mkv" if args.draft else "_vbase.mkv")
    abase = edit_dir / ("_abase_draft.mkv" if args.draft else "_abase.mkv")
    concat_streams(vpaths, vbase, edit_dir, "v")
    concat_streams(apaths, abase, edit_dir, "a")

    subs_path: Path | None = None
    if not args.no_subtitles:
        if args.build_subtitles:
            subs_path = edit_dir / "master.srt"
            build_master_srt(edl, fps, edit_dir, subs_path)
        elif edl.get("subtitles"):
            subs_path = resolve_path(edl["subtitles"], edit_dir)
            if not subs_path.exists():
                print(f"warning: subtitles path missing: {subs_path}")
                subs_path = None

    build_final(vbase, abase, fps, edl.get("overlays") or [], subs_path,
                out_path, edit_dir, preview=args.preview, draft=args.draft,
                loudnorm=not args.no_loudnorm)
    vbase.unlink(missing_ok=True); abase.unlink(missing_ok=True)

    # verify A/V lock
    vdur = ffprobe(["-select_streams", "v:0", "-show_entries", "stream=duration",
                    "-of", "csv=p=0", str(out_path)])
    adur = ffprobe(["-select_streams", "a:0", "-show_entries", "stream=duration",
                    "-of", "csv=p=0", str(out_path)])
    afr = ffprobe(["-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
                   "-of", "csv=p=0", str(out_path)])
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\ndone: {out_path}  ({size_mb:.1f} MB)")
    print(f"  expected {total_frames} frames = {total_frames/fps:.2f}s @ {fps}fps")
    print(f"  video={vdur}s  audio={adur}s  avg_fps={afr}  (should match within ~AAC priming)")


if __name__ == "__main__":
    main()
