#!/usr/bin/env python3
"""
AMI subtitle-style demo video.
- Full-frame utilization (1920x1080)
- Speaker-wise fixed lanes
- Text vertically centered in each lane
- Max TWO lines per utterance
- Larger font size, auto-shrinks if needed
- Speaker-colored subtitles
- MoviePy 2.x compatible
"""

import re
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ColorClip, TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

# ================= CONFIG =================

CSV_PATH = "/home/workspace/yoavellinson/LLM_SD/data/ami_synced/ami_utterances.csv"
OUT_MP4 = "ami_demo.mp4"
TMP_WAV = "tmp_audio.wav"

SR = 16000

MIN_LEN = 10.0
MAX_LEN = 25.0

PRE_PAD = 0.5
POST_PAD = 1.0

# ---- Full HD frame ----
W, H = 1920, 1080
BG_COLOR = (15, 15, 15)

LEFT_PAD = 80
RIGHT_PAD = 80
TOP_PAD = 80
BOTTOM_PAD = 80

LABEL_WIDTH = 140

BASE_FONT_SIZE = 56     # bigger font
MIN_FONT_SIZE = 26

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

SPEAKER_COLORS = {
    "A": "#1abc9c",
    "B": "#3498db",
    "C": "#e67e22",
    "D": "#e74c3c",
    "E": "#9b59b6",
}

MAX_WORDS_PER_UTT = 40

# =========================================


def resolve_font(path):
    try:
        ImageFont.truetype(path, 32)
        return path
    except Exception:
        print(f"[WARN] Font not found: {path}, using default.")
        return None


FONT = resolve_font(FONT_PATH)
FONT_BOLD = resolve_font(FONT_BOLD_PATH)

_space_re = re.compile(r"\s+")


def clean_text(txt: str) -> str:
    txt = str(txt).strip()
    return _space_re.sub(" ", txt)


def truncate_text(txt: str, max_words=MAX_WORDS_PER_UTT) -> str:
    words = txt.split()
    if len(words) <= max_words:
        return txt
    return " ".join(words[:max_words]) + " …"


def load_audio_segment(path, start, end):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    s = int(max(0, start - PRE_PAD) * SR)
    e = int(min(len(audio) / SR, end + POST_PAD) * SR)
    seg = audio[s:e]
    seg = seg/max(abs(seg))
    return seg.astype(np.float32)


# ---------- TEXT LAYOUT HELPERS ----------

def split_two_lines(text, font, max_width):
    """Force text into max TWO lines."""
    words = text.split()
    if not words:
        return text

    line1 = words[0]
    for i in range(1, len(words)):
        test = line1 + " " + words[i]
        w, _ = font.getbbox(test)[2:]
        if w <= max_width:
            line1 = test
        else:
            line2 = " ".join(words[i:])
            return line1 + "\n" + line2

    return line1


def measure_text(text, font):
    img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(img)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h


def fit_text_block(text, font_path, max_w, max_h):
    size = BASE_FONT_SIZE
    while size >= MIN_FONT_SIZE:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        wrapped = split_two_lines(text, font, max_w)
        w, h = measure_text(wrapped, font)
        if w <= max_w and h <= max_h:
            return wrapped, size, h
        size -= 2

    font = ImageFont.truetype(font_path, MIN_FONT_SIZE) if font_path else ImageFont.load_default()
    wrapped = split_two_lines(text, font, max_w)
    _, h = measure_text(wrapped, font)
    return wrapped, MIN_FONT_SIZE, h


# ---------- CLIP SELECTION ----------

def auto_select_clip(df):
    best = None
    best_score = -1

    for meeting, mdf in df.groupby("meeting"):
        mdf = mdf.sort_values("start_time")

        t0 = mdf["start_time"].iloc[0]
        t1 = mdf["end_time"].iloc[-1]
        t = t0

        while t + MIN_LEN <= t1:
            end = min(t + MAX_LEN, t1)
            chunk = mdf[(mdf.start_time >= t) & (mdf.start_time <= end)]

            speakers = set(chunk.speaker)
            if len(speakers) < 2:
                t += 5
                continue

            rows = chunk.to_dict("records")
            overlap = any(
                rows[i]["speaker"] != rows[i + 1]["speaker"]
                and rows[i]["end_time"] > rows[i + 1]["start_time"]
                for i in range(len(rows) - 1)
            )

            if not overlap:
                t += 5
                continue

            score = len(chunk) / (end - t) + len(speakers)
            if score > best_score:
                best_score = score
                best = {
                    "meeting": meeting,
                    "start": t,
                    "end": end,
                    "rows": rows,
                }

            t += 5

    return best


# ================= MAIN ===================

def main():
    df = pd.read_csv(CSV_PATH)

    clip = auto_select_clip(df)
    if clip is None:
        raise RuntimeError("No suitable clip found")

    print(f"Selected clip: {clip['meeting']} {clip['start']:.1f}–{clip['end']:.1f}s")

    audio = load_audio_segment(
        clip["rows"][0]["audio_path"],
        clip["start"],
        clip["end"],
    )
    sf.write(TMP_WAV, audio, SR)
    audio_clip = AudioFileClip(TMP_WAV)

    t0 = clip["start"] - PRE_PAD

    bg = ColorClip(
        size=(W, H),
        color=BG_COLOR,
        duration=audio_clip.duration,
    ).with_audio(audio_clip)

    clips = [bg]

    speakers = sorted({r["speaker"] for r in clip["rows"]})
    n_spk = len(speakers)

    usable_h = H - TOP_PAD - BOTTOM_PAD
    lane_h = usable_h // n_spk

    subtitle_w = W - LEFT_PAD - RIGHT_PAD - LABEL_WIDTH
    subtitle_h = lane_h - 20

    speaker_y = {
        spk: TOP_PAD + i * lane_h
        for i, spk in enumerate(speakers)
    }

    # Speaker labels
    for spk, y in speaker_y.items():
        clips.append(
            TextClip(
                text=f"{spk}:",
                font_size=56,
                color=SPEAKER_COLORS.get(spk, "white"),
                font=FONT_BOLD,
            )
            .with_start(0)
            .with_duration(audio_clip.duration)
            .with_position((LEFT_PAD, y + lane_h // 2 - 30))
        )

    # Subtitles
    for r in clip["rows"]:
        start = max(0.0, r["start_time"] - t0)
        end = min(audio_clip.duration, r["end_time"] - t0)
        if end <= start:
            continue

        txt = truncate_text(clean_text(r["text"]))
        spk = r["speaker"]

        wrapped, font_size, text_h = fit_text_block(
            txt, FONT, subtitle_w, subtitle_h
        )

        y_centered = speaker_y[spk] + (lane_h - text_h) // 2

        clips.append(
            TextClip(
                text=wrapped,
                font_size=font_size,
                color=SPEAKER_COLORS.get(spk, "white"),
                font=FONT,
                size=(subtitle_w, subtitle_h),
                method="caption",
            )
            .with_start(start)
            .with_duration(end - start)
            .with_position(
                (LEFT_PAD + LABEL_WIDTH, y_centered)
            )
        )

    video = CompositeVideoClip(clips, size=(W, H))

    video.write_videofile(
        "ami_demo.mp4",
        fps=24,
        codec="libx264",
        audio_codec="libmp3lame",
        preset="ultrafast",
        threads=0,  # use all CPU cores
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",
            "-tune", "zerolatency",
        ],
    )
    Path(TMP_WAV).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
