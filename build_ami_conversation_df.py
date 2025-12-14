#!/usr/bin/env python3
"""
Build a time-synchronized conversation dataset from AMI manual annotations.

Conversation-synchronous synchronization (FIXED):
- Preserve the order the conversation was conducted:
  global ordering per meeting by real segment time (transcriber_start, transcriber_end).
- start/end are strictly increasing per meeting (monotonic index time), but ordered by real time.
- Also writes start_real/end_real for debugging/validation.

Notes:
- NITE words <w> tags are NOT namespaced in AMI manual annotations.
- segment tags are not namespaced; <nite:child> is namespaced.
"""

import argparse
import csv
import re
from pathlib import Path
import xml.etree.ElementTree as ET

NITE_NS = "http://nite.sourceforge.net/"
WORD_ID_RE = re.compile(r"\.words(\d+)")


# -------------------------------------------------
# XML parsing (NITE-safe)
# -------------------------------------------------

def load_words_by_id(words_xml: Path):
    """
    Load NITE words.xml into dict: word_index -> text
    """
    root = ET.parse(words_xml).getroot()
    words = {}

    for w in root.iter("w"):  # 'w' is NOT namespaced in AMI
        wid = (
            w.attrib.get(f"{{{NITE_NS}}}id")
            or w.attrib.get("id")
        )

        if wid is None or w.text is None:
            continue

        if "words" not in wid:
            continue

        try:
            idx = int(wid.split("words")[-1])
        except ValueError:
            continue

        txt = w.text.strip()
        if txt:
            words[idx] = txt

    return words


def _read_float_attrib(elem, key: str):
    """Robust float reader for XML attributes; returns None if missing/unparseable."""
    v = elem.attrib.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def parse_segments_xml(seg_xml: Path, words_xml: Path):
    """
    Parse NITE segments.xml + words.xml into utterances.
    Extract real time from segment attributes:
      - transcriber_start / transcriber_end
    """
    fname = seg_xml.name           # EN2001a.A.segments.xml
    meeting = fname[:6]            # EN2001a (first 6 chars)
    speaker = fname[7].upper()     # 'A' in EN2001a.A...

    # If your filenames are actually EN2001a.A..., meeting should be 7 chars.
    # The original script used 6; AMI meetings are typically 7 like EN2001a.
    # We'll infer meeting as up to the first dot to be safe.
    meeting = fname.split(".")[0]  # e.g., EN2001a
    speaker = fname.split(".")[1].upper()  # e.g., A

    words = load_words_by_id(words_xml)
    root = ET.parse(seg_xml).getroot()

    utterances = []

    for seg in root.iter("segment"):  # 'segment' has NO namespace
        child = seg.find(f"{{{NITE_NS}}}child")
        if child is None:
            continue

        href = child.attrib.get("href")
        if not href:
            continue

        ids = WORD_ID_RE.findall(href)
        if not ids:
            continue

        start_id = int(ids[0])
        end_id = int(ids[-1])

        seg_words = [words[i] for i in range(start_id, end_id + 1) if i in words]
        if not seg_words:
            continue

        start_real = _read_float_attrib(seg, "transcriber_start")
        end_real = _read_float_attrib(seg, "transcriber_end")

        # Sometimes AMI also has "starttime"/"endtime" variants in other exports;
        # keep them as fallback if needed.
        if start_real is None:
            start_real = _read_float_attrib(seg, "starttime")
        if end_real is None:
            end_real = _read_float_attrib(seg, "endtime")

        utterances.append({
            "meeting": meeting,
            "speaker": speaker,
            "text": " ".join(seg_words),
            "start_real": start_real,
            "end_real": end_real,
        })

    return utterances


# -------------------------------------------------
# Dataset construction
# -------------------------------------------------

def load_all_utterances(ami_root: Path):
    segments_dir = ami_root / "segments"
    words_dir = ami_root / "words"

    all_utts = []

    for seg_xml in sorted(segments_dir.glob("*.segments.xml")):
        words_xml = words_dir / seg_xml.name.replace(".segments.xml", ".words.xml")
        if not words_xml.exists():
            continue
        all_utts.extend(parse_segments_xml(seg_xml, words_xml))

    return all_utts


def assign_tokens_and_time(utterances):
    """
    Conversation-synchronous time:
    - Sort globally per meeting by real time (start_real, end_real)
    - Then assign monotonic start/end indices per meeting (strictly increasing)

    Keeps:
      start_real/end_real columns in output for debugging/validation.
    """
    def sort_key(u):
        # Place None times at the end, but keep deterministic ordering.
        sr = u["start_real"]
        er = u["end_real"]
        sr_key = sr if sr is not None else float("inf")
        er_key = er if er is not None else float("inf")
        return (u["meeting"], sr_key, er_key, u["speaker"], u["text"])

    utterances.sort(key=sort_key)

    token_id = 0
    current_meeting = None
    idx = 0

    for u in utterances:
        if u["meeting"] != current_meeting:
            current_meeting = u["meeting"]
            idx = 0

        u["token"] = f"utt_{token_id:08d}"
        u["start"] = float(idx)
        u["end"] = float(idx + 1)

        token_id += 1
        idx += 1

    return utterances


# -------------------------------------------------
# Output
# -------------------------------------------------

def write_csv(utterances, out_csv: Path):
    fields = ["token", "meeting", "speaker", "start", "end", "start_real", "end_real", "text"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(utterances)


def write_txt(utterances, out_txt: Path):
    with open(out_txt, "w", encoding="utf-8") as f:
        for u in utterances:
            sr = u.get("start_real")
            er = u.get("end_real")
            sr_s = "NA" if sr is None else f"{sr:.3f}"
            er_s = "NA" if er is None else f"{er:.3f}"
            f.write(f"<{u['token']}> [{u['meeting']}] ({sr_s}-{er_s}) speaker {u['speaker']}: {u['text']}\n")


# -------------------------------------------------
# Optional sanity checks
# -------------------------------------------------

def sanity_check(utterances):
    """
    Basic checks:
    - start is strictly increasing per meeting
    - start_real is non-decreasing when available (per meeting)
    """
    last_by_meeting = {}
    last_real_by_meeting = {}

    for u in utterances:
        m = u["meeting"]

        # monotonic index time
        last = last_by_meeting.get(m)
        if last is not None and not (u["start"] > last):
            raise RuntimeError(f"Non-increasing start index in meeting {m}: {u['start']} after {last}")
        last_by_meeting[m] = u["start"]

        # non-decreasing real time when present
        sr = u.get("start_real")
        if sr is not None:
            last_sr = last_real_by_meeting.get(m)
            if last_sr is not None and sr < last_sr:
                raise RuntimeError(f"Real-time ordering violated in meeting {m}: {sr} after {last_sr}")
            last_real_by_meeting[m] = sr


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami_root", default="/home/workspace/yoavellinson/AMICorpusXML/data/ami_public_manual_1.6.2", type=Path)
    parser.add_argument("--out_dir", default="/home/workspace/yoavellinson/LLM_SD/ami_synced", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    utterances = load_all_utterances(args.ami_root)
    utterances = assign_tokens_and_time(utterances)

    # Optional: fail fast if something is inconsistent
    sanity_check(utterances)

    write_csv(utterances, args.out_dir / "ami_utterances.csv")
    write_txt(utterances, args.out_dir / "ami_utterances.txt")

    print("Done.")
    print(f"Utterances: {len(utterances)}")
    print(f"Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
