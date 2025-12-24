# ollama_segment_tiling.py
import json
import re
import requests
from nltk.tokenize import word_tokenize

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"   # use 3B for speed

_PUNCT_RE = re.compile(r"[^\w\s]")


def _extract_json_array(text: str):
    """
    Extract the outermost JSON array.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found")
    return text[start:end + 1]


def ollama_segment_tiling(text: str, num_speakers: int):
    """
    Returns:
        List[{"speaker": int, "length": int}]
    """
    clean_text = _PUNCT_RE.sub("", text)
    words = word_tokenize(clean_text)
    n_words = len(words)

    prompt = f"""
You perform text-only speaker diarization.

Return a JSON array of speaker segments.
Each segment MUST have:
- speaker: integer in [0, {num_speakers - 1}]
- length: number of words in this segment

Rules:
- Segments are contiguous.
- Sum of all lengths MUST equal {n_words}.
- Output JSON only.
- Do NOT include explanations.

Transcript:
{clean_text}
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 2000,  # small output → safe
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()

    raw = r.json()["response"]

    try:
        segments = json.loads(_extract_json_array(raw))
    except Exception:
        # Ultimate safe fallback: single speaker
        return [{"speaker": 0, "length": n_words}]

    # --- Safety: fix length mismatch ---
    total = sum(int(s.get("length", 0)) for s in segments)
    if total != n_words:
        segments[-1]["length"] += (n_words - total)

    return segments
