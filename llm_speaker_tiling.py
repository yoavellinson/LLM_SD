# llm_speaker_tiling.py
import json
import re
from typing import List, Optional

from openai import AsyncOpenAI
from nltk.tokenize import word_tokenize
from openai_key import key
# Same punctuation rule you used elsewhere
_PUNCT_RE = re.compile(r"[^\w\s]")


def _strip_code_fences(s: str) -> str:
    """
    Removes ```...``` wrappers if the model returns fenced JSON.
    """
    s = s.strip()
    if s.startswith("```"):
        # Handles ```json\n...\n``` or ```\n...\n```
        parts = s.split("```")
        # parts: ["", "json\n[...]\n", ""]
        if len(parts) >= 2:
            s = parts[1].strip()
            # remove optional leading "json"
            if s.lower().startswith("json"):
                s = s[4:].strip()
    return s.strip()


def _parse_labels(raw: str) -> List[int]:
    raw = _strip_code_fences(raw)
    labels = json.loads(raw)
    if not isinstance(labels, list) or not all(isinstance(x, int) for x in labels):
        raise ValueError(f"Expected JSON list[int], got: {type(labels)}")
    return labels


def _hard_align(labels: List[int], n_words: int) -> List[int]:
    """
    Never crash: pad/truncate to match n_words.
    """
    if n_words <= 0:
        return []
    if len(labels) == 0:
        return [0] * n_words
    if len(labels) < n_words:
        labels = labels + [labels[-1]] * (n_words - len(labels))
    elif len(labels) > n_words:
        labels = labels[:n_words]
    return labels


async def llm_speaker_tiling_async(
    client: AsyncOpenAI,
    text: str,
    num_speakers: int,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> List[int]:
    """
    Returns:
        labels: List[int] of length = number of tokenized words (after punctuation removal)
    """
    clean_text = _PUNCT_RE.sub("", text)
    words = word_tokenize(clean_text)

    prompt = (
        f"You are given a transcript of a conversation with {num_speakers} speakers.\n"
        f"Speakers are indexed from 0 to {num_speakers - 1}.\n\n"
        "Task:\n"
        "Assign a speaker index to EACH WORD in the transcript.\n\n"
        "Rules:\n"
        "- Speakers usually speak in contiguous turns.\n"
        "- Short acknowledgements (e.g., yeah, uh, mm) may interrupt turns.\n"
        "- Return ONLY a JSON list of integers.\n"
        "- Do NOT return words.\n"
        "- Do NOT add explanations.\n\n"
        "Transcript:\n"
        f"{clean_text}\n"
    )

    resp = await client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
    )

    raw = resp.output_text or ""
    labels = _parse_labels(raw)
    labels = _hard_align(labels, len(words))
    return labels


def llm_speaker_tiling(
    text: str,
    num_speakers: int,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
) -> List[int]:
    """
    Synchronous wrapper (kept for convenience).
    Prefer async evaluation script for speed.
    """
    import asyncio

    client = AsyncOpenAI(api_key=api_key)
    return asyncio.run(
        llm_speaker_tiling_async(
            client=client,
            text=text,
            num_speakers=num_speakers,
            model=model,
            temperature=temperature,
        )
    )
