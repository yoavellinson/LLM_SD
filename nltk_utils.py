# nltk_utils.py
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize.texttiling import TextTilingTokenizer
import re
from sklearn.metrics import precision_recall_fscore_support

# ------------------------------------------------------------------
# NLTK resources
# ------------------------------------------------------------------

nltk.download("punkt", quiet=True)
nltk.download("stopwords")
nltk.download("punkt_tab")

_PUNCT_RE = re.compile(r"[^\w\s]")


def ensure_nltk_resources():
    for r in ["punkt", "punkt_tab", "stopwords"]:
        try:
            nltk.data.find(
                f"tokenizers/{r}" if "punkt" in r else f"corpora/{r}"
            )
        except LookupError:
            nltk.download(r, quiet=True)


# ensure_nltk_resources()

# ------------------------------------------------------------------
# Text-only speaker tiling
# ------------------------------------------------------------------

def text_only_speaker_tiling(
    text: str,
    num_speakers: int,
    fallback_window: int = 40,
):
    """
    Text-only speaker tiling with robust fallback.
    Returns word-level speaker assignments.
    """

    words = word_tokenize(text)
    output = []

    # --- Try TextTiling ---
    try:
        tt = TextTilingTokenizer(w=20, k=10)
        segments = tt.tokenize(text)
        seg_words = [word_tokenize(s) for s in segments]

        idx = 0
        for seg_idx, sw in enumerate(seg_words):
            spk = seg_idx % num_speakers
            for _ in sw:
                if idx < len(words):
                    output.append({"word": words[idx], "speaker": spk})
                    idx += 1

        if len(output) == len(words):
            return output

    except Exception:
        pass  # expected on ASR text

    # --- Fallback: fixed-size windows ---
    output = []
    for i, w in enumerate(words):
        spk = (i // fallback_window) % num_speakers
        output.append({"word": w, "speaker": spk})

    return output

# ------------------------------------------------------------------
# Ground truth expansion
# ------------------------------------------------------------------

def target_segments_to_word_labels(target):
    """
    Converts segment-level GT to word-level labels.
    Uses the SAME punctuation handling you already had.
    """
    gt_words = []

    for seg in target:
        speaker = seg["speaker"]
        t = _PUNCT_RE.sub("", seg["text"])
        words = word_tokenize(t)

        for w in words:
            gt_words.append({
                "word": w,
                "speaker": speaker
            })

    return gt_words

# ------------------------------------------------------------------
# SAFE alignment (never fails)
# ------------------------------------------------------------------

def safe_align(pred, gt):
    """
    Align on longest common prefix.
    Never raises.
    """
    n = min(len(pred), len(gt))
    return pred[:n], gt[:n]

# ------------------------------------------------------------------
# Metrics (robust to mismatch)
# ------------------------------------------------------------------

def word_speaker_accuracy(pred, gt):
    pred, gt = safe_align(pred, gt)
    if len(gt) == 0:
        return 0.0

    correct = sum(
        p["speaker"] == g["speaker"]
        for p, g in zip(pred, gt)
    )
    return correct / len(gt)


def segment_consistency_accuracy(pred, target):
    """
    Segment-level accuracy evaluated only on aligned region.
    """
    idx = 0
    correct = 0
    total = 0

    for seg in target:
        t = _PUNCT_RE.sub("", seg["text"])
        words = word_tokenize(t)
        seg_len = len(words)

        if idx + seg_len > len(pred):
            break  # out of aligned region

        speakers = [pred[idx + i]["speaker"] for i in range(seg_len)]
        if speakers and all(s == seg["speaker"] for s in speakers):
            correct += 1

        idx += seg_len
        total += 1

    return correct / total if total > 0 else 0.0


def alignment_stats(pred, gt):
    """
    Optional diagnostic.
    """
    return {
        "pred_words": len(pred),
        "gt_words": len(gt),
        "coverage": min(len(pred), len(gt)) / max(len(pred), len(gt))
        if max(len(pred), len(gt)) > 0 else 0.0
    }

def pred_words_to_sentences(pred):
    sentences = []

    if not pred:
        return sentences

    cur_speaker = pred[0]["speaker"]
    cur_words = []

    for item in pred:
        spk = item["speaker"]
        word = item["word"]

        if spk != cur_speaker:
            sentences.append({
                "speaker": cur_speaker,
                "text": " ".join(cur_words),
            })
            cur_speaker = spk
            cur_words = [word]
        else:
            cur_words.append(word)

    if cur_words:
        sentences.append({
            "speaker": cur_speaker,
            "text": " ".join(cur_words),
        })

    return sentences

from pathlib import Path

def save_diarization(
    path: Path,
    step: int,
    pred_sentences,
    gt_segments,
    stats: dict,
):
    """
    Saves a single conversation comparison to disk.
    """

    path.mkdir(parents=True, exist_ok=True)
    out_file = path / f"step_{step:05d}.txt"

    with open(out_file, "w") as f:
        f.write("=== STATS ===\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== PREDICTED ===\n")
        for s in pred_sentences:
            f.write(f"[{s['speaker']}] {s['text']}\n")

        f.write("\n=== GROUND TRUTH ===\n")
        for s in gt_segments:
            f.write(f"[{s['speaker']}] {s['text']}\n")

    return out_file


def gt_boundaries_from_target(target):
    """
    Ground-truth boundaries from speaker turns.
    Returns a list of length (num_words - 1).
    """
    boundaries = []
    for seg in target:
        words = _PUNCT_RE.sub("", seg["text"])
        words = word_tokenize(words)
        if not boundaries:
            boundaries.extend([0] * (len(words) - 1))
        else:
            boundaries.append(1)              # boundary at segment start
            boundaries.extend([0] * (len(words) - 1))
    return boundaries

def pred_boundaries_from_word_labels(pred):
    """
    Predicted boundaries from word-level speaker predictions.
    """
    boundaries = []
    for i in range(len(pred) - 1):
        boundaries.append(
            1 if pred[i]["speaker"] != pred[i + 1]["speaker"] else 0
        )
    return boundaries

def boundary_segmentation_f1(pred, target):
    gt_b = gt_boundaries_from_target(target)
    pred_b = pred_boundaries_from_word_labels(pred)

    n = min(len(gt_b), len(pred_b))
    gt_b = gt_b[:n]
    pred_b = pred_b[:n]

    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_b, pred_b, average="binary", zero_division=0
    )
    return {
        "boundary_f1": f1,
        "boundary_precision": precision,
        "boundary_recall": recall,
    }
