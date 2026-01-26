import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from hf_token import HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

import whisperx
from whisperx.diarize import DiarizationPipeline

from data.ami_dataset import (
    AMIWordChunkDataset,
    load_conversations,
    split_conversations,
    ami_collate_fn,
)
from torch.utils.data import DataLoader
from pathlib import Path
from nltk_utils import *

import numpy as np
import random
import torch
import json
from datetime import datetime
from tqdm import tqdm
from difflib import SequenceMatcher
import string

# -------------------------------------------------------------
# Seed
# -------------------------------------------------------------
SEED = 42
MAX_SAMPLES = 1000
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
SAVE_DIR = Path("/home/workspace/yoavellinson/LLM_SD/diarization_logs_whisperx")
LOG_EVERY = 50

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Load data
# -------------------------------------------------------------
convs = load_conversations(
    "/home/workspace/yoavellinson/LLM_SD/data/ami_synced/ami_utterances.csv"
)
_, test_convs = split_conversations(convs)

test_ds = AMIWordChunkDataset(
    test_convs,
    word_budget=256,
    overlap_scramble_prob=0.0,   # IMPORTANT: do NOT scramble for WhisperX
    text_only=False
)

g = torch.Generator().manual_seed(SEED)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    collate_fn=ami_collate_fn,
    shuffle=True,
    generator=g,
)

# -------------------------------------------------------------
# Load WhisperX models (ONCE)
# -------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16"

asr_model = whisperx.load_model(
    "large-v2",
    device,
    compute_type=compute_type,
)

align_model_cache = {}

diarize_pipeline = DiarizationPipeline(
    use_auth_token=HF_TOKEN,
    device=device,
)

# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------
def whisperx_run(audio, sr, min_spk, max_spk):
    result = asr_model.transcribe(
        audio,
        batch_size=16,
        language="en",
    )

    lang = "en"
    if lang not in align_model_cache:
        align_model_cache[lang] = whisperx.load_align_model(
            language_code=lang,
            device=device,
        )

    model_a, metadata = align_model_cache[lang]

    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    diarize_segments = diarize_pipeline(
        audio,
        min_speakers=min_spk,
        max_speakers=max_spk,
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)
    segments = result["segments"]
    segments, speaker_map = remap_whisperx_speakers(segments)

    return segments



def whisperx_segments_to_words(segments):
    pred = []
    for seg in segments:
        for w in seg.get("words", []):
            if "speaker" not in w:
                continue
            pred.append({
                "word": w["word"].lower(),
                "speaker": w["speaker"],
            })
    return pred


def align_pred_to_gt(pred, gt):
    pred_words = [p["word"] for p in pred]
    gt_words = [g["word"].lower() for g in gt]

    sm = SequenceMatcher(None, gt_words, pred_words)
    mapping = {}

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for gi, pj in zip(range(i1, i2), range(j1, j2)):
                mapping[gi] = pj

    aligned = []
    for gi, g in enumerate(gt):
        if gi in mapping:
            p = pred[mapping[gi]]
            aligned.append({
                "word": g["word"],
                "speaker": p["speaker"],
            })

    return aligned


def remap_whisperx_speakers(segments):
    """
    Remap WhisperX word-level speaker IDs ('00','01',...)
    to ('A','B','C',...) IN PLACE.
    """

    seen = []
    for seg in segments:
        for w in seg.get("words", []):
            spk = w.get("speaker")
            if spk is not None and spk not in seen:
                seen.append(spk)

    mapping = {
        spk: string.ascii_uppercase[i]
        for i, spk in enumerate(seen)
    }

    # rewrite word-level speakers
    for seg in segments:
        for w in seg.get("words", []):
            if "speaker" in w:
                w["speaker"] = mapping[w["speaker"]]

    return segments, mapping


# -------------------------------------------------------------
# Run evaluation
# -------------------------------------------------------------
results = []

for step, batch in tqdm(enumerate(test_loader), total=MAX_SAMPLES):

    if step >= MAX_SAMPLES:
        break

    audio = batch["audio"][0].numpy().astype("float32")
    target = batch["target"][0]

    gt = target_segments_to_word_labels(target)
    if len(gt) == 0:
        continue

    gt_speakers = list({g["speaker"] for g in gt})
    num_speakers = len(gt_speakers)

    try:
        wx_segments = whisperx_run(
            audio=audio,
            sr=16000,
            min_spk=num_speakers,
            max_spk=num_speakers,
        )
    except Exception as e:
        print("⚠ WhisperX failed:", e)
        continue

    pred_raw = whisperx_segments_to_words(wx_segments)
    if len(pred_raw) == 0:
        continue

    pred = align_pred_to_gt(pred_raw, gt)
    if len(pred) == 0:
        continue

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    word_acc = word_speaker_accuracy(pred, gt)
    seg_acc = segment_consistency_accuracy(pred, target)
    align = alignment_stats(pred, gt)
    boundary_metrics = boundary_segmentation_f1(pred, target)

    results.append({
        "step": step,
        "word_acc": word_acc,
        "seg_acc": seg_acc,
        "coverage": align["coverage"],
        "boundary_f1": boundary_metrics["boundary_f1"],
        "boundary_precision": boundary_metrics["boundary_precision"],
        "boundary_recall": boundary_metrics["boundary_recall"],
    })

    if step % LOG_EVERY == 0:
        print(
            f"[{step:04d}] "
            f"WordAcc={word_acc:.3f} | "
            f"SegAcc={seg_acc:.3f} | "
            f"B-F1={boundary_metrics['boundary_f1']:.3f}"
        )

        pred_sentences = pred_words_to_sentences(pred)
        save_diarization(
            path=SAVE_DIR,
            step=step,
            pred_sentences=pred_sentences,
            gt_segments=target,
            stats={
                "word_acc": round(word_acc, 3),
                "seg_acc": round(seg_acc, 3),
                "coverage": round(align["coverage"], 3),
                "boundary_f1": round(boundary_metrics["boundary_f1"], 3),
                "boundary_precision": round(boundary_metrics["boundary_precision"], 3),
                "boundary_recall": round(boundary_metrics["boundary_recall"], 3),
                "num_words_pred": len(pred),
                "num_words_gt": len(gt),
            },
        )

# -------------------------------------------------------------
# Save means
# -------------------------------------------------------------
means = {
    "num_samples": len(results),
    "word_acc_mean": float(np.mean([r["word_acc"] for r in results])),
    "seg_acc_mean": float(np.mean([r["seg_acc"] for r in results])),
    "boundary_f1_mean": float(np.mean([r["boundary_f1"] for r in results])),
    "boundary_precision_mean": float(np.mean([r["boundary_precision"] for r in results])),
    "boundary_recall_mean": float(np.mean([r["boundary_recall"] for r in results])),
    "coverage_mean": float(np.mean([r["coverage"] for r in results])),
}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = SAVE_DIR / f"whisperx_means_{timestamp}.json"

with open(out_path, "w") as f:
    json.dump(means, f, indent=2)

print("=" * 60)
for k, v in means.items():
    print(f"{k:25s}: {v}")
print("=" * 60)
print(f"Saved means to: {out_path}")
