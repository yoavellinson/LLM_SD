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
import re
from nltk.tokenize import word_tokenize

from speaker_model import SpeakerExtractionModule
from speaker_model import encode_words_to_embs, encode_spans

_PUNCT_RE = re.compile(r"[^\w\s]")
from whisper_asr import WhisperASRWrapper


def expand_ranges_to_words(assignments, ranges, words, speaker_labels):
    """
    assignments: list[int] per range
    ranges: list[(a,b)]
    words: list[str]
    """
    pred = []
    for spk_idx, (a, b) in zip(assignments, ranges):
        spk = speaker_labels[spk_idx]
        for w in words[a:b]:
            pred.append({"word": w, "speaker": spk})
    return pred


@torch.no_grad()
def dynamic_segment_words(model, word_embs, tau=0.5, max_len=8):
    """
    word_embs: [T,H]
    Returns list of ranges (start,end) covering all words.
    Uses continuity head + memory, resets at predicted boundaries.
    """
    T = word_embs.size(0)
    if T == 0:
        return []

    ranges = []
    s = 0
    mem = torch.zeros(model.hidden, device=word_embs.device)

    for t in range(T - 1):
        # predict whether t+1 continues same speaker
        logit = model.cont_logit(word_embs[t], word_embs[t+1], mem)
        p_same = torch.sigmoid(logit).item()

        # update memory with current word
        mem = model.update_memory(word_embs[t], mem)

        cur_len = (t + 1) - s
        if (p_same < tau) or (cur_len >= max_len):
            ranges.append((s, t + 1))
            s = t + 1
            mem = torch.zeros(model.hidden, device=word_embs.device)

    ranges.append((s, T))
    return ranges


@torch.no_grad()
def global_label_spans_with_memory(model, tokenizer, span_texts, num_speakers, device):
    """
    Predict speaker for each span in sequence.
    Maintains a memory per speaker; updates memory of predicted speaker.
    """
    if len(span_texts) == 0:
        return []

    span_embs = encode_spans(tokenizer, model.encoder, span_texts, device)  # [N,H]

    mems = [torch.zeros(model.hidden, device=device) for _ in range(num_speakers)]
    assignments = []

    for t in range(span_embs.size(0)):
        z = span_embs[t]

        scores = []
        for k in range(num_speakers):
            scores.append(model.score_span(z.unsqueeze(0), mems[k])[0])
        scores = torch.stack(scores)  # [K]

        k_hat = int(torch.argmax(scores).item())
        assignments.append(k_hat)

        mems[k_hat] = model.update_memory(z, mems[k_hat])

    return assignments


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
SAVE_DIR = Path("/home/workspace/yoavellinson/LLM_SD/diarization_logs_hybrid_asr")
MODEL_CKPT = Path("/home/workspace/yoavellinson/LLM_SD/ckpt_hybrid/last.ckpt")

# Hybrid knobs
CONT_TAU = 0.5
MAX_LEN = 8

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
    overlap_scramble_prob=0.5,
    text_only=False
)
BATCH_SIZE=1
g = torch.Generator().manual_seed(SEED)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    collate_fn=ami_collate_fn,
    shuffle=True,
    generator=g,
)

# -------------------------------------------------------------
# Load trained model
# -------------------------------------------------------------
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
whisper_asr = WhisperASRWrapper(batch_size=BATCH_SIZE,device=device)

module = SpeakerExtractionModule.load_from_checkpoint(MODEL_CKPT, map_location=device)
module.eval()

model = module.model
tokenizer = module.tokenizer

# -------------------------------------------------------------
# Run evaluation
# -------------------------------------------------------------
results = []

for step, batch in tqdm(enumerate(test_loader), total=MAX_SAMPLES):

    audio = batch["audio"][0].numpy()
    text = whisper_asr(audio)[0]
    target = batch["target"][0]

    # speakers in order of appearance (GT labels)
    speaker_labels = []
    for seg in target:
        if seg["speaker"] not in speaker_labels:
            speaker_labels.append(seg["speaker"])
    num_speakers = len(speaker_labels)

    # ---------------------------------------------------------
    # Words (same as your current eval)
    # ---------------------------------------------------------
    clean_text = _PUNCT_RE.sub("", text)
    words = word_tokenize(clean_text)

    # ---------------------------------------------------------
    # 1) Dynamic segmentation
    # ---------------------------------------------------------
    word_embs = encode_words_to_embs(tokenizer, model.encoder, words, device)
    ranges = dynamic_segment_words(model, word_embs, tau=CONT_TAU, max_len=MAX_LEN)
    span_texts = [" ".join(words[a:b]) for (a, b) in ranges]

    # ---------------------------------------------------------
    # 2) Global labeling
    # ---------------------------------------------------------
    span_assignments = global_label_spans_with_memory(
        model=model,
        tokenizer=tokenizer,
        span_texts=span_texts,
        num_speakers=num_speakers,
        device=device,
    )

    # ---------------------------------------------------------
    # Expand to word-level
    # ---------------------------------------------------------
    pred = expand_ranges_to_words(span_assignments, ranges, words, speaker_labels)

    # ---------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------
    gt = target_segments_to_word_labels(target)

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
                "num_spans_pred": len(ranges),
            },
        )

    if step >= MAX_SAMPLES:
        break

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
out_path = SAVE_DIR / f"hybrid_means_{timestamp}.json"

with open(out_path, "w") as f:
    json.dump(means, f, indent=2)

print("=" * 60)
for k, v in means.items():
    print(f"{k:25s}: {v}")
print("=" * 60)
print(f"Saved means to: {out_path}")
