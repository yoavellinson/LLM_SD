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

# ---- model imports ----
from speaker_model import SpeakerExtractionModule_bert
from spans import split_into_spans  
from whisper_asr import WhisperASRWrapper

_PUNCT_RE = re.compile(r"[^\w\s]")


# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------

def expand_spans_to_words(span_assignments, spans, words, speaker_labels):
    """
    span_assignments: list[int] length = num_spans, value = speaker index
    spans: list[str]
    words: list[str]
    """
    pred = []
    w_idx = 0

    for spk_idx, span in zip(span_assignments, spans):
        spk = speaker_labels[spk_idx]
        span_words = span.split()

        for _ in span_words:
            if w_idx >= len(words):
                break
            pred.append({
                "word": words[w_idx],
                "speaker": spk
            })
            w_idx += 1

    return pred


@torch.no_grad()
def peel_with_trained_model(
    model,
    tokenizer,
    text,
    num_speakers,
    device,
    span_words=10,
    tau=0.5,
):
    """
    Iterative speaker peeling WITH online memory updates.
    This mirrors training-time behavior and prevents the
    'one sentence per speaker' collapse.
    """

    # --------------------------------------------------
    # 1) Split text into spans
    # --------------------------------------------------
    spans = split_into_spans(text, span_words)
    num_spans = len(spans)

    if num_spans == 0:
        return [], spans

    # --------------------------------------------------
    # 2) Encode all spans ONCE
    # --------------------------------------------------
    enc = tokenizer(
        spans,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    span_embs = (
        model.encoder(**enc)
        .last_hidden_state.mean(dim=1)
    )  # [N, H]

    # --------------------------------------------------
    # 3) Iterative peeling
    # --------------------------------------------------
    remaining = list(range(num_spans))
    assignments = [-1] * num_spans  # speaker index per span

    for k in range(num_speakers):

        if not remaining:
            break

        memory = torch.zeros(model.hidden, device=device)
        selected_local = []

        # ---- ONLINE scoring with memory updates ----
        for idx in remaining:
            z = span_embs[idx:idx+1]  # [1, H]

            logits = model.score(z, memory)
            p = torch.sigmoid(logits)[0].item()

            if p > tau:
                selected_local.append(idx)
                # 🔑 MEMORY UPDATE (THIS IS THE FIX)
                memory = model.mem_gru(z.squeeze(0), memory)

        # ---- fallback: force minimum recall ----
        if len(selected_local) == 0:
            # score all remaining spans WITHOUT memory
            base_memory = torch.zeros(model.hidden, device=device)
            logits_all = model.score(span_embs[remaining], base_memory)
            probs_all = torch.sigmoid(logits_all)

            k_top = min(2, len(remaining))
            topk_local = torch.topk(probs_all, k=k_top).indices.tolist()
            selected_local = [remaining[i] for i in topk_local]

        # ---- assign speaker index ----
        for idx in selected_local:
            assignments[idx] = k

        # ---- remove assigned spans ----
        remaining = [i for i in remaining if i not in selected_local]

    # --------------------------------------------------
    # 4) Any leftovers → last speaker
    # --------------------------------------------------
    last_spk = num_speakers - 1
    for idx in remaining:
        assignments[idx] = last_spk

    return assignments, spans


def non_peeling_assignment(model, tokenizer, text, speaker_labels, device, span_words=10):
    spans = split_into_spans(text, span_words)

    enc = tokenizer(
        spans,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        span_embs = model.encoder(**enc).last_hidden_state.mean(dim=1)

    all_probs = []

    for _ in speaker_labels:
        memory = torch.zeros(model.hidden, device=device)
        probs = torch.sigmoid(model.score(span_embs, memory))
        all_probs.append(probs.unsqueeze(1))

    all_probs = torch.cat(all_probs, dim=1)  # [num_spans, num_speakers]
    assignments = torch.argmax(all_probs, dim=1).tolist()

    return assignments, spans

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
SAVE_DIR = Path("/home/workspace/yoavellinson/LLM_SD/diarization_logs_trained")
MODEL_CKPT = Path("/home/workspace/yoavellinson/LLM_SD/ckpt/epoch09-loss0.9531.ckpt")
SPAN_WORDS = 10
TAU = 0.5
LOG_EVERY = 50
BATCH_SIZE=1
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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
whisper_asr = WhisperASRWrapper(batch_size=BATCH_SIZE,device=device)

module = SpeakerExtractionModule_bert.load_from_checkpoint(
    MODEL_CKPT,
    map_location=device,
)
module.eval()

model = module.model   # your SpeakerMemoryModel
tokenizer = module.tokenizer

# -------------------------------------------------------------
# Run evaluation
# -------------------------------------------------------------
results = []

for step, batch in tqdm(enumerate(test_loader), total=MAX_SAMPLES):

    # text = batch["input_text"][0]
    audio = batch["audio"][0].numpy()
    text = whisper_asr(audio)
    target = batch["target"][0]

    # speakers in order of appearance (GT)
    speaker_labels = []
    for seg in target:
        if seg["speaker"] not in speaker_labels:
            speaker_labels.append(seg["speaker"])

    num_speakers = len(speaker_labels)

    # ---------------------------------------------------------
    # Model → span assignments
    # ---------------------------------------------------------
    span_assignments, spans = peel_with_trained_model(
        model=model,
        tokenizer=tokenizer,
        text=text,
        num_speakers=num_speakers,
        device=device,
        span_words=SPAN_WORDS,
        tau=TAU,
    )
    # ---------------------------------------------------------
    # Expand to word-level
    # ---------------------------------------------------------
    clean_text = _PUNCT_RE.sub("", text)
    words = word_tokenize(clean_text)

    pred = expand_spans_to_words(
        span_assignments=span_assignments,
        spans=spans,
        words=words,
        speaker_labels=speaker_labels,
    )

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
out_path = SAVE_DIR / f"trained_model_means_{timestamp}.json"

with open(out_path, "w") as f:
    json.dump(means, f, indent=2)

print("=" * 60)
for k, v in means.items():
    print(f"{k:25s}: {v}")
print("=" * 60)
print(f"Saved means to: {out_path}")
