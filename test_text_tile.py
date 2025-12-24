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
SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# -------------------------------------------------------------
# Load data
# -------------------------------------------------------------
SAVE_DIR = Path("/home/workspace/yoavellinson/LLM_SD/diarization_logs")
LOG_EVERY = 50
SAVE_IF_WORD_ACC_BELOW = 0.4

convs = load_conversations(
    "/home/workspace/yoavellinson/LLM_SD/data/ami_synced/ami_utterances.csv"
)
_, test_convs = split_conversations(convs)

test_ds = AMIWordChunkDataset(
    test_convs,
    word_budget=256,
    overlap_scramble_prob=0.5,
)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    collate_fn=ami_collate_fn,
    shuffle=True,
)

# -------------------------------------------------------------
# Run baseline
# -------------------------------------------------------------

results = []

for step, batch in enumerate(test_loader):

    text = batch["input_text"][0]
    target = batch["target"][0]

    # speakers in order of appearance
    speaker_labels = []
    for seg in target:
        if seg["speaker"] not in speaker_labels:
            speaker_labels.append(seg["speaker"])

    num_speakers = len(speaker_labels)

    # predict
    pred = text_only_speaker_tiling(
        text=text,
        num_speakers=num_speakers,
    )

    # numeric → label
    pred = [
        {
            "word": p["word"],
            "speaker": speaker_labels[p["speaker"]],
        }
        for p in pred
    ]

    # GT
    gt = target_segments_to_word_labels(target)

    # metrics
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
    pred_sentences = pred_words_to_sentences(pred)

    if step % LOG_EVERY == 0:
        print(
            f"[{step:04d}] "
            f"WordAcc={word_acc:.3f} | "
            f"SegAcc={seg_acc:.3f} | "
            f"B-F1={boundary_metrics['boundary_f1']:.3f}"
            )
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
# Save means only
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
out_path = SAVE_DIR / f"a_means_{timestamp}.json"

with open(out_path, "w") as f:
    json.dump(means, f, indent=2)

print("=" * 60)
for k, v in means.items():
    print(f"{k:20s}: {v}")
print("=" * 60)
print(f"Saved means to: {out_path}")
