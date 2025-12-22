from data.ami_dataset import (
    AMIWordChunkDataset,
    load_conversations,
    split_conversations,
    ami_collate_fn,
)
from torch.utils.data import DataLoader
from pathlib import Path
from nltk_utils import *

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

    results.append({
        "step": step,
        "word_acc": word_acc,
        "seg_acc": seg_acc,
        "coverage": align["coverage"],
    })
    pred_sentences = pred_words_to_sentences(pred)

    if step % LOG_EVERY == 0:
        print(
            f"[{step:04d}] "
            f"WordAcc={word_acc:.3f} | "
            f"SegAcc={seg_acc:.3f} | "
            f"Coverage={align['coverage']:.2f}"
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
                "num_words_pred": len(pred),
                "num_words_gt": len(gt),
            },
        )
        

# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------

import numpy as np

print("=" * 60)
print("Mean word accuracy   :", np.mean([r["word_acc"] for r in results]))
print("Mean segment accuracy:", np.mean([r["seg_acc"] for r in results]))
print("Mean coverage        :", np.mean([r["coverage"] for r in results]))
print("=" * 60)
