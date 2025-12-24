# eval_llm_baseline_async.py
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import re

from openai import AsyncOpenAI
from openai_key import key
from nltk.tokenize import word_tokenize

from data.ami_dataset import (
    AMIWordChunkDataset,
    load_conversations,
    split_conversations,
    ami_collate_fn,
)
from torch.utils.data import DataLoader

from nltk_utils import *
from llm_speaker_tiling import llm_speaker_tiling_async

_PUNCT_RE = re.compile(r"[^\w\s]")

# -------------------------------------------------------------
# Seed
# -------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------
SAVE_DIR = Path("/home/workspace/yoavellinson/LLM_SD/diarization_logs_llm")
LOG_EVERY = 10
MODEL_NAME = "gpt-4o-mini"
MAX_CONCURRENCY = 10          # start with 10; increase carefully if needed
MAX_SAMPLES = 10            # set e.g. 5000 for cheaper runs; None = all

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
)

# Deterministic shuffle (optional)
g = torch.Generator()
g.manual_seed(SEED)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    collate_fn=ami_collate_fn,
    shuffle=True,
    generator=g,
)

# -------------------------------------------------------------
# Helper: run async jobs with bounded concurrency
# -------------------------------------------------------------
async def run_jobs(jobs, max_concurrency: int):
    sem = asyncio.Semaphore(max_concurrency)

    async def wrapped(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*(wrapped(j) for j in jobs))


async def main():
    # Create async OpenAI client once
    client = AsyncOpenAI(api_key=key)  # uses OPENAI_API_KEY from env

    # ---------------------------------------------------------
    # Collect samples (DataLoader is blocking -> convert to list)
    # ---------------------------------------------------------
    samples = []
    for step, batch in tqdm(enumerate(test_loader),total=len(test_loader)):
        samples.append((step, batch))
        if MAX_SAMPLES is not None and len(samples) >= MAX_SAMPLES:
            break

    # ---------------------------------------------------------
    # Build jobs
    # ---------------------------------------------------------
    jobs = []
    meta = []  # keep step, text, target, speaker_labels

    for step, batch in samples:
        text = batch["input_text"][0]
        target = batch["target"][0]

        # speakers in order of appearance (from GT)
        speaker_labels = []
        for seg in target:
            if seg["speaker"] not in speaker_labels:
                speaker_labels.append(seg["speaker"])

        num_speakers = len(speaker_labels)

        jobs.append(
            llm_speaker_tiling_async(
                client=client,
                text=text,
                num_speakers=num_speakers,
                model=MODEL_NAME,
                temperature=0.0,
            )
        )
        meta.append((step, text, target, speaker_labels))

    # ---------------------------------------------------------
    # Run LLM calls concurrently
    # ---------------------------------------------------------
    print(f"Running {len(jobs)} LLM calls with concurrency={MAX_CONCURRENCY} ...")
    labels_list = await run_jobs(jobs, max_concurrency=MAX_CONCURRENCY)

    # ---------------------------------------------------------
    # Compute metrics + logging (sync, fast)
    # ---------------------------------------------------------
    results = []

    for (labels, (step, text, target, speaker_labels)) in tqdm(
        zip(labels_list, meta),
        total=len(meta),
        desc="Scoring",
    ):
        clean_text = _PUNCT_RE.sub("", text)
        words = word_tokenize(clean_text)

        # numeric -> label
        pred = [
            {"word": w, "speaker": speaker_labels[labels[i]]}
            for i, w in enumerate(words)
        ]

        gt = target_segments_to_word_labels(target)

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

    # ---------------------------------------------------------
    # Save means only
    # ---------------------------------------------------------
    means = {
        "num_samples": len(results),
        "word_acc_mean": float(np.mean([r["word_acc"] for r in results])),
        "seg_acc_mean": float(np.mean([r["seg_acc"] for r in results])),
        "boundary_f1_mean": float(np.mean([r["boundary_f1"] for r in results])),
        "boundary_precision_mean": float(np.mean([r["boundary_precision"] for r in results])),
        "boundary_recall_mean": float(np.mean([r["boundary_recall"] for r in results])),
        "coverage_mean": float(np.mean([r["coverage"] for r in results])),
        "seed": SEED,
        "model": MODEL_NAME,
        "max_concurrency": MAX_CONCURRENCY,
        "max_samples": MAX_SAMPLES,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = SAVE_DIR / f"a_llm_means_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump(means, f, indent=2)

    print("=" * 60)
    for k, v in means.items():
        print(f"{k:25s}: {v}")
    print("=" * 60)
    print(f"Saved means to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
