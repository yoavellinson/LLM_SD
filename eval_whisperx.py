import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
from difflib import SequenceMatcher
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
from tqdm import tqdm

device = "cuda"
compute_type = "float16"


def whisperx_diarize(audio, sample_rate, min_speakers=None, max_speakers=None):
    # ASR
    result = asr_model.transcribe(audio, batch_size=16)

    # Alignment model (cached per language)
    lang = result["language"]
    if lang not in align_model_cache:
        align_model_cache[lang] = whisperx.load_align_model(
            language_code=lang,
            device=device,
        )
    model_a, metadata = align_model_cache[lang]

    # Word alignment
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Diarization
    diarize_segments = diarize_pipeline(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # Assign speaker labels to words
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result["segments"]  # word-level info inside

def whisperx_segments_to_words(segments):
    pred = []
    for seg in segments:
        for w in seg.get("words", []):
            if "speaker" not in w:
                continue
            pred.append({
                "word": w["word"].strip(),
                "speaker": w["speaker"],
            })
    return pred


def align_pred_to_gt(pred_words, gt_words):
    pred_tokens = [p["word"].lower() for p in pred_words]
    gt_tokens = [g["word"].lower() for g in gt_words]

    sm = SequenceMatcher(None, gt_tokens, pred_tokens)
    mapping = {}

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for gi, pj in zip(range(i1, i2), range(j1, j2)):
                mapping[gi] = pj

    aligned = []
    for gi, gt in enumerate(gt_words):
        if gi in mapping:
            p = pred_words[mapping[gi]]
            aligned.append({
                "word": gt["word"],
                "speaker": p["speaker"],
            })
    return aligned


asr_model = whisperx.load_model(
    "large-v2",
    device,
    compute_type=compute_type,
)

align_model_cache = {}  # language → (model, metadata)

diarize_pipeline = DiarizationPipeline(
    use_auth_token=True,
    device=device,
)

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
    overlap_scramble_prob=0.5,
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

results = []

for step, batch in enumerate(test_loader):
    if step >= MAX_SAMPLES:
        break

    audio = batch["audio"][0].numpy()
    sr = 16000
    target = batch["target"][0]

    # GT word-level
    gt = target_segments_to_word_labels(target)

    # Run WhisperX
    wx_segments = whisperx_diarize(
        audio,
        sr,
        min_speakers=len(set(g["speaker"] for g in gt)),
        max_speakers=len(set(g["speaker"] for g in gt)),
    )

    pred_raw = whisperx_segments_to_words(wx_segments)
    pred = align_pred_to_gt(pred_raw, gt)

    if len(pred) == 0:
        continue

    # Metrics (reuse yours)
    word_acc = word_speaker_accuracy(pred, gt)
    seg_acc = segment_consistency_accuracy(pred, target)
    boundary = boundary_segmentation_f1(pred, target)
    align = alignment_stats(pred, gt)

    results.append({
        "word_acc": word_acc,
        "seg_acc": seg_acc,
        "boundary_f1": boundary["boundary_f1"],
        "coverage": align["coverage"],
    })

    if step % 50 == 0:
        print(
            f"[{step:04d}] "
            f"WordAcc={word_acc:.3f} | "
            f"SegAcc={seg_acc:.3f} | "
            f"B-F1={boundary['boundary_f1']:.3f}"
        )
