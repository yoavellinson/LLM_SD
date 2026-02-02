# ContextDiarist    
# Text-Based Speaker Diarization with Speaker Memory

This repository contains the code for a **text-only speaker diarization system** that models speaker identity using a **speaker memory mechanism**, without relying on acoustic features.
The project explores whether speaker turns can be recovered from **text structure alone**, and compares the proposed methods against strong baselines, including WhisperX.

---

## Overview

Speaker diarization is traditionally treated as an **audio-based problem**.
In this project, we reformulate diarization as a **text-structural learning task**, where speaker identity is inferred from:

* lexical choice
* discourse structure
* consistency across conversational turns

We propose two methods based on a frozen BERT encoder combined with a **learnable speaker memory** that enforces long-range speaker consistency.

---

## Proposed Methods

### 1. BERT Encoder + Speaker Memory (Fixed Spans)

* The conversation text is split into fixed-length spans (e.g., 10 words).
* Each span is embedded using a frozen BERT encoder.
* Each speaker is represented by a **memory vector**, updated over time using a GRU.
* Spans are assigned to speakers based on compatibility with speaker memory.
* Training uses span-level cross-entropy loss with teacher-forced memory updates.

This method formulates diarization as **span classification with persistent speaker identity**.

---

### 2. Hybrid Incremental Span Construction (Next-Token Inspired)

* Speaker turns are built incrementally, starting from a single word.
* The model predicts whether the next word belongs to the same speaker.
* Spans grow dynamically based on model confidence.
* Speaker memory is updated online during span construction.

This approach better aligns with **natural speaker turns** and reduces fragmentation caused by fixed segmentation.

---

## Speaker Memory Model

The speaker memory is a **learned, persistent representation** of each speaker that evolves over time.

* Implemented using a GRU cell.
* Updated sequentially as new spans or words are assigned.
* Enforces speaker consistency across distant turns.
* Separates *what is said* (BERT embeddings) from *who is speaking* (speaker identity).

What the Speaker Memory Represents (Intuition)

The speaker memory is not another text embedding.
Instead, it is a dynamic identity state that summarizes how a specific speaker has spoken so far in the conversation.

Concretely:

BERT embeddings encode local linguistic content
- What is being said in this span?

Speaker memory encodes global speaker behavior
- Does this span sound like the same person as before?

The memory evolves over time and allows the model to reason about speaker continuity, even when turns are far apart or linguistically simila
---

## Dataset

* **AMI Meeting Corpus** (text-only version).
* Ground-truth speaker annotations are used for training and evaluation.
* Experiments are conducted on:

  * clean transcripts
  * ASR-generated transcripts (via WhisperX)

---

## Baselines

The following baselines are evaluated for comparison:

* TextTiling (classical segmentation)
* LLM-based text segmentation
* WhisperX (audio-based ASR + diarization pipeline)

WhisperX is used as a strong end-to-end reference system.

---

## Training

* Encoder: frozen `bert-base-uncased`
* Loss: span-level **cross-entropy over speaker identities**
* Speaker imbalance handled with per-speaker loss reweighting
* Speaker memory updated using teacher forcing during training


---

## Evaluation

Evaluation is performed at the **word level**, using:

* Word-level speaker accuracy
* Segment consistency accuracy
* Boundary F1 score
* Alignment coverage

Custom evaluation scripts are provided for:

* trained text-based models
* WhisperX-based diarization

---


## Key Takeaway

> Speaker diarization can be effectively approached as a **text-based identity modeling problem**, using persistent speaker memory rather than acoustic embeddings.

---

## Future Work

* Integrate acoustic features with the text-based speaker memory model.
* Extend the approach to long-context, full-meeting diarization.
* Evaluate generalization on conversational datasets beyond AMI.
