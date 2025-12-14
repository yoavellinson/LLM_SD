import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from collections import defaultdict
import random 


def ami_collate_fn(batch):
    max_num_speakers = max(b["num_speakers"] for b in batch)

    return {
        "input_text": [b["input_text"] for b in batch],
        "target": [b["target"] for b in batch],

        # per-sample
        "speakers": [b["speakers"] for b in batch],
        "num_speakers": [b["num_speakers"] for b in batch],
        "arrival_order": [b["arrival_order"] for b in batch],

        # batch-level
        "max_num_speakers": max_num_speakers,
    }


def meeting_to_words(df):
    """
    Convert meeting DataFrame into a per-word stream with real times.
    """
    words = []

    for _, row in df.iterrows():
        toks = row["text"].split()
        n = len(toks)
        if n == 0:
            continue

        # Distribute word times uniformly inside the utterance
        t0 = row["start_real"]
        t1 = row["end_real"]

        if t0 is None or t1 is None or t1 <= t0:
            # fallback: no timing, treat as non-overlapping
            for w in toks:
                words.append({
                    "word": w,
                    "speaker": row["speaker"],
                    "start": None,
                    "end": None,
                })
            continue

        dt = (t1 - t0) / n
        for i, w in enumerate(toks):
            words.append({
                "word": w,
                "speaker": row["speaker"],
                "start": t0 + i * dt,
                "end": t0 + (i + 1) * dt,
            })

    return words


def load_conversations(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure correct order
    df = df.sort_values(["meeting", "start"]).reset_index(drop=True)

    conversations = {}
    for meeting, g in df.groupby("meeting", sort=False):
        conversations[meeting] = g

    return conversations


def split_conversations(conversations, test_ratio=0.2, seed=42):
    meetings = sorted(conversations.keys())

    rng = random.Random(seed)
    rng.shuffle(meetings)

    n_test = int(len(meetings) * test_ratio)
    test_meetings = set(meetings[:n_test])
    train_meetings = set(meetings[n_test:])

    train_convs = {m: conversations[m] for m in train_meetings}
    test_convs = {m: conversations[m] for m in test_meetings}

    return train_convs, test_convs


def scramble_overlapping_words(words, overlap_idxs, rng, window=6):
    """
    Scramble words only inside overlap regions.
    """
    if not overlap_idxs:
        return words

    words = words.copy()
    overlap_idxs = sorted(overlap_idxs)

    i = 0
    while i < len(overlap_idxs):
        j = i
        while j + 1 < len(overlap_idxs) and overlap_idxs[j + 1] == overlap_idxs[j] + 1:
            j += 1

        # contiguous overlap block
        block = overlap_idxs[i:j+1]
        if len(block) > 1:
            start = block[0]
            end = min(block[-1] + 1, start + window)

            segment = words[start:end]
            rng.shuffle(segment)
            words[start:end] = segment

        i = j + 1

    return words

def find_overlap_indices(words):
    """
    Returns a set of word indices that are inside real overlap regions.
    """
    overlap_idxs = set()

    for i in range(len(words)):
        wi = words[i]
        if wi["start"] is None:
            continue

        for j in range(i + 1, len(words)):
            wj = words[j]
            if wj["start"] is None:
                continue

            # different speakers + time overlap
            if wi["speaker"] != wj["speaker"]:
                if max(wi["start"], wj["start"]) < min(wi["end"], wj["end"]):
                    overlap_idxs.add(i)
                    overlap_idxs.add(j)

    return overlap_idxs

def speaker_arrival_order(turns):
    """
    Returns speakers in order of first appearance.
    """
    order = []
    seen = set()

    for t in turns:
        spk = t["speaker"]
        if spk not in seen:
            order.append(spk)
            seen.add(spk)

    return order


def words_to_speaker_turn_dicts(words):
    """
    Convert a word list into speaker turns as dicts.
    """
    turns = []
    cur_speaker = None
    cur_words = []

    for w in words:
        spk = w["speaker"]
        if spk != cur_speaker:
            if cur_words:
                turns.append({
                    "speaker": cur_speaker,
                    "text": " ".join(cur_words),
                })
            cur_speaker = spk
            cur_words = [w["word"]]
        else:
            cur_words.append(w["word"])

    if cur_words:
        turns.append({
            "speaker": cur_speaker,
            "text": " ".join(cur_words),
        })

    return turns


class AMIWordChunkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        conversations,
        word_budget=256,
        overlap_scramble_prob=0.5,
        scramble_window=6,
        seed=42,
    ):
        self.word_budget = word_budget
        self.overlap_scramble_prob = overlap_scramble_prob
        self.scramble_window = scramble_window
        self.rng = random.Random(seed)

        self.meetings = []
        self.word_streams = {}

        for meeting, df in conversations.items():
            words = meeting_to_words(df)
            if len(words) >= word_budget:
                self.meetings.append(meeting)
                self.word_streams[meeting] = words

    def __len__(self):
        return len(self.meetings) * 1000  # virtual length

    def __getitem__(self, idx):
        meeting = self.meetings[idx % len(self.meetings)]
        stream = self.word_streams[meeting]

        # sample continuous span
        start = self.rng.randint(0, len(stream) - self.word_budget)
        chunk = stream[start:start + self.word_budget]

        # ---------- TARGET ----------
        target = words_to_speaker_turn_dicts(chunk)

        speakers = sorted({t["speaker"] for t in target})
        arrival_order = speaker_arrival_order(target)

        # ---------- INPUT ----------
        clean_words = [w["word"] for w in chunk]
        overlap_idxs = find_overlap_indices(chunk)

        input_words = clean_words.copy()
        if overlap_idxs and self.rng.random() < self.overlap_scramble_prob:
            input_words = scramble_overlapping_words(
                input_words,
                overlap_idxs,
                self.rng,
                self.scramble_window,
            )

        return {
            "input_text": " ".join(input_words),
            "target": target,
            "speakers": speakers,
            "num_speakers": len(speakers),
            "arrival_order": arrival_order,
        }




if __name__ == "__main__":
    convs = load_conversations("/home/workspace/yoavellinson/LLM_SD/data/ami_synced/ami_utterances.csv")
    train_convs, test_convs = split_conversations(convs)

    train_ds = AMIWordChunkDataset(
        train_convs,
        word_budget=256,
        overlap_scramble_prob=0.5,
    )

    train_loader = DataLoader(train_ds,batch_size=6,collate_fn=ami_collate_fn)
    for step,batch in enumerate(train_loader):
        print(batch)