import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import random 
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
from pathlib import Path
import re
import numpy as np
_PUNCT_RE = re.compile(r"[^\w\s]")
PRE_PAD  = 0.5   # seconds
POST_PAD = 1.0   # seconds

def remove_punctuation(words):
    """
    Remove punctuation from a list of word tokens.
    Keeps alphanumerics, drops empty tokens.
    """
    clean = []
    for w in words:
        w2 = _PUNCT_RE.sub("", w)
        if w2:
            clean.append(w2)
    return clean


def ami_collate_fn(batch):
    max_num_speakers = max(b["num_speakers"] for b in batch)

    audio = [torch.tensor(b["input_audio"]) for b in batch]
    audio_lens = torch.tensor([a.shape[-1] for a in audio])

    audio_padded = pad_sequence(audio, batch_first=True)
    meeting = [b['meeting'] for b in batch]
    return {
        'meeting': meeting,
        # text
        "input_text": [b["input_text"] for b in batch],
        "target": [b["target"] for b in batch],

        # audio
        "audio": audio_padded,          # [B, T_max]
        "audio_lens": audio_lens,        # [B]

        # speaker metadata
        "arrival_order": [b["arrival_order"] for b in batch],
        "num_speakers": [b["num_speakers"] for b in batch],
        "max_num_speakers": max_num_speakers,
    }

def load_audio_segment(
    path,
    start_time,
    end_time,
    sr=16000,
    pre_pad=PRE_PAD,
    post_pad=POST_PAD,
):
    """
    Load an audio segment with safety padding so the audio
    fully covers the text span.

    Args:
        path (str or Path): audio file path
        start_time (float): text-aligned start time (sec)
        end_time (float): text-aligned end time (sec)
        sr (int): expected sample rate
        pre_pad (float): seconds to pad before start
        post_pad (float): seconds to pad after end
    """

    audio, file_sr = sf.read(path, always_2d=False)

    if file_sr != sr:
        raise RuntimeError(
            f"Sample rate mismatch: file={file_sr}, expected={sr}"
        )

    # mono
    if audio.ndim > 1:
        audio = audio[:, 0]

    audio_len_sec = len(audio) / sr

    # ---- expand time window (Fix 1) ----
    audio_start = max(0.0, start_time - pre_pad)
    audio_end   = min(audio_len_sec, end_time + post_pad)

    # ---- convert to samples ----
    s = int(audio_start * sr)
    e = int(audio_end * sr)

    if e <= s:
        raise ValueError(
            f"Invalid audio slice: start={audio_start}, end={audio_end}"
        )

    return audio[s:e]


def meeting_to_words(df):
    """
    Build per-word stream with real times and audio_path.
    """
    words = []

    for _, row in df.iterrows():
        toks = row["text"].split()
        if not toks:
            continue

        t0 = row["start_time"]
        t1 = row["end_time"]
        n = len(toks)

        dt = (t1 - t0) / n if (t0 is not None and t1 is not None and t1 > t0) else 0.0

        for i, w in enumerate(toks):
            words.append({
                "word": w,
                "speaker": row["speaker"],

                # real timing
                "start_time": t0 + i * dt if t0 is not None else None,
                "end_time": t0 + (i + 1) * dt if t0 is not None else None,

                # 🔑 PROPAGATE MIXTURE AUDIO PATH
                "audio_path": row["audio_path"],
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
        if wi["start_time"] is None:
            continue

        for j in range(i + 1, len(words)):
            wj = words[j]
            if wj["start_time"] is None:
                continue

            # different speakers + time overlap
            if wi["speaker"] != wj["speaker"]:
                if max(wi["start_time"], wj["start_time"]) < min(wi["end_time"], wj["end_time"]):
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


class AMIWordChunkDataset(Dataset):
    def __init__(
        self,
        conversations,
        word_budget=256,
        overlap_scramble_prob=0.5,
        scramble_window=6,
        seed=42,
        sr=16000,
        text_only=False
    ):
        self.word_budget = word_budget
        self.overlap_scramble_prob = overlap_scramble_prob
        self.scramble_window = scramble_window
        self.rng = random.Random(seed)

        self.meetings = []
        self.word_streams = {}
        self.sample_rate= sr
        self.text_only=text_only
        for meeting, df in conversations.items():
            words = meeting_to_words(df)
            if len(words) >= word_budget:
                self.meetings.append(meeting)
                self.word_streams[meeting] = words
    
    def __len__(self):
        return len(self.meetings) * 1000  # virtual length

    def __getitem__(self, idx):
        # --------------------------------------------------
        # Select meeting and continuous word chunk
        # --------------------------------------------------
        meeting = self.meetings[idx % len(self.meetings)]
        stream = self.word_streams[meeting]

        start = self.rng.randint(0, len(stream) - self.word_budget)
        chunk = stream[start:start + self.word_budget]

        # --------------------------------------------------
        # TARGET: annotated conversation (clean)
        # --------------------------------------------------
        target = words_to_speaker_turn_dicts(chunk)

        arrival_order = speaker_arrival_order(target)
        num_speakers = len(arrival_order)

        # --------------------------------------------------
        # INPUT TEXT: ASR-like (no speaker info)
        # --------------------------------------------------
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

        input_words = remove_punctuation(input_words)
        input_text = " ".join(input_words)

        # --------------------------------------------------
        # INPUT AUDIO: mixture (Array1 channel 0)
        # --------------------------------------------------
        if not self.text_only:
            audio = load_audio_segment(
                path=chunk[0]["audio_path"],
                start_time=chunk[0]["start_time"],
                end_time=chunk[-1]["end_time"],
                sr=self.sample_rate,
            )
        else:
            audio= np.zeros(1)
        # --------------------------------------------------
        # Return sample
        # --------------------------------------------------
        return {
            "meeting": meeting,

            # inputs
            "input_audio": audio,        # 1D numpy array or torch tensor
            "input_text": input_text,

            # target
            "target": target,

            # speaker metadata (for permutation safety)
            "arrival_order": arrival_order,
            "num_speakers": num_speakers,
        }

def write_conversation_txt(conversation, path):
    with open(path, "w", encoding="utf-8") as f:
        for turn in conversation:
            speaker = turn["speaker"]
            text = turn["text"].strip()

            f.write(f"Speaker {speaker}:\n")
            f.write(f"  {text}\n\n")

if __name__ == "__main__":
    convs = load_conversations("/home/workspace/yoavellinson/LLM_SD/data/ami_synced/ami_utterances.csv")
    train_convs, test_convs = split_conversations(convs)

    train_ds = AMIWordChunkDataset(
        train_convs,
        word_budget=256,
        overlap_scramble_prob=0.5
    )

    train_loader = DataLoader(train_ds,batch_size=6,collate_fn=ami_collate_fn,shuffle=True)
    path = Path('/home/workspace/yoavellinson/LLM_SD/mock')
    ms =[]
    for step,batch in enumerate(train_loader):
        # print(batch)
        for i in range(len(batch['audio'])):
            a = batch['audio'][i]
            t = batch['target'][i]
            m =batch['meeting'][i]
            if not m[:2] in ms:
                ms.append(m[:2])
                sf.write(path/f'meeting_{m}_audio_{step+i}.wav',a,16000)
                write_conversation_txt(t,path/f'meeting_{m}_text_{step+i}.txt')
        if len(ms)==6:
            break