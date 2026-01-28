import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel

from spans import split_into_spans,span_mask_for_speaker
from difflib import SequenceMatcher


# ------------------------------
# Helpers
# ------------------------------
def encode_spans(tokenizer, encoder, spans, device):
    enc = tokenizer(
        spans,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        span_embs = encoder(**enc).last_hidden_state.mean(dim=1)  # [N, H]
    return span_embs


def encode_words_to_embs(tokenizer, encoder, words, device):
    """
    words: list[str]
    Return: [T, H] word embeddings using tokenizer word alignment.
    Requires a FAST tokenizer (AutoTokenizer usually is).
    """
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        hs = encoder(**enc).last_hidden_state.squeeze(0)  # [seq, H]

    word_ids = enc.word_ids(batch_index=0)
    H = hs.size(-1)
    T = len(words)

    word_embs = torch.zeros(T, H, device=device)
    counts = torch.zeros(T, device=device)

    for tok_i, wid in enumerate(word_ids):
        if wid is None:
            continue
        word_embs[wid] += hs[tok_i]
        counts[wid] += 1

    word_embs = word_embs / counts.clamp_min(1).unsqueeze(-1)
    return word_embs


def target_to_words_and_speakers(target_turns):
    """
    target_turns: list of {"speaker": X, "text": "..."}
    returns:
      words: list[str]
      speakers: list[Any] same length
    """
    words, spks = [], []
    for t in target_turns:
        toks = t["text"].split()
        words.extend(toks)
        spks.extend([t["speaker"]] * len(toks))
    return words, spks


# ------------------------------
# Core model
# ------------------------------
class SpeakerMemoryModel(nn.Module):
    """
    - Frozen encoder -> embeddings
    - mem_gru: builds speaker memory
    - speaker_head: scores span belongs-to-speaker(memory)
    - cont_head: predicts SAME-SPEAKER continuation at word level
    """
    def __init__(self, encoder_name="bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden = self.encoder.config.hidden_size

        # memory
        self.mem_gru = nn.GRUCell(self.hidden, self.hidden)

        # speaker scoring (span, mem, span*mem) -> logit
        self.speaker_head = nn.Linear(self.hidden * 3, 1)

        # continuity scoring (prev_word, next_word, mem) -> logit
        self.cont_head = nn.Sequential(
            nn.Linear(self.hidden * 3, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

        # freeze encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def score_span(self, span_embs, memory):
        """
        span_embs: [N, H] or [1, H]
        memory: [H]
        returns logits: [N]
        """
        mem = memory.unsqueeze(0).expand(len(span_embs), -1)
        feats = torch.cat([span_embs, mem, span_embs * mem], dim=-1)
        logits = self.speaker_head(feats).squeeze(-1)
        return logits

    def update_memory(self, emb, memory):
        """
        emb: [H]
        memory: [H]
        """
        return self.mem_gru(emb, memory)

    def cont_logit(self, prev_word_emb, next_word_emb, memory):
        """
        prev_word_emb: [H]
        next_word_emb: [H]
        memory: [H]
        returns scalar logit
        """
        feat = torch.cat([prev_word_emb, next_word_emb, memory], dim=-1)
        return self.cont_head(feat).squeeze(-1)


# ------------------------------
# Lightning module
# ------------------------------
class SpeakerExtractionModule(pl.LightningModule):
    def __init__(
        self,
        encoder_name="bert-base-uncased",
        lr=3e-5,
        span_words=3,          
        cont_lambda=0.5,       
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
        self.model = SpeakerMemoryModel(encoder_name)

        self.lr = lr
        self.span_words = span_words
        self.cont_lambda = cont_lambda

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr)

    def training_step(self, batch, batch_idx):
        """
        Total loss = global speaker softmax CE (balanced) + cont_lambda * continuity BCE
        """
        total_loss = 0.0
        total_global = 0.0
        total_cont = 0.0
        n_items = 0

        # --- iterate items in batch (texts are variable) ---
        for i in range(len(batch["input_text"])):

            text = batch["input_text"][i]
            target = batch["target"][i]
            arrival = batch["arrival_order"][i]
            num_speakers = len(arrival)

            # -------------------------------
            # A) GLOBAL labeling loss (span-level)
            # -------------------------------
            spans = split_into_spans(text, self.span_words)
            if len(spans) == 0 or num_speakers == 0:
                continue

            span_embs = encode_spans(self.tokenizer, self.model.encoder, spans, self.device)
            N = span_embs.size(0)

            # Build span_targets by matching spans to GT turns with your existing method:
            # We'll reuse your span_mask_for_speaker if you have it in spans.py.
            # If spans.py doesn't include it, keep your existing function there.
            from spans import span_mask_for_speaker

            span_targets = torch.full((N,), -1, dtype=torch.long, device=self.device)

            # assign once: first speaker that claims it (and leave ambiguous as-is)
            for spk_idx, spk in enumerate(arrival):
                mask = span_mask_for_speaker(spans, target, spk)
                for j, m in enumerate(mask):
                    if m == 1 and span_targets[j] == -1:
                        span_targets[j] = spk_idx

            valid = span_targets >= 0
            if valid.sum() == 0:
                continue

            span_embs_v = span_embs[valid]
            span_targets_v = span_targets[valid]
            Nv = span_embs_v.size(0)

            # speaker memories
            speaker_memories = [
                torch.zeros(self.model.hidden, device=self.device)
                for _ in range(num_speakers)
            ]

            logits = torch.zeros(Nv, num_speakers, device=self.device)

            # online: compute logits vs each speaker memory; update true speaker memory
            for t in range(Nv):
                z = span_embs_v[t]
                for k in range(num_speakers):
                    logits[t, k] = self.model.score_span(z.unsqueeze(0), speaker_memories[k])[0]

                true_k = span_targets_v[t].item()
                speaker_memories[true_k] = self.model.update_memory(z, speaker_memories[true_k])

            # balanced CE
            counts = torch.bincount(span_targets_v, minlength=num_speakers).float()
            weights = 1.0 / (counts + 1e-6)
            weights = weights / weights.sum() * num_speakers

            global_loss = F.cross_entropy(logits, span_targets_v, weight=weights)

            # diagnostics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                span_acc = (preds == span_targets_v).float().mean()

                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            self.log("train/span_acc", span_acc, prog_bar=True, sync_dist=True)
            self.log("train/entropy", entropy, sync_dist=True)

            # -------------------------------
            # B) CONTINUITY loss (word-level, GT-ordered)
            # -------------------------------
            # Use GT turns so continuity labels are clean.
            gt_words, gt_spks = target_to_words_and_speakers(target)
            cont_loss = torch.tensor(0.0, device=self.device)
            cont_acc = torch.tensor(0.0, device=self.device)

            if len(gt_words) >= 2:
                word_embs = encode_words_to_embs(self.tokenizer, self.model.encoder, gt_words, self.device)

                y_same = torch.tensor(
                    [1.0 if gt_spks[t+1] == gt_spks[t] else 0.0 for t in range(len(gt_spks) - 1)],
                    device=self.device,
                )  # [T-1]

                mem = torch.zeros(self.model.hidden, device=self.device)
                cont_logits = []

                for t in range(len(gt_words) - 1):
                    logit = self.model.cont_logit(word_embs[t], word_embs[t+1], mem)
                    cont_logits.append(logit)

                    # update memory with current word
                    mem = self.model.update_memory(word_embs[t], mem)

                    # if boundary, reset memory for next segment
                    if y_same[t].item() < 0.5:
                        mem = torch.zeros(self.model.hidden, device=self.device)

                cont_logits = torch.stack(cont_logits)  # [T-1]

                # balance same vs boundary
                pos = y_same.sum()
                neg = (1.0 - y_same).sum()
                pos_weight = (neg / (pos + 1e-6)).clamp(min=1.0)

                cont_loss = F.binary_cross_entropy_with_logits(cont_logits, y_same, pos_weight=pos_weight)

                with torch.no_grad():
                    pred_same = (torch.sigmoid(cont_logits) > 0.5).float()
                    cont_acc = (pred_same == y_same).float().mean()

            self.log("train/cont_loss", cont_loss, prog_bar=False, sync_dist=True)
            self.log("train/cont_acc", cont_acc, prog_bar=True, sync_dist=True)

            # -------------------------------
            # Combine
            # -------------------------------
            loss = global_loss + self.cont_lambda * cont_loss

            total_loss += loss
            total_global += global_loss
            total_cont += cont_loss
            n_items += 1

        if n_items == 0:
            # avoid Lightning complaining; return a zero loss
            loss0 = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log("train_loss", loss0, prog_bar=True, sync_dist=True)
            return loss0

        total_loss = total_loss / n_items
        total_global = total_global / n_items
        total_cont = total_cont / n_items

        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train/global_loss", total_global, prog_bar=False, sync_dist=True)
        self.log("train/cont_loss_mean", total_cont, prog_bar=False, sync_dist=True)

        return total_loss


class SpeakerMemoryModel_bert(nn.Module):
    def __init__(self, encoder_name="bert-base-uncased"):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden = self.encoder.config.hidden_size

        self.mem_gru = nn.GRUCell(self.hidden, self.hidden)
        self.head = nn.Linear(self.hidden * 3, 1)

        # Freeze encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def score(self, span_embs, memory):
        mem = memory.unsqueeze(0).expand(len(span_embs), -1)
        feats = torch.cat([span_embs, mem, span_embs * mem], dim=-1)
        logits = self.head(feats).squeeze(-1)
        return logits

    def update_memory(self, span_embs, mask, memory):
        for i in range(len(span_embs)):
            if mask[i] > 0.5:
                memory = self.mem_gru(span_embs[i], memory)
        return memory


class SpeakerExtractionModule_bert(pl.LightningModule):
    def __init__(
        self,
        encoder_name="bert-base-uncased",
        lr=3e-5,
        span_words=10,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.model = SpeakerMemoryModel_bert(encoder_name)

        self.lr = lr
        self.span_words = span_words
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.save_hyperparameters(
            {
                "span_words": self.span_words,
                "encoder": self.hparams.encoder_name,
                "frozen_encoder": True,
            }
        )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr)

    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        n_items = 0

        for i in range(len(batch["input_text"])):
            text = batch["input_text"][i]
            target = batch["target"][i]
            arrival = batch["arrival_order"][i]

            num_speakers = len(arrival)

            # --------------------------------------------------
            # 1) Split text into spans
            # --------------------------------------------------
            spans = split_into_spans(text, self.span_words)
            if len(spans) == 0:
                continue

            # --------------------------------------------------
            # 2) Encode spans ONCE
            # --------------------------------------------------
            enc = self.tokenizer(
                spans,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                span_embs = (
                    self.model.encoder(**enc)
                    .last_hidden_state.mean(dim=1)
                )  # [N, H]

            num_spans = span_embs.size(0)

            # --------------------------------------------------
            # 3) Build GT speaker id per span
            # --------------------------------------------------
            span_targets = torch.full(
                (num_spans,),
                fill_value=-1,
                dtype=torch.long,
                device=self.device,
            )

            for spk_idx, spk in enumerate(arrival):
                mask = span_mask_for_speaker(spans, target, spk)
                for j, m in enumerate(mask):
                    if m == 1:
                        span_targets[j] = spk_idx

            # Ignore unassigned spans
            valid = span_targets >= 0
            if valid.sum() == 0:
                continue

            span_embs = span_embs[valid]
            span_targets = span_targets[valid]

            # --------------------------------------------------
            # 4) Score spans against ALL speakers
            # --------------------------------------------------
            speaker_memories = [
                torch.zeros(self.model.hidden, device=self.device)
                for _ in range(num_speakers)
            ]

            logits = torch.zeros(
                span_embs.size(0),
                num_speakers,
                device=self.device,
            )

            # Online pass to update memories
            for t in range(span_embs.size(0)):
                z = span_embs[t]

                for k in range(num_speakers):
                    logits[t, k] = self.model.score(
                        z.unsqueeze(0),
                        speaker_memories[k],
                    )

                # Teacher-forced memory update
                true_k = span_targets[t].item()
                speaker_memories[true_k] = self.model.mem_gru(
                    z,
                    speaker_memories[true_k],
                )

            # --------------------------------------------------
            # 5) Cross-entropy loss
            # --------------------------------------------------
            counts = torch.bincount(
                span_targets,
                minlength=num_speakers,
            ).float()

            weights = 1.0 / (counts + 1e-6)
            weights = weights / weights.sum() * num_speakers

            loss = torch.nn.functional.cross_entropy(
                logits,
                span_targets,
                weight=weights,
            )

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                span_acc = (preds == span_targets).float().mean()

                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            self.log(
                "train/span_acc",
                span_acc,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/entropy",
                entropy,
                sync_dist=True,
            )

            total_loss += loss
            n_items += 1

        if n_items > 0:
            total_loss = total_loss / n_items

        self.log(
            "train_loss",
            total_loss,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss


