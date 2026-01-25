from data.ami_dataset import (
    AMIWordChunkDataset,
    load_conversations,
    split_conversations,
    ami_collate_fn,
)
from torch.utils.data import DataLoader

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from speaker_model import SpeakerExtractionModule


def main():
    # -------------------------------
    # Config
    # -------------------------------
    CSV_PATH = "/home/workspace/yoavellinson/LLM_SD/data/ami_synced/ami_utterances.csv"
    CKPT_DIR = "/home/workspace/yoavellinson/LLM_SD/ckpt_hybrid"

    BATCH_SIZE = 24
    NUM_WORKERS = 32

    MAX_EPOCHS = 25
    LR = 3e-5
    SPAN_WORDS = 3          # KEY
    CONT_LAMBDA = 0.5       # KEY

    # for now: disable scramble while hybrid stabilizes
    OVERLAP_SCRAMBLE_PROB = 0.0

    # -------------------------------
    # Data
    # -------------------------------
    convs = load_conversations(CSV_PATH)
    train_convs, _ = split_conversations(convs)

    train_ds = AMIWordChunkDataset(
        train_convs,
        word_budget=256,
        overlap_scramble_prob=OVERLAP_SCRAMBLE_PROB,
        text_only=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=ami_collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # -------------------------------
    # Model
    # -------------------------------
    module = SpeakerExtractionModule(
        encoder_name="bert-base-uncased",
        lr=LR,
        span_words=SPAN_WORDS,
        cont_lambda=CONT_LAMBDA,
    )

    # -------------------------------
    # Logging (W&B)
    # -------------------------------
    wandb_logger = WandbLogger(
        project="llm_speaker_diarization",
        log_model=False,
    )

    # -------------------------------
    # Checkpoints: keep best 3 + last
    # -------------------------------
    ckpt_cb = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename="epoch{epoch:02d}-loss{train_loss:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # -------------------------------
    # Trainer
    # -------------------------------
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp_find_unused_parameters_true",
        precision="bf16-mixed",
        max_epochs=MAX_EPOCHS,
        logger=wandb_logger,
        callbacks=[ckpt_cb],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )

    trainer.fit(module, train_loader)


if __name__ == "__main__":
    main()
