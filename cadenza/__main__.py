from argparse import ArgumentParser
import os
from typing import Literal
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from cadenza.data.dataset import MusicDataset
from cadenza.constants import Constants
from cadenza.data.preprocess import convert_tokens_to_midi
from cadenza.model.v2.rnn import CadenzaRNN, CadenzaRNNConfig
from cadenza.model.v2.transformer import (
    CadenzaTransformer,
    CadenzaTransformerConfig,
)

DATASET_FILE = "data/dataset.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_split(split: Literal["train", "validation", "test"]):
    df = pd.read_pickle(DATASET_FILE)
    df = df[df["split"] == split]

    data = np.concatenate(df["tokens"].values)

    return MusicDataset(data, seq_length=Constants.CONTEXT_LENGTH)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("-c", "--checkpoint", type=str, required=False)

    return parser.parse_args()


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("high")

    train_dataset = load_split("train")
    validation_dataset = load_split("validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Constants.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=Constants.BATCH_SIZE,
        shuffle=True,
    )

    # config = CadenzaRNNConfig(
    #     n_token=Constants.NUM_TOKENS,
    #     d_embed=512,
    #     hidden_size=512,
    #     n_layer=3,
    #     dropout=0.5,
    # )
    # model = CadenzaRNN(config)

    if args.checkpoint:
        model = CadenzaTransformer.load_from_checkpoint(args.checkpoint)
    else:
        config = CadenzaTransformerConfig(
            vocab_size=Constants.NUM_TOKENS,
            block_size=Constants.CONTEXT_LENGTH,
            d_model=512,
            n_head=8,
            n_attn_layer=6,
            dropout=0.1,
            lr=1e-3,
        )
        model = CadenzaTransformer(config)

    ckpt_callback = ModelCheckpoint(
        every_n_train_steps=100,
        monitor="train_loss",
        dirpath="checkpoints",
        filename="model-{epoch}-{train_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(max_epochs=1000, callbacks=[ckpt_callback])
    trainer.fit(model, train_loader, validation_loader)


if __name__ == "__main__":
    main()
