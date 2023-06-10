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


def save_model(model, folder_name, prompt=None):
    model.eval()

    os.makedirs(folder_name, exist_ok=True)

    if prompt is None:
        prompt = torch.randint(
            Constants.NUM_TOKENS,
            (1, 1),
            dtype=torch.long,
            device=device,
        )

    sample = model.generate(
        num_generate=1024,
        device=device,
        # prompt=torch.randint(
        #     Constants.NUM_TOKENS,
        #     (1, 1),
        #     dtype=torch.long,
        #     device=device,
        # ),
        prompt=prompt,
    )

    midi = convert_tokens_to_midi(sample[0])
    print(midi)

    midi.save(os.path.join(folder_name, "sample.mid"))
    torch.save(model.state_dict(), os.path.join(folder_name, "model.pt"))
    Constants.save(os.path.join(folder_name, "constants.json"))

    model.train()


def main():
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

    config = CadenzaRNNConfig(
        n_token=Constants.NUM_TOKENS,
        d_embed=512,
        hidden_size=512,
        n_layer=3,
        dropout=0.5,
    )
    model = CadenzaRNN(config)

    ckpt_callback = ModelCheckpoint(
        every_n_train_steps=500,
        monitor="train_loss",
        dirpath="checkpoints",
        filename="model-{epoch}-{train_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = pl.Trainer(profiler="simple", max_epochs=10, callbacks=[ckpt_callback])
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
