from typing import Literal
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from musicgen.dataset import MusicDataset
from musicgen.constants import Constants
from musicgen.model import MusicNet

DATASET_FILE = "data/dataset.pkl"


def load_split(split: Literal["train", "validation", "test"]):
    df = pd.read_pickle(DATASET_FILE)
    df = df[df["split"] == split]

    # join the tokens into one long sequence
    data = np.concatenate(df["tokens"].values)

    return MusicDataset(data, seq_length=Constants.SEQ_LENGTH)


def main():
    train_dataset = load_split("train")
    validation_dataset = load_split("validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Constants.BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=Constants.BATCH_SIZE,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MusicNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(Constants.EPOCHS):
        model.train()
        total_loss = 0
        batches = 0

        pb = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Constants.EPOCHS}", unit="b")

        for x, y in pb:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output, _ = model(x)

            loss = criterion(output.transpose(1, 2), y)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            batches += 1

            if batches % Constants.REPORT_INTERVAL == 0:
                pb.set_postfix(loss=total_loss / Constants.REPORT_INTERVAL)
                total_loss = 0

        pb.close()


if __name__ == "__main__":
    main()
