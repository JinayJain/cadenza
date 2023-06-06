import os
from typing import Literal
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from musicgen.dataset import MusicDataset
from musicgen.constants import Constants
from musicgen.model import MusicNet
from musicgen.preprocess import convert_tokens_to_midi

DATASET_FILE = "data/dataset.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(split: Literal["train", "validation", "test"]):
    df = pd.read_pickle(DATASET_FILE)
    df = df[df["split"] == split]

    # join the tokens into one long sequence
    data = np.concatenate(df["tokens"].values)[:10000]

    return MusicDataset(data, seq_length=Constants.SEQ_LENGTH)


# def save_model(model: MusicNet, folder_name, prompt=None):
#     os.makedirs(folder_name, exist_ok=True)

#     torch.save(model.state_dict(), os.path.join(folder_name, "model.pt"))

#     samples = model.generate(
#         num_generate=Constants.SEQ_LENGTH,
#         device=device,
#         prompt=prompt,
#     )

#     midi = convert_tokens_to_midi(samples)

#     midi.save(os.path.join(folder_name, "sample.mid"))


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

    model = MusicNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(Constants.EPOCHS):
        model.train()

        epoch_loss = 0
        report_loss = 0

        with tqdm(
            enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader)
        ) as pbar:
            for i, (X, y) in pbar:
                optimizer.zero_grad()

                X = X.to(device)
                y = y.to(device)

                y_pred, _ = model(X)

                loss = criterion(y_pred.transpose(1, 2), y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                report_loss += loss.item()

                if i % Constants.REPORT_INTERVAL == 0 or i == len(train_loader) - 1:
                    pbar.set_postfix({"loss": report_loss / Constants.REPORT_INTERVAL})
                    report_loss = 0

        epoch_loss /= len(train_loader)

        print(f"Epoch {epoch} loss: {epoch_loss}")


if __name__ == "__main__":
    main()
