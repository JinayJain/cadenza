import os
from typing import Literal
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cadenza.dataset import MusicDataset
from cadenza.constants import Constants
from cadenza.model import MusicNet
from cadenza.preprocess import convert_tokens_to_midi

DATASET_FILE = "data/dataset.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(split: Literal["train", "validation", "test"]):
    df = pd.read_pickle(DATASET_FILE)
    df = df[df["split"] == split]

    data = np.concatenate(df["tokens"].values)

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


def save_model(model: MusicNet, folder_name, prompt=None):
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

                if i % Constants.REPORT_INTERVAL == 0:
                    pbar.set_postfix({"loss": report_loss / Constants.REPORT_INTERVAL})
                    report_loss = 0

                    save_model(model, f"ckpt/latest", prompt=X[0, :100].unsqueeze(0))

                if i % Constants.SAVE_INTERVAL == 0:
                    model.eval()
                    save_model(
                        model,
                        f"ckpt/epoch_{epoch}_step_{i}",
                        prompt=X[0, :100].unsqueeze(0),
                    )
                    model.train()

        model.eval()

        print(f"Epoch {epoch} loss: {epoch_loss / len(train_loader)}")

        val_loss = 0

        with torch.no_grad():
            for X, y in validation_loader:
                X = X.to(device)
                y = y.to(device)

                y_pred, _ = model(X)

                loss = criterion(y_pred.transpose(1, 2), y)

                val_loss += loss.item()

        print(f"Epoch {epoch} validation loss: {val_loss / len(validation_loader)}")


if __name__ == "__main__":
    main()
