import os
from typing import Literal
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cadenza.data.dataset import MusicDataset
from cadenza.constants import Constants
from cadenza.model.rnn import CadenzaRNN
from cadenza.data.preprocess import convert_tokens_to_midi
from cadenza.model.transformer import CadenzaTransformer, CadenzaTransformerConfig

DATASET_FILE = "data/dataset.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_split(split: Literal["train", "validation", "test"]):
    df = pd.read_pickle(DATASET_FILE)
    df = df[df["split"] == split]

    data = np.concatenate(df["tokens"].values)[: Constants.CONTEXT_LENGTH * 100]

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
    seed_everything(Constants.SEED)

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

    # model = CadenzaRNN().to(device)
    model_config = CadenzaTransformerConfig(
        n_token=Constants.NUM_TOKENS,
        d_model=512,
        n_head=8,
        n_attn_layer=6,
        d_feedforward=1024,
        dropout=0.1,
        context_length=Constants.CONTEXT_LENGTH,
        norm_first=True,
    )
    model = CadenzaTransformer(model_config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Constants.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    attn_mask = CadenzaTransformer.generate_mask(Constants.CONTEXT_LENGTH, device)

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

                y_pred = model(X, attn_mask)

                loss = criterion(y_pred.transpose(1, 2), y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                report_loss += loss.item()

                if i % Constants.REPORT_INTERVAL == 0:
                    pbar.set_postfix({"loss": report_loss / Constants.REPORT_INTERVAL})
                    report_loss = 0

                    save_model(model, f"ckpt/latest", prompt=X[0, :20].unsqueeze(0))

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
