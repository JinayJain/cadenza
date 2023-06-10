from dataclasses import dataclass
from typing import Any
import lightning as l
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torch.nn import functional as F

# TODO: Learned initial hidden state


@dataclass
class CadenzaRNNConfig:
    n_token: int
    d_embed: int
    hidden_size: int
    n_layer: int
    dropout: float
    lr: float = 1e-3


class CadenzaRNN(l.LightningModule):
    def __init__(self, config: CadenzaRNNConfig) -> None:
        super(CadenzaRNN, self).__init__()

        self.save_hyperparameters()

        self.embedding = nn.Embedding(
            num_embeddings=config.n_token,
            embedding_dim=config.d_embed,
        )
        self.lstm = nn.LSTM(
            input_size=config.d_embed,
            hidden_size=config.hidden_size,
            num_layers=config.n_layer,
            dropout=config.dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.n_token,
        )
        self.config = config

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.linear(x)

        return x, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = F.cross_entropy(
            y_hat.view(-1, y_hat.size(-1)),
            y.view(-1),
        )
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(
                self.config.n_layer, batch_size, self.config.hidden_size, device=device
            ),
            torch.zeros(
                self.config.n_layer, batch_size, self.config.hidden_size, device=device
            ),
        )
