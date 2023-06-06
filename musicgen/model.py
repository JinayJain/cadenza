import torch
from torch import nn
from torch.nn import functional as F

from musicgen.constants import Constants


class MusicNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=Constants.NUM_TOKENS,
            embedding_dim=Constants.EMBEDDING_DIM,
        )

        self.lstm = nn.LSTM(
            input_size=Constants.EMBEDDING_DIM,
            hidden_size=Constants.HIDDEN_SIZE,
            num_layers=Constants.NUM_LAYERS,
            dropout=Constants.DROPOUT,
        )

        self.linear = nn.Linear(
            in_features=Constants.HIDDEN_SIZE,
            out_features=Constants.NUM_TOKENS,
        )

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.linear(x)

        return x, hidden

    def init_hidden(self, batch_size):
        return (
            torch.zeros(Constants.NUM_LAYERS, batch_size, Constants.HIDDEN_SIZE),
            torch.zeros(Constants.NUM_LAYERS, batch_size, Constants.HIDDEN_SIZE),
        )
