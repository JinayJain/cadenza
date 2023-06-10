import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

from cadenza.constants import Constants


@dataclass
class CadenzaRNNConfig:
    n_token: int
    d_embed: int
    hidden_size: int
    n_layer: int
    dropout: float


class CadenzaRNN(nn.Module):
    def __init__(self, config: CadenzaRNNConfig) -> None:
        super().__init__()

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

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.linear(x)

        return x, hidden

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(
                Constants.NUM_LAYERS,
                batch_size,
                Constants.HIDDEN_SIZE,
            ).to(device),
            torch.zeros(
                Constants.NUM_LAYERS,
                batch_size,
                Constants.HIDDEN_SIZE,
            ).to(device),
        )

    def generate(self, num_generate, device, prompt):
        """
        Generate a sequence of tokens.

        Parameters
        ----------

        num_generate: int
            The number of new tokens to generate.

        device: torch.device
            The device to run the model on.

        prompt: torch.Tensor
            The prompt to start generating from. Must be of shape
            (batch_size, seq_length).
        """

        batch_size, seq_length = prompt.shape

        hidden = self.init_hidden(batch_size, device)
        _, hidden = self.forward(prompt, hidden)

        tokens = prompt[:, -1].unsqueeze(1)

        for _ in range(num_generate):
            logits, hidden = self.forward(tokens, hidden)

            probs = F.softmax(logits[:, -1], dim=-1)
            tokens = torch.multinomial(probs, num_samples=1)

            prompt = torch.cat([prompt, tokens], dim=1)

        return prompt
