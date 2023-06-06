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
            batch_first=True,
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
