from torch import nn, Tensor
import torch
import math
from dataclasses import dataclass


@dataclass
class CadenzaTransformerConfig:
    d_model: int
    n_head: int
    n_attn_layer: int
    n_token: int
    d_feedforward: int
    dropout: float
    context_length: int
    norm_first: bool = False
    attn_mask: torch.Tensor = False


class CadenzaTransformer(nn.Module):
    def __init__(
        self,
        config: CadenzaTransformerConfig,
    ) -> None:
        super(CadenzaTransformer, self).__init__()

        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.n_token,
            embedding_dim=config.d_model,
        )

        self.pos_enc = PositionalEncoding(
            d_model=config.d_model,
            dropout=config.dropout,
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_head,
                dim_feedforward=config.d_feedforward,
                dropout=config.dropout,
                batch_first=True,
                norm_first=config.norm_first,
            ),
            num_layers=config.n_attn_layer,
        )

        self.fc = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_token,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x) * math.sqrt(self.config.d_model)
        x = self.pos_enc(x)
        x = self.encoder(x, self.config.attn_mask)
        x = self.fc(x)

        return x  # (batch_size, seq_len, n_token)

    def generate(self, num_generate, device, prompt):
        generated = prompt
        TEMP = 1.0
        for _ in range(num_generate):
            recent = generated[-self.config.context_length :,]
            output = self.forward(
                recent,
                mask=self.generate_mask(recent.shape[0], device=device),
            )

            # Sample from the output distribution
            next_token = torch.multinomial(
                torch.softmax(output[-1, -1, :] / TEMP, dim=-1), num_samples=1
            )

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

        return generated

    @staticmethod
    def generate_mask(sz: int, device="cpu") -> Tensor:
        return torch.triu(
            torch.full(
                (sz, sz),
                fill_value=float("-inf"),
                device=device,
            ),
            diagonal=1,
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
