import math
from typing import Optional
from torch import nn
from torch.nn import functional as F
import torch
from dataclasses import dataclass
import lightning as l

from cadenza.util import top_p_sample


@dataclass
class CadenzaTransformerConfig:
    vocab_size: int
    d_model: int
    n_head: int
    n_attn_layer: int
    dropout: float
    block_size: int
    lr: float


class CadenzaTransformer(l.LightningModule):
    def __init__(self, cfg: CadenzaTransformerConfig):
        super(CadenzaTransformer, self).__init__()

        self.save_hyperparameters()

        self.lr = cfg.lr
        self.vocab_size = cfg.vocab_size

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_encoder = PositionalEncoding(cfg.d_model, cfg.dropout)

        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(cfg) for _ in range(cfg.n_attn_layer)]
        )

        self.fc = nn.Linear(cfg.d_model, cfg.vocab_size)

        self.block_size = cfg.block_size

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = self.fc(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(
            y_hat.view(-1, y_hat.size(-1)),
            y.view(-1),
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(
            y_hat.view(-1, y_hat.size(-1)),
            y.view(-1),
        )
        self.log("val_loss", loss)

    def generate(
        self,
        prompt: Optional[torch.Tensor],
        num_tokens,
        top_p: float = 0.95,
        temperature: float = 1.0,
        show_progress: bool = False,
    ):
        if prompt is None:
            prompt = torch.randint(
                self.vocab_size,
                (1, 1),
                dtype=torch.long,
            )

        tokens = prompt

        if self.training:
            print("WARNING: You are running generate in training mode")

        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(range(num_tokens))
        else:
            iterator = range(num_tokens)

        with torch.no_grad():
            for i in iterator:
                model_input = tokens[:, -self.block_size :].to(self.device)
                token = self.generate_single(model_input, top_p, temperature).to(
                    tokens.device
                )
                tokens = torch.cat([tokens, token], dim=1)

        return tokens

    def generate_single(self, tokens, top_p, temperature):
        logits = self(tokens) / temperature

        token = top_p_sample(logits, top_p).to(tokens.device)

        return token


class TransformerLayer(nn.Module):
    def __init__(self, cfg: CadenzaTransformerConfig):
        super(TransformerLayer, self).__init__()

        self.attn = MultiHeadRelAttention(cfg)
        self.ff = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class MultiHeadRelAttention(nn.Module):
    def __init__(self, cfg: CadenzaTransformerConfig):
        super(MultiHeadRelAttention, self).__init__()

        assert cfg.d_model % cfg.n_head == 0

        self.n_head = cfg.n_head
        self.w_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # different per head
        self.Er = nn.Parameter(
            torch.randn(cfg.n_head, cfg.block_size, cfg.d_model // cfg.n_head)
        )

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
        )

    def forward(self, x):
        # x is shape (B, T, C) (batch size, sequence length, embedding size)
        B, T, C = x.size()

        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)  # (B, T, C)

        n_head = self.n_head
        q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
        k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
        v = v.view(B, T, n_head, C // n_head).transpose(1, 2)

        raw_attn = q @ k.transpose(-2, -1)
        Srel = q @ self.Er[:, -T:, :].transpose(-2, -1)

        # add a column of zeros to the left
        Srel = torch.cat([torch.zeros_like(Srel[:, :, :, :1]), Srel], dim=-1)
        Srel = Srel.view(B, n_head, T + 1, T)[:, :, 1:, :]
        attn = (raw_attn + Srel) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        y = attn @ v  # (B, H, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(y)

        return y


# Taken from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg: CadenzaTransformerConfig):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(cfg.d_model, cfg.d_model * 4)
        self.fc2 = nn.Linear(cfg.d_model * 4, cfg.d_model)
        self.act = NewGELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
