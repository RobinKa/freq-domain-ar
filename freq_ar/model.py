import torch
import torch.nn as nn
import math  # Added for positional encoding
from einops import rearrange  # Added for clearer reshaping
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x):
        return x + self.encoding[: x.size(0), :]


class LearnablePositionEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, embed_dim))

    def forward(self, x):
        return x + self.encoding[: x.size(1), :]


class FrequencyARModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        # self.positional_encoding = PositionalEncoding(embed_dim)
        self.positional_encoding = LearnablePositionEncoding(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation="relu",
            ),
            num_layers=num_layers,
        )

        self.embed_first = nn.Linear(input_dim, embed_dim)
        self.unembed_first = nn.Linear(embed_dim, input_dim)

        self.patchify = nn.Conv1d(
            kernel_size=1, stride=1, in_channels=input_dim, out_channels=embed_dim
        )
        self.unpatchify = nn.ConvTranspose1d(
            kernel_size=1, stride=1, in_channels=embed_dim, out_channels=input_dim
        )

    def forward(self, x):
        # Ensure input has a sequence dimension
        if x.dim() == 2:  # If input is (batch_size, input_dim)
            x = x.unsqueeze(1)  # Add a sequence dimension: (batch_size, 1, input_dim)

        # orig_x = x

        # Apply embedding and positional encoding
        # x = self.embedding(x)
        xx = x[:, 1:]
        xx = xx.permute(0, 2, 1)
        xx = self.patchify(xx)
        xx = xx.permute(0, 2, 1)
        yy = x[:, :1]
        yy = self.embed_first(yy)
        x = torch.cat([yy, xx], dim=1)

        x = self.positional_encoding(x)

        # Reshape to (seq_len, batch_size, embed_dim) for Transformer
        x = rearrange(
            x, "b s e -> s b e"
        )  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(0), device=x.device, dtype=torch.bool
        )

        # Apply transformer decoder with causal mask
        x = self.transformer(
            x,
            mask=causal_mask,
            is_causal=True,
        )  # Use x as both tgt and memory

        # Reshape back to (batch_size, seq_len, embed_dim)
        x = rearrange(
            x, "s b e -> b s e"
        )  # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)

        # Apply output layer
        # x = self.output_layer(x)

        xx = x[:, 1:]
        xx = xx.permute(0, 2, 1)
        xx = self.unpatchify(xx)
        xx = xx.permute(0, 2, 1)

        yy = x[:, :1]
        yy = self.unembed_first(yy)

        x = torch.cat([yy, xx], dim=1)

        # x = orig_x + x

        return x
