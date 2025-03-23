import math

import torch
import torch.nn as nn


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
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x):
        return x + self.encoding[: x.size(1), :]


class LearnablePositionEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, embed_dim))

    def forward(self, x):
        return x + self.encoding[: x.size(1), :]


class FrequencyARModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.positional_encoding = PositionalEncoding(embed_dim)
        # self.positional_encoding = LearnablePositionEncoding(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation="relu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.embed_first = nn.Linear(input_dim, embed_dim)
        self.unembed_first = nn.Linear(embed_dim, input_dim)

        self.patchify = nn.Conv1d(
            kernel_size=15, stride=15, in_channels=input_dim, out_channels=embed_dim
        )
        self.unpatchify = nn.ConvTranspose1d(
            kernel_size=15, stride=15, in_channels=embed_dim, out_channels=input_dim
        )

    def forward(self, x):
        # Ensure input has a sequence dimension
        if x.dim() == 2:  # If input is (batch_size, input_dim)
            x = x.unsqueeze(1)  # Add a sequence dimension: (batch_size, 1, input_dim)

        orig_shape = x.shape

        # Apply input layers
        xx = x[:, 1:]
        xx = xx.permute(0, 2, 1)
        xx = self.patchify(xx)
        xx = xx.permute(0, 2, 1)
        yy = x[:, :1]
        yy = self.embed_first(yy)
        x = torch.cat([yy, xx], dim=1)

        # Apply positional encoding
        x = self.positional_encoding(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device, dtype=torch.bool
        )

        # Apply transformer decoder with causal mask
        x = self.transformer(
            x,
            mask=causal_mask,
            is_causal=True,
        )

        # Apply output layers
        x = x.permute(0, 2, 1)
        x = self.unpatchify(x)
        x = x.permute(0, 2, 1)

        # x.shape[1] is n * 15 instead of (n-1) * 15 + 1.
        # Remove the excess.
        x = x[:, : orig_shape[1]]

        return x
