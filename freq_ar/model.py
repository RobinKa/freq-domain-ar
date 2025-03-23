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


class FrequencyARModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation="relu",
            ),
            num_layers=num_layers,
        )
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        # Ensure input has a sequence dimension
        if x.dim() == 2:  # If input is (batch_size, input_dim)
            x = x.unsqueeze(1)  # Add a sequence dimension: (batch_size, 1, input_dim)

        # Apply embedding and positional encoding
        x = self.embedding(x)
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
        x = self.output_layer(x)
        return x
