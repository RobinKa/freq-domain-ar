import torch
import torch.nn as nn
from einops import rearrange
from transformers import LlamaConfig, LlamaModel


class FrequencyARModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        patchify: int,
        label_count: int = 10,
    ):
        super().__init__()

        llama_config = LlamaConfig(
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            intermediate_size=embed_dim * 4,
            use_cache=False,
        )
        self.transformer = LlamaModel(llama_config)

        self.embed_label = nn.Embedding(label_count, embed_dim)
        self.embed_image = nn.Conv1d(
            kernel_size=patchify,
            stride=patchify,
            in_channels=input_dim,
            out_channels=embed_dim,
        )
        self.unembed_image = nn.ConvTranspose1d(
            kernel_size=patchify,
            stride=patchify,
            in_channels=embed_dim,
            out_channels=input_dim,
        )

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 2  # ...SC
        assert label.ndim == x.ndim - 2

        # Keep track of original shape because we want to reshape back to it in the end.
        # Add 1 to the sequence dimension to account for the label.
        orig_shape = x.shape

        # Flatten batch dimension
        x = rearrange(x, "... s c -> (...) s c")
        label = rearrange(label, "... -> (...)")
        orig_seq_len = x.shape[1] + 1  # image + label

        # Apply input layers
        # Embed label (B -> BC -> B1C)
        label = self.embed_label(label).unsqueeze(1)

        # Embed image
        x = x.permute(0, 2, 1)
        x = self.embed_image(x)
        x = x.permute(0, 2, 1)
        # Combine label and image
        x = torch.cat([label, x], dim=1)
        del label

        # Apply LLaMA model
        x = self.transformer(inputs_embeds=x).last_hidden_state

        # Apply output layers
        # Unembed image
        x = x.permute(0, 2, 1)
        x = self.unembed_image(x)
        x = x.permute(0, 2, 1)

        # x.shape[1] is n * patchify instead of (n-1) * patchify + 1.
        # Remove the excess.
        x = x[..., :orig_seq_len, :]

        x = x.reshape(
            *orig_shape[:-2],
            orig_seq_len,
            x.shape[-1],
        )

        return x
