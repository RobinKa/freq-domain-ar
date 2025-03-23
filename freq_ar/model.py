import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaConfig


class FrequencyARModel(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim,
        num_heads,
        num_layers,
        patchify,
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

        self.embed_first = nn.Linear(input_dim, embed_dim)

        self.patchify = nn.Conv1d(
            kernel_size=patchify,
            stride=patchify,
            in_channels=input_dim,
            out_channels=embed_dim,
        )
        self.unpatchify = nn.ConvTranspose1d(
            kernel_size=patchify,
            stride=patchify,
            in_channels=embed_dim,
            out_channels=input_dim,
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

        # Apply LLaMA model
        x = self.transformer(inputs_embeds=x).last_hidden_state

        # Apply output layers
        x = x.permute(0, 2, 1)
        x = self.unpatchify(x)
        x = x.permute(0, 2, 1)

        # x.shape[1] is n * patchify instead of (n-1) * patchify + 1.
        # Remove the excess.
        x = x[:, : orig_shape[1]]

        return x
