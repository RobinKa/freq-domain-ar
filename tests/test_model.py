import pytest
import torch

from freq_ar.model import FrequencyARModel

# Hyperparameters
input_dim = 2
embed_dim = 16
num_heads = 2
num_layers = 1
patchify = 5


@pytest.fixture
def model():
    return torch.compile(
        FrequencyARModel(input_dim, embed_dim, num_heads, num_layers, patchify)
    )


def test_model_forward(model):
    batch_size = 2
    height = 28
    width = 15
    dummy_input = torch.rand(batch_size, height * width + 1, input_dim)

    # Perform a forward pass
    output = model(dummy_input)

    # Assert the output shape matches the input shape
    assert output.shape == (batch_size, height * width + 1, input_dim), (
        f"Expected output shape {(batch_size, height * width + 1, input_dim)}, "
        f"but got {output.shape}"
    )


def test_model_backward(model):
    batch_size = 2
    height = 28
    width = 15
    dummy_input = torch.rand(
        batch_size, height * width + 1, input_dim, requires_grad=True
    )

    # Perform a forward pass
    output = model(dummy_input)

    # Create a dummy loss
    loss = output.mean()

    # Perform a backward pass
    loss.backward()

    # Assert gradients are computed for the input
    assert dummy_input.grad is not None, "Gradients were not computed for the input"
