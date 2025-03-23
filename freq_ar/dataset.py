import numpy as np
import torch
from torchvision import datasets, transforms
from einops import rearrange


class FrequencyMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True, image_dtype=torch.bfloat16):
        self.dataset = datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        self.image_dtype = image_dtype

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Convert image to frequency domain using FFT
        freq_image = torch.fft.rfft2(image.squeeze())

        # Apply log1p to complex freq_image's magnitude while keeping its angle
        # freq_image = torch.log1p(torch.abs(freq_image)) * torch.exp(
        #     1j * torch.angle(freq_image)
        # )

        freq_image_real = freq_image.real
        freq_image_imag = freq_image.imag
        freq_image = torch.stack((freq_image_real, freq_image_imag), dim=-1)
        assert freq_image.shape == (28, 15, 2), freq_image.shape

        freq_image = rearrange(
            freq_image.to(dtype=self.image_dtype), "h w c -> (h w) c"
        )

        return freq_image, label
