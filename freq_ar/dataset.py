import numpy as np
import torch
from torchvision import datasets, transforms


class FrequencyMNIST(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = datasets.MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Convert image to frequency domain using FFT
        freq_image = torch.fft.fft2(image.squeeze())
        freq_image_real = freq_image.real
        freq_image_imag = freq_image.imag
        freq_image = torch.stack((freq_image_real, freq_image_imag), dim=-1)

        return freq_image.flatten(), label
