import torch
from torchvision import datasets, transforms
import numpy as np

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
        freq_image = torch.fft.fft2(image.squeeze()).abs()
        freq_image = torch.log1p(freq_image)  # Log scale for stability
        return freq_image.flatten(), label
