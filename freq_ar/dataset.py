from functools import lru_cache

import torch
from einops import rearrange
from torchvision import datasets, transforms


@lru_cache(maxsize=1)
def get_frequency_sorting_index():
    freq_x = torch.fft.fftfreq(28)
    freq_y = torch.fft.rfftfreq(28)
    freqs = torch.meshgrid(freq_x, freq_y, indexing="ij")
    coord_length_squared = freqs[0] ** 2 + freqs[1] ** 2
    return torch.argsort(coord_length_squared.view(-1))


def sort_by_frequency(freq_image):
    # Sort the image by frequency from low to high.
    sort_idx = get_frequency_sorting_index()
    return freq_image.view(-1)[sort_idx].view(28, 15)


def unsort_by_frequency(sorted_freq_image):
    """Undoes sort_by_frequency"""
    # Get the sorting indices used in sort_by_frequency.
    sort_idx = get_frequency_sorting_index()
    # Compute the inverse permutation.
    unsort_idx = torch.argsort(sort_idx)
    # Reorder the sorted image back to its original order.
    return sorted_freq_image.view(-1)[unsort_idx].view(28, 15)


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
        freq_image = torch.log1p(torch.abs(freq_image)) * torch.exp(
            1j * torch.angle(freq_image)
        )

        # Sort the image by frequency from low to high.
        freq_image = sort_by_frequency(freq_image)

        # Split the complex tensor into real and imaginary parts
        freq_image_real = freq_image.real
        freq_image_imag = freq_image.imag
        freq_image = torch.stack((freq_image_real, freq_image_imag), dim=-1)
        assert freq_image.shape == (28, 15, 2), freq_image.shape

        freq_image = rearrange(
            freq_image.to(dtype=self.image_dtype), "h w c -> (h w) c"
        )

        return freq_image, label
