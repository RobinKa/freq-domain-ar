from functools import lru_cache

import torch
from einops import rearrange
from torchvision import datasets, transforms


@lru_cache(maxsize=1)
def get_frequency_sorting_index(height, width):
    # Get frequencies of the indices.
    # We only have the width and height after doing RFFT, which throws away half of the redundant data.
    # w_rfft = w / 2 + 1 -> w = (w_rfft - 1) * 2
    width = (width - 1) * 2
    freq_x = torch.fft.fftfreq(width)
    freq_y = torch.fft.rfftfreq(height)

    # argsort the frequencies by their squared length.
    freqs = torch.meshgrid(freq_x, freq_y, indexing="ij")
    coord_length_squared = freqs[0] ** 2 + freqs[1] ** 2
    return torch.argsort(coord_length_squared.view(-1))


def sort_by_frequency(freq_image):
    # Sort the image by frequency from low to high.
    sort_idx = get_frequency_sorting_index(
        height=freq_image.shape[-2], width=freq_image.shape[-1]
    )
    return rearrange(freq_image, "... h w -> ... (h w)")[..., sort_idx].view(
        *freq_image.shape
    )


def unsort_by_frequency(sorted_freq_image):
    """Undoes sort_by_frequency"""
    # Get the sorting indices used in sort_by_frequency.
    sort_idx = get_frequency_sorting_index(
        height=sorted_freq_image.shape[-2], width=sorted_freq_image.shape[-1]
    )
    # Compute the inverse permutation.
    unsort_idx = torch.argsort(sort_idx)
    # Reorder the sorted image back to its original order.
    return rearrange(sorted_freq_image, "... h w -> ... (h w)")[..., unsort_idx].view(
        *sorted_freq_image.shape
    )


def freq_to_time(complex_image: torch.Tensor) -> torch.Tensor:
    # Apply expm1 to complex freq_image's magnitude while keeping its angle
    complex_image = torch.expm1(complex_image.abs()) * torch.exp(
        1j * complex_image.angle()
    )

    time_image = torch.fft.irfft2(complex_image).real
    return time_image


def split_to_complex(freq_image: torch.Tensor) -> torch.Tensor:
    freq_image = rearrange(freq_image.float(), "h w (d c) -> d c h w", d=2)

    freq_image_complex = torch.complex(freq_image[0], freq_image[1])

    # Undo the sorting by frequency
    freq_image_complex = unsort_by_frequency(freq_image_complex)

    return freq_image_complex


class FrequencyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.VisionDataset, image_dtype=torch.bfloat16):
        self.dataset = dataset
        self.image_dtype = image_dtype

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]  # CHW

        # Convert image to frequency domain using FFT
        freq_image = torch.fft.rfft2(image)

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

        # Make the color and complex dimensions into the channels dimension.
        freq_image = rearrange(
            freq_image.to(dtype=self.image_dtype), "c h w d -> h w (d c)"
        )

        return freq_image, label


class FrequencyMNIST(FrequencyDataset):
    def __init__(self, train: bool = True, image_dtype=torch.bfloat16):
        super().__init__(
            datasets.MNIST(
                root="data",
                train=train,
                download=True,
                transform=transforms.ToTensor(),
            ),
            image_dtype=image_dtype,
        )


class FrequencyCIFAR10(FrequencyDataset):
    def __init__(self, train: bool = True, image_dtype=torch.bfloat16):
        super().__init__(
            datasets.CIFAR10(
                root="data",
                train=train,
                download=True,
                transform=transforms.ToTensor(),
            ),
            image_dtype=image_dtype,
        )
