import pytest
import torch

from freq_ar.dataset import (
    FrequencyCIFAR10,
    FrequencyMNIST,
    sort_by_frequency,
    unsort_by_frequency,
)


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_sort_unsort_frequency(device: str):
    # Create a random frequency image on the specified device
    freq_image = torch.rand(28, 15, dtype=torch.complex64, device=device)

    # Sort and then unsort the frequency image
    sorted_freq_image = sort_by_frequency(freq_image)
    unsorted_freq_image = unsort_by_frequency(sorted_freq_image)

    # Check if the original and unsorted images are the same
    assert torch.allclose(freq_image, unsorted_freq_image), (
        "Unsorted image does not match the original"
    )


@pytest.mark.parametrize("dataset_class", [FrequencyMNIST, FrequencyCIFAR10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_datasets_are_iterable(dataset_class, dtype: torch.dtype):
    dataset = dataset_class(train=True, image_dtype=dtype)
    for idx, (freq_image, label) in enumerate(dataset):
        assert freq_image is not None, "Frequency image should not be None"
        assert label is not None, "Label should not be None"

        assert freq_image.ndim == 3  # HW(2C)
        assert freq_image.dtype == dtype
        assert freq_image.device.type == "cpu"

        if idx > 10:
            break
