import io

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from PIL import Image


def render_complex_image(complex_image: np.ndarray) -> Image.Image:
    """
    Visualizes a complex image and returns it as a PIL Image.

    freq_image: complex CHW
    """
    assert complex_image.ndim in {2, 3}, (
        f"freq_image must be 2D or 3D but was {complex_image.ndim}: {complex_image.shape}"
    )
    complex_image = rearrange(complex_image, "c h w -> h w c").squeeze(-1)

    plt.figure(figsize=(4, 4))
    plt.imshow(complex_image, cmap="gray" if complex_image.ndim == 2 else "jet")
    plt.colorbar()
    plt.axis("off")

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # Convert buffer to PIL Image
    return Image.open(buf)
