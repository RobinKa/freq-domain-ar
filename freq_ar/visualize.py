import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def visualize_frequency_image(freq_image):
    """
    Visualizes a frequency image and returns it as a PIL Image for logging.
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(freq_image.reshape(28, 28), cmap="gray")
    plt.colorbar()
    plt.axis("off")
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    
    # Convert buffer to PIL Image
    return Image.open(buf)
