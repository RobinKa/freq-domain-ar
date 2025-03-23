import matplotlib.pyplot as plt
import torch
from freq_ar.visualize import visualize_frequency_image

def visualize_frequency_image(freq_image):
    plt.imshow(freq_image.reshape(28, 28), cmap="gray")
    plt.colorbar()
    plt.show()

def evaluate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x_hat = model(x)
            visualize_frequency_image(x_hat[0].cpu().numpy())
            break
