import numpy as np
import pytorch_lightning as pl
import torch
from einops import repeat
from jsonargparse import ActionConfigFile, ArgumentParser
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from freq_ar.dataset import FrequencyMNIST, unsort_by_frequency
from freq_ar.model import FrequencyARModel
from freq_ar.visualize import visualize_frequency_image


class FrequencyARTrainer(pl.LightningModule):
    def __init__(
        self, input_dim, embed_dim, num_heads, num_layers, patchify, lr, compile_model
    ):
        super().__init__()
        model = FrequencyARModel(input_dim, embed_dim, num_heads, num_layers, patchify)
        self.model = (
            torch.compile(model, fullgraph=True) if compile_model else model
        )  # Conditionally compile the model
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr

    def forward(self, x, label):
        # convert label from B to B 1 2 (repeated across last dim)
        label_tensor = repeat(label.to(dtype=x.dtype) / 9, "B -> B 1 2")
        assert x.shape[1:] == (28 * 15, 2)

        x = torch.cat([label_tensor, x], dim=1)  # Concatenate label with input
        output = self.model(x)
        assert output.shape[1:] == (28 * 15 + 1, 2), output.shape
        return output[:, :-1]

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x, y)  # Pass label `y` to the forward method
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def freq_to_time(complex_image: torch.Tensor) -> torch.Tensor:
    # Apply expm1 to complex freq_image's magnitude while keeping its angle
    complex_image = torch.expm1(complex_image.abs()) * torch.exp(
        1j * complex_image.angle()
    )

    time_image = torch.fft.irfft2(complex_image).real
    return time_image


def split_to_complex(freq_image: torch.Tensor) -> torch.Tensor:
    freq_image = freq_image.float().view(28, 15, 2)
    freq_image_complex = torch.complex(freq_image[..., 0], freq_image[..., 1])

    # Undo the sorting by frequency
    freq_image_complex = unsort_by_frequency(freq_image_complex)

    return freq_image_complex


class ImageLoggingCallback(Callback):
    def __init__(self, log_every_n_steps):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps == 0:
            trainer.model.eval()

            x, y = batch  # Extract a sample from the batch
            with torch.no_grad():
                freq_image = pl_module(
                    x[0].unsqueeze(0), y[0].unsqueeze(0)
                )  # Process the sample through the model

            complex_image = split_to_complex(freq_image)
            time_image = freq_to_time(complex_image)
            freq_image_vis = visualize_frequency_image(
                complex_image.abs().cpu().numpy()
            )
            time_image_vis = visualize_frequency_image(time_image.cpu().numpy())

            input_complex_image = split_to_complex(x[0])
            input_time_image = freq_to_time(input_complex_image)
            input_image_vis = visualize_frequency_image(
                input_complex_image.abs().cpu().numpy()
            )
            input_time_image_vis = visualize_frequency_image(
                input_time_image.cpu().numpy()
            )

            # Log to Wandb
            trainer.logger.experiment.log(
                {
                    "frequency_image": wandb.Image(
                        freq_image_vis, caption=f"Label: {y[0].item()}"
                    ),
                    "time_image": wandb.Image(
                        time_image_vis, caption=f"Label: {y[0].item()}"
                    ),
                    "input_image": wandb.Image(
                        input_image_vis, caption=f"Label: {y[0].item()}"
                    ),
                    "input_time_image": wandb.Image(
                        input_time_image_vis, caption=f"Label: {y[0].item()}"
                    ),
                    "ground_truth": y[0].item(),  # Log ground truth as a scalar value
                }
            )

            trainer.model.train()


class AutoRegressiveSamplingCallback(Callback):
    def __init__(
        self, num_samples, patchify, log_every_n_steps=250
    ):  # Default to 250 steps
        self.num_samples = num_samples
        self.patchify = patchify
        self.log_every_n_steps = log_every_n_steps

    @staticmethod
    def create_video(images):
        """Create a video from a list of images."""
        video = np.array([np.array(image) for image in images])
        # THWC -> TCHW
        video = np.transpose(video, (0, 3, 1, 2))
        return video

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps == 0:
            trainer.model.eval()

            with torch.no_grad():
                for sample_index in range(self.num_samples):
                    random_label = torch.randint(0, 10, (1,), device=pl_module.device)
                    generated_sequence = torch.zeros(
                        1, 28 * 15, 2, device=pl_module.device
                    )  # Initialize an empty sequence

                    freq_images = []
                    time_images = []

                    for i in range(
                        0, 28 * 15, self.patchify
                    ):  # Generate pixel by pixel
                        output = pl_module(generated_sequence, random_label)
                        next_pixels = output[
                            :, i : i + self.patchify
                        ]  # Take the next predicted pixels
                        generated_sequence[:, i : i + self.patchify] = (
                            next_pixels  # Update the sequence
                        )

                        freq_image = generated_sequence  # The full generated sequence
                        complex_image = split_to_complex(freq_image)
                        time_image = freq_to_time(complex_image)

                        freq_images.append(
                            visualize_frequency_image(complex_image.abs().cpu().numpy())
                        )
                        time_images.append(
                            visualize_frequency_image(time_image.cpu().numpy())
                        )

                    # Create videos, TCHW
                    freq_video = self.create_video(freq_images)
                    time_video = self.create_video(time_images)
                    freq_and_time_video = np.concatenate(
                        [freq_video, time_video], axis=-1
                    )

                    # Log videos to Wandb
                    trainer.logger.experiment.log(
                        {
                            f"ar_video_{sample_index}": wandb.Video(
                                freq_and_time_video,
                                fps=5,
                                format="mp4",
                                caption=f"Label: {random_label.item()}",
                            ),
                        }
                    )

            trainer.model.train()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--input_dim", type=int, default=2, help="Input dimension")
    parser.add_argument(
        "--embed_dim", type=int, default=768, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=6, help="Number of attention heads"
    )
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="Accelerator type"
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument(
        "--project_name", type=str, default="freq-ar", help="Wandb project name"
    )
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--log_every_n_steps", type=int, default=250, help="Log images every n steps"
    )
    parser.add_argument(
        "--image_dtype",
        type=str,
        default="float32",
        help="Data type for images (e.g., float32, bfloat16)",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Enable model compilation for optimization",
    )
    parser.add_argument(
        "--patchify",
        type=int,
        default=15,
        help="Patchify the input image for the transformer",
    )

    args = parser.parse_args()

    image_dtype = getattr(torch, args.image_dtype)

    train_dataset = FrequencyMNIST(train=True, image_dtype=image_dtype)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    model = FrequencyARTrainer(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        patchify=args.patchify,
        lr=args.lr,
        compile_model=args.compile_model,
    ).to(dtype=image_dtype)

    wandb_logger = WandbLogger(project=args.project_name, name=args.run_name)

    image_logging_callback = ImageLoggingCallback(
        log_every_n_steps=args.log_every_n_steps
    )

    autoregressive_sampling_callback = AutoRegressiveSamplingCallback(
        num_samples=1, patchify=args.patchify, log_every_n_steps=args.log_every_n_steps
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=wandb_logger,
        callbacks=[
            image_logging_callback,
            autoregressive_sampling_callback,
        ],
    )
    trainer.fit(model, train_loader)
