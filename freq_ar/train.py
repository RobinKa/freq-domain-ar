import pytorch_lightning as pl
import torch
from jsonargparse import ActionConfigFile, ArgumentParser
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from freq_ar.dataset import FrequencyMNIST
from freq_ar.model import FrequencyARModel
from freq_ar.visualize import visualize_frequency_image


class FrequencyARTrainer(pl.LightningModule):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, lr):
        super().__init__()
        self.model = FrequencyARModel(input_dim, embed_dim, num_heads, num_layers)
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def freq_to_time(complex_image: torch.Tensor) -> torch.Tensor:
    time_image = torch.fft.ifft2(complex_image).real
    time_image = (time_image - time_image.min()) / (time_image.max() - time_image.min())
    return time_image


def split_to_complex(freq_image: torch.Tensor) -> torch.Tensor:
    freq_image = torch.expm1(freq_image)
    freq_image = freq_image.view(28, 28, 2)
    freq_image_complex = torch.complex(freq_image[..., 0], freq_image[..., 1])
    return freq_image_complex


class ImageLoggingCallback(Callback):
    def __init__(self, log_every_n_steps):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.log_every_n_steps == 0:
            x, y = batch  # Extract a sample from the batch
            with torch.no_grad():
                freq_image = pl_module(
                    x[0].unsqueeze(0)
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
                    "frequency_image": wandb.Image(freq_image_vis),
                    "time_image": wandb.Image(time_image_vis),
                    "input_image": wandb.Image(input_image_vis),
                    "input_time_image": wandb.Image(input_time_image_vis),
                    "ground_truth": y[0].item(),  # Log ground truth as a scalar value
                }
            )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument(
        "--input_dim", type=int, default=2 * 784, help="Input dimension"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="Accelerator type"
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument(
        "--project_name", type=str, default="freq-ar", help="Wandb project name"
    )
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--log_every_n_steps", type=int, default=1000, help="Log images every n steps"
    )

    args = parser.parse_args()

    train_dataset = FrequencyMNIST(train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = FrequencyARTrainer(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
    )

    wandb_logger = WandbLogger(
        project=args.project_name, name=args.run_name
    )  # Initialize WandbLogger

    image_logging_callback = ImageLoggingCallback(
        log_every_n_steps=args.log_every_n_steps
    )  # Initialize callback

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=wandb_logger,  # Pass WandbLogger to the trainer
        callbacks=[image_logging_callback],  # Add callback to trainer
    )
    trainer.fit(model, train_loader)
