import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from freq_ar.dataset import FrequencyMNIST  # Updated import path
from freq_ar.model import FrequencyARModel
from freq_ar.visualize import visualize_frequency_image
from jsonargparse import ArgumentParser, ActionConfigFile
from pytorch_lightning.loggers import WandbLogger  # Added import
from pytorch_lightning.callbacks import Callback  # Added import
import wandb

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

class ImageLoggingCallback(Callback):  # Updated callback class
    def __init__(self, log_every_n_steps):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # Updated method
        if trainer.global_step % self.log_every_n_steps == 0:
            x, _ = batch  # Extract a sample from the batch
            with torch.no_grad():
                freq_image = pl_module(x[0].unsqueeze(0))  # Process the sample through the model
            sample_image = visualize_frequency_image(freq_image.cpu().numpy())  # Generate visualization
            trainer.logger.experiment.log({"frequency_image": wandb.Image(sample_image)})  # Log to Wandb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--input_dim", type=int, default=784, help="Input dimension")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--project_name", type=str, default="freq-ar", help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--log_every_n_steps", type=int, default=1000, help="Log images every n steps")  # Added argument

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

    wandb_logger = WandbLogger(project=args.project_name, name=args.run_name)  # Initialize WandbLogger

    image_logging_callback = ImageLoggingCallback(log_every_n_steps=args.log_every_n_steps)  # Initialize callback

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=wandb_logger,  # Pass WandbLogger to the trainer
        callbacks=[image_logging_callback],  # Add callback to trainer
    )
    trainer.fit(model, train_loader)
