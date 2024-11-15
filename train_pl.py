import os
import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, loggers
from dataset import VCTKDEMANDDataset  # Your custom dataset
from gtcrn import GTCRN  # Your model
from loss import HybridLoss  # Your custom loss function

# Set Tensor Core precision (for GPUs that support it)
torch.set_float32_matmul_precision('medium')

# Define the LightningModule for training
class SpeechEnhancementModel(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super(SpeechEnhancementModel, self).__init__()
        self.model = GTCRN()
        self.loss_func = HybridLoss()
        self.learning_rate = learning_rate

    def forward(self, noisy_spectrogram):
        return self.model(noisy_spectrogram)

    def training_step(self, batch, batch_idx):
        noisy_waveform, clean_waveform = batch

        # Convert waveform to spectrogram
        noisy_spectrogram = torch.stft(
            noisy_waveform.squeeze(1), n_fft=512, hop_length=256, return_complex=False
        )
        clean_spectrogram = torch.stft(
            clean_waveform.squeeze(1), n_fft=512, hop_length=256, return_complex=False
        )

        # Forward pass and calculate loss
        outputs = self(noisy_spectrogram)
        loss = self.loss_func(outputs, clean_spectrogram)
        squared_loss = loss ** 2
        
        # Log loss for monitoring
        self.log('train_loss_squared', squared_loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Custom collate function for padding
def collate_fn(batch):
    max_length = max([x[0].shape[1] for x in batch])
    padded_noisy = torch.zeros(len(batch), 1, max_length)
    padded_clean = torch.zeros(len(batch), 1, max_length)

    for i, (noisy, clean) in enumerate(batch):
        length = noisy.shape[1]
        padded_noisy[i, 0, :length] = noisy
        padded_clean[i, 0, :length] = clean

    return padded_noisy, padded_clean

# Set up data
train_dataset = VCTKDEMANDDataset(root_dir='VCTK-DEMAND')
train_loader = DataLoader(
    train_dataset, batch_size=50, shuffle=True, collate_fn=collate_fn, num_workers=4
)

# Directory to save checkpoints
checkpoint_dir = 'pl_checkpoints'

# EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='train_loss_squared',  # Metric to monitor
    min_delta=0.00,        # Minimum change to qualify as improvement
    patience=10,            # Number of epochs with no improvement
    verbose=True,
    mode='min',
    stopping_threshold=0.0005 # Set the desired loss threshold
)

# Custom logger setup
logger = loggers.TensorBoardLogger(
    save_dir=os.path.join("exp", "logs"),    # Save logs in the logs directory
    name="gtcrn_28spk",  # Set custom name for version folder
    version="3a"
)

# Trainer setup with custom early stopping
trainer = pl.Trainer(
    max_epochs=200,
    accelerator="gpu",
    devices=2,  # Use both GPUs
    logger=logger,
    callbacks=[early_stopping_callback],
    default_root_dir="logs"
)

# Initialize the model and start training
model = SpeechEnhancementModel(learning_rate=0.001)
trainer.fit(model, train_loader)
