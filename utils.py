import config

import os
import pprint
import time
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl

class InfoPrinterCallback(pl.Callback):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"CPU cores: {os.cpu_count()}, Device: {device}, GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"CPU cores: {os.cpu_count()}, Device: {device}, No GPU available.")
            
        # Print hyperparameters for records
        print("Hyperparameters: ")
        pprint.pprint(hyperparameters, indent=4)
        
    def setup(self, trainer, pl_module, stage):
        self.start_time = time.time()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip sanity check
        if trainer.sanity_checking:
            return
        
        epoch = trainer.current_epoch
        total_epochs = trainer.max_epochs
        
        elapsed_time = time.time() - self.start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        avg_time_per_epoch_min, avg_time_per_epoch_sec = divmod(avg_time_per_epoch, 60)
        
        remaning_epochs = total_epochs - epoch - 1
        remaining_time = remaning_epochs * avg_time_per_epoch
        remaining_time_min, remaining_time_sec = divmod(remaining_time, 60)
        
        print(f"Epoch {epoch + 1}/{total_epochs}: ", end=' ')
        
        if "val_loss" in trainer.callback_metrics:
            validation_loss = trainer.callback_metrics["val_loss"].cpu().numpy()
            print(f"Validation Loss: {validation_loss:.4f}, ", end=' ')
        else:
            print("Validation Loss N/A", end=' ')
            
        if "train_loss_epoch" in trainer.logged_metrics:
            train_loss = trainer.logged_metrics["train_loss_epoch"].cpu().numpy()
            print(f", TrainL Loss = {train_loss:.4f}", end='')
        else:
            print(", Train Loss N/A", end='')
            
        print(f", Epoch Time: {avg_time_per_epoch_min:.0f}m {avg_time_per_epoch_sec:02.0f}s, Remaining Time: {remaining_time_min:.0f}m {remaining_time_sec:02.0f}s")
        
        def plot_losses(self):
            plt.plot(self.training_losses, label="Training Loss")
            plt.plot(self.validation_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Training and Validation Losses")
            plt.show()

class PlotTestConfusionMatrixCallback(pl.Callback):
    pass

class PlotTrainingLogsCallback(pl.Callback):
    pass

