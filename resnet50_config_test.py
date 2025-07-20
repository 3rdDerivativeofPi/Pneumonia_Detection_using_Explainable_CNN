hyperparameters = {
    "num_epochs": 10,
    "batch_size": 64,
    "image_size": 224,
    "lr": 0.001,
    "model": "resnet50",
    "scheduler": "CosineAnnealingLR10",
    "balance": True,
    "early_stopping_patience": float("inf"),
    "use_best_checkpoint": True
}