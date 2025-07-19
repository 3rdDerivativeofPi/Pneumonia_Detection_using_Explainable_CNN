from config import hyperparameters

import torch
import torch.nn as nn
from torchvision.models import resnet50, densenet121, vgg16
from sklearn.metrics import f1_score
import timm
import pytorch_lightning as pl

class PneumoniaModel(pl.LightningModule):
    def __init__(self, hyperparams=hyperparameters):
        super().__init__()
        self.hyperparams = hyperparams
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.test_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        metrics = {"val_loss": loss, "val_acc": acc}
        self.log_dict(metrics, on_epoch=True, on_step=True, prog_bar=True)
        return metrics
    
    def on_test_epoch_start(self):
        self.test_outputs = []
        
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        preds = torch.argmax(outputs, dim=1)
        self.test_outputs.append({"test_loss": loss, "test_acc": acc, "preds": preds, "labels": labels})
        return {"test_loss": loss, "test_acc": acc, "preds": preds, "labels": labels}
    
    def on_test_epoch_end(self):
        test_loss_mean = torch.stack([x["test_loss"] for x in self.test_outputs]).mean()
        test_acc_mean = torch.stack([x["test_acc"] for x in self.test_outputs]).mean()
        
        self.test_predicted_labels = torch.cat([x["preds"] for x in self.test_outputs], dim=0).cpu().numpy()
        self.test_true_labels = torch.cat([x["labels"] for x in self.test_outputs], dim=0).cpu().numpy()
        
        f1 = f1_score(self.test_true_labels, self.test_predicted_labels)
        
        self.test_f1 = f1
        self.test_acc = test_acc_mean.cpu().numpy()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparams["lr"])
        scheduler_dic = self._configure_scheduler(optimizer)

        if (scheduler_dic["scheduler"]):
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_dic
            }            
        else:
            return optimizer

    def _configure_scheduler(self, optimizer):
        scheduler_name = self.hyperparams["scheduler"]
        lr = self.hyperparams["lr"]
        if (scheduler_name==""):
            return {
                "scheduler": None
            }
        if (scheduler_name=="CosineAnnealingLR10"):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hyperparams["num_epochs"], eta_min=lr*0.1) # If scheduler interval is "step", use T_max = num_epochs * len(train_loader)
            return {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        print (f"Error. Unknown scheduler name '{scheduler_name}'")
        return None
    
    def _create_model(self):
        model_name = self.hyperparams["model"]
        num_classes = 2

        if model_name == "resnet50":
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        elif model_name == "densenet121":
            model = densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model

        elif model_name == "efficientnet_b1":
            model = timm.create_model("efficientnet_b1", pretrained=True, num_classes=num_classes)
            return model

        elif model_name == "vgg16":
            model = vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            return model

        else:
            raise ValueError(f"Unknown model name '{model_name}' in config.")
            