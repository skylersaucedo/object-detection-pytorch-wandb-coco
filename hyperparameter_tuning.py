"""
Use this to run a hyperparameter search on the RetinaNet object detector

"""

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from shared_utilites import CustomDataModule, LightningModel,PyTorchMLP
from watermark import watermark

if __name__ == "__main__":
    print(watermark(packages="torch, ligtning"))

    cli = LightningCLI(
        model_class = LightningModel,
        datamodule_class = CustomDataModule,
        run=False, #don't run upon instantiation
        save_config_callback=None,
        seed_everything_default=123,
        trainer_defaults={
            "max_epochs":10,
            "accelerator":"gpu",
            "callbacks": [ModelCheckpoint(monitor="val_map", mode="max")],
        },
    )

    pytorch_model = PyTorchMLP(num_features=100, num_classes=2)
    lightning_model = LightningModel(
        pytorch_model=pytorch_model, learning_rate=cli.model.learning_rate,
        hidden_units = 3  #add your args here
    )