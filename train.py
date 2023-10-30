# https://towardsdatascience.com/biomedical-image-segmentation-u-net-a787741837fa#:~:text=Dice%20coefficient,-A%20common%20metric&text=The%20calculation%20is%202%20*%20the,denotes%20perfect%20and%20complete%20overlap.

import os
import torch
from glob import glob
import src.data.constants as c
from src.data.prepare_data import DataPreparer
from src.data.prepare_datasets import DatasetPreparer
from src.model.unet import UNet
from src.model.losses import DiceLoss
from src.trainer import Trainer

data_preparer = DataPreparer()
data_preparer.prepare_data()

# Check number of image files
train_files = glob(os.path.join(c.SEGMENTATION_IMAGE_DIR, "*.png"))
test_files = glob(os.path.join(c.SEGMENTATION_TEST_DIR, "*.png"))
mask_files = glob(os.path.join(c.SEGMENTATION_MASK_DIR, "*.png"))
print(len(train_files), len(test_files), len(mask_files))

dataset_preparer = DatasetPreparer()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = UNet(
    in_channels=3,
    out_channels=2,
    n_blocks=4,
    start_filters=32,
    activation="relu",
    normalization="batch",
    conv_mode="same",
    dim=2,
).to(device)

# criterion = torch.nn.CrossEntropyLoss()
criterion = DiceLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

trainer = Trainer(
    model=model,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    training_dataloader=dataset_preparer.dataloader_training,
    validation_dataloader=dataset_preparer.dataloader_validation,
    lr_scheduler=None,
    epochs=4,
    epoch=0,
)

training_losses, validation_losses, lr_rates = trainer.run_trainer()
