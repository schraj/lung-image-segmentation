import os
import torch
from torchinfo import summary
import numpy as np
from glob import glob
import src.data.constants as c
from src.data.prepare_data import DataPreparer
from src.data.prepare_datasets import DatasetPreparer
from src.model.unet import UNet
from src.model.losses import DiceLoss, dice_loss
from src.model_api.trainer import Trainer
from src.model_api.inferencer import Inferencer
from src.model.lifecycle import ModelLifecycle
from src.data.data_utils import get_test_inputs_and_targets

class Processor:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = UNet(
            in_channels=3,
            out_channels=2,
            n_blocks=4,
            start_filters=32,
            activation="relu",
            normalization="batch",
            conv_mode="same",
            dim=2,
        ).to(self.device)
    def summarize(self):
        # TODO - get size from input params
        summary(self.model, input_size=(2, 3, 512, 512))

    def run_training(self, update_data = True):
        if (update_data): 
            data_preparer = DataPreparer()
            data_preparer.prepare_data()

        # Check number of image files
        train_files = glob(os.path.join(c.SEGMENTATION_IMAGE_DIR, "*.png"))
        test_files = glob(os.path.join(c.SEGMENTATION_TEST_DIR, "*.png"))
        mask_files = glob(os.path.join(c.SEGMENTATION_MASK_DIR, "*.png"))
        print(len(train_files), len(test_files), len(mask_files))

        dataset_preparer = DatasetPreparer()
        dataset_preparer.prepare_training_dataloaders()

        # criterion = torch.nn.CrossEntropyLoss()
        criterion = DiceLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        trainer = Trainer(
            model=self.model,
            device=self.device,
            criterion=criterion,
            optimizer=optimizer,
            training_dataloader=dataset_preparer.dataloader_training,
            validation_dataloader=dataset_preparer.dataloader_validation,
            lr_scheduler=None,
            epochs=50,
            epoch=0,
        )

        training_losses, validation_losses, lr_rates = trainer.run_trainer()

        model_lifecycle = ModelLifecycle(self.model)
        model_lifecycle.save_model()

        print('training_losses', training_losses)
        print('validation_losses', validation_losses)

    def run_test_phase(self):
        inputs, targets = get_test_inputs_and_targets()
        dataset_preparer = DatasetPreparer()
        dataloader = dataset_preparer.prepare_test_dataloader(inputs, targets)

        inferencer = Inferencer(self.model, self.device)
        test_losses = []
        for input, target in dataloader:
            result = inferencer.predict(input)    
            test_losses.append(dice_loss(result, target))
        test_loss = np.mean(np.array(test_losses))
        print('test_loss', test_loss)