from typing import Callable

import numpy as np
import torch
from src.model.lifecycle import ModelLifecycle

class Inferencer:
    def __init__(
            self,
            model: torch.nn.Module,
            device: str,
            preprocess: Callable = None,
            postprocess: Callable = None
        ):
        self.model = model
        self.device = device
        self.preprocess = preprocess
        self.postprocess: postprocess
        self.model_lifecycle = ModelLifecycle(model)
        self.model_lifecycle.load_model()

    def predict(self,
        img: np.ndarray,
    ) -> np.ndarray:
        self.model.eval()
        # img = self.preprocess(img)  # preprocess image
        x = img.to(self.device)  # to torch, send to device
        with torch.no_grad():
            out = self.model(x)  # send through model/network

        out_softmax = torch.softmax(out, dim=1)  # perform softmax on outputs
        # result = self.postprocess(out_softmax)  # postprocess outputs
        result = out_softmax
        return result
