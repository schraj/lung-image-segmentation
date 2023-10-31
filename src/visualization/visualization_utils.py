import torch
from torchvision.utils import draw_segmentation_masks

def add_mask(image, mask_image):
    mask_bool = (mask_image!=0)

    ret = draw_segmentation_masks(image, mask_bool, 0.7)

    return ret