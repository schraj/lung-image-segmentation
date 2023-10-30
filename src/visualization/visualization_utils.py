from torchvision.utils import draw_segmentation_masks

def add_mask(image, mask_image):
    mask_image_gray = v2.Grayscale()(mask_image)

    mask = mask_image_gray.squeeze()

    mask_bool = (mask!=0)

    ret = draw_segmentation_masks(image, mask_bool, 0.7)

    return ret