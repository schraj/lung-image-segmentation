import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import src.data.constants as c
from src.data.data_utils import get_file_list
from src.utils.image_utils import permute
from visualization.visualization_utils import add_mask

def show_montgomery_images():  
  base_file = os.path.basename(get_file_list(c.MONTGOMERY_IMAGE_DIR)[0])

  image_file = os.path.join(c.SEGMENTATION_IMAGE_DIR, base_file)
  mask_image_file = os.path.join(c.SEGMENTATION_MASK_DIR, base_file)

  image = imread(image_file)
  mask_image = imread(mask_image_file)
  merged_image = add_mask(image, mask_image)
                            
  fig, axs = plt.subplots(2, 3, figsize=(15, 8))

  axs[0, 0].set_title("X-Ray")
  axs[0, 0].imshow(permute(image))

  axs[0, 1].set_title("Mask")
  axs[0, 1].imshow(permute(mask_image))

  axs[0, 2].set_title("Merged")
  axs[0, 2].imshow(permute(merged_image))

  base_file = os.path.basename(get_file_list(c.MONTGOMERY_IMAGE_DIR)[0])
  filename, fileext = os.path.splitext(base_file)
  image_file = os.path.join(c.SEGMENTATION_TEST_DIR, base_file)
  mask_image_file = os.path.join(c.SEGMENTATION_TEST_DIR, \
                                "%s_mask%s" % (filename, fileext))

  image = imread(image_file)
  mask_image = imread(mask_image_file)
  merged_image = add_mask(image, mask_image)

  axs[1, 0].set_title("X-Ray")
  axs[1, 0].imshow(permute(image))

  axs[1, 1].set_title("Mask")
  axs[1, 1].imshow(permute(mask_image))

  axs[1, 2].set_title("Merged")
  axs[1, 2].imshow(permute(merged_image))

def show_sample_shenszhen_images():
  base_file = os.path.basename(get_file_list(c.SHENZHEN_IMAGE_DIR).replace("_mask", ""))

  image_file = os.path.join(c.SEGMENTATION_IMAGE_DIR, base_file)
  mask_image_file = os.path.join(c.SEGMENTATION_MASK_DIR, base_file)

  image = imread(image_file)
  mask_image = imread(mask_image_file)
  merged_image = add_mask(image, mask_image)
                            
  fig, axs = plt.subplots(2, 3, figsize=(15, 8))

  axs[0, 0].set_title("X-Ray")
  axs[0, 0].imshow(permute(image))

  axs[0, 1].set_title("Mask")
  axs[0, 1].imshow(permute(mask_image))

  axs[0, 2].set_title("Merged")
  axs[0, 2].imshow(permute(merged_image))

  base_file = os.path.basename(get_file_list(c.SHENZHEN_IMAGE_DIR).replace("_mask", ""))
  image_file = os.path.join(c.SEGMENTATION_TEST_DIR, base_file)
  filename, fileext = os.path.splitext(base_file)
  mask_image_file = os.path.join(c.SEGMENTATION_TEST_DIR, \
                                "%s_mask%s" % (filename, fileext))

  filename, fileext = os.path.splitext(base_file)
  image_file = os.path.join(c.SEGMENTATION_TEST_DIR, base_file)
  mask_image_file = os.path.join(c.SEGMENTATION_TEST_DIR, \
                                "%s_mask%s" % (filename, fileext))
  image = imread(image_file)
  mask_image = imread(mask_image_file)
  merged_image = add_mask(image, mask_image)

  axs[1, 0].set_title("X-Ray")
  axs[1, 0].imshow(permute(image))

  axs[1, 1].set_title("Mask")
  axs[1, 1].imshow(permute(mask_image))

  axs[1, 2].set_title("Merged")
  axs[1, 2].imshow(permute(merged_image))