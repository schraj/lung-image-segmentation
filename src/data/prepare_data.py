import os
import numpy as np
from glob import glob
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from torchvision.transforms import v2
import src.data.constants as c

def format_image_tensor(image):
  return image.unsqueeze(0).permute(0, 1, 2, 3)/255

class DataPreparer():
    def __init__(self):
        pass
    
    def prepare_data(self):
       self.reset_data()
       print('data reset')
       self.prepare_montgomery_data()
       print('montgomery data prepared')
       self.prepare_shenzhen_data()
       print('shenzhen data prepared')

    def reset_data(self):
        for dir in [c.SEGMENTATION_DIR, c.SEGMENTATION_TRAIN_DIR, c.SEGMENTATION_TEST_DIR, c.SEGMENTATION_IMAGE_DIR, c.SEGMENTATION_MASK_DIR]:
            if os.path.exists(dir):
                for file in glob(os.path.join(dir, "*")):
                    os.remove(file)
            else:
                os.mkdir(dir)

    def prepare_montgomery_data(self):
      montgomery_left_mask_dir = glob(os.path.join(c.MONTGOMERY_LEFT_MASK_DIR, '*.png'))
      montgomery_test = montgomery_left_mask_dir[0:2]
      montgomery_train= montgomery_left_mask_dir[2:5]

      for left_image_file in montgomery_left_mask_dir:
          base_file = os.path.basename(left_image_file)
          image_file = os.path.join(c.MONTGOMERY_IMAGE_DIR, base_file)
          right_image_file = os.path.join(c.MONTGOMERY_RIGHT_MASK_DIR, base_file)

          image = read_image(image_file)
          left_mask = read_image(left_image_file, ImageReadMode.GRAY)
          right_mask = read_image(right_image_file, ImageReadMode.GRAY)
          
          image = v2.Resize(size=[512, 512])(image)
          left_mask = v2.Resize(size=[512,512])(left_mask)
          right_mask = v2.Resize(size=[512,512])(right_mask)
          
          mask = np.maximum(left_mask, right_mask)

          image = format_image_tensor(image)
          mask = format_image_tensor(mask)

          if (left_image_file in montgomery_train):
              save_image(image,os.path.join(c.SEGMENTATION_IMAGE_DIR, base_file))
              save_image(mask, os.path.join(c.SEGMENTATION_MASK_DIR, base_file))
          else:
              filename, fileext = os.path.splitext(base_file)
              save_image(image,os.path.join(c.SEGMENTATION_TEST_DIR, base_file))
              save_image(mask, os.path.join(c.SEGMENTATION_TEST_DIR, "%s_mask%s" % (filename, fileext)))
        
    def prepare_shenzhen_data(self):
      shenzhen_mask_dir = glob(os.path.join(c.SHENZHEN_MASK_DIR, '*.png'))
      shenzhen_test = shenzhen_mask_dir[0:5]
      shenzhen_train= shenzhen_mask_dir[5:20]

      for mask_file in shenzhen_mask_dir[0:20]:
          base_file = os.path.basename(mask_file).replace("_mask", "")
          image_file = os.path.join(c.SHENZHEN_IMAGE_DIR, base_file)

          image = read_image(image_file)
          mask = read_image(mask_file, ImageReadMode.GRAY)
              
          image = v2.Resize([512, 512])(image)
          mask = v2.Resize([512, 512])(mask)
          
          image = format_image_tensor(image)
          mask = format_image_tensor(mask)

          if (mask_file in shenzhen_train):
              save_image(image,os.path.join(c.SEGMENTATION_IMAGE_DIR, base_file))
              save_image(mask, os.path.join(c.SEGMENTATION_MASK_DIR, base_file))
          else:
              filename, fileext = os.path.splitext(base_file)

              save_image(image,os.path.join(c.SEGMENTATION_TEST_DIR, base_file))
              save_image(mask,os.path.join(c.SEGMENTATION_TEST_DIR, "%s_mask%s" % (filename, fileext)))