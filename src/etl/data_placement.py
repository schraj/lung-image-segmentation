import os
import boto3
import kaggle
from glob import glob
import src.data.constants as c
from src.data.prepare_data import DataPreparer
from src.etl.file_loader import FileLoader

class DataPlacement:
  def __init__(self):
    session = boto3.Session()
    self.s3resource = session.resource('s3', region_name='us-east-1')
    
  def create_bucket(self, bucket_name):
    self.s3resource.create_bucket(Bucket=bucket_name)

  def upload_files(self, bucket_name, file_name):
    file_loader = FileLoader(bucket_name, self.s3resource)
    file_loader.upload_files(file_name)

  def download_files(self, bucket_name, dir_name):
    self.prepare_directories()
    file_loader = FileLoader(bucket_name, self.s3resource)
    file_loader.download_files(dir_name)

  def prepare_directories(self):
    for dir in [
      c.INPUT_DIR,
      c.SEGMENTATION_SOURCE_DIR,
      c.SHENZHEN_TRAIN_INTER_DIR,
      c.SHENZHEN_TRAIN_DIR,
      c.SHENZHEN_IMAGE_DIR,
      c.SHENZHEN_TOP_MASK_DIR,
      c.SHENZHEN_MASK_INTER_DIR,
      c.SHENZHEN_MASK_DIR,
      c.MONTGOMERY_TRAIN_INTER_DIR,
      c.MONTGOMERY_TRAIN_DIR,
      c.MONTGOMERY_IMAGE_DIR,
      c.MONTGOMERY_MASK_DIR,
      c.MONTGOMERY_LEFT_MASK_DIR,
      c.MONTGOMERY_RIGHT_MASK_DIR
    ]:
        if os.path.exists(dir):
            for file in glob(os.path.join(dir, "*")):
                os.remove(file)
        else:
            os.mkdir(dir)

  # TODO: deal with cert issue
  #   For now, just downloading the dataset manually
  def get_kaggle_dataset(self):
    pass