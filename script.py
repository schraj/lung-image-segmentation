import pathlib
from src.processor import Processor
from src.etl.data_placement import DataPlacement
# processor = Processor()
# processor.summarize()
# processor.run_training(update_data=False)
# processor.run_test_phase()

TRAINING_DATA_DIR = 'input'
BUCKET_NAME = 'lung-image-data'
DOWNLOAD_DIR = 'download'
data_placement = DataPlacement()
# data_placement.create_bucket(BUCKET_NAME)
# data_placement.upload_files(BUCKET_NAME, pathlib.Path.cwd() / TRAINING_DATA_DIR)
data_placement.download_files(BUCKET_NAME, DOWNLOAD_DIR)