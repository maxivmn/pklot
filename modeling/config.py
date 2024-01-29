import logging
import parsenvy

logger = logging.getLogger(__name__)

# mlflow
# sometimes when saving links in text.. there is a new line .. strip removes that
try:
    TRACKING_URI = open(".mlflow_uri").read().strip()
except:
    TRACKING_URI = parsenvy.str("MLFLOW_URI")

EXPERIMENT_NAME = "0-template-ds-modeling"

# specify path to dataset
DATASET_PATH_RAW = "../data/PKLot/PKLotSegmented/PUC/Sunny/"
DATASET_PATH = "../data/PKLot/PKLotSegmented_rearranged/PUC/Sunny/"
# specify the paths to our training and validation set 
TRAIN = "train"
TEST = "test"
VAL = "val"
# set the input height and width
INPUT_HEIGHT = 128
INPUT_WIDTH = 128
# set the batch size and validation data split
BATCH_SIZE = 32
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2