import os

# define the path to the original input dataset
ORIG_INPUT_DATASET = "datasets/orig"

# define the base path to the output dataset after computing training and testing splits
BASE_PATH = "datasets/idc"

# define the path to the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define amount of data used for training
TRAIN_SPLIT = 0.8

# define amount of data used for validation
VAL_SPLIT = 0.1