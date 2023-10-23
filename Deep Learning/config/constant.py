import os

# cuda: boolean
cuda = True

# data
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

FIRE_RAW_DATASET = os.path.join("data/fire_smoke", "*")
NO_FIRE_RAW_DATASET = os.path.join("data/no_fire_smoke", "*")