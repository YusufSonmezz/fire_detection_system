import os

# cuda: boolean
cuda = False

# data
CLASSES = {
    "non-fire": 0,
    "fire": 1
}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

FIRE_RAW_DATASET = os.path.join("data/fire_smoke", "*")
NO_FIRE_RAW_DATASET = os.path.join("data/no_fire_smoke", "*")
NO_FIRE_RAW_DATASET_AUG = os.path.join("data/non_fire_augmented", "*")
shuffle = True

# Train config
EPOCH = 1
LEARNING_RATE = 0.00001

valid_size = 0.15
test_size = 0.15