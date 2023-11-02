import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import albumentations.pytorch
import os
import glob
import tqdm

from config import constant

non_fire_dataset = glob.glob(constant.NO_FIRE_RAW_DATASET)

for image_path in tqdm.tqdm(non_fire_dataset):
    image = cv2.imread(image_path)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformImage = A.Compose(
        [
            A.HorizontalFlip(p = 1),
            A.ColorJitter(brightness=0.5, p = 0.75),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.4, p = 0.6),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=4, p = 0.8)
        ]
    )

    augmented_image = transformImage(image = imageRGB)['image']
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

    os.makedirs("data/non_fire_augmented", exist_ok=True)
    file_name = os.path.basename(image_path) + "_augmented.jpg"
    augmented_image_path = os.path.join("data/non_fire_augmented", file_name)
    cv2.imwrite(augmented_image_path, augmented_image) 