import glob
from config import constant
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
from datetime import datetime
import json

from src.logger import setup_logging, logging


def train_test_split():
    setup_logging(logging.INFO)
    # data_ingestion ile farklı kaynaklardan gelen datanın tek bir formatta kullanılması
    fire_dataset = glob.glob(constant.FIRE_RAW_DATASET)
    non_fire_dataset = glob.glob(constant.NO_FIRE_RAW_DATASET)
    aug_non_fire_dataset = glob.glob(constant.NO_FIRE_RAW_DATASET_AUG)

    logging.info(f"Datasets has been read and added to lists: fire images -> {len(fire_dataset)}, no fire images -> {len(non_fire_dataset)},\
 augmented no fire images -> {len(aug_non_fire_dataset)}.")
    logging.info(f"Summary of dataset: fire -> {len(fire_dataset)}, no fire -> {len(non_fire_dataset) + len(aug_non_fire_dataset)}.")

    non_fire_dataset += aug_non_fire_dataset

    image_path_and_label_dict = {}

    # Match image path and image label
    for fire_image in fire_dataset:           image_path_and_label_dict[fire_image] = 1
    for non_fire_image in non_fire_dataset :  image_path_and_label_dict[non_fire_image] = 0
    logging.info(f"For fire images '1' has been assigned, for no fire images '0' has been assigned.")

    input_path_list = fire_dataset + non_fire_dataset
        
    if constant.shuffle:
        random.shuffle(input_path_list)
        
    test_ind = int(len(input_path_list) * constant.test_size)
    valid_ind = int(test_ind + len(input_path_list) * constant.valid_size)
        
    test_path_list = input_path_list[:test_ind]
    valid_path_list = input_path_list[test_ind:valid_ind]
    train_path_list = input_path_list[valid_ind:]
    logging.info(f"Dataset has been shuffled and splitted to train, test and valid. test ratio -> {constant.test_size}, valid ratio -> {constant.valid_size}")
    logging.info(f"train size -> {len(train_path_list)}, test size -> {len(test_path_list)}, validation size -> {len(valid_path_list)}")

    return train_path_list, valid_path_list, test_path_list, image_path_and_label_dict

def save_json(information_dict, json_file_path):
    json_str = json.dumps(information_dict, indent=4, separators=(", ", ": "))

    with open(json_file_path, 'w') as json_file:
        json_file.write(json_str)

def read_json(json_file_path):
    
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    return data

def save_the_best_model(model, optimizer, information_dict):
    # Generate a unique file name for the model
    MODEL_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"

    # Define the path for the model
    model_folder_path = os.path.join(os.getcwd(), "models", MODEL_FILE)

    os.makedirs(model_folder_path, exist_ok=True)

    model_file_name = os.path.join(model_folder_path, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.pt")

    # Save the model state and optimizer state to the model file
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_file_name)

    # Define the path for the JSON file
    json_file_name = "model_info.json"
    json_file_path = os.path.join(model_folder_path, json_file_name)

    save_json(information_dict, json_file_path)

def load_the_model(model_path):
    ...

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(writer, net, images, labels):
    '''
    Logs images, predicted labels, and class probabilities for each image
    to TensorBoard using the SummaryWriter.
    '''
    classes = list(constant.CLASSES.keys())
    preds, probs = images_to_probs(net, images)
    
    for idx, (image, label, pred, prob) in enumerate(zip(images, labels, preds, probs)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.axis('off')
        ax.set_title(f'Predicted: {classes[pred]}\nProbability: {probs[pred] * 100:.2f}%\nTrue Label: {classes[label]}',
                     color=("green" if pred == label.item() else "red"))
        image = image.cpu().detach().numpy()
        # Change tensor size to numpy size -> (C, H, W) to (H, W, C) and convert dtype to np.uint8 for plotting.
        image = image.transpose((1, 2, 0)).astype(np.uint8)
        # Convert BGR image to RGB image to understand image better
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        
        # Log the image, its predicted label, and probabilities to TensorBoard
        writer.add_figure(f'Predictions/Image_{idx}', fig, global_step=idx)
        writer.add_text(f'Predictions/Class_{idx}', f'Predicted: {classes[pred]}, True: {classes[label]}', global_step=idx)
        writer.add_text(f'Predictions/Probability_{idx}', f'Probability: {probs[pred] * 100:.2f}%', global_step=idx)