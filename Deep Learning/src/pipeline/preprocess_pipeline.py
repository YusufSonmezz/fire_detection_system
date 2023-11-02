import os
import glob
import cv2
import numpy as np
import torch

from config import constant


class PreprocessPipeline:
    def __init__(self):
        ...
    
    def initate_preprocess(self, input_path_list):
        '''
        Get paths of images. After read image by image, resize to normalize and convert them into
        tensors which are proper to be used in GPU.

        Parameters
        -------------
        input_path_list: List[str]

        Returns
        -------------
        torch_image_list: List[tensor]
        '''

        local_image_list = []
        
        for input_path in input_path_list:

            image = cv2.imread(input_path)

            image = cv2.resize(image, constant.IMAGE_SIZE)

            torchlike_image = self.torchlike_data(image)

            local_image_list.append(torchlike_image)
        
        image_array = np.array(local_image_list, dtype = np.float32)
        torch_image_list = torch.from_numpy(image_array).float()

        if constant.cuda:
            torch_image_list = torch_image_list.cuda()
        
        return torch_image_list


    def torchlike_data(self, data):
        '''
        Change data structure according to Torch Tensor structure where the first
        dimension corresponds to the data depth.


        Parameters
        ----------
        data : Array of uint8
            Shape : HxWxC.

        Returns
        -------
        torchlike_data_output : Array of float64
            Shape : CxHxW.
        '''

        n_channels = data.shape[2]

        torchlike_data_output = np.empty((n_channels, data.shape[0], data.shape[1]))

        for channel in range(n_channels):
            torchlike_data_output[channel] = data[:, :, channel]

        return torchlike_data_output
    
    def one_hot_encoder(self, label_list, num_class = len(constant.CLASSES)):
        
        ohe_list = []

        fire     = [1, 0]
        non_fire = [0, 1]

        for label in label_list:
            if label == 1: ohe_list.append(fire)
            elif label == 0: ohe_list.append(non_fire)
        
        return torch.from_numpy(np.array(ohe_list)).float().cuda() if constant.cuda else torch.from_numpy(np.array(ohe_list)).float()



if __name__ == "__main__":

    preprocessPipeline = PreprocessPipeline()

    image_list = glob.glob(constant.FIRE_RAW_DATASET)

    batch_image_list = image_list[:constant.BATCH_SIZE]

    batch_image_tensor = preprocessPipeline.initate_preprocess(batch_image_list)

    print("For features:\ndtype is "+str(batch_image_tensor.dtype))
    print("Type is "+str(type(batch_image_tensor)))
    print("Size is "+str(batch_image_tensor.shape)+"\n")

    label_test = [1, 0, 1, 0, 1, 1, 1, 0]

    label_ohe = preprocessPipeline.one_hot_encoder(label_test)
    print(label_ohe)
    
    