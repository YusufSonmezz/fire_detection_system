import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from app.dl.dl_utils import PreprocessPipeline
from app.dl.dl_utils import ResNet
from app.dl import constant

class DLController:
    def __init__(self, dl_model_path: str):
        self.preprocessor = PreprocessPipeline()

        self.model = ResNet(len(constant.CLASSES))
        self.model.load_state_dict(torch.load(dl_model_path, map_location='cpu'))
        self.model.eval()
    
    def predict_image(self, image):
        batch_input = self.preprocessor.initiate_preprocess_image([image])

        output = self.model(batch_input)
        
        # Use the .item() method to get the integer value of the index
        idx = torch.argmax(output, dim=1).item()
        
        output_prob_list = nn.Sigmoid()(output)
        
        # Use idx.item() to convert the tensor index to an integer
        output_prob = output_prob_list[0, idx].item()

        return idx, output_prob
    
    def save_image(self, image, output, output_prob, path, iter):
        # Convert OpenCV BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(image_rgb)

        # Assuming 'result_text' is defined somewhere
        result_text = f"Fire: {output}, Prob: {output_prob}"

        # Calculate the position for the top of the image
        position = (0.02, 0.92)  # Adjust as needed

        # Add a text annotation
        ax.text(position[0], position[1], result_text, color='white', fontsize=10, ha='left', va='center',
                transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.7))

        # Turn off axis labels
        ax.axis('off')

        output_path = path + f'/{iter}_output_image_matplotlib.jpg'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)



