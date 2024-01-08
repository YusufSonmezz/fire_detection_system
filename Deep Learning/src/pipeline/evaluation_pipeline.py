import os
import glob
import importlib
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report

from src.logger import setup_logging, logging
from config import constant
from src.utils import read_json, plot_classes_preds
from src.pipeline.preprocess_pipeline import PreprocessPipeline

class EvaluationUtils:
    def __init__(self):
        ...
    
    def get_model_folder(self, model_time_name):

        model_folder = os.path.join(os.getcwd(), "models", model_time_name, "*")
        components = glob.glob(model_folder)
        return components

    def plot_confusion_matrix(self, matrix, cmap='Blues'):
        title = "Confusion Matrix"

        classes = constant.CLASSES.keys()
        
        plt.figure(figsize=(len(classes) + 1, len(classes) + 1))
        sns.set(font_scale=1.2)
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


class EvaluationPipeline:
    def __init__(self, model_time_name):
        if constant.cuda: device = torch.device("cuda")
        else: device = torch.device("cpu")

        self.evaluation_utils = EvaluationUtils()
        self.preprocess_pipeline = PreprocessPipeline()

        self.criterion = nn.CrossEntropyLoss()

        self.writer = SummaryWriter()

        self.model_components = self.evaluation_utils.get_model_folder(model_time_name)
        
        for path in self.model_components:
            if ".json" in path:
                self.model_json = read_json(path)
            elif ".pth" in path:
                model_path = path
        
        #model_name = self.model_json["Model Name"]
        model_name = "ResNet"
        model_module = importlib.import_module(f"src.components.model")
        model_class = getattr(model_module, model_name)

        self.model = model_class(len(constant.CLASSES))
        self.model = torch.load(model_path, map_location='cpu')
        self.model.to(device)
        self.model.eval()

        #self.test_path_list = self.model_json['test_path_list']
        self.test_path_list = glob.glob(os.path.join("data/mixed/", "*"))
        self.predicted_folder_path = os.path.join(os.getcwd(), "data/predicted_images")

        os.makedirs(self.predicted_folder_path, exist_ok=True)
    
    def initiate_evaluation(self):
        self.test_model()

        self.writer.flush()
    
    def test_model(self):
        label_list = []
        prediction_list = []
        test_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():
            for idx, test_path in enumerate(tqdm.tqdm(self.test_path_list)):
                batch_input = self.preprocess_pipeline.initate_preprocess([test_path])
                #batch_label = self.preprocess_pipeline.one_hot_encoder([label])

                output = self.model(batch_input)

                _, output = torch.max(output, axis = 1)

                result_text = "Predicted Result: {}".format(list(constant.CLASSES.keys())[output])

                fig, ax = plt.subplots()

                # Display the image
                ax.imshow(plt.imread(test_path))

                # Calculate the position for the top of the image
                position = (0.02, 0.92)  # Adjust as needed

                # Add a text annotation
                ax.text(position[0], position[1], result_text, color='white', fontsize=10, ha='left', va='center',
                        transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.7))

                # Turn off axis labels
                ax.axis('off')

                output_path = self.predicted_folder_path + f'/output_image_matplotlib_{idx}.jpg'
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.show()

                #plot_classes_preds(self.writer, self.model, batch_input, batch_label, global_step=idx * len(self.test_path_list))
                
        



if __name__ == "__main__":
    evaluation = EvaluationPipeline("11_12_2023_18_08_30")
    evaluation.initiate_evaluation()