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
        
        model_name = self.model_json["Model Name"]
        model_module = importlib.import_module(f"src.components.model")
        model_class = getattr(model_module, model_name)

        self.model = model_class(len(constant.CLASSES))
        self.model = torch.load(model_path)
        self.model.to(device)
        self.model.eval()

        self.test_path_list = self.model_json['test_path_list']
    
    def initiate_evaluation(self):
        self.test_model()

        self.writer.flush()
    
    def test_model(self):
        label_list = []
        prediction_list = []
        test_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():
            for idx, (test_path, label) in enumerate(tqdm.tqdm(self.test_path_list.items())):
                batch_input = self.preprocess_pipeline.initate_preprocess([test_path])
                batch_label = self.preprocess_pipeline.one_hot_encoder([label])

                output = self.model(batch_input)
                loss = self.criterion(output, batch_label)

                _, predicted = torch.max(output, 1)
                prediction_list.extend(predicted.cpu().numpy())
                _, true_label = torch.max(batch_label, 1)
                label_list.extend(true_label.cpu().numpy())
                correct_predictions += (predicted == true_label).sum().float()

                test_loss += loss.float()

                plot_classes_preds(self.writer, self.model, batch_input, batch_label, global_step=idx * len(self.test_path_list))
                
        accuracy = accuracy_score(label_list, prediction_list)
        confusion = confusion_matrix(label_list, prediction_list)
        classification_report_str = classification_report(label_list, prediction_list)

        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(confusion)
        self.evaluation_utils.plot_confusion_matrix(confusion)
        print("Classification Report:")
        print(classification_report_str)
        
        average_loss = test_loss / len(self.test_path_list)
        accuracy = correct_predictions / len(self.test_path_list)

        print(f"Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    evaluation = EvaluationPipeline("11_12_2023_18_08_30")
    evaluation.initiate_evaluation()