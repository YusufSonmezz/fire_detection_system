import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import tqdm
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report

from src.components.model import VGGModel
from config import constant
from src.utils import train_test_split, plot_classes_preds
from src.pipeline.preprocess_pipeline import PreprocessPipeline

class TrainPipeline:
    def __init__(self):
        self.classes = len(constant.CLASSES)
        self.model = VGGModel(self.classes)
        self.preprocessPipeline = PreprocessPipeline()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = constant.LEARNING_RATE, weight_decay=0.001)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.writer = SummaryWriter()

        if constant.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        
        self.train_list, self.valid_list, self.test_list, self.label_dict = train_test_split()

        ########
        self.prediction = []
        self.label = []
        ########
    
    def initiate_train(self):

        for epoch in range(constant.EPOCH):
            train_loss, train_accuracy = self.train_one_epoch(epoch)
            val_loss, val_accuracy = self.validate(epoch)
            ####
            self.label = []
            self.prediction = []
            ####
        
        self.writer.flush()
    
    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        steps_per_epoch = len(self.train_list)//constant.BATCH_SIZE

        for ind in tqdm.tqdm(range(steps_per_epoch)):
            batch_input_list = self.train_list[constant.BATCH_SIZE*ind:constant.BATCH_SIZE*(ind+1)]
            batch_label_list = [self.label_dict[input_path] for input_path in batch_input_list]

            batch_input = self.preprocessPipeline.initate_preprocess(batch_input_list)
            batch_label = self.preprocessPipeline.one_hot_encoder(batch_label_list)

            outputs = self.model(batch_input)
            '''

            # Apply last layer outside of model
            outputs = nn.Sigmoid()(outputs)'''

            loss = self.criterion(outputs, batch_label)
            
            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.float()

            # Calculate Accuracy
            _, predicted = torch.max(outputs, 1)
            _, batch_label_predict = torch.max(batch_label, 1)
            correct_predictions += (predicted == batch_label_predict).sum().float()
            total_samples += batch_label.size(0)

            
            '''self.writer.add_figure(
                'predictions vs. actuals',
                plot_classes_preds(self.writer, self.model, batch_input, batch_label_predict),
                global_step=epoch
            )'''
        
        epoch_loss = running_loss / len(self.train_list)
        accuracy = correct_predictions / total_samples

        self.writer.add_scalar("Loss/train", epoch_loss, epoch)

        print(f"Epoch {epoch + 1} - Training Loss: {epoch_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

        return epoch_loss, accuracy
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():
            for i, valid_path in enumerate(self.valid_list):
                batch_input = self.preprocessPipeline.initate_preprocess([valid_path])
                batch_label = self.preprocessPipeline.one_hot_encoder([self.label_dict[valid_path]])

                output = self.model(batch_input)
                loss = self.criterion(output, batch_label)

                val_loss += loss.float()

                _, predicted = torch.max(output, 1)
                self.prediction.extend(predicted.cpu().numpy())
                _, predicted_label = torch.max(batch_label, 1)
                self.label.extend(predicted_label.cpu().numpy())
                correct_predictions += (predicted == predicted_label).sum().float()
                
                self.writer.add_scalar("Loss/Validate", val_loss, i)
                
        accuracy = accuracy_score(self.label, self.prediction)
        confusion = confusion_matrix(self.label, self.prediction)
        classification_report_str = classification_report(self.label, self.prediction)

        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(confusion)
        print("Classification Report:")
        print(classification_report_str)
        
        average_loss = val_loss / len(self.valid_list)
        accuracy = correct_predictions / len(self.valid_list)
        self.scheduler.step()

        print(f"Epoch {epoch + 1} - Validation Loss: {average_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

        return average_loss, accuracy

    



if __name__ == "__main__":
    trainPipeline = TrainPipeline()
    trainPipeline.initiate_train()