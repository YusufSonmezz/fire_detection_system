import os
import sys
from src.exception import CustomException
from src.logger import logging, setup_logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.threshold = 15
    
    def initiate_data_ingestion(self):
        setup_logging(logging.INFO)
        logging.info("Data Ingestion has been started.")

        try:
            df = pd.read_csv("notebook/data/nasa_dataset/output_area_2.csv")
            logging.info("Read the dataset as DataFrame.")

            # Add threshold to detect if there is a fire or not.
            df["burned_area"] = (df["burned_area"] > 15).astype(int)    

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            removed_columns = ["year", "frp", "perc_frp"]
            df = df.drop(columns=removed_columns)
            logging.info(f"Irrelivant columns are dropped -> {removed_columns}.")

            df.to_csv(self.ingestion_config.raw_data_path, index = False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df["burned_area"])

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion of the data is completed.")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))