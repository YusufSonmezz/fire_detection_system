import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging, setup_logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        setup_logging(logging.INFO)

        self.target_column = 'area'
        self.numerical_columns = []
        self.categorical_columns = []
    
    def get_data_transformation_obj(self):
        '''
        This functions prepare preprocess object to use. Here the object can be configurated.
        '''
        try:
            logging.info("Creation of preprocess object has been started.")

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical columns preprocess has been completed.")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical columns preprocess has been completed.")

            logging.info(f"Numerical Columns -> {self.numerical_columns} Categorical Columns -> {self.categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, self.numerical_columns),
                    ("cat_pipeline", cat_pipeline, self.categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed.")

            # Extract numerical and categorical columns automatically
            self.numerical_columns = [column for column in train_df.columns if train_df[column].dtype != 'O']
            self.categorical_columns = [column for column in train_df.columns if train_df[column].dtype == 'O']

            # Remove target column from preprocess. Because target column is something we want to predict.
            # We don't want any information about target column inside other columns.
            if self.target_column in self.numerical_columns: self.numerical_columns.remove(self.target_column)
            if self.target_column in self.categorical_columns: self.categorical_columns.remove(self.target_column)
            logging.info("Numerical and Categorical Columns has been detected and target column has been removed.")

            preprocessing_obj = self.get_data_transformation_obj()

            input_feature_train_df = train_df.drop(columns=[self.target_column], axis = 1)
            target_feature_train_df = train_df[self.target_column]

            input_feature_test_df = test_df.drop(columns=[self.target_column], axis = 1)
            target_feature_test_df = test_df[self.target_column]

            logging.info("Applying preprocess object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)