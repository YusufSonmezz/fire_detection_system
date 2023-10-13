import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    RandomForestClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging, setup_logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        setup_logging(logging.INFO)
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model trainer has been started")
            logging.info("Split train and test input array")

            x_train, y_train, x_test, y_test = (
                train_array[:, : -1],
                train_array[:, -1],

                test_array[:, : -1],
                test_array[:, -1]
            )

            logging.info(f"x_train shape -> {x_train.shape} y_train shape -> {y_train.shape} x_test shape -> {x_test.shape} y_test shape -> {y_test.shape}")

            models = {
                "Random Forest": RandomForestRegressor(),
                "Random Forest Classifier": RandomForestClassifier(),
                "Decision Tree": DecisionTreeRegressor(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Linear Regression": LinearRegression(),
                "CatBoosting Regressor": CatBoostRegressor(verbose = False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Logistic Regression": LogisticRegressionCV()
            }

            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Logistic Regression":{},
                "Decision Tree Classifier":{
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                },
                "Random Forest Classifier":{
                    'criterion': ['gini', 'entropy', 'log_loss'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report: dict = evaluate_models(
                x_train = x_train,
                y_train = y_train,
                x_test = x_test,
                y_test = y_test,
                models = models,
                params = params
            )

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                ...#raise CustomException("No best model found!!")
            
            logging.info([f"{model} -> {model_report[model]}" for model in model_report.keys()])

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            y_pred = best_model.predict(x_test)

            accuracy = accuracy_score(y_test, [int(y > 0.5) for y in y_pred])
            confusion = confusion_matrix(y_test, [int(y > 0.5) for y in y_pred])
            classification_report_str = classification_report(y_test, [int(y > 0.5) for y in y_pred])

            print(f"Accuracy: {accuracy}")
            print("Confusion Matrix:")
            print(confusion)
            print("Classification Report:")
            print(classification_report_str)

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)