import os
import sys
import numpy as np

import dill

from src.exception import CustomException

from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path:sys):
    '''
    This function returns the object in certain file path.
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model, param, cv = 3)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            accuracy = accuracy_score(y_test, [int(y > 0.5) for y in y_test_pred])
            test_model_score = accuracy

            report[list(models.keys())[i]] = test_model_score
        
        return report

    except Exception as e:
        raise CustomException(e, sys)