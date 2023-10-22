import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        ...
    
    def predict(self, features):
        '''
        This function returns the result of custom datas result from pretrained model. 
        '''
        try:
            model_path = 'artifacts/model.pkl'
            preprocess_path = 'artifacts/preprocessor.pkl'

            model = load_object(model_path)
            preprocessor = load_object(preprocess_path)

            preprocessed_features = preprocessor.transform(features)
            
            raw_result = model.predict(preprocessed_features)
            classified_result = int(raw_result > 0.5)

            return classified_result

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 month: str,
                 temp: float,
                 RH: float,
                 wind: float,
                 rain: float):
        
        self.month = month
        self.temp = temp
        self.RH = RH
        self.wind = wind
        self.rain = rain
    
    def get_data_as_data_frame(self):

        try:
            custom_data_input_dict = {
                "month": [self.month],
                "temp" : [self.temp],
                "RH"   : [self.RH],
                "wind" : [self.wind],
                "rain" : [self.rain]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
