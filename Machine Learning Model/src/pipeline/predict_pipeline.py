import sys
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
            
            classified_result = model.predict(preprocessed_features)
            

            return classified_result

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 month: int,
                 temp: float,
                 RH: float,
                 wind: float,
                 day_night: int):
        
        self.month = month
        self.temp = temp
        self.RH = RH
        self.wind = wind
        self.day_night = day_night

        base_temperature = 10.0

        # Calculate the temperature difference
        temp_diff = self.temp - base_temperature

        # Calculate Cooling Degree Days (CDD)
        self.daily_cdd_customer = max(temp_diff, 0)

        # Calculate Heating Degree Days (HDD)
        self.daily_hdd_customer = max(-temp_diff, 0)
    
    def get_data_as_data_frame(self):

        try:
            custom_data_input_dict = {
                "month": [self.month],
                "temp" : [self.temp],
                "RH"   : [self.RH],
                "wind" : [self.wind],
                "day_night" : [self.day_night],
                "daily_cdd": [self.daily_cdd_customer],
                "daily_hdd": [self.daily_hdd_customer]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
    
if __name__ == "__main__":
    predict = PredictPipeline()
    df = pd.read_csv("notebook/data/synthetic/synthetic.csv")

    x = df.drop(columns=["burned_area"])
    y = df["burned_area"]
    
    y_pred = predict.predict(x)

    print("Accuracy score is..: ", accuracy_score(y, y_pred))
    print("Confusion Matrix\n", confusion_matrix(y, y_pred))
    print("Classification Report\n", classification_report(y, y_pred))


