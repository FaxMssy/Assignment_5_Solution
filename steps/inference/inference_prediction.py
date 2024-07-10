from zenml import step
from typing_extensions import Annotated
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
@step
def inference_prediction(batch: pd.DataFrame, model: RandomForestRegressor, drift: bool) -> Annotated[pd.DataFrame, "predictions"]:
    """
    Perform inference on a batch of data using a trained model.
    """
    predictions = model.predict(batch)
    batch["predictions"] = predictions
    batch["drift"] = drift


    inference_data_with_timestamp = pd.read_csv("data/inference_data_with_timestamp.csv")
    inference_data_with_timestamp["predictions"] = predictions
    inference_data_with_timestamp.to_csv('data/result_with_timestamp.csv')
    #batch.to_csv('data/result.csv', index=False) 
    return batch