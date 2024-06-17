from zenml import step
import pandas as pd
from datetime import datetime
@step
def calculate_age(dataset: pd.DataFrame, inference: bool = False) -> pd.DataFrame:
    """
    Calculate the age of players based on their date of birth.
    """
    dataset["date_of_birth"] = pd.to_datetime(dataset["date_of_birth"])
    if inference:
        comparison_date = pd.to_datetime(datetime.now())
    else:
        comparison_date = pd.to_datetime(dataset.index) 
    dataset["player_age"] = (comparison_date - dataset["date_of_birth"]).dt.days // 365
    dataset.drop(columns=["date_of_birth"], inplace=True, axis=1)
    return dataset
    