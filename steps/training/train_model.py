from zenml import step 
from sklearn.ensemble import RandomForestRegressor
from typing_extensions import Annotated
import pandas as pd


@step 
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Annotated[RandomForestRegressor, "model"]:
    """
    Trains a random forest regressor model using the given training data.
    """
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model