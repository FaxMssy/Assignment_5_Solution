from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
import pandas as pd


@step
def data_splitter(data: pd.DataFrame, target: str) -> Tuple[Annotated[pd.DataFrame, 'x_train'], Annotated[pd.DataFrame, "x_test"], Annotated[pd.Series, "y_train"], Annotated[pd.Series, "y_test"]]:
    """
        Split the given data into training and testing sets.
    """
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test