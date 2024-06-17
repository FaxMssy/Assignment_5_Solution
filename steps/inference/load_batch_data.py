from zenml import step
from typing_extensions import Annotated
import pandas as pd


@step
def load_batch_data(data_path: str) -> Annotated[pd.DataFrame,"batch_data"]:
    """
    Load batch data from a CSV file.
    """
    return pd.read_csv(data_path)