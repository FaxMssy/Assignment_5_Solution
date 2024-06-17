from zenml import step
import pandas as pd
from typing_extensions import Annotated

@step
def data_loader(data_path: str) -> Annotated[pd.DataFrame,"input_data"]:
    """
    Loads data from a CSV file located at the given data_path and returns it as a pandas DataFrame.
    """
    return pd.read_csv(data_path,index_col="date")