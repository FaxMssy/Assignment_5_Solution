from zenml import step
import pandas as pd
from typing_extensions import Annotated
from sklearn.pipeline import Pipeline


@step
def inference_preprocessing(data: pd.DataFrame, pipeline: Pipeline) -> Annotated[pd.DataFrame, "inference_preprocessed_data"]:
    """
    Preprocesses the input data using the provided pipeline.
    """
    transformed_data = pipeline.transform(data)
    cat_features_after_encoding = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(data.select_dtypes(include=['object']).columns)
    all_features = list(data.select_dtypes(exclude=['object']).columns) + list(cat_features_after_encoding)
    data_df = pd.DataFrame(transformed_data, columns=all_features)
    return data_df