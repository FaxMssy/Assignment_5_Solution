from zenml import step
from typing_extensions import Annotated
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple
from sklearn.pipeline import Pipeline

@step
def feature_engineering_preprocessing(train: pd.DataFrame, test: pd.DataFrame, pipeline: Pipeline) -> Tuple[Annotated[pd.DataFrame, "X_train_preprocessed"], Annotated[pd.DataFrame, "X_test_preprocessed"], Annotated[Pipeline, "pipeline"]]:
    """
    Preprocesses the training and test data using the provided pipeline.
    """
    X_train_transformed = pipeline.fit_transform(train)
    X_test_transformed = pipeline.transform(test)
    cat_features_after_encoding = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(train.select_dtypes(include=['object']).columns)
    print("Cat features: " + cat_features_after_encoding)
    
    all_features = list(train.select_dtypes(exclude=['object']).columns) + list(cat_features_after_encoding)
    print("All features: " + ", ".join(all_features))
    

    X_train_df = pd.DataFrame(X_train_transformed, columns=all_features)
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_features)
   
    X_train_df.to_csv('data/X_train_df.csv', index=False) 
    X_test_df.to_csv('data/X_test_df.csv', index=False) 

    return X_train_df, X_test_df, pipeline
