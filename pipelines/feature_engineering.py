from zenml import pipeline
from steps import data_loader, calculate_age,create_preprocessing_pipeline,feature_engineering_preprocessing,data_splitter
@pipeline(enable_cache=False)
def feature_engineering_pipeline():
    """
    Executes the feature engineering pipeline.

    This function loads the dataset, calculates the age, creates a preprocessing pipeline,
    splits the data into training and testing sets, and performs feature engineering preprocessing.
    """
    dataset = data_loader("./data/train.csv")
    dataset = calculate_age(dataset)
    pipeline = create_preprocessing_pipeline(dataset,"market_value_in_eur")
    X_train,X_test,y_train,y_test = data_splitter(dataset,"market_value_in_eur")
    X_train,X_test,pipeline = feature_engineering_preprocessing(X_train,X_test,pipeline)