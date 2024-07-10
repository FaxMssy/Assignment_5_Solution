from zenml import pipeline
from steps import data_loader,create_preprocessing_pipeline,feature_engineering_preprocessing,data_splitter,prepare_data
@pipeline(enable_cache=False)
def feature_engineering_pipeline():
    """
    Executes the feature engineering pipeline.

    This function loads the dataset, creates a preprocessing pipeline,
    splits the data into training and testing sets, and performs feature engineering preprocessing.
    """

    data_loader.after(prepare_data)
    prepare_data()
    dataset = data_loader("./data/wue_data.csv")

    pipeline = create_preprocessing_pipeline(dataset,"pedestrians_count")
    X_train,X_test,y_train,y_test = data_splitter(dataset,"pedestrians_count")
    X_train,X_test,pipeline = feature_engineering_preprocessing(X_train,X_test,pipeline)