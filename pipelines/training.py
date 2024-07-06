from zenml import pipeline
from zenml.client import Client
from steps import train_model,evaluate_model

@pipeline(enable_cache=False)
def training_pipeline():
    """
    Executes the training pipeline.

    This function retrieves preprocessed data from a client, trains a model using the training data,
    and evaluates the model using the test data.
    """
    client = Client()
    X_train = client.get_artifact_version("X_train_preprocessed")
    X_test = client.get_artifact_version("X_test_preprocessed")
    y_train = client.get_artifact_version("y_train")
    y_test = client.get_artifact_version("y_test")
    model = train_model(X_train, y_train)
    mae = evaluate_model(model, X_test, y_test)

    return mae
