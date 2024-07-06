from zenml import pipeline
from zenml.client import Client
from steps import load_batch_data, inference_prediction, inference_preprocessing, drift_detection, build_inference_data
from evidently.metrics import DatasetDriftMetric
from zenml.integrations.evidently.metrics import EvidentlyMetricConfig
from zenml.integrations.evidently.steps import evidently_report_step

@pipeline(enable_cache=False)
def inference_pipeline():
    """
    Runs the inference pipeline for making predictions on new data.
    """
    load_batch_data.after(build_inference_data)
    build_inference_data()
    client = Client()
    model = client.get_artifact_version("model")
    preprocessing_pipeline = client.get_artifact_version("pipeline")
    train_dataset = client.get_artifact_version("x_train")
    batch_data = load_batch_data("data/inference_data.csv")
    drift, report = drift_detection(train_dataset, batch_data)
    if drift:
        print("Drift detected")
        # Handle the drift if needed
    batch_data = inference_preprocessing(batch_data, preprocessing_pipeline)
    predictions = inference_prediction(batch_data, model, drift)
    return predictions
