from zenml import step
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from zenml.types import HTMLString
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset



@step()
def drift_detection(reference_data: pd.DataFrame, inference_data: pd.DataFrame) -> Tuple[Annotated[bool, "drift_detected"],Annotated[HTMLString, "drift_report"]]:
  """
    Detects drift between reference data and inference data.
  """
  data_drift_report = Report(metrics=[
    DataDriftPreset(),
  ])

  data_drift_report.run(current_data=inference_data, reference_data=reference_data, column_mapping=None)
  drift_detected = data_drift_report.as_dict()["metrics"][0]["result"]["dataset_drift"]
  return drift_detected, HTMLString(data_drift_report.get_html())
