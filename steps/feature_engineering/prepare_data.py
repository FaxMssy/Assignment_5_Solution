import pandas as pd
from zenml import step
import pandas as pd


@step 
def prepare_data():
    data = pd.read_csv("https://opendata.wuerzburg.de/api/explore/v2.1/catalog/datasets/passantenzaehlung_stundendaten/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B", delimiter=";")
    data.drop(["min_temperature","details","GeoShape","GeoPunkt","location_id", "unverified"], axis=1, inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)

    # Extrahieren neuer Features aus dem Timestamp
    for df in [data]:
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month #probieren was funktioniert
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        #df['year'] = df['timestamp'].dt.year
        #data.drop(["timestamp"], axis=1, inplace=True)

    data['weather_condition'] = data['weather_condition'].replace(['partly-cloudy-night', 'partly-cloudy-day'], 'cloudy')
    data.to_csv('data/wue_data.csv', index=False) 