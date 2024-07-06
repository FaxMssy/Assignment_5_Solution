import pandas as pd
from zenml import step
import pandas as pd
import requests
import datetime
from datetime import timedelta
import pandas as pd
from dateutil import parser

@step 
def build_inference_data():

    # Koordinaten von Würzburg
    lat = 49.7833
    lon = 9.9333
    api_key = '0c8cb9eaefd487f928f2245b334c520e'  # Hier deinen tatsächlichen API-Schlüssel einfügen

    # URL für die API-Anfrage
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}"

    # API-Anfrage senden und die Antwort erhalten
    response = requests.get(url)
    data = response.json()

    # Überprüfen, ob die Anfrage erfolgreich war
    if response.status_code == 200:
        # Neues Format für die reduzierten Daten
        formatted_data = []

        for entry in data['list']:
            timestamp = datetime.datetime.utcfromtimestamp(entry['dt']).isoformat() + 'Z'
            weather_main = entry['weather'][0]['main']
            temperature = entry['main']['temp']
            
            formatted_entry = {
                "timestamp": timestamp,
                "weather_condition": weather_main,
                "temperature": temperature
            }

            formatted_data.append(formatted_entry)

        # Erstellen eines Pandas DataFrame
        df = pd.DataFrame(formatted_data)

        # Mapping der Wetterbedingungen zu den Klassen
        weather_mapping = {
            "Clear": "clear-day",
            "Clouds": "cloudy",
            "Rain": "rain",
            "Drizzle": "rain",
            "Thunderstorm": "rain",
            "Snow": "snow",
            "Mist": "fog",
            "Smoke": "fog",
            "Haze": "wind",
            "Dust": "fog",
            "Fog": "fog",
            "Sand": "fog",
            "Squall": "fog",
            "Tornado": "wind"
        }

        # Funktion zur Zuordnung der Klassen und zur Bestimmung von Tag/Nacht
        def map_weather_to_class(weather, timestamp):
            mapped_class = weather_mapping.get(weather, "wind")
            dt = parser.parse(timestamp)
            hour = dt.hour
            if "clear" in mapped_class or "partly-cloudy" in mapped_class:
                if 6 <= hour < 18:
                    mapped_class = mapped_class.replace('night', 'day')
                else:
                    mapped_class = mapped_class.replace('day', 'night')
            return mapped_class

        # Temperatur von Kelvin in Celsius umrechnen
        df['temperature'] = df['temperature'] - 273.15

        # Ersetzen der Werte von 'weather_condition' und Konvertieren der Temperatur
        df['weather_condition'] = df.apply(lambda row: map_weather_to_class(row['weather_condition'], row['timestamp']), axis=1)

        # Ausgabe des DataFrames zur Überprüfung
        #print(df)
    else:
        print(f"Fehler bei der API-Anfrage. Statuscode: {response.status_code}")

    # Timestamps in datetime umwandeln
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Start- und Endzeit für die nächsten 24 Stunden
    start_time = df['timestamp'].min()
    end_time = start_time + timedelta(hours=24)

    # Neuer DataFrame mit stündlichen Timestamps
    new_timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
    new_df = pd.DataFrame({'timestamp': new_timestamps})

    # 'weather_condition' mit vorwärts-fill auffüllen
    df.set_index('timestamp', inplace=True)
    new_df = new_df.merge(df[['weather_condition', 'temperature']], left_on='timestamp', right_index=True, how='left')
    new_df['weather_condition'] = new_df['weather_condition'].fillna(method='ffill')

    # 'temperature' mit linearer Interpolation auffüllen
    new_df['temperature'] = new_df['temperature'].interpolate(method='linear')

    # DataFrame auf 24 Stunden limitieren und als CSV speichern
    new_df = new_df.iloc[:24]

    # DataFrame erstellen
    df2 = pd.DataFrame(new_df)

    # Timestamps in datetime umwandeln
    df2['timestamp'] = pd.to_datetime(df2['timestamp'])

    # Extrahieren von hour, day, month und dayofweek
    df2['hour'] = df2['timestamp'].dt.hour
    df2['day'] = df2['timestamp'].dt.day
    df2['month'] = df2['timestamp'].dt.month
    df2['dayofweek'] = df2['timestamp'].dt.dayofweek

    # Droppen der timestamp-Spalte
    df2.drop(columns=['timestamp'], inplace=True)

    # Hinzufügen der location_name-Spalte
    locations = ['Schönbornstraße', 'Spiegelstraße', 'Kaiserstraße']
    expanded_data = []

    for _, row in df2.iterrows():
        for location in locations:
            new_row = row.copy()
            new_row['location_name'] = location
            expanded_data.append(new_row)

    expanded_df = pd.DataFrame(expanded_data)

    # Speichern des DataFrames als CSV-Datei ohne Index
    expanded_df.to_csv('data/inference_data.csv', index=False) 


