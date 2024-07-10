import pandas as pd
from zenml import step
import requests
import datetime
from datetime import timedelta
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
        print(f"Fehler bei der API-Anfrage.[Inference] Statuscode: {response.status_code}")

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

    # Timestamps in datetime umwandeln
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

    # Extrahieren von hour, day, month und dayofweek
    new_df['hour'] = new_df['timestamp'].dt.hour
    new_df['day'] = new_df['timestamp'].dt.day
    new_df['month'] = new_df['timestamp'].dt.month
    new_df['dayofweek'] = new_df['timestamp'].dt.dayofweek

    # Hinzufügen der location_name-Spalte
    locations = ['Schönbornstraße', 'Spiegelstraße', 'Kaiserstraße']
    expanded_data = []

    for _, row in new_df.iterrows():
        for location in locations:
            new_row = row.copy()
            new_row['location_name'] = location
            expanded_data.append(new_row)

    expanded_df = pd.DataFrame(expanded_data)

    # URL der API
    url = "https://ferien-api.de/api/v1/holidays/"

    # Anfrage an die API senden
    response = requests.get(url)

    # Überprüfen, ob die Anfrage erfolgreich war
    if response.status_code == 200:
        # JSON-Daten aus der Antwort extrahieren
        holidays = response.json()
        
        # Einträge für das Jahr 2024 filtern
        holidays_2024 = [holiday for holiday in holidays if holiday['year'] == 2024]
        
        # Dictionary zum Speichern der Ferien pro Bundesland
        holiday_dict = {}
        
        # Daten in das Dictionary einfügen
        for holiday in holidays_2024:
            state = holiday['stateCode']
            start = pd.to_datetime(holiday['start'])
            end = pd.to_datetime(holiday['end'])
            date_range = pd.date_range(start, end)
            if state not in holiday_dict:
                holiday_dict[state] = []
            holiday_dict[state].extend(date_range)

        # Daten in ein DataFrame umwandeln
        for state in holiday_dict:
            holiday_dict[state] = pd.to_datetime(holiday_dict[state]).date
        
        # Konvertiere 'timestamp' in ein Datum
        expanded_df['date'] = expanded_df['timestamp'].dt.date

        # Für jedes Bundesland eine Spalte hinzufügen
        for state in holiday_dict.keys():
            col_name = f'ferien_{state.lower()}'
            expanded_df[col_name] = expanded_df['date'].apply(lambda x: 1 if x in holiday_dict[state] else 0)
    else:
        print(f"Fehler bei der Anfrage. [Inference2] Statuscode: {response.status_code}")

    # URL der Fußballspiele-API
    fussball_url = "https://api.openligadb.de/getmatchdata/em2024/2024"

    # Anfrage an die Fußballspiele-API senden
    fussball_response = requests.get(fussball_url)

    if fussball_response.status_code == 200:
        # JSON-Daten aus der Antwort extrahieren
        matches = fussball_response.json()

        # Fußballspiel-Spalten hinzufügen
        expanded_df['fussballspiel'] = 0
        expanded_df['deutschlandspiel'] = 0

        # Sicherstellen, dass die 'timestamp'-Spalte timezone-naiv ist
        expanded_df['timestamp'] = expanded_df['timestamp'].dt.tz_localize(None)

        # Fußballspiel-Zeiten verarbeiten
        for match in matches:
            match_datetime = pd.to_datetime(match['matchDateTime']).tz_localize(None)  # Zeitzone entfernen
            match_start = match_datetime - pd.Timedelta(hours=2)
            match_end = match_datetime + pd.Timedelta(hours=4)
            team1 = match['team1']['teamName']
            team2 = match['team2']['teamName']
            
            is_deutschland = team1 == 'Deutschland' or team2 == 'Deutschland'

            # Update der 'fussballspiel' und 'deutschlandspiel' Spalten
            mask = (expanded_df['timestamp'] >= match_start) & (expanded_df['timestamp'] <= match_end)
            expanded_df.loc[mask, 'fussballspiel'] = 1
            if is_deutschland:
                expanded_df.loc[mask, 'deutschlandspiel'] = 1

    

    expanded_df.drop("date", axis=1, inplace=True)
    expanded_df.to_csv('data/inference_data_with_timestamp.csv', index=False)
    # Speichern des DataFrames als CSV-Datei ohne Index
    expanded_df.drop("timestamp", axis=1, inplace=True)
    expanded_df.to_csv('data/inference_data.csv', index=False)
