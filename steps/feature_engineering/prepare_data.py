import pandas as pd
from zenml import step
import requests

@step 
def prepare_data():
    # Daten laden und vorbereiten
    data = pd.read_csv("https://opendata.wuerzburg.de/api/explore/v2.1/catalog/datasets/passantenzaehlung_stundendaten/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B", delimiter=";")
    data.drop(["min_temperature","details","GeoShape","GeoPunkt","location_id", "unverified"], axis=1, inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)

    # Extrahieren neuer Features aus dem Timestamp
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month
    data['dayofweek'] = data['timestamp'].dt.dayofweek
    data['date'] = data['timestamp'].dt.date

    data['weather_condition'] = data['weather_condition'].replace(['partly-cloudy-night', 'partly-cloudy-day'], 'cloudy')
    
    # URL der Ferien-API
    url = "https://ferien-api.de/api/v1/holidays/"
    
    # Anfrage an die API senden
    response = requests.get(url)

    if response.status_code == 200:
        holidays = response.json()
        
        # Dictionary zum Speichern der Ferien pro Bundesland
        holiday_dict = {state: [] for state in ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV', 'NI', 'NW', 'RP', 'SL', 'ST', 'SN', 'SH', 'TH']}
        
        # Daten in das Dictionary einfügen
        for holiday in holidays:
            state = holiday['stateCode']
            start = pd.to_datetime(holiday['start']).date()
            end = pd.to_datetime(holiday['end']).date()
            holiday_dict[state].extend(pd.date_range(start, end).date)
        
        # Für jedes Bundesland eine Spalte hinzufügen
        for state in holiday_dict:
            col_name = f'ferien_{state.lower()}'
            data[col_name] = data['date'].apply(lambda x: 1 if x in holiday_dict[state] else 0)
    else:
        print(f"Fehler bei der Anfrage. Statuscode: {response.status_code}")

    # URL der Fußballspiele-API
    fussball_url = "https://api.openligadb.de/getmatchdata/em2024/2024"

    # Anfrage an die Fußballspiele-API senden
    fussball_response = requests.get(fussball_url)

    if fussball_response.status_code == 200:
        matches = fussball_response.json()

        # Fußballspiel-Spalten hinzufügen
        data['fussballspiel'] = 0
        data['deutschlandspiel'] = 0

        # Sicherstellen, dass die 'timestamp'-Spalte timezone-naiv ist
        data['timestamp'] = data['timestamp'].dt.tz_localize(None)

        # Fußballspiel-Zeiten verarbeiten
        for match in matches:
            match_datetime = pd.to_datetime(match['matchDateTime']).tz_localize(None)  # Zeitzone entfernen
            match_start = match_datetime - pd.Timedelta(hours=2)
            match_end = match_datetime + pd.Timedelta(hours=4)
            team1 = match['team1']['teamName']
            team2 = match['team2']['teamName']
            
            is_deutschland = team1 == 'Deutschland' or team2 == 'Deutschland'

            # Update der 'fussballspiel' und 'deutschlandspiel' Spalten
            mask = (data['timestamp'] >= match_start) & (data['timestamp'] <= match_end)
            data.loc[mask, 'fussballspiel'] = 1
            if is_deutschland:
                data.loc[mask, 'deutschlandspiel'] = 1
    else:
        print(f"Fehler bei der Anfrage. Statuscode: {fussball_response.status_code}")

    data.drop("date", axis=1, inplace=True)
    # Daten in CSV-Datei speichern
    data.to_csv('data/wue_data.csv', index=False)
