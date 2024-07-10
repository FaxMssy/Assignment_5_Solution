import pandas as pd
from zenml import step
import requests
import datetime
from datetime import timedelta
from dateutil import parser

@step 
def build_inference_data():
    """
    Getting the weather prediction from a weather API for the prediction. Building a dataset with predicted weather data and temperature 
    for the next 24 hours to enable the AI to predict the pedestrian count.
    """
    # Koordinates for Würzburg, as they are needed for the weather API
    lat = 49.7833
    lon = 9.9333
    api_key = '0c8cb9eaefd487f928f2245b334c520e'  # Personal API key needed for API usage

    # URL for API Request
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}"

    # API Request
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        formatted_data = []
        # Parsing the weather data response
        for entry in data['list']:
            timestamp = datetime.datetime.utcfromtimestamp(entry['dt']).isoformat() + 'Z'
            weather_main = entry['weather'][0]['main']
            temperature = entry['main']['temp']

            # The information needed for the inference Dataset. 
            formatted_entry = {
                "timestamp": timestamp,
                "weather_condition": weather_main,
                "temperature": temperature
            }

            formatted_data.append(formatted_entry)

        df = pd.DataFrame(formatted_data)

        # Mapping of the weather_conditions from the API response to the existing weather_condition from the Würzburg Dataset
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

        # Based on the time the class gets changed to clear-night or clear-day in accordance to the Würzburg Data
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

        # Temperature from Kelvin to Celsius
        df['temperature'] = df['temperature'] - 273.15

        # Addition of weather_condition values to the dataset
        df['weather_condition'] = df.apply(lambda row: map_weather_to_class(row['weather_condition'], row['timestamp']), axis=1)

    else:
        print(f"Fehler bei der API-Anfrage.[Inference] Statuscode: {response.status_code}")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Prediction for the next 24 hours from the point of request
    start_time = df['timestamp'].min()
    end_time = start_time + timedelta(hours=24)

    new_timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
    new_df = pd.DataFrame({'timestamp': new_timestamps})

    df.set_index('timestamp', inplace=True)
    new_df = new_df.merge(df[['weather_condition', 'temperature']], left_on='timestamp', right_index=True, how='left')
    new_df['weather_condition'] = new_df['weather_condition'].fillna(method='ffill')

    # Linear interpolation of temperature values as the API only delivers them in 3 hour time steps.
    new_df['temperature'] = new_df['temperature'].interpolate(method='linear')

    # Dataframe limited to 24 hours
    new_df = new_df.iloc[:24]

    """
    Bringing the inference dataset into the same format as the Würzburg dataset
    """

    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

    # Feature engineering according to Würzburg Dataset
    new_df['hour'] = new_df['timestamp'].dt.hour
    new_df['day'] = new_df['timestamp'].dt.day
    new_df['month'] = new_df['timestamp'].dt.month
    new_df['dayofweek'] = new_df['timestamp'].dt.dayofweek

    locations = ['Schönbornstraße', 'Spiegelstraße', 'Kaiserstraße']
    expanded_data = []

    for _, row in new_df.iterrows():
        for location in locations:
            new_row = row.copy()
            new_row['location_name'] = location
            expanded_data.append(new_row)

    expanded_df = pd.DataFrame(expanded_data)

    """
    Also adding the holidays like with the Würzburg Dataset.
    """

    # URL der API
    url = "https://ferien-api.de/api/v1/holidays/"

    response = requests.get(url)

    if response.status_code == 200:
        holidays = response.json()

        holidays_2024 = [holiday for holiday in holidays if holiday['year'] == 2024]

        holiday_dict = {}

        for holiday in holidays_2024:
            state = holiday['stateCode']
            start = pd.to_datetime(holiday['start'])
            end = pd.to_datetime(holiday['end'])
            date_range = pd.date_range(start, end)
            if state not in holiday_dict:
                holiday_dict[state] = []
            holiday_dict[state].extend(date_range)

        for state in holiday_dict:
            holiday_dict[state] = pd.to_datetime(holiday_dict[state]).date

        expanded_df['date'] = expanded_df['timestamp'].dt.date

        for state in holiday_dict.keys():
            col_name = f'ferien_{state.lower()}'
            expanded_df[col_name] = expanded_df['date'].apply(lambda x: 1 if x in holiday_dict[state] else 0)
    else:
        print(f"Fehler bei der Anfrage. [Inference2] Statuscode: {response.status_code}")

    """
    Also adding the football data like with the Würzburg Dataset.
    """

    fussball_url = "https://api.openligadb.de/getmatchdata/em2024/2024"

    fussball_response = requests.get(fussball_url)

    if fussball_response.status_code == 200:
        matches = fussball_response.json()

        expanded_df['fussballspiel'] = 0
        expanded_df['deutschlandspiel'] = 0

        expanded_df['timestamp'] = expanded_df['timestamp'].dt.tz_localize(None)

        for match in matches:
            match_datetime = pd.to_datetime(match['matchDateTime']).tz_localize(None)  # Zeitzone entfernen
            match_start = match_datetime - pd.Timedelta(hours=2)
            match_end = match_datetime + pd.Timedelta(hours=4)
            team1 = match['team1']['teamName']
            team2 = match['team2']['teamName']
            
            is_deutschland = team1 == 'Deutschland' or team2 == 'Deutschland'

            mask = (expanded_df['timestamp'] >= match_start) & (expanded_df['timestamp'] <= match_end)
            expanded_df.loc[mask, 'fussballspiel'] = 1
            if is_deutschland:
                expanded_df.loc[mask, 'deutschlandspiel'] = 1

    expanded_df.drop("date", axis=1, inplace=True)
    # The data with timestamp is also saved to merge it with the predictions later
    expanded_df.to_csv('data/inference_data_with_timestamp.csv', index=False)

    expanded_df.drop("timestamp", axis=1, inplace=True)
    expanded_df.to_csv('data/inference_data.csv', index=False)
