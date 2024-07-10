import pandas as pd
from zenml import step
import requests

@step 
def prepare_data():

    """
    Loading of the whole Dataset from the Würzburg City API. Feature Engineering of the Dataset 
    and also some formatting and typechanges based on the exploratory data analysis.
    """
    #Retrieve Data via API
    data = pd.read_csv("https://opendata.wuerzburg.de/api/explore/v2.1/catalog/datasets/passantenzaehlung_stundendaten/exports/csv?lang=de&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B", delimiter=";")
    
    #Drop unimportant colums based on exploratory data analysis
    data.drop(["min_temperature","details","GeoShape","GeoPunkt","location_id", "unverified"], axis=1, inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)

    #Feature engineering of the Dataset
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month
    data['dayofweek'] = data['timestamp'].dt.dayofweek
    data['date'] = data['timestamp'].dt.date

    #Map categories that aren't important (according to exploratory data analysis) to a more abstract category 'cloudy'
    data['weather_condition'] = data['weather_condition'].replace(['partly-cloudy-night', 'partly-cloudy-day'], 'cloudy')

    """
    In this section, a german holiday API is called to extract information about the school holidays and national holidays. 
    These get added to the original dataset to add features. 
    """

    #URL of the API used for holidays    
    url = "https://ferien-api.de/api/v1/holidays/"
    
    # API request to holiday API
    response = requests.get(url)


    #If the request is succesful, the API response gets parsed
    if response.status_code == 200:
        holidays = response.json()
        
        #Creation of a dictionary including all the state name abbreviations
        holiday_dict = {state: [] for state in ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV', 'NI', 'NW', 'RP', 'SL', 'ST', 'SN', 'SH', 'TH']}
        
        #Extraction of the holiday timeframes for each state
        for holiday in holidays:
            state = holiday['stateCode']
            start = pd.to_datetime(holiday['start']).date()
            end = pd.to_datetime(holiday['end']).date()
            holiday_dict[state].extend(pd.date_range(start, end).date)
        
        #Add a column for each state in the dataset, if there is a holiday at the specific day in a specific state, than a "1" is placed in the corresponding state holiday column
        for state in holiday_dict:
            col_name = f'ferien_{state.lower()}'
            data[col_name] = data['date'].apply(lambda x: 1 if x in holiday_dict[state] else 0)
    else:
        print(f"Fehler bei der Anfrage. Statuscode: {response.status_code}")


    """
    Additionally to the holiday API, in this section a football API is called. Because of the European Football tournament in Germany 
    there are more people in the city of Würzburg (backed by the exploratory data analysis). To help the AI calculating the impact,
    2 additional colums get added. One for a EM game overall and one specifically for games of the german football team, as the 
    data analysis showed that these games had an even larger impact. 
    """

    # URL of the football API
    fussball_url = "https://api.openligadb.de/getmatchdata/em2024/2024"

    # API request to the football API
    fussball_response = requests.get(fussball_url)

    if fussball_response.status_code == 200:
        matches = fussball_response.json()

        # Addition of the 2 additional columns
        data['fussballspiel'] = 0
        data['deutschlandspiel'] = 0

        # Adapting the timestamp accordingly
        data['timestamp'] = data['timestamp'].dt.tz_localize(None)

        # Parsing the API response to analyze if there is a game and if germany played
        for match in matches:
            match_datetime = pd.to_datetime(match['matchDateTime']).tz_localize(None) 
            match_start = match_datetime - pd.Timedelta(hours=2) # Two hours before the game and after the game are also labelled as if there is a game,
            match_end = match_datetime + pd.Timedelta(hours=2) #to also grasp people coming early to bars and also parties afterwards
            team1 = match['team1']['teamName']
            team2 = match['team2']['teamName']
            
            #Check if germany is one of the teams playing
            is_deutschland = team1 == 'Deutschland' or team2 == 'Deutschland'

            #Updating the columns regarding the game
            mask = (data['timestamp'] >= match_start) & (data['timestamp'] <= match_end)
            data.loc[mask, 'fussballspiel'] = 1
            if is_deutschland:
                data.loc[mask, 'deutschlandspiel'] = 1
    else:
        print(f"Fehler bei der Anfrage. Statuscode: {fussball_response.status_code}")

    
    data.drop("date", axis=1, inplace=True)

    data.to_csv('data/wue_data.csv', index=False)
