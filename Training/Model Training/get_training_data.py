import pandas as pd
import numpy as np
import requests

def get_matches():
    url = 'https://kyykka.com/api/matches/'
    response = requests.get(url).json()
    home_ids = []
    away_ids = []
    for match in response:
        home_ids.append(match['home_team']['id'])
        away_ids.append(match['away_team']['id'])
        
get_matches()