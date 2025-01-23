import pandas as pd
import numpy as np
import requests

def get_matches():
    url = 'https://kyykka.com/api/matches/'
    response = requests.get(url).json()
    home_ids = []
    away_ids = []
    winners = []
    for match in response:
        if match['home_score_total'] != None:
            home_ids.append(match['home_team']['id'])
            away_ids.append(match['away_team']['id'])
            if match['home_score_total'] > match['away_score_total']:
                winners.append(1)
            elif match['home_score_total'] < match['away_score_total']:
                winners.append(-1)
            else:
                winners.append(0)
        else:
            break
            
    return home_ids, away_ids, winners

def get_data(home_ids, away_ids, winners, step):
    with open('Training\Teams_data.csv', 'r') as f:
        teams_data = pd.read_csv(f)
        with open(f'Training\{step}_data.csv', 'w') as f2:
            for home_id, away_id, winner in zip(home_ids, away_ids, winners):
                home_data = teams_data[teams_data['Team ID'] == home_id]
                away_data = teams_data[teams_data['Team ID'] == away_id]
                f2.write(f"{home_data.iloc[0, 1]},{home_data.iloc[0, 2]},{home_data.iloc[0, 3]},{home_data.iloc[0, 4]},{home_data.iloc[0, 5]},{home_data.iloc[0, 6]},{home_data.iloc[0, 7]},{home_data.iloc[0, 8]},")
                f2.write(f"{away_data.iloc[0, 1]},{away_data.iloc[0, 2]},{away_data.iloc[0, 3]},{away_data.iloc[0, 4]},{away_data.iloc[0, 5]},{away_data.iloc[0, 6]},{away_data.iloc[0, 7]},{away_data.iloc[0, 8]},")
                f2.write(f"{winner}\n")       
    
        
def datas():
    home_ids, away_ids, winners = get_matches()
    N = len(home_ids)
    # Split the data 70/15/15
    train_data_idxs = np.random.choice(N, int(0.7*N), replace=False)
    validation_data_idxs = np.random.choice(np.setdiff1d(np.arange(N), train_data_idxs), int(0.15*N), replace=False)
    test_data_idxs = np.setdiff1d(np.arange(N), np.concatenate([train_data_idxs, validation_data_idxs]))
    train_home_ids = np.array(home_ids)[train_data_idxs]
    train_away_ids = np.array(away_ids)[train_data_idxs]
    train_winners = np.array(winners)[train_data_idxs]
    get_data(train_home_ids, train_away_ids, train_winners, 'train')
    validation_home_ids = np.array(home_ids)[validation_data_idxs]
    validation_away_ids = np.array(away_ids)[validation_data_idxs]
    validation_winners = np.array(winners)[validation_data_idxs]
    get_data(train_home_ids, train_away_ids, train_winners, 'validation')
    test_home_ids = np.array(home_ids)[test_data_idxs]
    test_away_ids = np.array(away_ids)[test_data_idxs]
    test_winners = np.array(winners)[test_data_idxs]
    get_data(train_home_ids, train_away_ids, train_winners, 'test')
    