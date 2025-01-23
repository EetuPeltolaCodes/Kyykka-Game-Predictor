import requests

response = requests.get('https://kyykka.com/api/teams/').json()

with open('Training/Teams_data.csv', 'w') as file:
    for res in response:
        id_response = requests.get(f'https://kyykka.com/api/teams/{res["id"]}/').json()
        stats_year = id_response['2025']
        match_played = stats_year['match_count']
        score_total = stats_year['score_total']
        throws_total = stats_year['throws_total']
        average_score = score_total / throws_total
        pike_percentage = stats_year['pike_percentage']
        zeros_percentage = stats_year['zero_percentage']
        match_average = stats_year['match_average']
        players = stats_year['players']
        scaled_points_per_throw = []
        for player in players:
            if player['throws_total'] != 0:
                scaled_points_per_throw.append(player['scaled_points_per_throw'])
        average_scaled_points_per_throw = sum(scaled_points_per_throw) / len(scaled_points_per_throw)
        print(stats_year['current_abbreviation'])
        file.write(f"{res['id']},{match_played},{score_total},{throws_total},{average_score},{pike_percentage},{zeros_percentage},{match_average},{average_scaled_points_per_throw}\n")

        
        
# Fetch more years than just 2025      
'''
        year = 2025
        while True:
            try:
                stats_year = id_response[str(year)]
                match_played = stats_year['match_count']
                score_total = stats_year['score_total']
                throws_total = stats_year['throws_total']
                average_score = score_total / throws_total
                pike_percentage = stats_year['pike_percentage']
                zeros_percentage = stats_year['zero_percentage']
                match_average = stats_year['match_average']
                players = stats_year['players']
                scaled_points_per_throw = []
                for player in players:
                    if player['throws_total'] != 0:
                        scaled_points_per_throw.append(player['scaled_points_per_throw'])
                average_scaled_points_per_throw = sum(scaled_points_per_throw) / len(scaled_points_per_throw)
                print(stats_year['current_abbreviation'])
                file.write(f"{res['id']},{match_played},{score_total},{throws_total},{average_score},{pike_percentage},{zeros_percentage},{match_average},{average_scaled_points_per_throw}\n")
            except KeyError:
                break
            year -= 1
'''
