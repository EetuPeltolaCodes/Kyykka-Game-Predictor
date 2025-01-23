import requests

response = requests.get('https://kyykka.com/api/teams/').json()

for res in response:
    id_response = requests.get(f'https://kyykka.com/api/teams/{res["id"]}/').json()
    stats_2025 = id_response['2025']
    mathc_played = stats_2025['match_count']
    score_total = stats_2025['score_total']
    throws_total = stats_2025['throws_total']
    average_score = score_total / throws_total
    pike_percentage = stats_2025['pike_percentage']
    zeros_percentage = stats_2025['zero_percentage']
    match_average = stats_2025['match_average']
    players = stats_2025['players']
    scaled_points_per_throw = []
    for player in players:
        if player['throws_total'] != 0:
            scaled_points_per_throw.append(player['scaled_points_per_throw'])
    average_scaled_points_per_throw = sum(scaled_points_per_throw) / len(scaled_points_per_throw)
    print(f'Team: {stats_2025["current_abbreviation"]}')
    print(f'Score total: {score_total}')
    print(f'Pike percentage: {pike_percentage}')
    print(f'Zeros percentage: {zeros_percentage}')
    print(f'Match average: {match_average}')
    print(f'Average scaled points per throw: {average_scaled_points_per_throw}')
    print(f'Average score: {average_score}')
    print('-----------------------------------')
    
