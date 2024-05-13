import os
import pandas as pd

def calc_cards (row):
    # Grab the column values that are relevent to calculating cards per team per match
    num_cards = str(row['card_count'])
    card_team = str(row['card_team'])
    card_detail = str(row['card_detail'])
    
    # Remove unwanted items so the original string can be turned into a list
    remove = ['[', ']', '\'', ',']
    for i in remove:
        card_team = card_team.replace(i, '')
        card_detail = card_detail.replace(i, '')
    
    card_team = card_team.split()
    card_detail = card_detail.split()

    
    # Count number of yellow & red cards per team
    home_id = str(row['home_team_api_id'])
    away_id = str(row['away_team_api_id'])
    home_yellow_card_count = 0
    away_yellow_card_count = 0
    home_red_card_count = 0
    away_red_card_count = 0

    count = 0
    for card in card_detail:
        if card == 'y' or card == 'y2':
            if card_team[count] == home_id:
                home_yellow_card_count += 1
            if card_team[count] == away_id:
                away_yellow_card_count += 1
        if card == 'r':
            if card_team[count] == home_id:
                home_red_card_count += 1
            if card_team[count] == away_id:
                away_red_card_count += 1
        count += 1
    return pd.Series({'home_yellow_card_count': home_yellow_card_count, 'away_yellow_card_count': away_yellow_card_count, \
                      'home_red_card_count' : home_red_card_count, 'away_red_card_count': away_red_card_count})

def calc_foul (row):
    # Grab the column values that are relevent to calculating fouls per team per match
    foul_team = str(row['foulcommit_team'])
    
    # Remove unwanted items so the original string can be turned into a list
    remove = ['[', ']', '\'', ',']
    for i in remove:
        foul_team = foul_team.replace(i, '')
    
    foul_team = foul_team.split()
    
    # Count number of fouls per team
    home_id = str(row['home_team_api_id'])
    away_id = str(row['away_team_api_id'])
    home_foul_count = 0
    away_foul_count = 0

    count = 0
    for foul in foul_team:
        if foul == home_id:
            home_foul_count += 1
        if foul == away_id:
            away_foul_count += 1
        count += 1
    return pd.Series({'home_foul_count': home_foul_count, 'away_foul_count': away_foul_count})


def calc_stats(match_df):

    print('Creating Rolling Goals Team Data...')

    # Ensure the data is sorted by date to make the calculations chronological
    match_df['date'] = pd.to_datetime(match_df['date'])
    match_df.sort_values(by='date', inplace=True)

    # Initialize columns for rolling goal counts for home and away teams
    match_df['home_goals_last_10'] = 0
    match_df['away_goals_last_10'] = 0

    # Calculate the rolling sum of the last 10 matches for home and away teams
    for team_id in pd.concat([match_df['home_team_api_id'], match_df['away_team_api_id']]).unique():
        # Home matches
        home_mask = match_df['home_team_api_id'] == team_id
        match_df.loc[home_mask, 'home_goals_last_10'] = match_df.loc[home_mask, 'home_team_goal'].rolling(window=10, min_periods=1).sum().shift(1)
        
        # Away matches
        away_mask = match_df['away_team_api_id'] == team_id
        match_df.loc[away_mask, 'away_goals_last_10'] = match_df.loc[away_mask, 'away_team_goal'].rolling(window=10, min_periods=1).sum().shift(1)

    
    # Create Team Card Stats
    print('Creating Rolling Card Team Data...')

    match_df[['home_yellow_card_count', 'away_yellow_card_count', 'home_red_card_count', 'away_red_card_count']] = match_df.apply(calc_cards, axis = 1)
    
    # Initialize columns for rolling goal counts for home and away teams
    match_df['home_yellow_card_last_10'] = 0
    match_df['away_yellow_card_last_10'] = 0
    match_df['home_red_card_last_10'] = 0
    match_df['away_red_card_last_10'] = 0
    
    # Calculate the rolling sum of the last 10 matches for home and away teams
    for team_id in pd.concat([match_df['home_team_api_id'], match_df['away_team_api_id']]).unique():
        # Home matches
        home_mask = match_df['home_team_api_id'] == team_id
        match_df.loc[home_mask, 'home_yellow_card_last_10'] = match_df.loc[home_mask, 'home_yellow_card_count'].rolling(window=10, min_periods=1).sum().shift(1)
        match_df.loc[home_mask, 'home_red_card_last_10'] = match_df.loc[home_mask, 'home_red_card_count'].rolling(window=10, min_periods=1).sum().shift(1)
        
        
        # Away matches
        away_mask = match_df['away_team_api_id'] == team_id
        match_df.loc[away_mask, 'away_yellow_card_last_10'] = match_df.loc[away_mask, 'away_yellow_card_count'].rolling(window=10, min_periods=1).sum().shift(1)
        match_df.loc[away_mask, 'away_red_card_last_10'] = match_df.loc[away_mask, 'away_red_card_count'].rolling(window=10, min_periods=1).sum().shift(1)


    # Create Team Foul Stats
    print('Creating Rolling Card Foul Data...')

    match_df[['home_foul_count', 'away_foul_count']] = match_df.apply(calc_foul, axis = 1)
    
    # Initialize columns for rolling goal counts for home and away teams
    match_df['home_foul_last_10'] = 0
    match_df['away_foul_last_10'] = 0

    
    # Calculate the rolling sum of the last 10 matches for home and away teams
    for team_id in pd.concat([match_df['home_team_api_id'], match_df['away_team_api_id']]).unique():
        # Home matches
        home_mask = match_df['home_team_api_id'] == team_id
        match_df.loc[home_mask, 'home_foul_last_10'] = match_df.loc[home_mask, 'home_foul_count'].rolling(window=10, min_periods=1).sum().shift(1)
        
        
        # Away matches
        away_mask = match_df['away_team_api_id'] == team_id
        match_df.loc[away_mask, 'away_foul_last_10'] = match_df.loc[away_mask, 'away_foul_count'].rolling(window=10, min_periods=1).sum().shift(1)


    # At this point, matches_df contains two new columns:
    # 'home_goals_last_10' and 'away_goals_last_10' with the sum of goals scored in the last 10 matches
    # Note: The first few matches for each team will have NaN values for these columns until they have played 10 matches
    # You might want to fill NaN values with 0 or leave as is, depending on your analysis needs

    # Example to fill NaN values with 0
    match_df['home_goals_last_10'].fillna(0, inplace=True)
    match_df['away_goals_last_10'].fillna(0, inplace=True)
    match_df['home_yellow_card_last_10'].fillna(0, inplace=True)
    match_df['home_red_card_last_10'].fillna(0, inplace=True)
    match_df['away_yellow_card_last_10'].fillna(0, inplace=True)
    match_df['away_red_card_last_10'].fillna(0, inplace=True)
    match_df['home_foul_last_10'].fillna(0, inplace=True)
    match_df['away_foul_last_10'].fillna(0, inplace=True)    



    return match_df
    # Now you can use match_df for further analysis or merge these stats with your teams or players dataset as needed