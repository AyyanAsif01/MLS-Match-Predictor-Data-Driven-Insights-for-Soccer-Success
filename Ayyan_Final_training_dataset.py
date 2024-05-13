#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 00:13:12 2024

@author: ayyanasif
"""
import os
import pandas as pd
#new_directory = r'C:\Users\ethan\OneDrive\Documents\Ethan NMSU Stuff\Current Classes\C S-519 Applied Machine Learning I\Group Project\GP_Stage 4\Main'
#sos.chdir(new_directory)

# Load the match data
#matches = pd.read_csv('match.csv')
#team_attributes = pd.read_csv('team_attributes.csv')
#player_attributes = pd.read_csv('player_attributes.csv')


def combine_tables(matches, team_attributes, player_attributes):
    print("Combining Tables & Cleaning Data...")
    # Process team_attributes to get the most recent attributes for each team
    team_attributes.sort_values(by='date', ascending=False, inplace=True)
    latest_team_attributes = team_attributes.drop_duplicates(subset=['team_api_id']).copy()


    latest_team_attributes.rename(columns={'id': 'team_attr_id'}, inplace=True)

    # Process player_attributes to get the most recent attributes for each player
    player_attributes.sort_values(by='date', ascending=False, inplace=True)
    latest_player_attributes = player_attributes.drop_duplicates(subset=['player_api_id']).copy()

    # Merge the above with the latest team attributes (for both home and away teams)
    matches_with_team_attributes = matches.merge(latest_team_attributes, left_on='home_team_api_id', right_on='team_api_id', how='left', suffixes=('_match', '_home_attr'))
    matches_with_team_attributes = matches_with_team_attributes.merge(latest_team_attributes, left_on='away_team_api_id', right_on='team_api_id', how='left', suffixes=('_home', '_away_attr'))

    # player columns for home and away
    player_columns_home = ['home_player_' + str(i) for i in range(1, 12)]
    player_columns_away = ['away_player_' + str(i) for i in range(1, 12)]
    player_columns = player_columns_home + player_columns_away

    # Merge the latest player attributes for the identified players in each match
    for column in player_columns:
        if column in matches.columns:
            matches_with_team_attributes = matches_with_team_attributes.merge(
                latest_player_attributes, left_on=column, right_on='player_api_id', how='left', suffixes=('', '_' + column)
            )

    # Address PerformanceWarning by creating new columns using assign
    goals_penalties_info = {
        'home_team_goals': [0] * len(matches_with_team_attributes), 
        'away_team_goals': [0] * len(matches_with_team_attributes), 
        'home_team_penalties': [0] * len(matches_with_team_attributes),  
        'away_team_penalties': [0] * len(matches_with_team_attributes),  
    }

    # efficiently add multiple columns
    matches_with_team_attributes = matches_with_team_attributes.assign(**goals_penalties_info)

    # Combine the data to form one row per match with high dimensionality
    final_dataset = matches_with_team_attributes

    # Remove unwanted columns
    final_dataset = final_dataset.drop(columns = ['team_attr_id_home', 'team_fifa_api_id_home', 'team_api_id_home', 'date_home_attr',
                                                'team_attr_id_away_attr', 'team_fifa_api_id_away_attr', 'team_api_id_away_attr', 'date'])


    final_dataset = final_dataset.drop(columns= ['id_home_player_' + str(i) for i in range(1, 12)])
    final_dataset = final_dataset.drop(columns= ['player_fifa_api_id'])
    final_dataset = final_dataset.drop(columns= ['player_api_id'])
    final_dataset = final_dataset.drop(columns= ['date_home_player_' + str(i) for i in range(1, 12)])
    final_dataset = final_dataset.drop(columns= ['player_fifa_api_id_home_player_' + str(i) for i in range(2, 12)])
    final_dataset = final_dataset.drop(columns= ['player_api_id_home_player_' + str(i) for i in range(2, 12)])

    final_dataset = final_dataset.drop(columns= ['id_away_player_' + str(i) for i in range(1, 12)])
    final_dataset = final_dataset.drop(columns= ['player_fifa_api_id_away_player_' + str(i) for i in range(1, 12)])
    final_dataset = final_dataset.drop(columns= ['player_api_id_away_player_' + str(i) for i in range(1, 12)])
    final_dataset = final_dataset.drop(columns= ['date_away_player_' + str(i) for i in range(1, 12)])
  
    
    # This portion was added by Ethan later
    non_prediction_columns = ['id', 'country_id', 'league_id', 'season', 'stage', 'date_match', 'match_api_id', 'home_team_api_id', 'away_team_api_id',\
                            'home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7',\
                                'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1', 'away_player_2', 'away_player_3',\
                                    'away_player_4', 'away_player_5', 'away_player_6', 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10',\
                                        'away_player_11', 'goal_elapsed', 'goal_player1', 'goal_team', 'goal_type', 'goal_detail', 'shoton_elapsed', 'shoton_player1',\
                                            'shoton_team', 'shoton_type', 'shoton_detail', 'shotoff_elapsed', 'shotoff_player1', 'shotoff_team', 'shotoff_type',\
                                                'shotoff_detail', 'foulcommit_elapsed', 'foulcommit_player1', 'foulcommit_team', 'foulcommit_type', 'foulcommit_detail',\
                                                    'card_elapsed', 'card_player1', 'card_team', 'card_type', 'card_detail', 'cross_elapsed', 'cross_player1',\
                                                        'cross_team', 'cross_type', 'cross_detail', 'corner_elapsed', 'corner_player1', 'corner_team', 'corner_type',\
                                                            'corner_detail', 'possession_elapsed', 'possession_player1', 'possession_team', 'possession_type',\
                                                                'possession_detail', 'shoton_count', 'shotoff_count', 'cross_count', 'corner_count', 'possession_count',\
                                                                    'home_team_goals',	'away_team_goals',	'home_team_penalties',	'away_team_penalties']

    final_dataset = final_dataset.drop(non_prediction_columns, axis=1)

    # This portion turns string catagorical data columns into integers
    final_dataset['buildUpPlaySpeedClass_home'] = final_dataset['buildUpPlaySpeedClass_home'].map({'Balanced': 0, 'Fast': 1, 'Slow': 2})   
    final_dataset['buildUpPlayDribblingClass_home'] = final_dataset['buildUpPlayDribblingClass_home'].map({'Little': 0, 'Lots': 1, 'Normal': 2})
    final_dataset['buildUpPlayPassingClass_home'] = final_dataset['buildUpPlayPassingClass_home'].map({'Long': 0, 'Mixed': 1, 'Short': 2})
    final_dataset['buildUpPlayPositioningClass_home'] = final_dataset['buildUpPlayPositioningClass_home'].map({'Free Form': 0, 'Organised': 1})
    final_dataset['chanceCreationPassingClass_home'] = final_dataset['chanceCreationPassingClass_home'].map({'Normal': 0, 'Safe': 1, 'Risky': 2})
    final_dataset['chanceCreationCrossingClass_home'] = final_dataset['chanceCreationCrossingClass_home'].map({'Little': 0, 'Lots': 1, 'Normal': 2})
    final_dataset['chanceCreationShootingClass_home'] = final_dataset['chanceCreationShootingClass_home'].map({'Little': 0, 'Lots': 1, 'Normal': 2})
    final_dataset['chanceCreationPositioningClass_home'] = final_dataset['chanceCreationPositioningClass_home'].map({'Free Form': 0, 'Organised': 1})
    final_dataset['defencePressureClass_home'] = final_dataset['defencePressureClass_home'].map({'Deep': 0, 'High': 1, 'Medium': 2})
    final_dataset['defenceAggressionClass_home'] = final_dataset['defenceAggressionClass_home'].map({'Contain': 0, 'Double': 1, 'Press': 2})
    final_dataset['defenceTeamWidthClass_home'] = final_dataset['defenceTeamWidthClass_home'].map({'Narrow': 0, 'Normal': 1, 'Wide': 2})
    final_dataset['defenceDefenderLineClass_home'] = final_dataset['defenceDefenderLineClass_home'].map({'Offside Trap': 0, 'Cover': 1})
    final_dataset['buildUpPlaySpeedClass_away_attr'] = final_dataset['buildUpPlaySpeedClass_away_attr'].map({'Balanced': 0, 'Fast': 1, 'Slow': 2})   
    final_dataset['buildUpPlayDribblingClass_away_attr'] = final_dataset['buildUpPlayDribblingClass_away_attr'].map({'Little': 0, 'Lots': 1, 'Normal': 2})
    final_dataset['buildUpPlayPassingClass_away_attr'] = final_dataset['buildUpPlayPassingClass_away_attr'].map({'Long': 0, 'Mixed': 1, 'Short': 2})
    final_dataset['buildUpPlayPositioningClass_away_attr'] = final_dataset['buildUpPlayPositioningClass_away_attr'].map({'Free Form': 0, 'Organised': 1})
    final_dataset['chanceCreationPassingClass_away_attr'] = final_dataset['chanceCreationPassingClass_away_attr'].map({'Normal': 0, 'Safe': 1, 'Risky': 2})
    final_dataset['chanceCreationCrossingClass_away_attr'] = final_dataset['chanceCreationCrossingClass_away_attr'].map({'Little': 0, 'Lots': 1, 'Normal': 2})
    final_dataset['chanceCreationShootingClass_away_attr'] = final_dataset['chanceCreationShootingClass_away_attr'].map({'Little': 0, 'Lots': 1, 'Normal': 2})
    final_dataset['chanceCreationPositioningClass_away_attr'] = final_dataset['chanceCreationPositioningClass_away_attr'].map({'Free Form': 0, 'Organised': 1})
    final_dataset['defencePressureClass_away_attr'] = final_dataset['defencePressureClass_away_attr'].map({'Deep': 0, 'High': 1, 'Medium': 2})
    final_dataset['defenceAggressionClass_away_attr'] = final_dataset['defenceAggressionClass_away_attr'].map({'Contain': 0, 'Double': 1, 'Press': 2})
    final_dataset['defenceTeamWidthClass_away_attr'] = final_dataset['defenceTeamWidthClass_away_attr'].map({'Narrow': 0, 'Normal': 1, 'Wide': 2})
    final_dataset['defenceDefenderLineClass_away_attr'] = final_dataset['defenceDefenderLineClass_away_attr'].map({'Offside Trap': 0, 'Cover': 1})







    return final_dataset

#final_dataset.to_csv('ayyan_test_final_training dataset.csv', index=False)

