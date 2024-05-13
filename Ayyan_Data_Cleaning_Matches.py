import os
import numpy as np
import pandas as pd


def clean_match_data (match = None):
    print('Cleaning Initial "Match" Data...')
    
    # Remove data in which XLM data (goals, shoton, etc.) is missing
    match.dropna(subset=['goal'], inplace=True)

    # Remove rows before 2010 since there are no team attribute data for before this date:
    index_date = match[ (match['date'] < '2010-02-22')].index
    #indexAge = df[ (df['Age'] >= 20) & (df['Age'] <= 25) ].index
    match.drop(index_date , inplace=True)

    # Remove columns in which 10% or more of the rows are missing
    rows = len(match.axes[0])
    threshhold = round(rows * 0.1)

    for i in match.columns:
        nulls = match[i].isna().sum()
        if nulls >= threshhold:
            match = match.drop(str(i), axis=1)

    # Drop remaining rows that have incomplete data
    match.dropna(inplace=True)

    # Drop irrelevant columns
    match = match.drop(columns = ['home_player_X1', 'home_player_X2', 'home_player_X3', 'home_player_X4', 'home_player_X5',
                                  'home_player_X6', 'home_player_X7', 'home_player_X8', 'home_player_X9', 'home_player_X10',
                                  'home_player_X11', 'away_player_X1', 'away_player_X2', 'away_player_X3', 'away_player_X4', 'away_player_X5',
                                  'away_player_X6', 'away_player_X7', 'away_player_X8', 'away_player_X9', 'away_player_X10', 'away_player_X11',
                                  'home_player_Y1', 'home_player_Y2', 'home_player_Y3', 'home_player_Y4', 'home_player_Y5', 'home_player_Y6',
                                  'home_player_Y7',	'home_player_Y8', 'home_player_Y9', 'home_player_Y10', 'home_player_Y11', 'away_player_Y1', 
                                  'away_player_Y2',	'away_player_Y3', 'away_player_Y4', 'away_player_Y5', 'away_player_Y6', 'away_player_Y7', 
                                  'away_player_Y8',	'away_player_Y9', 'away_player_Y10', 'away_player_Y11'])
    print('Finished Cleaning Intial "Match" Data')
    return match
