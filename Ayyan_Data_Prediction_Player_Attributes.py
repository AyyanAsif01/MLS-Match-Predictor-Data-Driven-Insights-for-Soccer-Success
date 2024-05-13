import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from random import sample
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Create the linear regression model that will predict each column with null values
def Reg(column):
    # Data preperation
    target = str(column)
    X = known_rows[features].values
    y = known_rows[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    
    # Linear Regression
    lr = LinearRegression()  
    lr.fit(X_train, y_train)
    y_train_pred_lr = lr.predict(X_train)
    y_test_pred_lr = lr.predict(X_test)
    
    # Measure mean squared error for Linear Regression
    mse_train_lr = mean_absolute_error(y_train, y_train_pred_lr)
    mse_test_lr = mean_absolute_error(y_test, y_test_pred_lr)
    print('Linear Regression', column, f'MSE train: {mse_train_lr:.2f}, test: {mse_test_lr:.2f}')

    # Measure R^2 for Linear Regression
    train_r2_lr = r2_score(y_train, y_train_pred_lr)
    test_r2_lr = r2_score(y_test, y_test_pred_lr)
    print('Linear Regression', column, f'R^2 train: {train_r2_lr:.2f}, test: {test_r2_lr:.2f}')

    # RANSAC Regressor
    ransac = RANSACRegressor(LinearRegression(), max_trials = 100, min_samples=0.95, residual_threshold=None, random_state=1)
    ransac.fit(X_train, y_train)
    y_train_pred_ransac = ransac.predict(X_train)
    y_test_pred_ransac = ransac.predict(X_test)

    # Measure mean squared error for RANSAC
    mse_train_ransac = mean_absolute_error(y_train, y_train_pred_ransac)
    mse_test_ransac = mean_absolute_error(y_test, y_test_pred_ransac)
    print('RANSAC', column, f'MSE train: {mse_train_ransac:.2f}, test: {mse_test_ransac:.2f}')

    # Measure R^2 for RANSAC
    train_r2_ransac = r2_score(y_train, y_train_pred_ransac)
    test_r2_ransac = r2_score(y_test, y_test_pred_ransac)
    print('RANSAC', column, f'R^2 train: {train_r2_ransac:.2f}, test: {test_r2_ransac:.2f}')

    # Ridge Regressor
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_train_pred_ridge = ridge.predict(X_train)
    y_test_pred_ridge = ridge.predict(X_test)

    # Measure mean squared error for Ridge
    mse_train_ridge = mean_absolute_error(y_train, y_train_pred_ridge)
    mse_test_ridge = mean_absolute_error(y_test, y_test_pred_ridge)
    print('Ridge', column, f'MSE train: {mse_train_ridge:.2f}, test: {mse_test_ridge:.2f}')

    # Measure R^2 for Ridge
    train_r2_ridge = r2_score(y_train, y_train_pred_ridge)
    test_r2_ridge = r2_score(y_test, y_test_pred_ridge)
    print('Ridge', column, f'R^2 train: {train_r2_ridge:.2f}, test: {test_r2_ridge:.2f}')    

    # ElasticNet Regressor
    elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)
    elanet.fit(X_train, y_train)
    y_train_pred_elanet = elanet.predict(X_train)
    y_test_pred_elanet = elanet.predict(X_test)

    # Measure mean squared error for Elanet
    mse_train_elanet = mean_absolute_error(y_train, y_train_pred_elanet)
    mse_test_elanet = mean_absolute_error(y_test, y_test_pred_elanet)
    print('Elanet', column, f'MSE train: {mse_train_elanet:.2f}, test: {mse_test_elanet:.2f}')

    # Measure R^2 for Elanet
    train_r2_elanet = r2_score(y_train, y_train_pred_elanet)
    test_r2_elanet = r2_score(y_test, y_test_pred_elanet)
    print('Elanet', column, f'R^2 train: {train_r2_elanet:.2f}, test: {test_r2_elanet:.2f}')  

    # Random Forest Regressor
    forest = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=1, n_jobs=-1)
    forest.fit(X_train, y_train)
    y_train_pred_forest = forest.predict(X_train)
    y_test_pred_forest = forest.predict(X_test)

    # Measure mean squared error for Forest
    mse_train_forest = mean_absolute_error(y_train, y_train_pred_forest)
    mse_test_forest = mean_absolute_error(y_test, y_test_pred_forest)
    print('Forest', column, f'MSE train: {mse_train_forest:.2f}, test: {mse_test_forest:.2f}')

    # Measure R^2 for Forest 
    train_r2_forest = r2_score(y_train, y_train_pred_forest)
    test_r2_forest = r2_score(y_test, y_test_pred_forest)
    print('Forest', column, f'R^2 train: {train_r2_forest:.2f}, test: {test_r2_forest:.2f}')  

    # Predict and fill missing values in unknown rows
    R2_test_dict = {'lr':test_r2_lr, 'ransac':test_r2_ransac, 'ridge':test_r2_ridge, 'elanet':test_r2_elanet, 'forest':test_r2_forest}
    R2_train_dict = {'lr':train_r2_lr, 'ransac':train_r2_ransac, 'ridge':train_r2_ridge, 'elanet':train_r2_elanet, 'forest':train_r2_forest}
    MSE_test_dict = {'lr':mse_test_lr, 'ransac':mse_test_ransac, 'ridge':mse_test_ridge, 'elanet':mse_test_elanet, 'forest':mse_test_forest}
    MSE_train_dict = {'lr':mse_train_lr, 'ransac':mse_train_ransac, 'ridge':mse_train_ridge, 'elanet':mse_train_elanet, 'forest':mse_train_forest}

    for index, row in unknown_rows.iterrows():
        prediction_features = row[features].values.reshape(1, -1)
        algorithm, best_R2_test, best_MSE_test, best_R2_train, best_MSE_train = select_algorithm(R2_test_dict, MSE_test_dict, R2_train_dict, MSE_train_dict)
        if algorithm == 'lr':
            prediction = lr.predict(prediction_features)
            alg_print = "Linear Regression"
        if algorithm == 'ransac':
            prediction = ransac.predict(prediction_features)
            alg_print = "RANSAC"
        if algorithm == 'ridge':
            prediction = ridge.predict(prediction_features)
            alg_print = "Ridge"
        if algorithm == 'elanet':
            prediction = elanet.predict(prediction_features)
            alg_print = "Elanet"
        if algorithm == 'forest':
            prediction = forest.predict(prediction_features)
            alg_print = "Random Forest"

        
        player_attributes.loc[index, column] = np.round(prediction) # Note: The number may still be considered a float while other data points are integers. I'm not sure if this will cause any issues, but this is a place to look.

    print("Best algorithm for", column, ":", alg_print)
    print("R^2 Test:", best_R2_test, "MSE Test:", best_MSE_test, "R^2 Train:", best_R2_train, "MSE Train", best_MSE_train)
    print()

def select_algorithm(R2_test_dict, MSE_test_dict, R2_train_dict, MSE_train_dict):
    best_algorithm = None
    best_R2_test = 0
    best_MSE_test = float('inf')
    best_R2_train = 0
    best_MSE_train = float('inf')

    for algorithm, R2_test in R2_test_dict.items():
        R2_test = round(R2_test, 2)
        MSE_test = round(MSE_test_dict[algorithm],2)
        R2_train = round(R2_train_dict[algorithm],2)
        MSE_train = round(MSE_train_dict[algorithm],2)

        if R2_test > best_R2_test:
            best_algorithm = algorithm
            best_R2_test = R2_test
            best_MSE_test = MSE_test
            best_R2_train = R2_train
            best_MSE_train = MSE_train
        elif R2_test == best_R2_test:
            if MSE_test < best_MSE_test:
                best_algorithm = algorithm
                best_R2_test = R2_test
                best_MSE_test = MSE_test
                best_R2_train = R2_train
                best_MSE_train = MSE_train
            elif MSE_test == best_MSE_test:
                if R2_train > best_R2_train:
                    best_algorithm = algorithm
                    best_R2_test = R2_test
                    best_MSE_test = MSE_test
                    best_R2_train = R2_train
                    best_MSE_train = MSE_train
                elif R2_train == best_R2_train:
                    if MSE_train < best_MSE_train:
                        best_algorithm = algorithm
                        best_R2_test = R2_test
                        best_MSE_test = MSE_test
                        best_R2_train = R2_train
                        best_MSE_train = MSE_train
    
    return best_algorithm, best_R2_test, best_MSE_test, best_R2_train, best_MSE_train



def clean_player_data(data):
    global player_attributes
    player_attributes = data

    # Drop unclear data
    player_attributes = player_attributes.drop(['attacking_work_rate', 'defensive_work_rate'], axis=1)

    # Seperate data used for prediction from data not used for prediction
    non_predictable_columns = ['id', 'player_fifa_api_id', 'player_api_id', 'date']
    predictable_columns = ['overall_rating', 'potential', 'preferred_foot', 'crossing', 'finishing', 'heading_accuracy',
        'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
        'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
        'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
        'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
        'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
        'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
        'gk_reflexes']

    # Convert string data to numerical
    player_attributes['preferred_foot'] = player_attributes['preferred_foot'].map({'left': 0, 'right': 1})


    # Drop player_attributes rows with no data
    player_attributes.dropna(subset=['overall_rating'], inplace=True)

    # Iterate over each variable column and find those with missing values
    null_columns = []
    for i in predictable_columns:
        if pd.isnull(player_attributes[i]).values.any():
            null_columns.append(i)

    # Identify feature columns as those without missing values
    global features
    features = []
    for j in predictable_columns:
        if j not in(null_columns):
            features.append(j)

    # Create linear regression model for each null_column and predict missing values
    print("Predicting Missing Values in player_attributes Table...")
    for i in null_columns:
        global known_rows
        global unknown_rows
        known_rows =  player_attributes[player_attributes[i].notnull()] # Identify rows with a known target
        unknown_rows = player_attributes[player_attributes[i].isnull()] # Identify rows that need prediction
        Reg(i)


    return player_attributes

    # Write the data to a new file called "player_attributes_processed"
    #test_write = player_attributes
    #test_write.to_csv(r'C:\Users\ethan\OneDrive\Documents\Ethan NMSU Stuff\Current Classes'\
    #                                r'\C S-519 Applied Machine Learning I\Group Project\GP_Stage 4\player_attributes_processed.csv', sep=',',)