import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, RANSACRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVC





def prediction(df, y):
    # Load the dataset
    X_std_lda = pd.read_csv('lda_reduced_X.csv')
    #y_options = df[['home_team_goal', 'away_team_goal', 'goal_count', 'foulcommit_count', 'card_count', 'home_yellow_card_count', 'away_yellow_card_count',\
    #    'home_red_card_count', 'away_red_card_count', 'home_foul_count', 'away_foul_count']]
    y = y.values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_std_lda, y, test_size=0.3, random_state=1)

    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=15, p = 1, metric='minkowski')
    knn.fit(X_train, y_train)
    y_pred_train_knn = knn.predict(X_train)
    y_pred_test_knn = knn.predict(X_test)

    # Determine algorithm accuracy
    print()
    print('Train Accuracy KNN: %.3f' % accuracy_score(y_train, y_pred_train_knn))
    print('Test Accuracy KNN: %.3f' % accuracy_score(y_test, y_pred_test_knn))


    # Create Random Forest Classifier
    forest_c = RandomForestClassifier(n_estimators=500, random_state=1, n_jobs=-1)
    forest_c.fit(X_train, y_train)
    y_pred_train_forest_c = forest_c.predict(X_train)
    y_pred_test_forest_c = forest_c.predict(X_test)

    # Determine algorithm accuracy
    print()
    print('Train Accuracy Forest Classifier: %.3f' % accuracy_score(y_train, y_pred_train_forest_c))
    print('Test Accuracy Forest Classifier: %.3f' % accuracy_score(y_test, y_pred_test_forest_c))
    print()

    # Create SVM Classifier
    svm = SVC(kernel = 'linear', random_state=1, C=1)
    svm.fit(X_train, y_train)
    y_pred_train_svm = svm.predict(X_train)
    y_pred_test_svm = svm.predict(X_test)

    # Determine algorithm accuracy
    print('Train Accuracy SVM: %.3f' % accuracy_score(y_train, y_pred_train_svm))
    print('Test Accuracy SVM: %.3f' % accuracy_score(y_test, y_pred_test_svm))
    print()

    # Create Random Forest Regressor
    forest_r = RandomForestRegressor(n_estimators=25, random_state=1, n_jobs=-1, criterion='squared_error')
    forest_r.fit(X_train, y_train)
    y_pred_train_forest_r = forest_r.predict(X_train)
    y_pred_test_forest_r = forest_r.predict(X_test)

    y_pred_test_forest_r_rounded = np.round(y_pred_test_forest_r, 0)
    y_pred_test_forest_r_rounded = y_pred_test_forest_r_rounded.astype(int)
    y_pred_train_forest_r_rounded = np.round(y_pred_train_forest_r, 0)
    y_pred_train_forest_r_rounded = y_pred_train_forest_r_rounded.astype(int)

    # Measure mean squared error for Forest
    mse_train_forest_r = mean_absolute_error(y_train, y_pred_train_forest_r)
    mse_test_forest_r = mean_absolute_error(y_test, y_pred_test_forest_r)
    print('Forest Regressor', f'MSE train: {mse_train_forest_r:.2f}, test: {mse_test_forest_r:.2f}')

    # Measure R^2 for Forest 
    train_r2_forest_r = r2_score(y_train, y_pred_train_forest_r)
    test_r2_forest_r = r2_score(y_test, y_pred_test_forest_r)
    print('Forest Regressor', f'R^2 train: {train_r2_forest_r:.2f}, test: {test_r2_forest_r:.2f}')  

    # Determine algorithm accuracy
    print('Train Accuracy Forest Regressor: %.3f' % accuracy_score(y_train, y_pred_train_forest_r_rounded))
    print('Test Accuracy Forest Regressor: %.3f' % accuracy_score(y_test, y_pred_test_forest_r_rounded))
    print()


    # Create Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_train_lr = lr.predict(X_train)
    y_pred_test_lr = lr.predict(X_test)

    y_pred_test_lr_rounded = np.round(y_pred_test_lr, 0)
    y_pred_test_lr_rounded = y_pred_test_lr_rounded.astype(int)
    y_pred_train_lr_rounded = np.round(y_pred_train_lr, 0)
    y_pred_train_lr_rounded = y_pred_train_lr_rounded.astype(int)

    # Measure mean squared error for Linear Regression
    mse_train_lr = mean_absolute_error(y_train, y_pred_train_lr)
    mse_test_lr = mean_absolute_error(y_test, y_pred_test_lr)
    print('Linear Regression', f'MSE train: {mse_train_lr:.2f}, test: {mse_test_lr:.2f}')

    # Measure R^2 for Linear Regression 
    train_r2_lr = r2_score(y_train, y_pred_train_lr)
    test_r2_lr = r2_score(y_test, y_pred_test_lr)
    print('Linear Regression', f'R^2 train: {train_r2_lr:.2f}, test: {test_r2_lr:.2f}') 

    # Determine algorithm accuracy
    print('Train Accuracy Linear Regression: %.3f' % accuracy_score(y_train, y_pred_train_lr_rounded))
    print('Test Accuracy Linear Regression: %.3f' % accuracy_score(y_test, y_pred_test_lr_rounded))
    print()



