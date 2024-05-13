import os

def prediction_selection(data):
    while True:
        class_options = [1, 2, 3, 4]
        team_options = [1, 2]
        print('Choose Prediction: Goals = 1, Yellow Flags = 2, Red Flags = 3,  Fouls = 4')
        prediction_class = int(input('Enter here: '))
        if not prediction_class in class_options:
            print()
            print("Invalid selection. Please try again...")
            print()        
            continue
        print('Choose team: Home = 1, Away = 2')
        prediction_team = int(input('Enter here: '))
        if not prediction_team in team_options:
            print()
            print("Invalid selection. Please try again...")
            print()    
            continue
        else:
            break

    if prediction_class == 1 and prediction_team == 1:
        y = data['home_team_goal']
    if prediction_class == 1 and prediction_team == 2:
        y = data['away_team_goal']
    if prediction_class == 2 and prediction_team == 1:
        y = data['home_yellow_card_count']
    if prediction_class == 2 and prediction_team == 2:
        y = data['away_yellow_card_count']
    if prediction_class == 3 and prediction_team == 1:
        y = data['home_red_card_count']
    if prediction_class == 3 and prediction_team == 2:
        y = data['away_red_card_count']
    if prediction_class == 4 and prediction_team == 1:
        y = data['home_foul_count']
    if prediction_class == 4 and prediction_team == 2:
        y = data['away_foul_count']
    
    n_classes = len(y.unique())

    return y, n_classes