Major League Soccer Match Outcome Prediction

Objective

Develop a machine learning model that accurately predicts the outcomes of Major League Soccer matches, including the final score, based on comprehensive datasets encompassing match conditions, team dynamics, and individual player performance metrics.

Potential Value

Providing sports analysts, betting companies, and soccer enthusiasts with predictive insights into match outcomes.
Enhancing strategic betting, fan engagement, and broadcasting strategies.
Offering teams and coaches data-driven insights into performance, vulnerabilities, and strategy effectiveness.

Data

Overview
Source: European Soccer Database from Kaggle.

Contents:

11 countries, 11 leagues, 25,979 matches, 11,060 players, and 299 teams.
7 tables with varying dimensions and instances.
Record dates range from 2007 to 2016.

Details of Each Dataset

Country: Unique country ID, country name.
League: Unique league ID, country ID, league name.
Match: Preliminary data including team, league, country, and match IDs; date; season; final score; player positions; player IDs; XML data for goals, shots, fouls, cards, etc.; betting odds.
Player: Player data including ID, name, birthday, height, weight.
Player Attributes: Player ratings on a scale of 1-100 for various attributes (sourced from FIFA video game).
Team: Team data including IDs, full name, shortened name.
Team Attributes: Various team metrics (sourced from FIFA video game).
Data Completeness
Missing Values:
Country: None
League: None
Match: Several columns contain missing values, but enough data remains for analysis.
Player: None
Player Attributes: Missing values exist but can be predicted.
Team: 11 missing FIFA IDs
Team Attributes: Missing values for “buildUpPlayDribbling” column only.

Code & Methodology

Loading & Cleaning Match Data: Removed rows missing vital data, removed irrelevant columns, and filtered matches before February 22, 2010.
Extracting XML Data: Parsed XML data for goals, fouls, and cards into new columns.
Creating New Features: Calculated team performance from past 10 matches for goals, fouls, and cards.
Predicting Missing Data: Used various regression models to predict missing player attribute values.
Combining Datasets: Merged player and team attributes with match data.
Reducing Dimensionality: Tested PCA, LDA, and t-SNE, with LDA performing the best.
Prediction: Utilized SVM for classification and regression, achieving the best results.

Conclusion

The dataset is comprehensive but required extensive cleaning and preprocessing. SVM performed best for prediction, offering valuable insights into match outcomes.

