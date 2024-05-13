import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

def lda_reduction (df, n_classes, y):   
    print("Reducing Dimensions Using LDA...")
    # Read in the data and rearrange the columns so that the prediction variables can be selected easily
    df = df.iloc[:, [0, 1, 20, 21, 22, 25, 26, 27, 28, 33, 34] + list(range(2, 20)) + [23, 24] + list(range(29, 33)) + list(range(35, len(df.columns)))]
    X = df.iloc[:, list(range(11, len(df.columns)))]

    # Standardize the data
    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    # Apply LDA with 20 components
    lda = LDA(n_components=(n_classes-1))
    X_std_lda = lda.fit_transform(X_std, y)

    # Convert LDA results into a new DataFrame
    lda_df = pd.DataFrame(X_std_lda, columns=[f'PC{i+1}' for i in range(n_classes-1)])
    lda_df.to_csv('lda_reduced_X.csv', sep =',', index=False)

    return lda_df
