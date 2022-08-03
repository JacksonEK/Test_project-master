import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class MinMaxScalerCustom:
    """
    Class to scale each feature to a given range
    Normalizes the data
    """

    def __init__(self):
        self.columns_scaler_dict = {}

    def fit(self, X, y=None):

        for column in X.columns:
            column_values_array = np.array(X[column]).reshape(-1, 1)
            scaler = MinMaxScaler().fit(column_values_array)
            self.columns_scaler_dict[column] = scaler

        return self
    """
    Reshape the data into one column
    Fit the scaler to the data and create a new column of scaled data: columns_scaler_dict
    """

    def transform(self, X, y=None):
        X_copy = pd.DataFrame.copy(X)
        for column in X_copy.columns:
            column_values_array = np.array(X_copy[column]).reshape(-1, 1)
            X_copy[column] = self.columns_scaler_dict[column].transform(
                column_values_array)

        return X_copy
    """
    Create a copy of the data and scale the copy
    Args:
        X: Data table to scale.
    Returns:
        x_copy: Scaled copy of the data.
    """

    def inverse_transform(self, X, y=None):
        X_copy = pd.DataFrame.copy(X)
        for column in X_copy.columns:  # TODO: 'if' statement is not necessary here (only temporary fix)
            if column in self.columns_scaler_dict.keys():
                column_values_array = np.array(X_copy[column]).reshape(-1, 1)
                X_copy[column] = self.columns_scaler_dict[column].inverse_transform(column_values_array)
        return X_copy
    """
    Create a copy of the scaled data and inversely transform it to return it to its non-scaled form
    Args:
        X: Data table to undo scaling.
    Returns:
        x_copy: Copy of the data with scaling reversed.
    """