
class ImputerByMedian:
    """
    Class to impute missing data
    Missing values are replaced with the median of their column
    """

    def __init__(self):
        self.medians_dict = {}

    def fit(self, X, y=None):

        for col in X.columns:  # todo to var X[self.num_features_ls].columns
            median = X[col].median()
            self.medians_dict.update({col: median})

        return self
    """
    Calculate the median value of each column and collect them in a dictionary: medians_dict
    """

    def transform(self, X, y=None):
        X_ = X.copy()
        new_ls = X_.columns
        for col in new_ls:
            X_.loc[:, col] = X_[col].fillna(self.medians_dict[col])

        return X_
    """
    Replace missing values with the median values of their columns
    Args:
        X: Data table to impute.

    Returns:
        X_: Copy of the data with missing values imputed.
    """