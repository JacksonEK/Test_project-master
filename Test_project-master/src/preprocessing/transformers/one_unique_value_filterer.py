from pandas import DataFrame


class OneUniqueValueFilterer:
    """
    Class to filter out all columns that only have 1 unique value
    Removes columns that are not helpful for model training
    """

    COL_AX = 'columns'

    def __init__(self):
        self.features_to_drop_list = list()

    def fit(self, df, y=None):
        df = df.loc[:, ~df.columns.duplicated()]
        for col in df.columns:
            temp_col = df[col]
            try:
                temp_col = df[col].astype(str)
            except:
                pass
            if temp_col.nunique(dropna=False) == 1:
                self.features_to_drop_list.append(col)

        return self
    """
    Add columns that only have 1 unique value to a list
            Args:
                df (DataFrame): data table to filter columns.
            Returns:
                features_to_drop_list: list of columns to drop
    """

    def transform(self, df):
        df_ = df.copy()

        columns_ls = list(filter(lambda x: x in set(self.features_to_drop_list), list(df.columns)))
        df_ = df_.drop(columns=columns_ls, axis=self.COL_AX)

        return df_

    """
            Reassigning dataframe to the new with corresponding
             columns from self.feature_list.

            Args:
                df (DataFrame): data table to choose columns from.

            Returns:
                DataFrame. Contains only columns with more than
                    one unique value.

    """