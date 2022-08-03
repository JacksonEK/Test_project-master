
class TestTargetEncoderCustom:
    """
    Test the target encoder to ensure it is transforming the data as intended
    """

    def test_categorical_columns_result_encoded(self):
        df, encoded = self.fit_transform()

    def fit_transform(self, df):
        encoder = TargetEncoderCustom()
        encoder.fit(X=df['X'], y=df['y'])
        encoded = encoder.transform(df['X'])

        return df, encoded
