from src.preprocessing.transformers.one_unique_value_filterer import OneUniqueValueFilterer
from tests.preprocessing.data.transformers import one_unique_value_filterer_datasets as datasets
import collections


class TestOneUniqueValueFilterer:
    """
    Test the one unique value filter on various datasets to see ensure it
    properly identifies and drops features with only one unique value
    """

    def test_int_column_with_all_uniq_result_drop_column(self):
        df, res = self.fit_transform(datasets.int_column_with_one_uniq)
        assert list(res.columns) == ["Age", "Country"]
        assert len(df) == len(res)

    def test_string_column_with_all_uniq_result_drop_column(self):
        df, res = self.fit_transform(datasets.string_column_with_one_uniq)
        assert list(res.columns) == ["id", "Age"]
        assert len(df) == len(res)

    def test_int_column_with_all_uniq_and_missing_result_not_drop_column(self):
        df, res = self.fit_transform(datasets.int_column_with_one_uniq_and_missing)
        assert list(res.columns) == ["id", "Age", "Country"]
        assert len(df) == len(res)

    def test_float_column_with_all_missing_result_drop_column(self):
        df, res = self.fit_transform(datasets.float_column_with_one_missing)
        assert collections.Counter(res.columns) == collections.Counter(["id", "Country"])
        assert len(df) == len(res)

    def fit_transform(self, df):
        filterer = OneUniqueValueFilterer()
        filterer.fit(df)
        res = filterer.transform(df)

        return df, res
