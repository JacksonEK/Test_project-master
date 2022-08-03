import pandas as pd
import numpy as np

"""
Create datasets to test the one unique value filter
"""

int_column_with_one_uniq = pd.DataFrame(
    data={"id": [1, 1, 1, 1, 1],
          "Age": [1, 1, 22, 33, 95],
          "Country": ['Japan', 'Indonesia', 'India', 'Philippines', 'Brazil']}
)

string_column_with_one_uniq = pd.DataFrame(
    data={"id": [1, 12, 13, 31, 14],
          "Age": [1, 1, 22, 33, 95],
          "Country": ['Japan', 'Japan', 'Japan', 'Japan', 'Japan']}
)

int_column_with_one_uniq_and_missing = pd.DataFrame(
    data={"id": [1, 12, 13, 31, 14],
          "Age": [1, 1, 1, 1, np.nan],
          "Country": ['Georgia', 'Ukraine', 'China', 'Belarus', 'Japan']}
)

float_column_with_one_missing = pd.DataFrame(
    data={"id": [1, 12, 13, 31, 14],
          "Age": [np.nan, np.nan, np.nan, np.nan, np.nan],
          "Country": ['Georgia', 'Ukraine', 'China', 'Belarus', 'Japan']}
)
