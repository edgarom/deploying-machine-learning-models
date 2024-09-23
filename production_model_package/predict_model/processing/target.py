from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class YesNoToBinaryTargetTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert 'Yes'/'No' values to binary 1/0 for the target."""

    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            # Convert 'Yes' to 0 and 'No' to 1 for the target
            X[feature] = (X[feature] == 'Yes') * 1  # No -> 1, Yes -> 0
        return X


class ModeImputerForTarget(BaseEstimator, TransformerMixin):
    """Custom transformer to impute missing values in the target using the mode."""

    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # Calculate the mode for each variable in the target
        self.mode_values_ = {}
        for feature in self.variables:
            self.mode_values_[feature] = X[feature].mode()[0]  # Mode of the column
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            # Fill missing values with the mode calculated in fit
            X[feature] = X[feature].fillna(self.mode_values_[feature])
        return X
