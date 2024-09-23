from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



class YesNoToBinaryTransformer(BaseEstimator, TransformerMixin):
    """Transformer to convert 'Yes'/'No' values to binary 1/0."""

    def __init__(self, variables: List[str]):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables

    def fit(self,X: pd.DataFrame, y: pd.Series = None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            # Convert 'Yes' to 1 and anything else (including 'No') to 0
            X[feature] = (X[feature] == 'Yes') * 1
        return X

class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: List[str], mappings: dict):

        if not isinstance(variables, list):
            raise ValueError("variables should be a list")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings).astype(float)

        return X