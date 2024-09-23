import pytest
import pandas as pd
from predict_model.processing import features as pp  # Importing the transformers from features.py


@pytest.fixture
def sample_input_data():
    """Fixture that returns sample input data for testing."""
    return pd.DataFrame(
        {
            'Date': ['2016-01-01'],
            'Location': ['Albury'],
            'MinTemp': [20.4],
            'MaxTemp': [37.6],
            'Rainfall': [0.0],
            'Evaporation': [None],  # Missing value
            'Sunshine': [None],  # Missing value
            'WindGustDir': ['ENE'],
            'WindGustSpeed': [54],
            'WindDir9am': [None],  # Missing value
            'WindDir3pm': ['ESE'],
            'WindSpeed9am': [0],
            'WindSpeed3pm': [7],
            'Humidity9am': [46],
            'Humidity3pm': [17],
            'Pressure9am': [1013.4],
            'Pressure3pm': [1009.2],
            'Cloud9am': [7],
            'Cloud3pm': [3],
            'Temp9am': [26.1],
            'Temp3pm': [36.7],
            'RainToday': ['No'],
            'RainTomorrow': ['Yes'],  # Target column
        }
    )

def test_yes_no_to_binary_transformer(sample_input_data):
    """Test the YesNoToBinaryTransformer to ensure 'Yes'/'No' values are correctly converted."""
    transformer = pp.YesNoToBinaryTransformer(variables=['RainToday', 'RainTomorrow'])

    transformed_data = transformer.fit_transform(sample_input_data)

    # Assert that the transformation was applied correctly
    assert transformed_data['RainToday'].iloc[0] == 0, "RainToday should be transformed to 0"
    assert transformed_data['RainTomorrow'].iloc[0] == 1, "RainTomorrow should be transformed to 1"

def test_mapper(sample_input_data):
    """Test the Mapper to ensure categorical values are correctly mapped to numeric."""
    wind_mapping = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5,
        'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    transformer = pp.Mapper(variables=['WindGustDir', 'WindDir3pm'], mappings=wind_mapping)

    transformed_data = transformer.fit_transform(sample_input_data)

    # Assert that the mapping was applied correctly
    assert transformed_data['WindGustDir'].iloc[0] == 67.5, "WindGustDir should be mapped to 67.5 (ENE)"
    assert transformed_data['WindDir3pm'].iloc[0] == 112.5, "WindDir3pm should be mapped to 112.5 (ESE)"


