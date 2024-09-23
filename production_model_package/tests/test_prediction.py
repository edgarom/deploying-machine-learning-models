import pytest
import pandas as pd
from predict_model.config.core import config
from predict_model.predict import make_prediction

@pytest.fixture()
def test_prediction_on_sample_data(sample_input_data):
    """
    Test the prediction function to ensure it runs without errors
    and produces valid output on the sample data.
    """
    # Act: Make predictions
    predictions = make_prediction(sample_input_data)
    
    # Assert: Check predictions are valid
    assert predictions is not None
    assert len(predictions) == len(sample_input_data)