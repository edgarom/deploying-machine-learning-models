import logging
import pytest
from predict_model.config.core import config
from predict_model.processing.data_manager import load_dataset

@pytest.fixture(scope="module")
def sample_input_data():
    """
    Fixture to load sample input data for tests.
    
    This fixture will load the dataset specified by the 'test_data_file'
    in the configuration file, and return the loaded DataFrame.
    The 'scope' is set to 'module' to ensure the fixture is only called
    once per test module.
    """
    # Load the test data specified in the config file
    data = load_dataset(file_name=config.app_config.test_data_file)
    
    # Return the loaded dataset
    return data