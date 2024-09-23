import pandas as pd
from typing import List, Tuple, Optional, Dict
from pydantic import BaseModel
from predict_model.config.core import config

def apply_wind_direction_mapping(input_data: pd.DataFrame) -> pd.DataFrame:
    """Map wind direction features to their corresponding numeric values."""
    
    # Wind direction mapping from config
    wind_mapping = config.modeldt_parameters.wind_to_assign
    
    # Apply mapping to wind direction columns
    for col in config.modeldt_parameters.wind_vars:
        input_data[col] = input_data[col].map(wind_mapping)
    
    return input_data

def drop_na_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values in non-optional fields."""

    # List required fields for validation
    required_columns = config.modeldt_parameters.features  # Ensure this is defined in your config

    # Only drop rows where the essential fields have missing values
    validated_data = input_data.copy()
    validated_data.dropna(subset=required_columns, inplace=True)
    
    return validated_data

def validate_inputs(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict[str, int]]]:
    """Check model inputs for unprocessable values (like NaNs) and return errors if found."""
    
    # Apply mapping for wind directions
    relevant_data = input_data.copy()
    relevant_data = apply_wind_direction_mapping(relevant_data)
    
    # Drop rows with missing values in non-optional fields
    validated_data = drop_na_inputs(input_data=relevant_data)
    
    # Check for remaining missing values in all columns (numeric and non-numeric)
    missing_values = validated_data.isnull().sum()
    
    errors = None
    if missing_values.sum() > 0:
        # Return errors as a dictionary of columns with missing values
        errors = missing_values[missing_values > 0].to_dict()
    
    return validated_data, errors

class WeatherInputSchema(BaseModel):
    MinTemp: Optional[float]
    MaxTemp: Optional[float]
    Rainfall: Optional[float]
    Evaporation: Optional[float]
    Sunshine: Optional[float]
    WindGustDir: Optional[str]
    WindGustSpeed: Optional[float]
    WindDir9am: Optional[str]
    WindDir3pm: Optional[str]
    WindSpeed9am: Optional[float]
    WindSpeed3pm: Optional[float]
    Humidity9am: Optional[float]
    Humidity3pm: Optional[float]
    Pressure9am: Optional[float]
    Pressure3pm: Optional[float]
    Cloud9am: Optional[float]
    Cloud3pm: Optional[float]
    Temp9am: Optional[float]
    Temp3pm: Optional[float]
    RainToday: Optional[str]

class WeatherDataInputs(BaseModel):
    inputs: List[WeatherInputSchema]