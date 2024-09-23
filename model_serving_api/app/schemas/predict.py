from typing import Any, List, Optional

from pydantic import BaseModel
from production_model_package.predict_model.processing.validation import WeatherInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class MultipleHouseDataInputs(BaseModel):
    inputs: List[WeatherInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "MinTemp": 10.0,
                        "MaxTemp": 25.0,
                        "Rainfall": 5.0,
                        "Evaporation": 2.0,
                        "Sunshine": 7.0,
                        "WindGustDir": "N",
                        "WindGustSpeed": 35,
                        "WindDir9am": "SE",
                        "WindDir3pm": "N",
                        "WindSpeed9am": 10,
                        "WindSpeed3pm": 15,
                        "Humidity9am": 80,
                        "Humidity3pm": 60,
                        "Pressure9am": 1010,
                        "Pressure3pm": 1005,
                        "Cloud9am": 5,
                        "Cloud3pm": 3,
                        "Temp9am": 15.0,
                        "Temp3pm": 20.0,
                        "RainToday": "Yes"
                        
                    }
                ]
            }
        }