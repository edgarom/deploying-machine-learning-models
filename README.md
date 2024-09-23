# Weather Prediction Model

This repository contains a weather prediction model built using machine learning techniques and a model serving API. The project is structured into two main components: a model training package (`production_model_package`) and a model serving API (`model_serving_api`). 


Why do I structure this project things this way?

I tried to implement the best practices DevOps, what mean that implement `convetions` like versions, linting tools and config. `packing mandatory files` setup and MANIFEST. by the other hand `software engineering best practices` in two layers packing (split training, predict, evaluation) and general (test abilility, manager data validations). the modules could be connected using a setup `pip install -e` 


## Table of Contents

- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Model Serving](#model-serving)
- [Setup Instructions](#setup-instructions)
- [Running Tests](#running-tests)
- [Usage](#usage)

---

## Project Structure

```bash
.
├── evaluacion-mle-wm
│   ├── modelo_DS.ipynb          # Jupyter Notebook for Data Science model evaluation shared by D.S Team
│   ├── modelo_DS.zip            # Compressed Notebook file
│   ├── README.md                # Readme file for model evaluation, Check Insighs Section  
│   └── weatherAUS.csv           # Dataset for weather in Australia
├── model_serving_api
│   ├── app
│   │   ├── api.py               # API endpoints for model serving
│   │   ├── config.py            # Configuration for API
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── schemas              # Schema definitions for API
│   │   │   ├── health.py        # Health check schema
│   │   │   ├── __init__.py
│   │   │   └── predict.py       # Schema for prediction request
│   │   └── tests
│   │       ├── conftest.py      # Pytest fixtures
│   │       ├── __init__.py
│   │       └── test_api.py      # Tests for API
│   ├── mypy.ini                 # Static type checker configuration
│   ├── Procfile                 # Procfile for deployment
│   ├── requirements.txt         # Python dependencies
│   ├── test_requirements.txt    # Test dependencies
│   └── tox.ini                  # Tox configuration for testing
├── production_model_package
│   ├── predict_model
│   │   ├── config               # Model configuration files
│   │   │   ├── core.py
│   │   │   ├── __init__.py
│   │   ├── datasets             # Datasets for training and testing
│   │   ├── pipeline.py          # Pipeline for model training
│   │   ├── predict.py           # Prediction functions
│   │   ├── processing           # Data processing modules
│   │   ├── trained_models       # Directory for storing trained models
│   │   ├── train_pipeline.py    # Script for training the pipeline
│   │   └── VERSION              # Version of the model
│   ├── requirements             # Python dependency files
│   ├── tests                    # Unit tests for the model
│   ├── tox.ini                  # Tox configuration for testing
└── README.md                    # This README file

18 directories, 65 files
```


## Model Training

The production_model_package contains all the necessary components to train and evaluate the machine learning model. The pipeline includes data processing, feature engineering, model training, and evaluation.

Key components:

config/: Contains configuration settings for the model.
datasets/: Includes the dataset for training and testing the model.
processing/: Contains transformers for feature engineering and data validation.
train_pipeline.py: Script to train the model using a decision tree.
predict.py: Used for making predictions with the trained model.

How to Train the Model
To train the model, run the following command from the root directory:
`python production_model_package/predict_model/train_pipeline.py`

# Model Serving

The model_serving_api directory contains a FastAPI application that serves the trained model and exposes an endpoint for making predictions.

Key components:

api.py: Defines the prediction endpoint.
main.py: Entry point for running the FastAPI server.
schemas/: Defines the input and output schema for the prediction endpoint.
tests/: Unit tests for the API.

# How to Run the API
To run the FastAPI server locally, execute:

`uvicorn model_serving_api.app.main:app --reload`
Once running, the API can be accessed at http://127.0.0.1:8000. The prediction endpoint is located at /predict.

## Setup Instructions

Clone the Repository
`git clone <repository-url>`
`cd <repository-directory>`

## Install Dependencies

You can install the required dependencies for the model training and API serving using:
`pip install -r production_model_package/requirements/requirements.txt`
`pip install -r model_serving_api/requirements.txt`


## Running Tests with Tox

We have a tox.ini file set up for running tests in isolated environments. You can run tests using Tox:

`cd production_model_package`
`tox -e train`

---

### Key Sections:
- **Project Structure**: A detailed breakdown of the directory and its contents.
- **Model Training**: Instructions on how to train the model.
- **Model Serving**: Details on running the API to serve the trained model.
- **Setup Instructions**: Steps to set up the environment, install dependencies, and run tests.
- **Usage**: Explanation of how to train and serve the model. 

