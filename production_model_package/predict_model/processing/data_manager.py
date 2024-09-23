import typing as t
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from predict_model.processing.target import YesNoToBinaryTargetTransformer, ModeImputerForTarget
from predict_model import __version__ as _version
from predict_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


# Function to count null values in the dataframe
def count_nulls_by_line(dataframe: pd.DataFrame) -> pd.Series:
    return dataframe.isnull().sum().sort_values(ascending=False)


# Function to calculate the percentage of null values in the dataframe
def null_percent_by_line(dataframe: pd.DataFrame) -> pd.Series:
    return (dataframe.isnull().sum() / dataframe.isnull().count()).sort_values(ascending=False)


# Function to clean and prepare the dataframe before the pipeline processing
def pre_pipeline_preparation(*, dataframe: pd.DataFrame) -> pd.DataFrame:
    zeros_cnt = count_nulls_by_line(dataframe)
    percent_zeros = null_percent_by_line(dataframe)
    
    # Combine total and percentage of nulls in one dataframe
    missing_data = pd.concat([zeros_cnt, percent_zeros], axis=1, keys=["Total", "Percent"])
    
    # Drop columns with more than 30% missing data
    drop_list = list(missing_data[missing_data["Percent"] > 0.3].index)
    dataframe.drop(drop_list, axis=1, inplace=True)
    
    # Drop columns specified in the config
    dataframe.drop(labels=config.modeldt_parameters.unused_fields, axis=1, inplace=True)

    return dataframe


# Load the dataset and apply pre-processing before the pipeline
def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file, ensuring only valid columns (present in both the data and config) are selected.
    """
    # Load the raw data
    raw_data = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))


    # Perform preprocessing (e.g., drop columns with too many missing values)
    preprocessed_data = pre_pipeline_preparation(dataframe=raw_data)
    
    # Get the list of features from the config
    required_features = config.modeldt_parameters.features
    
    # Extract the valid features that exist in both the preprocessed data and the config
    valid_features = [feature for feature in required_features if feature in preprocessed_data.columns]

    # Include the target column if it still exists in the data
    #if config.model_parameters.target in preprocessed_data.columns:
    #   valid_features.append(config.model_parameters.target)
    
    # Select only the valid features
    X = preprocessed_data[valid_features]

    # Process target (y)
    y = raw_data[[config.modeldt_parameters.target]].copy()

    # === Transform the target variable (y) ===
    target_transformer = YesNoToBinaryTargetTransformer(variables=[config.modeldt_parameters.target])
    y_transformed = target_transformer.fit_transform(y)

    # === Impute missing values in the target ===
    target_imputer = ModeImputerForTarget(variables=[config.modeldt_parameters.target])
    y_imputed = target_imputer.fit_transform(y_transformed)
    
    return X, y_imputed

def _load_test_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


# Save the pipeline, versioned, and remove old pipelines
def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


# Load a previously saved pipeline
def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


# Remove old pipelines, keeping the specified files
def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
