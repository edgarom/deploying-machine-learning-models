from config.core import config
from pipeline import weather_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split



def run_training() -> None:
    """Train the model."""

    # Load training data and extract predictors (X) and target (y)
    X,y = load_dataset(file_name=config.app_config.raw_data_file)
    
    

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.modeldt_parameters.test_size,
        random_state=config.modeldt_parameters.random_state,  # For reproducibility
    )


    # Train the model
    weather_pipe.fit(X_train, y_train)

    # Save the trained model
    save_pipeline(pipeline_to_persist=weather_pipe)


if __name__ == "__main__":
    run_training()
