from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from predict_model.config.core import config
from predict_model.processing import features as pp

# Feature transformation pipeline (without target)
weather_pipe = Pipeline(
    [   
        # === Transform categorical 'Yes/No' to numerical values ===
        (
            "yes_no_to_binary",
            pp.YesNoToBinaryTransformer(
                variables=config.modeldt_parameters.categorical_vars
            ),
        ),
        # === MAPPER FIRST ===
        (
            "mapper_wind",
            pp.Mapper(
                variables=config.modeldt_parameters.wind_vars,
                mappings=config.modeldt_parameters.wind_to_assign,
            ),
        ),
        # ===== IMPUTATION =====
        # Add missing indicator for numerical features
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.modeldt_parameters.numerical_vars_with_na),
        ),
        # Use median imputation for numerical features due to skewness
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.modeldt_parameters.numerical_vars_with_na,
            ),
        ),
        # === MODEL ===
        (
            "DecisionTree",
            DecisionTreeClassifier(
                max_leaf_nodes=config.modeldt_parameters.max_leaf_nodes,
                random_state=config.modeldt_parameters.random_state,
            ),
        ),
    ]
)