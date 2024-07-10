import json
from zenml import step
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from typing_extensions import Annotated
import pandas as pd

CONFIG_FILE = 'config.json'

def load_config():
    try:
        with open(CONFIG_FILE, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        config = {
            "test_all_models": True,
            "best_model_name": None,
            "best_params": None
        }
    return config

def save_config(config):
    with open(CONFIG_FILE, 'w') as file:
        json.dump(config, file)

@step
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Annotated[RandomForestRegressor, "model"]:
    """
    Trains a model using the given training data with hyperparameter tuning.
    """
    config = load_config()

    # Define the models and their hyperparameters
    models = {
        'RandomForest': RandomForestRegressor(),
        'LinearRegression': LinearRegression(),
        'SVR': SVR()
    }

    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'LinearRegression': {
            'fit_intercept': [True, False],
        },
        'SVR': {
            'kernel': ['linear', 'rbf'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    }

    if config["test_all_models"]:
        best_model = None
        best_score = -float('inf')
        best_params = None
        best_model_name = None
        i = 1
        
        for model_name in models:
            model = models[model_name]
            param_grid = param_grids[model_name]

            print("Durchlauf: " + str(i) + " Modelname: " + model_name)
            i += 1

            # Use GridSearchCV for hyperparameter tuning
            search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
            search.fit(X_train, y_train)

            # Output the best parameters and score for each model
            print(f"Best parameters for {model_name}: {search.best_params_}")
            print(f"Best score for {model_name}: {search.best_score_}")

            # Check if this model is better than the best one found so far
            if search.best_score_ > best_score:
                best_model = search.best_estimator_
                best_score = search.best_score_
                best_params = search.best_params_
                best_model_name = model_name

        config["best_model_name"] = best_model_name
        config["best_params"] = best_params
        config["test_all_models"] = False
        save_config(config)

        print(f"Best model: {best_model_name}")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score}")

    else:
        best_model_name = config["best_model_name"]
        best_params = config["best_params"]
        best_model = models[best_model_name].set_params(**best_params)
        best_model.fit(X_train, y_train)

        print(f"Using best model from previous run: {best_model_name}")
        print(f"With parameters: {best_params}")

    return best_model
