import json
from zenml import step
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from typing_extensions import Annotated
import pandas as pd

#Defining the config file name
CONFIG_FILE = 'config.json'


"""
A config file is loaded (or created if it doesnt exist yet)
The config file is needed for the model evaluation and hyperparameter tuning.
It safes the best working model out of multiple different variants and has an 
option to decide if all variants get trained or only the one that worked best.

"""
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
"""
train_model loads the config to decide which model to use or if it uses all models and tests which is best. 
IF "test_all_models": True then it only trains with this one, otherwise it trains all.
"""
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

    """
    Different model variations to identify the best working model and the best hyperparameters
    """

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

    """
    Training all or just the best model based on the config file
    """

    if config["test_all_models"]:
        best_model = None
        best_score = -float('inf')
        best_params = None
        best_model_name = None

        
        for model_name in models:
            model = models[model_name]
            param_grid = param_grids[model_name]



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

        #Save the config with the best model and best hyperparamets according to the trainings from before
        save_config(config)


    else:
        best_model_name = config["best_model_name"]
        best_params = config["best_params"]
        best_model = models[best_model_name].set_params(**best_params)
        best_model.fit(X_train, y_train)


    #The best model is returned for training
    return best_model
