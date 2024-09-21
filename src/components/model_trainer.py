import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join('artifacts', "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Entered the model training method or component")
        try:
            logging.info("Loading the preprocessor object")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "Linear": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Ridge": Ridge(),
            }
            
            params = {
                "DecisionTree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "RandomForest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            
            # Evaluating models using some custom utility function
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            
            # Find the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            
            # Predict using the best model
            predicted = best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicted)
            
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
