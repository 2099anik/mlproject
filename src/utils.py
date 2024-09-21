import os 
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        for model_name, model in models.items():
            #logging.info(f"Training {model_name}...")
            para = param.get(model_name, {})
            
            # GridSearchCV with 3-fold cross-validation
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)
            
            # Set best parameters to the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
            
            #logging.info(f"{model_name} | Train R2 Score: {train_model_score}, Test R2 Score: {test_model_score}")
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
