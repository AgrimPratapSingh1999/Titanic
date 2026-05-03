import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os 
import sys
from src.Titanic.utils import save_object, evaluate_models
from src.Titanic.exception import CustomException
from src.Titanic.logger import logging
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score)
from src.Titanic.components.data_transformation import DataTransformation


@dataclass
class Model_Trainer_Config:
    train_model_file_path = os.path.join ("artifacts","model.pkl")
    model_report_file_path = os.path.join("artifacts","model_report.csv")

logging.info("Model evaluation started")


class Model_Trainer:
    def __init__(self):
        self.model_trainer_config =Model_Trainer_Config()

    

    def evalution (self,actual,predicted):
        accu = accuracy_score(actual,predicted)
        precision = precision_score(actual,predicted)
      
        recall = recall_score(actual,predicted)
        f1 = f1_score(actual,predicted)
        roc=roc_auc_score(actual,predicted)

        return( accu,precision,recall,f1,roc)
    
    def Initiating_Training(self,x_train,y_train,x_test,y_test):
        try:
            models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "xgboost": XGBClassifier(),
            "Logistic regression": LogisticRegression()

            }

            param_grids = {    
              "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
              },
 
              "Decision Tree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
              },

             "xgboost": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
               },
    
             "Logistic regression": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"],
            "max_iter": [200, 500, 1000]
                           }
                   }
            
            model_report = evaluate_models(x_train,y_train,x_test,y_test,models,param_grids)
            print( model_report)
            model_report_df= pd.DataFrame(list(model_report.items()),columns=["model","score"])
            model_report_df.to_csv(self.model_trainer_config.model_report_file_path,index= True)

            return model_report,model_report_df
        


        except Exception as e:
            raise CustomException(e,sys)
        logging.info("model report save in artifact folder")