import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
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
import dagshub
import mlflow.sklearn



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
                         #     Best model name based on highest test accuracy
            best_model_name = max(
                          model_report,
                          key=lambda model: model_report[model]["test_accuracy"]
                      )

                                   # Best model score
            best_model_score = model_report[best_model_name]["test_accuracy"]

                   # Best model object
            best_model_object = models[best_model_name]

                      # Best hyperparameters
            best_params = model_report[best_model_name]["best_params"]

                      # Print results
            print(f"Best Model: {best_model_name}")
            print(f"Best Test Accuracy: {best_model_score}")
            print(f"Best Params: {best_params}")


            dagshub.init(repo_owner='agrimsingh19992207-web', repo_name='Titanic', mlflow=True)

            mlflow.set_experiment("Titanic_Classification")

            with mlflow.start_run():
              

              best_model_object.set_params(**best_params)
              best_model_object.fit(x_train, y_train)
              predict_qualities = best_model_object.predict(x_test)


              accu,precision,recall,f1,roc = self.evalution(y_test,predict_qualities)
              mlflow.log_metric("accu", accu)
              mlflow.log_metric("precision",precision)
              mlflow.log_metric("recall",recall)
              mlflow.log_metric("f1",f1)
              mlflow.log_metric("roc",roc)
              mlflow.log_param("best_params",best_params)
              

              if best_model_name == "xgboost":
                  mlflow.xgboost.log_model(best_model_object,artifact_path="best_model")
              else:
                 mlflow.sklearn.log_model(best_model_object,artifact_path="best_model")


            logging.info("model , params and metrics  log into mlflow")


            save_object(
                          file_path=self.model_trainer_config.train_model_file_path,
                          obj=best_model_object
                                                   )

        except Exception as e:
            raise CustomException(e,sys)