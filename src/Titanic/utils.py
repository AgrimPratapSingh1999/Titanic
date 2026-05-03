import pymysql
from dotenv import load_dotenv
import pandas as pd
import os
import sys
from src.Titanic.exception import CustomException
from src.Titanic.logger import logging
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
load_dotenv()
host = os.getenv("MYSQL_HOST")
password = os.getenv("MYSQL_PASSWORD")
user = os.getenv("MYSQL_USER")
port = os.getenv("MYSQL_PORT")
db = os.getenv("MYSQL_DB")


def read_sql_data():
    logging.info("start reading sql database ")

    try:
        titanic = pymysql.connect(
            host=host,
            user = user,
            port=int(port),
            db=db,
            password=password
        )
        logging.info("connection established")
        
        df= pd.read_sql_query("SELECT * FROM `titanic-dataset`",titanic)
        logging.info("Data loaded successfully")
        print(df.head(3))
        

        return df
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise CustomException (e,sys)
    

def save_object(file_path,obj):
    try:
        dir_path  = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb")as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:

        logging.error(f" error reasion is {e}")
        raise CustomException(e,sys)
    


   
def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}

        # Loop through each model
        for model_name, model in models.items():

            # Get corresponding parameter grid
            para = param[model_name]

            # GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )

            # Train GridSearchCV
            gs.fit(x_train, y_train)

            # Best model after tuning
            best_model = gs.best_estimator_

            # Predictions
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            # Accuracy scores
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            # Store results
            report[model_name] = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "best_params": gs.best_params_
            }

        return report

    except Exception as e:
        raise Exception(f"Error in model evaluation: {e}")    

