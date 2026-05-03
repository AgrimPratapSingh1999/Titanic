import pymysql
from dotenv import load_dotenv
import pandas as pd
import os
import sys
from src.Titanic.exception import CustomException
from src.Titanic.logger import logging
import pickle

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
    

