from src.Titanic.logger import logging
from src.Titanic.exception import CustomException
from src.Titanic.utils import read_sql_data
from src.Titanic.components.data_ingestion import DataIngestionConfig, DataIngestion
import sys


if __name__=="__main__":
    logging.info("Model training Pipeline execution started")

    try:
        data_ingestion = DataIngestion()
        x_test_path,y_test_path,x_train_path,y_train_path = data_ingestion.initiate_data_ingestion()
        

    except Exception as e:
        logging.error(f"this is reasion of error{e}")
        raise CustomException (e,sys)