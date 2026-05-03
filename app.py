from src.Titanic.logger import logging
from src.Titanic.exception import CustomException
from src.Titanic.utils import read_sql_data
from src.Titanic.components.data_ingestion import DataIngestionConfig, DataIngestion
import sys
from src.Titanic.components.data_transformation import DataTransformationConfig,DataTransformation
from src.Titanic.components.model_trainer import Model_Trainer


if __name__=="__main__":
    logging.info("Model training Pipeline execution started")

    try:
        data_ingestion = DataIngestion()
        x_train_path,y_train_path,x_test_path,y_test_path = data_ingestion.initiate_data_ingestion()

        data_transformation_config = DataTransformationConfig()

        data_transformation = DataTransformation()
        x_train,y_train,x_test,y_test = data_transformation.transformation_apply(x_train_path,y_train_path,x_test_path,y_test_path)

        model_trainer = Model_Trainer()
        model_trainer.Initiating_Training(x_train,y_train,x_test,y_test)


    except Exception as e:
        logging.error(f"this is reasion of error{e}")
        raise CustomException (e,sys)