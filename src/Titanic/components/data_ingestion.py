import pandas as pd
from sklearn.model_selection import train_test_split
from src.Titanic.logger import logging
from src.Titanic.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os
from src.Titanic.utils import read_sql_data
import sys



@dataclass
class DataIngestionConfig:
    x_train_path = os.path.join('artifacts',"x_train.csv")
    y_train_path = os.path.join ("artifacts","y_train.csv")
    x_test_path = os.path.join ("artifacts","x_test.csv")
    y_test_path = os.path.join ("artifacts","y_test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self): 
            
            try:
                df = read_sql_data()
                logging.info("reading complete data")

                os.makedirs(os.path.dirname(self.ingestion_config.x_train_path),exist_ok=True)
                logging.info("artifact created")
                
                df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

                #train test split

                x_train,x_test,y_train,y_test = train_test_split(df.drop(columns =["Survived"]),df["Survived"] ,test_size=0.25,random_state=42)

                                  #save splited data to csv

                x_train.to_csv(self.ingestion_config.x_train_path,index=False,header=True)
                x_test.to_csv(self.ingestion_config.x_test_path,index=False,header=True)

                y_train.to_csv(self.ingestion_config.y_train_path,index = False,header = True)
                y_test.to_csv(self.ingestion_config.y_test_path,index = False,header= True)

                logging.info("data ingestion completed")

                return (
                     
                    self.ingestion_config.x_train_path,
                    self.ingestion_config.y_train_path,

                    self.ingestion_config.x_test_path,
                    self.ingestion_config.y_test_path
                    )
            except Exception as e:
                 logging.error(f"unexcepted error occured {e}")
                 raise CustomException (e,sys)
            



cof =DataIngestion()



if __name__ == "__main__":
    logging.info("Data ingestion pipeline started")
    cof.initiate_data_ingestion()