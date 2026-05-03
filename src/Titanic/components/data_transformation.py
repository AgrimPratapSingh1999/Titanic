import pandas as pd 
from dataclasses import dataclass
import numpy as np
from  src.Titanic.logger import logging
from src.Titanic.exception import CustomException
from src.Titanic.components.data_ingestion import DataIngestion
from src.Titanic.utils import save_object
import os
from sklearn.pipeline import Pipeline
from sklearn .compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import sys
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_object(self):
        try:

            numerical_columns  = ["Age","SibSp","Parch","Fare","Pclass"]
            categorical_columns = ["Sex","Embarked"]


            num_pipeline = Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="mean")),
               ( "scalar",StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("simputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scalar",StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")


            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ],remainder="drop")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def transformation_apply(self,x_train_path,y_train_path ,x_test_path,y_test_path):
        try:
            x_train_df = pd.read_csv(x_train_path)
            y_train_df = pd.read_csv(y_train_path)
            
            x_test_df = pd.read_csv(x_test_path)
            y_test_df = pd.read_csv(y_test_path)



            logging.info("Reading the all dataset ")
            
            preprocessing_object = self.get_data_transformation_object()

            training_features = preprocessing_object.fit_transform(x_train_df)
            testing_features = preprocessing_object.transform(x_test_df)

            training_labels = np.array(y_train_df).ravel()
            testing_labels = np.array(y_test_df).ravel()


            logging.info("saved preprocessing obejct")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )


            return (training_features,training_labels,testing_features,testing_labels)


        except Exception as e:
            logging.error(f"something bad happen{e}")

            raise CustomException(e , sys)
        
        

        