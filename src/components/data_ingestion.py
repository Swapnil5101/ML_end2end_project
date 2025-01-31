import os
import sys
from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))
     ## OR ##
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # And Run this way: python -m src.components.data_ingestion (to avoid ModuleNotFoundError)


from logger import logging
from exception import CustomException
from src.components.data_transformation import DataTransformation

import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass                     # Define variables without __init__ method
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the Data ingestion component")
        
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Data loaded successfully as df in data_ingestion component")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
        
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
        
            logging.info("Initiating train-test split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
        
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )    
            
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    data_transformation_obj = DataTransformation()
    data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)