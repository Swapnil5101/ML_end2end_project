import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # And Run this way: python -m src.components.model_trainer (to avoid ModuleNotFoundError)
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models

# Modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor    
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
        
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting the data into X_train, y_train, X_test, y_test")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1], 
                train_arr[:, -1], 
                test_arr[:, :-1], 
                test_arr[:, -1])
            
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNN": KNeighborsRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "XGBoost": XGBRegressor()
            }
            
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, 
                                               y_test=y_test, models=models)
                                            
            # Get best model score and the model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No model achieving r2 score >= 0.6!!")
                
            logging.info("Best model finding task executed successfully.")
            
            save_obj(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            
            best_prediction = best_model.predict(X_test)
            best_r2_score = r2_score(y_test, best_prediction)
            
            print("Model achieving best r2 score is:" + '\033[1m' + f"{best_model_name}" + '\033[0m')
            return (best_prediction, best_r2_score)
        
        except Exception as e:
            raise CustomException(e, sys)