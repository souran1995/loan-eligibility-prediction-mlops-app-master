from typing import List
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

import glob
import os

# from app import preprocessing
from pipelines.preprocess import preprocess


def make_predictions(data_filepath: str, models_dir: Path) -> np.ndarray:
    """
    data_df (pd.DataFrame): a pandas dataframe with the inference data
    data_filepath (str): the path to the inference data
    :returns (np.ndarray) the model predictions (output of the model.predict() method)
    """
    X = pd.read_excel(data_filepath)
    X.pop('Loan_Status')
    X_preprocessed = preprocess(X, models_dir, False)
    train_model = load_model(models_dir)
    predictions = train_model.predict(X_preprocessed)
    return predictions

def make_dag_predictions(X:pd.DataFrame, models_dir: Path) -> np.ndarray:
    """
    data_df (pd.DataFrame): a pandas dataframe with the inference data
    data_filepath (str): the path to the inference data
    :returns (np.ndarray) the model predictions (output of the model.predict() method)
    """
    if 'Loan_Status' in X.columns:
        X.pop('Loan_Status')
    if 'prediction_type' in X.columns:
        X.pop('prediction_type')

    X_preprocessed = preprocess(X, models_dir, False)
    train_model = load_model(models_dir)
    predictions = train_model.predict(X_preprocessed)

    predictions = predictions.tolist()
    return predictions

def load_model(models_dir: Path):

    #get the latest model object
    # list_of_files = glob.glob('C:/Users/Wakar/loan_prediction_production-master/models/*')
    # latest_file = max(list_of_files, key=os.path.getctime)
    # model_object = os.path.basename(latest_file)

    model_filepath = models_dir / 'model.pkl'
   
    if model_filepath.is_file():
       with open(model_filepath, 'rb') as file:  
           model = pickle.load(file)
           #model = joblib.load(file)
    else:
       raise Exception('model not found')
    
    return model
    

def make_single_prediction(X: dict,  models_dir: Path) -> int:
    df = pd.DataFrame([X])
    if 'Loan_Status' in df.columns:
        df.pop('Loan_Status')
    if 'prediction_type' in df.columns:
        df.pop('prediction_type')
    X_preprocessed = preprocess(df, models_dir, False)
    #train_model = load_model(models_dir / 'model.joblib')
    train_model = load_model(models_dir)
    prediction = train_model.predict(X_preprocessed)
    
    prediction = prediction.tolist()
    return prediction[0]

# if __name__ == '__main__':
#     ROOT_DIR = Path('./')
#     MODELS_DIR = ROOT_DIR / 'models'
#     response = make_predictions('./data/loan_eligibility.xlsx', MODELS_DIR)
#     print(response)