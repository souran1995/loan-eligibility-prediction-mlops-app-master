from datetime import timedelta
from datetime import datetime

import logging
import requests

import pandas as pd

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

import sys
sys.path.append('../')
sys.path.append('../pipelines')
sys.path.append('../data')
sys.path.append('../models')

import glob
import os
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME')

#please replace this url with localhost if not using docker
retrain_url = "http://host.docker.internal:8000/retrain"
drift_url = "http://host.docker.internal:8000/drift"

@dag(
    dag_id="retraining_model",
    description="Detect drift and Retrain the model",
    tags=["retrain_model"],
    default_args={'owner': 'airflow'},
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(n=0, hour=1),
    catchup=False
)
def retrain_model():
    @task
    def detect_drift_in_data_task():
        return detect_drift_in_data()
    
    @task
    def retrain_model_task(dict):
        return retrain_model(dict)


    # Task relationships
    drift_dict = detect_drift_in_data_task()
    if drift_dict['retrain']:
        drift_dict = retrain_model_task(drift_dict)


model_retraining = retrain_model()


#####
def detect_drift_in_data() -> dict:
    path = AIRFLOW_HOME + "/dags/data/output_data/"
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    # loop over the list of csv files
    for f in csv_files:
        
        # read the csv file
        df_for_pred = pd.read_csv(f)

        #preprocess continuous data
        X_preprocessed = df_for_pred.copy()
        continuous_columns = X_preprocessed.select_dtypes(include='number').columns
        for column_name in continuous_columns:
            X_preprocessed[column_name] = X_preprocessed[column_name].fillna(X_preprocessed[column_name].mean())
        df_for_pred =  X_preprocessed 
        #preprocessing finish

        #preprocess categorical data
        X_preprocessed = df_for_pred.copy()
        categorical_columns = list(X_preprocessed.select_dtypes(include='object').columns)
        for column_name in categorical_columns:
            X_preprocessed[column_name] = X_preprocessed[column_name].fillna(X_preprocessed[column_name].mode()[0])
        df_for_pred = X_preprocessed
        #preprocessing finish

        #nd_arr = df_for_pred.to_numpy().tolist()
        #logging.info('List ND', nd_arr)

        #convert to dict
        #df_dict = df_for_pred.to_dict()
        #if 'Loan_Status' in df_for_pred.columns:
        #    df_for_pred.pop('Loan_Status')

        df_for_pred['prediction_type'] = "dag"
        
        df_dict = df_for_pred.to_dict(orient='records')
        
        retrain = requests.post(drift_url, json=df_dict)

        
        dict = {
            "file_path": f,
            "retrain": retrain.json(),
            "df": df_for_pred
        }

        return dict

def retrain_model(dict: dict) -> dict:

    retrain_dict = {
        "file_path": dict['file_path'],
        "retrain": dict['retrain']
    }
    model_path = requests.post(retrain_url, json=retrain_dict)
    pred_dict = {
        "df": dict['df'],
        "retrain": dict['retrain'],
        "model_path": model_path
    }
    return pred_dict
    