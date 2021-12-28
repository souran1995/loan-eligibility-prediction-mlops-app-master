from datetime import timedelta
from datetime import datetime

import logging
from pathlib import Path
from typing import Dict, List
import requests
import psycopg2

import pandas as pd
import numpy as np

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
url = "http://host.docker.internal:8000/dag_predictions"
db_url = "http://host.docker.internal:8000/save_predictions"

@dag(
    dag_id="make_prediction",
    description="Make prediction on the Ingested data by the other DAG",
    tags=["make_prediction"],
    default_args={'owner': 'airflow'},
    schedule_interval=timedelta(minutes=3),
    start_date=days_ago(n=0, hour=1),
    catchup=False
)
def make_prediction():

    @task
    def make_pred_from_ingested_data_task():
        return make_pred_from_ingested_data()

    @task
    def save_predictions_task(predictions):
        save_predictions(predictions)


    # Task relationships
    pred_dict = make_pred_from_ingested_data_task()
    save_predictions_task(pred_dict)


make_pred_dag = make_prediction()


#####
def make_pred_from_ingested_data() -> dict:

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

        predictions = requests.post(url, json=df_dict)
    
        # logging.info('Json Dict', json.dumps(df_dict[0]))
        # logging.info('DF Full Dictionary', json.dumps(df_dict))
        # logging.info('Predictions',predictions.json())

        dict ={
            "predictions": predictions.json(),
            "df": df_for_pred
        }
        return dict



def save_predictions(dict: dict):

    predictions = dict["predictions"]
    df = dict["df"]
    
    if 'Loan_Status' not in df.columns:
        df['Loan_Status'] = predictions
        df['Loan_Status'] = df['Loan_Status'].map(
                   {1:'Fully Paid' ,0:'Charged Off'})

        df['prediction_type'] = "dag"

    df_dict = df.to_dict(orient='records')

    res = requests.post(db_url, json=df_dict)

    return res
    