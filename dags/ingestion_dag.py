from datetime import timedelta
from datetime import datetime

import logging
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

import sys
sys.path.append('../pipelines')
sys.path.append('../data')

import os
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME')

#import global_vars
#from pipelines.inference import make_predictions

@dag(
    dag_id="ingest_data",
    description="Ingest data from a file to another DAG",
    tags=["loan_pred"],
    default_args={'owner': 'airflow'},
    schedule_interval=timedelta(minutes=3),
    start_date=days_ago(n=0, hour=1),
    catchup=False
)
def ingest_data():
    @task
    def get_data_to_ingest_from_local_file_task():
        return get_data_to_ingest_from_local_file()

    @task
    def save_data_task(data_to_ingest_df):
        save_data(data_to_ingest_df)

    # Task relationships
    data_to_ingest = get_data_to_ingest_from_local_file_task()
    save_data_task(data_to_ingest)


ingest_data_dag = ingest_data()


#####
def get_data_to_ingest_from_local_file() -> pd.DataFrame:
    #data_to_ingest_df = pd.read_excel("../data/loan_eligibility.xlsx", skiprows=global_vars.skip_rows_index)
    input_data_df = pd.read_csv(AIRFLOW_HOME + "/dags/data/loan_eligibility.csv")
    data_to_ingest_df = input_data_df.sample(n=1000)
    #global_vars.skip_rows_index+=1000
    #data_to_ingest_df = input_data_df.head(1000)
    return data_to_ingest_df


def save_data(data_to_ingest_df: pd.DataFrame):
    #X = pd.DataFrame(data_to_ingest_df, index=[0])
    filepath = AIRFLOW_HOME + f"/dags/data/output_data/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    data_to_ingest_df.to_csv(filepath, index=False)