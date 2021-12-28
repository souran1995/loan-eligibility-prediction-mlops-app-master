from typing import List
from fastapi import FastAPI
import uvicorn

import numpy as np
from pydantic import BaseModel
import sys
import psycopg2
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from river.drift import PageHinkley

from decouple import config

ROOT_DIR = Path('./')
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'

from pipelines.inference import make_dag_predictions, make_single_prediction
from pipelines.train_mlflow import train_model

sys.path.insert(0, '../api')
sys.path.insert(0, '../pipelines')
sys.path.insert(0, '../models')

app = FastAPI()

# Routes
@app.get("/")
async def index():
   return {"message": "Hello World"}


class request_body(BaseModel):
    Loan_ID: str
    Customer_ID: str
    #Loan_Status: int 
    Current_Loan_Amount : int
    Term : str
    Credit_Score : float
    Annual_Income : float
    Years_in_current_job : str
    Home_Ownership : str
    Purpose : str
    Monthly_Debt : float
    Years_of_Credit_History : float
    Months_since_last_delinquent: float
    Number_of_Open_Accounts : int
    Number_of_Credit_Problems : int
    Current_Credit_Balance : int
    Maximum_Open_Credit : float
    Bankruptcies : float
    Tax_Liens : float
    prediction_type: str
    Loan_Status: str


class retrain_model(BaseModel):
    file_path: str
    retrain : int
    

    #class Config:
	#    orm_mode=True

@app.post("/retrain")
async def retraining_model(data: retrain_model) -> Path:
   
   received = data.dict()

   training_data_filepath = received['file_path']
   retrain = received['retrain']

   dict = train_model(training_data_filepath, MODELS_DIR, retrain)
   return dict['model_path']

@app.post("/drift")
async def drift_detection(data: List[request_body]) -> int:
   
   first = data[0]
   received = first.dict()
   df = pd.DataFrame([received])
   dict = {}

   for index, row  in enumerate(data):
      row_data = row.dict()
      if(index!=0):
         df2 = pd.DataFrame([row_data])
         df = df.append(df2, ignore_index=True)

   retrain = 0
   #preprocess for drift
   df=df.apply(LabelEncoder().fit_transform)

   #detecting drift
   df_salary_low=df[df['Annual_Income']<=500000] 
   df_salary_high=df[df['Annual_Income']>500000]
   counter = 0
   
   np.random.seed(100)
   ph = PageHinkley(threshold=10 ,min_instances=100)
   for col in df.columns:
      data_stream=[]
      a = np.array(df_salary_low[col])
      b = np.array(df_salary_high[col])
      data_stream = np.concatenate((a,b))

      
      for i, val in enumerate(data_stream):
            in_drift, in_warning = ph.update(val)
            if in_drift:
               #print(f"Data Drift detected for column: {col}")
               counter = counter + 1
            #else:
               #print("No Data Drift")
   if counter >= 2:
      retrain = counter
   
   return retrain

@app.post("/dag_predictions")
async def dag_predictions(data: List[request_body]) -> np.ndarray:
   
   first = data[0]
   received = first.dict()
   df = pd.DataFrame([received])
   dict = {}

   for index, row  in enumerate(data):
      row_data = row.dict()
      if(index!=0):
         df2 = pd.DataFrame([row_data])
         df = df.append(df2, ignore_index=True)
   
   predictions = make_dag_predictions(df, MODELS_DIR)
   #print(predictions)
   return predictions


@app.post("/save_predictions")
async def save_db_predictions(data: List[request_body]) -> bool:

   #create a df out of data
   first = data[0]
   received = first.dict()
   df = pd.DataFrame([received])
   dict = {}

   for index, row  in enumerate(data):
      row_data = row.dict()
      if(index!=0):
         df2 = pd.DataFrame([row_data])
         df = df.append(df2, ignore_index=True)

   # Create a list of tupples from the dataframe values
   tuples = [tuple(x) for x in df.to_numpy()]
   # Comma-separated dataframe columns
   cols = ','.join(list(df.columns))
   
   # SQL quert to execute
   #query  = "INSERT INTO %s(%s) VALUES(%%s,%%s,%%s)" % ('future_value', cols)
   query = "INSERT INTO future_value VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

   connection = psycopg2.connect(user=config('PG_USER'),
                              password=config('PG_PWD'),
                              host=config('PG_HOST'),
                              database=config('PG_DB'))
   # Create a cursor
   cusrsor = connection.cursor()

   execution=-1
   try:
      cusrsor.executemany(query, tuples)
      connection.commit()
      execution=0
   except (Exception, psycopg2.DatabaseError) as error:
      print("Error: %s" % error)
      connection.rollback()
      cusrsor.close()
      #return 1
      execution=1
   print("execute_many() done")
   cusrsor.close()

   return execution

@app.post("/predict")
async def prediction(data: request_body):
   
    received = data.dict()

    prediction = make_single_prediction(received, MODELS_DIR)

    #prediction = dict(enumerate(prediction.flatten(), 1))

    Loan_Status = ""
    if(prediction==1):
       Loan_Status="Fully Paid"
    else:
       Loan_Status="Charged Off"

    Loan_ID = received['Loan_ID']
    Customer_ID = received['Customer_ID']
    Current_Loan_Amount = received['Current_Loan_Amount']
    Term = received['Term']
    Credit_Score = received['Credit_Score']
    Annual_Income = received['Annual_Income']
    Years_in_current_job = received['Years_in_current_job']
    Home_Ownership = received['Home_Ownership']
    Purpose = received['Purpose']
    Monthly_Debt = received['Monthly_Debt']
    Years_of_Credit_History = received['Years_of_Credit_History']
    Months_since_last_delinquent = received['Months_since_last_delinquent']
    Number_of_Open_Accounts = received['Number_of_Open_Accounts']
    Number_of_Credit_Problems = received['Number_of_Credit_Problems']
    Current_Credit_Balance = received['Current_Credit_Balance']
    Maximum_Open_Credit = received['Maximum_Open_Credit']
    Bankruptcies = received['Bankruptcies']
    Tax_Liens = received['Tax_Liens']
    prediction_type = received['prediction_type']
    
    connection = psycopg2.connect(user=config('PG_USER'),
                                password=config('PG_PWD'),
                                host=config('PG_HOST'),
                                database=config('PG_DB'))
    # Create a cursor
    cusrsor = connection.cursor()

    # Define the query
    #select_query = "select * from future_value"
    insert_query = "INSERT INTO future_value VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    #data_list = list(result.to_records())
    db_values = [Loan_ID, Customer_ID, Current_Loan_Amount, Term, Credit_Score, Annual_Income, Years_in_current_job, Home_Ownership,
    Purpose, Monthly_Debt, Years_of_Credit_History, Months_since_last_delinquent, Number_of_Open_Accounts, Number_of_Credit_Problems,
    Current_Credit_Balance, Maximum_Open_Credit, Bankruptcies, Tax_Liens, Loan_Status, prediction_type]
    # Perform the query
    cusrsor.execute(insert_query, db_values)
    cusrsor.close()
    connection.commit()
    
    
    #return {'prediction' : prediction}
    return Loan_Status


if __name__ == '__main__':
   uvicorn.run(app, host='127.0.0.1', port=8000)
